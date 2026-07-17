"""
AI Provider — Multi-provider support (Groq + Cerebras) with circuit breakers.
"""
import json
import time
import logging
from config.settings import AI_PROVIDERS, CIRCUIT_BREAKER, TOKEN_BUDGET

logger = logging.getLogger(__name__)


class CircuitBreaker:
    def __init__(self):
        self.failures = 0
        self.max_failures = CIRCUIT_BREAKER["max_failures"]
        self.cooldown = CIRCUIT_BREAKER["cooldown_seconds"]
        self.last_failure_time = 0
        self.state = "closed"

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.max_failures:
            self.state = "open"
            logger.warning(f"Circuit breaker OPEN — {self.failures} failures")

    def record_success(self):
        self.failures = 0
        self.state = "closed"

    def can_proceed(self) -> bool:
        if self.state == "closed":
            return True
        if time.time() - self.last_failure_time > self.cooldown:
            self.state = "half-open"
            return True
        return False


class TokenBudget:
    def __init__(self):
        self.tokens_used = 0
        self.hour_start = time.time()
        self.max_per_hour = TOKEN_BUDGET["max_per_hour"]

    def _reset_if_new_hour(self):
        if time.time() - self.hour_start > 3600:
            self.tokens_used = 0
            self.hour_start = time.time()

    def can_spend(self, priority: str = "medium") -> bool:
        self._reset_if_new_hour()
        threshold = TOKEN_BUDGET["priority_thresholds"].get(priority, 0.80)
        return self.tokens_used < self.max_per_hour * threshold

    def record_usage(self, tokens: int):
        self._reset_if_new_hour()
        self.tokens_used += tokens


class AIProvider:
    def __init__(self):
        self.clients = {}
        self.models = {}
        self.circuit_breaker = CircuitBreaker()
        self.token_budget = TokenBudget()
        self._init_groq()
        self._init_cerebras()

    def _init_groq(self):
        config = AI_PROVIDERS.get("groq", {})
        if config.get("api_key"):
            try:
                from groq import Groq
                self.clients["groq"] = Groq(api_key=config["api_key"])
                self.models["groq"] = config["models"]
                logger.info("Groq provider initialized")
            except ImportError:
                logger.warning("groq package not installed")

    def _init_cerebras(self):
        config = AI_PROVIDERS.get("cerebras", {})
        if config.get("api_key"):
            try:
                from openai import OpenAI
                self.clients["cerebras"] = OpenAI(
                    api_key=config["api_key"],
                    base_url=config["base_url"],
                )
                self.models["cerebras"] = config["models"]
                logger.info("Cerebras provider initialized")
            except ImportError:
                logger.warning("openai package not installed")

    def call(self, model_key: str, prompt: str, system_prompt: str = "",
             priority: str = "medium", max_tokens: int = 1024,
             provider_name: str = "groq") -> dict:
        client = self.clients.get(provider_name)
        if not client:
            return {"success": False, "error": f"Provider {provider_name} not configured", "content": ""}
        if not self.circuit_breaker.can_proceed():
            return {"success": False, "error": "Circuit breaker open", "content": ""}
        if not self.token_budget.can_spend(priority):
            return {"success": False, "error": "Token budget exceeded", "content": ""}

        provider_models = self.models.get(provider_name, {})
        model = provider_models.get(model_key, provider_models.get("fast", model_key))
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=0.3,
            )
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            self.token_budget.record_usage(tokens_used)
            self.circuit_breaker.record_success()
            return {"success": True, "content": content, "tokens": tokens_used,
                    "model": model, "provider": provider_name}
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate" in error_str.lower()
            if is_rate_limit:
                logger.warning(f"Rate limited ({provider_name}:{model}), retrying...")
                time.sleep(3)
                try:
                    response = client.chat.completions.create(
                        model=model, messages=messages, max_tokens=max_tokens, temperature=0.3,
                    )
                    content = response.choices[0].message.content
                    tokens_used = response.usage.total_tokens if response.usage else 0
                    self.token_budget.record_usage(tokens_used)
                    self.circuit_breaker.record_success()
                    return {"success": True, "content": content, "tokens": tokens_used,
                            "model": model, "provider": provider_name}
                except Exception:
                    pass
            else:
                self.circuit_breaker.record_failure()
            logger.error(f"AI call failed ({provider_name}:{model}): {e}")
            return {"success": False, "error": error_str, "content": ""}

    def call_json(self, model_key: str, prompt: str, system_prompt: str = "",
                  priority: str = "medium", provider_name: str = "groq") -> dict:
        result = self.call(model_key, prompt, system_prompt, priority, provider_name=provider_name)
        if not result["success"]:
            return {**result, "parsed": None}
        try:
            text = result["content"]
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                return {**result, "parsed": parsed}
            return {**result, "parsed": None, "error": "No JSON found"}
        except json.JSONDecodeError as e:
            return {**result, "parsed": None, "error": f"JSON parse error: {e}"}
