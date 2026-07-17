"""
Task-aware model routing — routes through Cerebras primary, Groq fallback.
"""
import logging
from brain.providers import AIProvider
from config.settings import TASK_ROUTING

logger = logging.getLogger(__name__)


class TaskRouter:
    def __init__(self, provider: AIProvider):
        self.provider = provider

    def route(self, task_type: str, prompt: str, system_prompt: str = "",
              priority: str = "medium") -> dict:
        routes = TASK_ROUTING.get(task_type, ["cerebras:fast", "groq:fast"])
        for route in routes:
            provider_name, model_key = route.split(":")
            result = self.provider.call(
                model_key, prompt, system_prompt, priority,
                provider_name=provider_name,
            )
            if result["success"]:
                return result
            logger.warning(f"Route {route} failed for {task_type}, trying next...")
        return {"success": False, "error": "All routes failed", "content": ""}

    def route_json(self, task_type: str, prompt: str, system_prompt: str = "",
                   priority: str = "medium") -> dict:
        routes = TASK_ROUTING.get(task_type, ["cerebras:fast", "groq:fast"])
        for route in routes:
            provider_name, model_key = route.split(":")
            result = self.provider.call_json(
                model_key, prompt, system_prompt, priority,
                provider_name=provider_name,
            )
            if result["success"] and result.get("parsed"):
                return result
            logger.warning(f"Route {route} failed for {task_type}, trying next...")
        return {"success": False, "error": "All routes failed", "content": "", "parsed": None}
