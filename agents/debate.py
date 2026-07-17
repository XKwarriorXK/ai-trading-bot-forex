"""
Adversarial Bull/Bear Debate Agent for forex.
"""
import logging
from brain.router import TaskRouter

logger = logging.getLogger(__name__)


class DebateAgent:
    def __init__(self, router: TaskRouter):
        self.router = router

    def debate(self, instrument: str, tech_result: dict) -> dict:
        signal = tech_result.get("signal", "SKIP")
        confidence = tech_result.get("confidence", 0)
        regime = tech_result.get("regime", "unknown")
        indicators = tech_result.get("indicators", {})
        reasons = tech_result.get("reasons", [])

        context = (
            f"Instrument: {instrument} | Signal: {signal} | Confidence: {confidence:.0%} | Regime: {regime}\n"
            f"Price: {indicators.get('price', 'N/A')} | RSI: {indicators.get('rsi', 'N/A')} | "
            f"ADX: {indicators.get('adx', 'N/A')} | MACD: {indicators.get('macd_histogram', 'N/A')}\n"
            f"ATR: {indicators.get('atr', 'N/A')} | Stoch: {indicators.get('stoch_k', 'N/A')}\n"
            f"Reasons: {', '.join(reasons)}"
        )

        bull_result = self.router.route(
            "debate",
            f"You are the BULL. Argue why this forex trade SHOULD be taken.\n\n{context}\n\n"
            f"Consider session timing, trend strength, support/resistance, and risk/reward. Max 100 words.",
            system_prompt="You are a forex trading bull. Be persuasive but data-driven.",
        )

        bear_result = self.router.route(
            "debate",
            f"You are the BEAR. Argue why this forex trade should NOT be taken.\n\n{context}\n\n"
            f"Consider session overlap risks, false breakouts, news events, and correlation exposure. Max 100 words.",
            system_prompt="You are a forex trading bear. Be skeptical and data-driven.",
        )

        bull_case = bull_result.get("content", "No argument") if bull_result["success"] else "Bull failed"
        bear_case = bear_result.get("content", "No argument") if bear_result["success"] else "Bear failed"

        judge_result = self.router.route_json(
            "trade_decision",
            f"Two agents debated this forex trade:\n\n"
            f"BULL: {bull_case}\n\nBEAR: {bear_case}\n\n"
            f"Original signal: {signal} at {confidence:.0%} confidence.\n"
            f"This signal required 2+ technical indicators to agree.\n\n"
            f"Who wins? Respond in JSON:\n"
            f'{{"verdict": "BUY" or "SELL" or "SKIP", '
            f'"adjusted_confidence": 0.0 to 1.0, '
            f'"reasoning": "one sentence"}}',
            system_prompt="You are a forex trading judge. Approve trades unless the bear "
                         "identifies a specific dealbreaker. General uncertainty is not "
                         "enough to reject.",
        )

        if judge_result["success"] and judge_result.get("parsed"):
            parsed = judge_result["parsed"]
            return {
                "verdict": parsed.get("verdict", "SKIP"),
                "adjusted_confidence": parsed.get("adjusted_confidence", confidence * 0.8),
                "reasoning": parsed.get("reasoning", ""),
                "bull_case": bull_case[:200],
                "bear_case": bear_case[:200],
            }

        return {
            "verdict": signal,
            "adjusted_confidence": confidence * 0.75,
            "reasoning": "AI unavailable — original signal with confidence haircut",
            "bull_case": bull_case[:200],
            "bear_case": bear_case[:200],
        }
