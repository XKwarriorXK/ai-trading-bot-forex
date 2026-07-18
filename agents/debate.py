"""
AI Trade Review Brain — institutional-grade trade validation.
Only called on A+ setups (85%+ confidence). Three-step process:
1. Bull builds the case WITH specific data points
2. Bear stress-tests for hidden risks
3. Judge evaluates against institutional checklist
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
        categories = tech_result.get("categories", [])
        num_categories = tech_result.get("num_categories", 0)
        strategies = tech_result.get("agreeing_strategies", [])
        grade = tech_result.get("grade", "?")
        structure = tech_result.get("structure", {})
        session = tech_result.get("session", {})
        news = tech_result.get("news", {})

        rsi = indicators.get("rsi", "N/A")
        adx = indicators.get("adx", "N/A")
        atr = indicators.get("atr", "N/A")
        macd_hist = indicators.get("macd_histogram", "N/A")
        stoch_k = indicators.get("stoch_k", "N/A")
        price = indicators.get("price", "N/A")
        bb_upper = indicators.get("bb_upper", "N/A")
        bb_lower = indicators.get("bb_lower", "N/A")

        struct_bias = structure.get("bias", "unknown")
        struct_pattern = structure.get("pattern", "unknown")
        support = structure.get("support", "N/A")
        resistance = structure.get("resistance", "N/A")

        active_sessions = session if isinstance(session, list) else []
        news_events = news.get("events", [])
        news_risk = news.get("risk_level", "low")

        context = (
            f"=== A+ TRADE REVIEW REQUEST ===\n"
            f"Instrument: {instrument}\n"
            f"Signal: {signal} | Grade: {grade} | Confidence: {confidence:.0%}\n"
            f"Regime: {regime}\n\n"
            f"=== CONFLUENCE BREAKDOWN ===\n"
            f"Strategies agreeing: {len(strategies)} ({', '.join(strategies)})\n"
            f"Independent categories: {num_categories} ({', '.join(categories)})\n"
            f"Strategy reasons: {', '.join(reasons[:10])}\n\n"
            f"=== INDICATORS ===\n"
            f"Price: {price} | RSI: {rsi} | ADX: {adx} | MACD Hist: {macd_hist}\n"
            f"ATR: {atr} | Stochastic K: {stoch_k}\n"
            f"Bollinger Upper: {bb_upper} | Lower: {bb_lower}\n\n"
            f"=== MARKET STRUCTURE ===\n"
            f"Bias: {struct_bias} | Pattern: {struct_pattern}\n"
            f"Support: {support} | Resistance: {resistance}\n\n"
            f"=== SESSION & NEWS ===\n"
            f"Active sessions: {', '.join(active_sessions) if active_sessions else 'unknown'}\n"
            f"News risk: {news_risk}\n"
            f"Upcoming events: {[e.get('event', '?') for e in news_events] if news_events else 'None'}\n"
        )

        bull_result = self.router.route(
            "debate",
            f"You are an institutional forex analyst building the BULL case for this trade.\n\n"
            f"{context}\n\n"
            f"Analyze these specific points:\n"
            f"1. Does the {regime} regime support a {signal} on {instrument}?\n"
            f"2. Are the {num_categories} independent confluence categories sufficient?\n"
            f"3. Do the indicators confirm (RSI not overextended for entry direction, ADX showing trend strength)?\n"
            f"4. Is market structure aligned with the signal direction?\n"
            f"5. What is the realistic risk-to-reward based on support/resistance levels?\n\n"
            f"Use SPECIFIC numbers from the data. No vague statements. Max 120 words.",
            system_prompt="You are an institutional forex analyst. Only cite specific indicator values and levels. No fluff.",
        )

        bear_result = self.router.route(
            "debate",
            f"You are a risk manager stress-testing this trade for hidden problems.\n\n"
            f"{context}\n\n"
            f"Check for these SPECIFIC dealbreakers:\n"
            f"1. Is RSI overextended for the signal direction? (BUY with RSI>70 or SELL with RSI<30 = bad)\n"
            f"2. Is ADX below 20? (no trend = breakout likely to fail)\n"
            f"3. Does market structure CONTRADICT the signal? ({signal} against {struct_bias} bias)\n"
            f"4. Are we near a major news event that could reverse the move?\n"
            f"5. Is the signal chasing a move that already happened? (entry far from support/resistance)\n"
            f"6. Is ATR showing declining volatility that could kill momentum?\n\n"
            f"Only flag problems backed by SPECIFIC numbers. Max 120 words.",
            system_prompt="You are a forex risk manager. Find real problems with data, not vague concerns. If it's clean, say so.",
        )

        bull_case = bull_result.get("content", "No argument") if bull_result["success"] else "Bull unavailable"
        bear_case = bear_result.get("content", "No argument") if bear_result["success"] else "Bear unavailable"

        judge_result = self.router.route_json(
            "trade_decision",
            f"You are the HEAD TRADER making the final call on this A+ setup.\n\n"
            f"TRADE: {signal} {instrument} | Grade {grade} | {confidence:.0%} confidence\n"
            f"Regime: {regime} | {num_categories} independent confluence categories\n\n"
            f"BULL CASE:\n{bull_case}\n\n"
            f"BEAR CASE:\n{bear_case}\n\n"
            f"DECISION CRITERIA — reject if ANY are true:\n"
            f"- RSI overextended against signal direction\n"
            f"- Market structure contradicts signal (e.g. BUY in bearish structure)\n"
            f"- Major news event within next 2 hours\n"
            f"- Bear identified a specific data-backed problem\n"
            f"- Entry is chasing (price already moved significantly)\n\n"
            f"If none of those apply and the bull case has specific supporting data, APPROVE.\n\n"
            f"Respond in JSON:\n"
            f'{{"verdict": "BUY" or "SELL" or "SKIP", '
            f'"adjusted_confidence": 0.0 to 0.95, '
            f'"reasoning": "one sentence with specific reason"}}',
            system_prompt=(
                "You are a head trader at a prop firm. Your job is to protect capital. "
                "Approve trades that have data-backed confluence. Reject trades where the "
                "bear found a specific problem. Do NOT reject based on vague uncertainty — "
                "only reject for concrete reasons. Adjust confidence up if bull case is "
                "strong with clean structure, down if there are minor concerns."
            ),
        )

        if judge_result["success"] and judge_result.get("parsed"):
            parsed = judge_result["parsed"]
            verdict = parsed.get("verdict", "SKIP")
            adj_conf = parsed.get("adjusted_confidence", confidence * 0.8)
            reasoning = parsed.get("reasoning", "")

            logger.info(
                f"AI BRAIN | {instrument} | Verdict: {verdict} | "
                f"Adjusted conf: {adj_conf:.0%} | Reason: {reasoning}"
            )

            return {
                "verdict": verdict,
                "adjusted_confidence": adj_conf,
                "reasoning": reasoning,
                "bull_case": bull_case[:300],
                "bear_case": bear_case[:300],
            }

        logger.warning(f"AI BRAIN | {instrument} | Judge failed — applying confidence haircut")
        return {
            "verdict": signal,
            "adjusted_confidence": confidence * 0.75,
            "reasoning": "AI judge unavailable — original signal with 25% confidence haircut",
            "bull_case": bull_case[:300],
            "bear_case": bear_case[:300],
        }
