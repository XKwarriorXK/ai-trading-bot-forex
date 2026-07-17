"""
Trade Pipeline — orchestrates the full analysis flow for each instrument.
Technical → Session Filter → AI Brain → Debate → Risk Manager
"""
import logging

logger = logging.getLogger(__name__)


class TradePipeline:
    def __init__(self, technical, debate, risk, session_filter, router=None):
        self.technical = technical
        self.debate = debate
        self.risk = risk
        self.session = session_filter
        self.router = router

    def evaluate(self, df, instrument: str, price: float = 0,
                 mode: str = "full") -> dict:
        result = {
            "instrument": instrument,
            "final_decision": "SKIP",
            "confidence": 0,
            "units": 0,
        }

        # 1. Technical Analysis
        tech = self.technical.analyze(df, instrument)
        if tech["signal"] == "SKIP":
            result["reason"] = f"Technical: {tech.get('reason', 'No agreement')}"
            logger.info(f"SKIP | {instrument} | {result['reason']}")
            return result

        # 2. Session Filter
        session = self.session.check(instrument)
        if not session.get("tradeable", False):
            result["reason"] = f"Session: {session.get('reason', 'No active session')}"
            logger.info(f"SKIP | {instrument} | {result['reason']}")
            return result

        confidence = tech["confidence"]
        confidence += session.get("confidence_boost", 0)

        # 3. AI Debate (full mode only)
        if mode == "full" and self.debate:
            debate_result = self.debate.debate(instrument, tech)
            verdict = debate_result.get("verdict", "SKIP")
            if verdict == "SKIP":
                result["reason"] = f"Debate rejected: {debate_result.get('reasoning', '')}"
                logger.info(f"SKIP | {instrument} | {result['reason']}")
                return result
            confidence = debate_result.get("adjusted_confidence", confidence)
            result["debate"] = debate_result

        # 4. Risk Check
        atr = tech["indicators"].get("atr", 0)
        risk_check = self.risk.check_trade(
            instrument, tech["signal"], confidence,
            atr=atr, price=price,
        )
        if not risk_check["approved"]:
            result["reason"] = f"Risk: {risk_check['reason']}"
            logger.info(f"SKIP | {instrument} | {result['reason']}")
            return result

        result["final_decision"] = tech["signal"]
        result["confidence"] = round(confidence, 4)
        result["units"] = risk_check["units"]
        result["stop_loss_price"] = risk_check.get("stop_loss_price")
        result["take_profit_price"] = risk_check.get("take_profit_price")
        result["stop_loss_pips"] = risk_check.get("stop_loss_pips")
        result["take_profit_pips"] = risk_check.get("take_profit_pips")
        result["risk_amount"] = risk_check.get("risk_amount")
        result["regime"] = tech.get("regime")
        result["reasons"] = tech.get("reasons", [])
        result["indicators"] = tech.get("indicators", {})

        logger.info(
            f"SIGNAL | {instrument} | {tech['signal']} | Conf: {confidence:.0%} | "
            f"Units: {risk_check['units']} | SL: {risk_check.get('stop_loss_pips')} pips | "
            f"TP: {risk_check.get('take_profit_pips')} pips"
        )
        return result
