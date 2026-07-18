"""
Trade Pipeline — Full institutional-grade analysis flow.
Spread → Session → Multi-TF → Market Structure → Strategy Ensemble →
News Risk → AI Debate → Risk Manager → Execute
"""
import logging
from config.settings import INSTRUMENTS, RISK

logger = logging.getLogger(__name__)


class TradePipeline:
    def __init__(self, technical, strategy_selector, debate, risk,
                 session_filter, spread_filter, news_agent,
                 market_structure, multi_tf, journal, router=None,
                 entry_sniper=None):
        self.technical = technical
        self.strategy_selector = strategy_selector
        self.debate = debate
        self.risk = risk
        self.session = session_filter
        self.spread = spread_filter
        self.news = news_agent
        self.structure = market_structure
        self.multi_tf = multi_tf
        self.journal = journal
        self.router = router
        self.entry_sniper = entry_sniper

    def evaluate(self, df, instrument: str, price_data: dict = None,
                 mode: str = "full") -> dict:
        result = {
            "instrument": instrument,
            "final_decision": "SKIP",
            "confidence": 0,
            "units": 0,
        }

        # 1. SPREAD FILTER — don't trade with bad spreads
        if price_data and self.spread:
            spread = price_data.get("spread", 0)
            spread_check = self.spread.check(instrument, spread)
            if not spread_check.get("tradeable", True):
                result["reason"] = f"Spread: {spread_check.get('reason')}"
                self._log_skip(instrument, result["reason"])
                return result

        # 2. SESSION FILTER — are we in a good trading window?
        bar_timestamp = price_data.get("timestamp") if price_data else None
        if self.session:
            session_check = self.session.check(instrument, timestamp=bar_timestamp)
            if not session_check.get("tradeable", True):
                result["reason"] = f"Session: {session_check.get('reason')}"
                self._log_skip(instrument, result["reason"])
                return result
            result["session"] = session_check.get("sessions", [])

        # 3. TECHNICAL ANALYSIS — indicators + regime detection
        tech = self.technical.analyze(df, instrument)
        regime = tech.get("regime", "unknown")

        # 4. MARKET STRUCTURE — HH/HL/LL/LH, support/resistance
        structure = {}
        if self.structure:
            structure = self.structure.analyze(df)
            result["structure"] = structure

        # 5. MULTI-TIMEFRAME — check higher TF alignment
        mtf = {}
        if self.multi_tf and mode == "full":
            try:
                mtf = self.multi_tf.analyze(instrument)
                result["multi_tf"] = mtf
            except Exception as e:
                logger.warning(f"Multi-TF failed for {instrument}: {e}")

        # 6. STRATEGY ENSEMBLE — 7 proven strategies vote
        if self.strategy_selector:
            ensemble = self.strategy_selector.evaluate(df, regime)
        else:
            ensemble = {"signal": tech.get("signal", "SKIP"),
                       "confidence": tech.get("confidence", 0),
                       "reasons": tech.get("reasons", [])}

        if ensemble["signal"] == "SKIP":
            result["reason"] = ensemble.get("reason", "No strategy agreement")
            self._log_skip(instrument, result["reason"])
            return result

        confidence = ensemble["confidence"]

        # Apply multi-TF modifier
        if mtf:
            modifier = mtf.get("confidence_modifier", 0)
            alignment = mtf.get("alignment", "mixed")
            if alignment != "mixed":
                if (alignment == "bullish" and ensemble["signal"] == "SELL") or \
                   (alignment == "bearish" and ensemble["signal"] == "BUY"):
                    result["reason"] = f"Signal conflicts with {alignment} multi-TF bias"
                    self._log_skip(instrument, result["reason"])
                    return result
            confidence += modifier

        # Apply session overlap boost
        if self.session:
            confidence += session_check.get("confidence_boost", 0)

        # Apply structure bias check
        if structure:
            struct_bias = structure.get("bias", "neutral")
            if struct_bias != "neutral":
                if (struct_bias == "bullish" and ensemble["signal"] == "SELL") or \
                   (struct_bias == "bearish" and ensemble["signal"] == "BUY"):
                    confidence -= 0.10

        # 7. NEWS RISK — reduce confidence near high-impact events
        news_risk = {}
        if self.news:
            news_risk = self.news.check_risk(instrument, timestamp=bar_timestamp)
            result["news"] = news_risk
            confidence += news_risk.get("confidence_modifier", 0)

        # 8. AI DEBATE — only on A+ setups (85%+ confidence)
        #    The brain reviews the best trades, not every signal
        AI_DEBATE_THRESHOLD = 0.85
        if self.debate and confidence >= AI_DEBATE_THRESHOLD:
            logger.info(f"AI DEBATE | {instrument} | Conf {confidence:.0%} >= {AI_DEBATE_THRESHOLD:.0%} — requesting brain review")

            price = price_data.get("bid", 0) if price_data else 0
            spread = price_data.get("spread", 0) if price_data else 0
            atr = tech["indicators"].get("atr", 0)
            spec = INSTRUMENTS.get(instrument, {})
            pip_value = 10 ** spec.get("pip_location", -4)
            pre_sl_pips = max(atr / pip_value * 2.0, 15) if atr > 0 else RISK["default_stop_loss_pips"]
            pre_tp_pips = pre_sl_pips * (2.5 if regime == "trending" else 2.0)
            if price > 0:
                if ensemble["signal"] == "BUY":
                    pre_sl_price = round(price - (pre_sl_pips * pip_value), 5)
                    pre_tp_price = round(price + (pre_tp_pips * pip_value), 5)
                else:
                    pre_sl_price = round(price + (pre_sl_pips * pip_value), 5)
                    pre_tp_price = round(price - (pre_tp_pips * pip_value), 5)
            else:
                pre_sl_price = "N/A"
                pre_tp_price = "N/A"

            open_trades = len(self.risk.open_instruments) if self.risk else 0

            debate_result = self.debate.debate(instrument, {
                "signal": ensemble["signal"],
                "confidence": confidence,
                "regime": regime,
                "indicators": tech.get("indicators", {}),
                "reasons": ensemble.get("reasons", []),
                "categories": ensemble.get("categories", []),
                "num_categories": ensemble.get("num_categories", 0),
                "agreeing_strategies": ensemble.get("agreeing_strategies", []),
                "grade": ensemble.get("grade", "?"),
                "structure": structure,
                "session": result.get("session", []),
                "news": news_risk,
                "stop_loss_pips": round(pre_sl_pips, 1),
                "take_profit_pips": round(pre_tp_pips, 1),
                "stop_loss_price": pre_sl_price,
                "take_profit_price": pre_tp_price,
                "spread": round(spread / pip_value, 1) if pip_value else 0,
                "open_trades": open_trades,
            })
            verdict = debate_result.get("verdict", "SKIP")
            if verdict == "SKIP":
                result["reason"] = f"Debate rejected: {debate_result.get('reasoning', '')}"
                self._log_skip(instrument, result["reason"])
                return result
            confidence = debate_result.get("adjusted_confidence", confidence)
            result["debate"] = debate_result
        elif self.debate and confidence < AI_DEBATE_THRESHOLD:
            logger.info(f"SKIP DEBATE | {instrument} | Conf {confidence:.0%} < {AI_DEBATE_THRESHOLD:.0%} — not A+ grade")

        # Clamp confidence
        confidence = max(0, min(confidence, 0.95))

        grade = ensemble.get("grade", "C")
        if grade in ("C", "B"):
            result["reason"] = f"Grade {grade} ({confidence:.0%}) — only A and A+ setups trade"
            self._log_skip(instrument, result["reason"])
            return result

        # 9. RISK CHECK — position sizing + limits
        price = price_data.get("bid", 0) if price_data else 0
        atr = tech["indicators"].get("atr", 0)
        risk_check = self.risk.check_trade(
            instrument, ensemble["signal"], confidence,
            atr=atr, price=price, regime=regime,
        )
        if not risk_check["approved"]:
            result["reason"] = f"Risk: {risk_check['reason']}"
            self._log_skip(instrument, result["reason"])
            return result

        # 10. ENTRY SNIPER — drop to M15 for precise entry (live only)
        if self.entry_sniper:
            sniper = self.entry_sniper.snipe_entry(
                instrument, ensemble["signal"],
                h1_price=price, h1_atr=atr,
            )
            result["sniper"] = sniper
            if not sniper["confirmed"]:
                result["reason"] = f"Sniper: {sniper.get('reject_reason', 'M15 not confirmed')}"
                self._log_skip(instrument, result["reason"])
                return result
            logger.info(
                f"SNIPER | {instrument} | {ensemble['signal']} confirmed | "
                f"M15 score: {sniper['score']}/{sniper['max_score']} | "
                f"{', '.join(sniper.get('reasons', []))}"
            )

        # News size reduction
        if news_risk.get("size_reduction_pct"):
            reduction = news_risk["size_reduction_pct"] / 100
            risk_check["units"] = int(risk_check["units"] * (1 - reduction))

        # BUILD FINAL RESULT
        result["final_decision"] = ensemble["signal"]
        result["confidence"] = round(confidence, 4)
        result["units"] = risk_check["units"]
        result["stop_loss_price"] = risk_check.get("stop_loss_price")
        result["take_profit_price"] = risk_check.get("take_profit_price")
        result["stop_loss_pips"] = risk_check.get("stop_loss_pips")
        result["take_profit_pips"] = risk_check.get("take_profit_pips")
        result["risk_amount"] = risk_check.get("risk_amount")
        result["regime"] = regime
        result["reasons"] = ensemble.get("reasons", [])
        result["strategies"] = ensemble.get("agreeing_strategies", [])
        result["indicators"] = tech.get("indicators", {})

        # Log signal
        if self.journal:
            self.journal.log_signal(
                instrument, ensemble["signal"], confidence,
                regime, ensemble.get("reasons", []), executed=True,
            )

        logger.info(
            f"SIGNAL | {instrument} | {ensemble['signal']} | "
            f"Conf: {confidence:.0%} | Regime: {regime} | "
            f"Units: {risk_check['units']} | "
            f"Strategies: {ensemble.get('agreeing_strategies', [])} | "
            f"SL: {risk_check.get('stop_loss_pips')} pips | "
            f"TP: {risk_check.get('take_profit_pips')} pips"
        )
        return result

    def _log_skip(self, instrument, reason):
        logger.info(f"SKIP | {instrument} | {reason}")
        if self.journal:
            self.journal.log_signal(
                instrument, "SKIP", 0, "", [],
                executed=False, skip_reason=reason,
            )
