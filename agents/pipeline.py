"""
Trade Pipeline — Full institutional-grade analysis flow.
Spread → Session → Multi-TF → Market Structure → Strategy Ensemble →
News Risk → AI Debate → Risk Manager → Execute
"""
import logging

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
        if self.session:
            session_check = self.session.check(instrument)
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

        # 5b. 200 EMA TREND FILTER — only trade with the major trend
        import ta as _ta
        ema_200 = tech["indicators"].get("ema_200")
        if ema_200 is None and len(df) >= 200:
            ema_200 = _ta.trend.ema_indicator(df["close"], window=200).iloc[-1]
        if ema_200:
            price_now = float(df["close"].iloc[-1])
            result["ema_200"] = ema_200

        # 5c. ATR VOLATILITY FILTER — skip dead markets
        atr_val = tech["indicators"].get("atr", 0)
        if atr_val > 0 and len(df) >= 50:
            atr_series = _ta.volatility.average_true_range(
                df["high"], df["low"], df["close"], window=14)
            avg_atr = atr_series.rolling(50).mean().iloc[-1]
            if atr_val < avg_atr * 0.7:
                result["reason"] = "ATR below 70% of 50-bar avg — dead market"
                self._log_skip(instrument, result["reason"])
                return result

        # 6. STRATEGY ENSEMBLE — multiple strategies vote
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

        # 200 EMA DIRECTION CHECK — trade must align with major trend
        if ema_200:
            if ensemble["signal"] == "BUY" and price_now < ema_200:
                result["reason"] = "BUY rejected: price below 200 EMA (bearish trend)"
                self._log_skip(instrument, result["reason"])
                return result
            if ensemble["signal"] == "SELL" and price_now > ema_200:
                result["reason"] = "SELL rejected: price above 200 EMA (bullish trend)"
                self._log_skip(instrument, result["reason"])
                return result

        # PRICE ACTION CONFIRMATION — last bar must close in trade direction
        if len(df) >= 3:
            last_close = df["close"].iloc[-1]
            prev_close = df["close"].iloc[-2]
            last_open = df["open"].iloc[-1]
            if ensemble["signal"] == "BUY" and (last_close < last_open or last_close < prev_close):
                result["reason"] = "Price action: bar closed bearish, skipping BUY"
                self._log_skip(instrument, result["reason"])
                return result
            if ensemble["signal"] == "SELL" and (last_close > last_open or last_close > prev_close):
                result["reason"] = "Price action: bar closed bullish, skipping SELL"
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
            news_risk = self.news.check_risk(instrument)
            result["news"] = news_risk
            confidence += news_risk.get("confidence_modifier", 0)

        # 8. AI DEBATE (full mode only)
        if mode == "full" and self.debate:
            debate_result = self.debate.debate(instrument, {
                "signal": ensemble["signal"],
                "confidence": confidence,
                "regime": regime,
                "indicators": tech.get("indicators", {}),
                "reasons": ensemble.get("reasons", []),
            })
            verdict = debate_result.get("verdict", "SKIP")
            if verdict == "SKIP":
                result["reason"] = f"Debate rejected: {debate_result.get('reasoning', '')}"
                self._log_skip(instrument, result["reason"])
                return result
            confidence = debate_result.get("adjusted_confidence", confidence)
            result["debate"] = debate_result

        # Clamp confidence
        confidence = max(0, min(confidence, 0.95))

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
