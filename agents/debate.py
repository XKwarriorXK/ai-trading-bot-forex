"""
Multi-Level Trade Approval Chain — institutional-grade validation.

Level 0: Data Validation (spread, session, holidays — pipeline handles)
Level 1: Strategy Ensemble (11 strategies, 7 categories, A+ grade)
Level 2: Fast Screening — 3 Groq models, quick sanity checks
         - GPT-OSS 20B: Trend screener
         - Llama 3.1 8B: Momentum screener
         - Llama 3.3 70B: Risk screener
         Need 2/3 to pass.
Level 3: Senior Panel — 2 models do deep analysis
         - GPT-OSS 120B: Senior technical analyst
         - Gemini 2.5 Flash: Senior structure analyst
         Need 2/2 (both must agree).
Level 4: Final Approver — best model makes the call
         - Gemini 2.5 Pro: Head trader
         Sees all L2 + L3 findings. EXECUTE / REJECT / WAIT.
Level 5: Risk Engine (hard limits — pipeline handles after debate)
"""
import logging
from brain.providers import AIProvider
from config.settings import APPROVAL_CHAIN

logger = logging.getLogger(__name__)


def _build_context(instrument, tech_result):
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
    sl_pips = tech_result.get("stop_loss_pips", "N/A")
    tp_pips = tech_result.get("take_profit_pips", "N/A")
    sl_price = tech_result.get("stop_loss_price", "N/A")
    tp_price = tech_result.get("take_profit_price", "N/A")
    spread = tech_result.get("spread", "N/A")
    open_trades = tech_result.get("open_trades", 0)

    active_sessions = session if isinstance(session, list) else []
    news_events = news.get("events", [])

    price = indicators.get('price', 'N/A')
    ema_9 = indicators.get('ema_9', 'N/A')
    ema_21 = indicators.get('ema_21', 'N/A')
    ema_50 = indicators.get('ema_50', 'N/A')
    ema_200 = indicators.get('ema_200', 'N/A')

    ema_alignment = "N/A"
    if all(v != 'N/A' and v is not None for v in [price, ema_9, ema_21, ema_50]):
        if price > ema_9 > ema_21 > ema_50:
            ema_alignment = "BULLISH (price > EMA9 > EMA21 > EMA50)"
        elif price < ema_9 < ema_21 < ema_50:
            ema_alignment = "BEARISH (price < EMA9 < EMA21 < EMA50)"
        else:
            ema_alignment = "MIXED (no clean alignment)"

    rr_ratio = "N/A"
    if sl_pips != "N/A" and tp_pips != "N/A" and sl_pips > 0:
        rr_ratio = f"1:{tp_pips / sl_pips:.1f}"

    return (
        f"=== A+ TRADE UNDER REVIEW ===\n"
        f"Instrument: {instrument} | Signal: {signal}\n"
        f"Grade: {grade} | Confidence: {confidence:.0%} | Regime: {regime}\n\n"
        f"=== WHY IT HIT A+ ===\n"
        f"Strategies agreeing: {len(strategies)} ({', '.join(strategies)})\n"
        f"Independent categories: {num_categories} ({', '.join(categories)})\n"
        f"Reasons: {', '.join(reasons[:10])}\n\n"
        f"=== RAW INDICATORS ===\n"
        f"Price: {price}\n"
        f"RSI(14): {indicators.get('rsi', 'N/A')}\n"
        f"ADX: {indicators.get('adx', 'N/A')}\n"
        f"MACD Histogram: {indicators.get('macd_histogram', 'N/A')}\n"
        f"ATR: {indicators.get('atr', 'N/A')}\n"
        f"Stochastic K: {indicators.get('stoch_k', 'N/A')}\n"
        f"Bollinger Upper: {indicators.get('bb_upper', 'N/A')}\n"
        f"Bollinger Lower: {indicators.get('bb_lower', 'N/A')}\n"
        f"Bollinger Width: {indicators.get('bb_width', 'N/A')}\n"
        f"EMA 9: {ema_9}\n"
        f"EMA 21: {ema_21}\n"
        f"EMA 50: {ema_50}\n"
        f"EMA 200: {ema_200}\n"
        f"EMA Alignment: {ema_alignment}\n\n"
        f"=== TRADE PLAN ===\n"
        f"Entry: {price} | Stop Loss: {sl_price} ({sl_pips} pips)\n"
        f"Take Profit: {tp_price} ({tp_pips} pips)\n"
        f"Risk:Reward: {rr_ratio}\n"
        f"Spread: {spread} pips\n\n"
        f"=== MARKET STRUCTURE ===\n"
        f"Bias: {structure.get('bias', 'unknown')}\n"
        f"Pattern: {structure.get('pattern', 'unknown')}\n"
        f"Support: {structure.get('support', 'N/A')}\n"
        f"Resistance: {structure.get('resistance', 'N/A')}\n\n"
        f"=== SESSION & NEWS ===\n"
        f"Active sessions: {', '.join(active_sessions) if active_sessions else 'unknown'}\n"
        f"News risk: {news.get('risk_level', 'low')}\n"
        f"Events: {[e.get('event', '?') for e in news_events] if news_events else 'None'}\n\n"
        f"=== RISK CONTEXT ===\n"
        f"Currently open trades: {open_trades}\n"
    )


# ── LEVEL 2: FAST SCREENING ─────────────────────────────────────────
L2_PROMPTS = {
    "trend_screener": {
        "system": "You are a trend analyst. Quick yes/no only. Be concise.",
        "prompt": (
            "QUICK SCREEN: {signal} {instrument}\n\n"
            "{context}\n\n"
            "Does the TREND support this {signal}?\n"
            "Check EMA alignment, ADX strength, and regime.\n"
            "Respond ONLY in JSON:\n"
            "{{\"pass\": true or false, \"flag\": \"one sentence why\"}}"
        ),
    },
    "momentum_screener": {
        "system": "You are a momentum analyst. Quick yes/no only. Be concise.",
        "prompt": (
            "QUICK SCREEN: {signal} {instrument}\n\n"
            "{context}\n\n"
            "Does MOMENTUM support this {signal}?\n"
            "Check RSI, MACD histogram direction, Stochastic position.\n"
            "Respond ONLY in JSON:\n"
            "{{\"pass\": true or false, \"flag\": \"one sentence why\"}}"
        ),
    },
    "risk_screener": {
        "system": "You are a risk screener. Flag red flags only. Be concise.",
        "prompt": (
            "QUICK SCREEN: {signal} {instrument}\n\n"
            "{context}\n\n"
            "Any OBVIOUS RED FLAGS that should kill this trade?\n"
            "Check session, news, spread, volatility extremes, R:R ratio.\n"
            "Respond ONLY in JSON:\n"
            "{{\"pass\": true or false, \"flag\": \"one sentence why\"}}"
        ),
    },
}


# ── LEVEL 3: SENIOR PANEL ───────────────────────────────────────────
L3_PROMPTS = {
    "senior_technical": {
        "system": (
            "You are a senior technical analyst at a prop firm. You ONLY approve "
            "trades where indicators genuinely confirm the signal direction."
        ),
        "prompt": (
            "SENIOR REVIEW: {signal} {instrument} | Grade {grade}\n\n"
            "{context}\n\n"
            "=== L2 SCREENING RESULTS ===\n{l2_summary}\n\n"
            "YOUR JOB — Deep technical verification:\n"
            "1. RSI confirming? (BUY: 40-65 ideal, >75 overextended. SELL: 35-60, <25 overextended)\n"
            "2. MACD histogram expanding in signal direction?\n"
            "3. ADX > 25 = confirmed trend? < 20 = no trend?\n"
            "4. EMA alignment clean? Price relation to 50 & 200?\n"
            "5. R:R at least 1:2? SL/TP placement logical?\n"
            "6. Any L2 screening flags that need deeper investigation?\n\n"
            "Respond in JSON:\n"
            "{{\"approve\": true or false, "
            "\"confidence_adjustment\": -0.10 to +0.05, "
            "\"key_finding\": \"one sentence with specific numbers\", "
            "\"additional_data\": \"deeper insight you derived\"}}"
        ),
    },
    "senior_structure": {
        "system": (
            "You are a market structure analyst at a prop firm specializing in "
            "institutional order flow, supply/demand zones, and liquidity."
        ),
        "prompt": (
            "SENIOR REVIEW: {signal} {instrument} | Grade {grade}\n\n"
            "{context}\n\n"
            "=== L2 SCREENING RESULTS ===\n{l2_summary}\n\n"
            "YOUR JOB — Deep structural verification:\n"
            "1. Price at key support/resistance for this direction?\n"
            "2. Structure bias (HH/HL=bullish, LH/LL=bearish) matches signal?\n"
            "3. Entry chasing? Too far from the key level?\n"
            "4. SL placement makes structural sense?\n"
            "5. R:R minimum 1:2 to next structural level?\n"
            "6. Liquidity trap risk? Stop hunt potential?\n\n"
            "Respond in JSON:\n"
            "{{\"approve\": true or false, "
            "\"confidence_adjustment\": -0.10 to +0.05, "
            "\"key_finding\": \"one sentence with specific levels\", "
            "\"additional_data\": \"structural insight you derived\"}}"
        ),
    },
}


# ── LEVEL 4: FINAL APPROVER ─────────────────────────────────────────
L4_SYSTEM = (
    "You are the head trader at an institutional prop firm. "
    "You make the final execution decision. You have seen what the "
    "screening panel and senior analysts found. Trust their data but "
    "apply your own judgment. Protect capital above all."
)

L4_PROMPT = (
    "FINAL DECISION: {signal} {instrument} | Grade {grade} | {confidence:.0%}\n"
    "Regime: {regime} | {num_categories} independent confluence categories\n\n"
    "=== LEVEL 2 SCREENING ({l2_pass}/{l2_total} passed) ===\n"
    "{l2_summary}\n\n"
    "=== LEVEL 3 SENIOR REVIEWS ({l3_pass}/{l3_total} approved) ===\n"
    "{l3_summary}\n\n"
    "YOUR JOB as final approver:\n"
    "1. Did L2 screeners flag anything L3 seniors didn't address?\n"
    "2. Are L3 findings thorough and data-backed?\n"
    "3. If both L3 approved — any hidden risk they both missed?\n"
    "4. If any L3 rejected with a concrete finding — weigh it heavily\n"
    "5. Set final confidence based on ALL evidence\n\n"
    "Respond in JSON:\n"
    "{{\"verdict\": \"BUY\" or \"SELL\" or \"SKIP\", "
    "\"adjusted_confidence\": 0.0 to 0.95, "
    "\"reasoning\": \"one sentence citing specific L2/L3 findings\"}}"
)


class DebateAgent:
    def __init__(self, router=None, provider: AIProvider = None):
        self.router = router
        self.provider = provider or (router.provider if router else None)

    def _call_reviewer(self, model_key, provider_name, prompt, system):
        return self.provider.call_json(
            model_key, prompt, system,
            priority="high", provider_name=provider_name,
        )

    def _format_l2_summary(self, results):
        if not results:
            return "No screening data available."
        lines = []
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            lines.append(f"  {r['role']} ({r['provider']}): {status} — {r['flag']}")
        return "\n".join(lines)

    def _format_l3_summary(self, results):
        if not results:
            return "No senior review data available."
        lines = []
        for r in results:
            status = "APPROVED" if r["approved"] else "REJECTED"
            lines.append(
                f"  {r['role']} ({r['provider']}): {status}\n"
                f"    Finding: {r['key_finding']}\n"
                f"    Conf adj: {r['confidence_adjustment']:+.0%}\n"
                f"    Extra: {r.get('additional_data', 'None')}"
            )
        return "\n".join(lines)

    def debate(self, instrument: str, tech_result: dict) -> dict:
        context = _build_context(instrument, tech_result)
        signal = tech_result.get("signal", "SKIP")
        confidence = tech_result.get("confidence", 0)
        grade = tech_result.get("grade", "?")

        # ── LEVEL 2: FAST SCREENING ──────────────────────────────
        l2_config = APPROVAL_CHAIN["level_2"]
        l2_results = []

        for reviewer in l2_config["reviewers"]:
            role = reviewer["role"]
            prov = reviewer["provider"]
            model = reviewer["model"]
            prompts = L2_PROMPTS.get(role, {})
            if not prompts:
                continue

            prompt = prompts["prompt"].format(
                context=context, signal=signal, instrument=instrument,
            )
            logger.info(f"L2 SCREEN | {role} | {prov}:{model}")

            result = self._call_reviewer(model, prov, prompt, prompts["system"])

            if result["success"] and result.get("parsed"):
                parsed = result["parsed"]
                passed = parsed.get("pass", False)
                flag = parsed.get("flag", "No comment")
                logger.info(f"L2 SCREEN | {role} ({prov}) | {'PASS' if passed else 'FAIL'} | {flag}")
                l2_results.append({
                    "role": role, "provider": prov,
                    "passed": passed, "flag": flag,
                })
            else:
                logger.warning(f"L2 SCREEN | {role} ({prov}) | API FAILED — excluded")

        l2_passed = sum(1 for r in l2_results if r["passed"])
        l2_failed = sum(1 for r in l2_results if not r["passed"])
        l2_total = len(l2_results)
        l2_min = l2_config["min_pass"]

        logger.info(f"L2 COMPLETE | {l2_passed} pass, {l2_failed} fail, {len(l2_config['reviewers']) - l2_total} unavailable (need {l2_min})")

        if l2_total == 0:
            logger.warning("L2 | No screeners available — passing with haircut")
            return self._haircut_result(signal, confidence, "No L2 screeners available", 2)

        if l2_passed < l2_min:
            fail_flags = "; ".join(r["flag"] for r in l2_results if not r["passed"])
            logger.info(f"L2 REJECTED | {fail_flags}")
            return {
                "verdict": "SKIP",
                "adjusted_confidence": confidence * 0.5,
                "reasoning": f"L2 screening failed ({l2_passed}/{l2_total} pass, need {l2_min}): {fail_flags}",
                "l2_results": l2_results, "l3_results": [], "level": 2,
            }

        # ── LEVEL 3: SENIOR PANEL ────────────────────────────────
        l2_summary = self._format_l2_summary(l2_results)
        l3_config = APPROVAL_CHAIN["level_3"]
        l3_results = []

        for reviewer in l3_config["reviewers"]:
            role = reviewer["role"]
            prov = reviewer["provider"]
            model = reviewer["model"]
            prompts = L3_PROMPTS.get(role, {})
            if not prompts:
                continue

            prompt = prompts["prompt"].format(
                context=context, signal=signal, instrument=instrument,
                grade=grade, l2_summary=l2_summary,
            )
            logger.info(f"L3 SENIOR | {role} | {prov}:{model}")

            result = self._call_reviewer(model, prov, prompt, prompts["system"])

            if result["success"] and result.get("parsed"):
                parsed = result["parsed"]
                approved = parsed.get("approve", False)
                conf_adj = parsed.get("confidence_adjustment", 0)
                finding = parsed.get("key_finding", "No finding")
                extra = parsed.get("additional_data", "")
                logger.info(
                    f"L3 SENIOR | {role} ({prov}) | "
                    f"{'APPROVED' if approved else 'REJECTED'} | "
                    f"Conf adj: {conf_adj:+.0%} | {finding}"
                )
                l3_results.append({
                    "role": role, "provider": prov,
                    "approved": approved,
                    "confidence_adjustment": conf_adj,
                    "key_finding": finding,
                    "additional_data": extra,
                })
            else:
                logger.warning(f"L3 SENIOR | {role} ({prov}) | API FAILED — excluded")

        l3_approved = sum(1 for r in l3_results if r["approved"])
        l3_rejected = sum(1 for r in l3_results if not r["approved"])
        l3_total = len(l3_results)
        l3_min = l3_config["min_pass"]

        logger.info(f"L3 COMPLETE | {l3_approved} approved, {l3_rejected} rejected, {len(l3_config['reviewers']) - l3_total} unavailable (need {l3_min})")

        if l3_total == 0:
            logger.warning("L3 | No seniors available — using L2 consensus with haircut")
            return self._haircut_result(signal, confidence, "No L3 seniors available — L2 passed", 3)

        if l3_approved < l3_min:
            reject_findings = "; ".join(r["key_finding"] for r in l3_results if not r["approved"])
            logger.info(f"L3 REJECTED | {reject_findings}")
            return {
                "verdict": "SKIP",
                "adjusted_confidence": confidence * 0.5,
                "reasoning": f"L3 senior panel rejected ({l3_approved}/{l3_total} approve, need {l3_min}): {reject_findings}",
                "l2_results": l2_results, "l3_results": l3_results, "level": 3,
            }

        # ── LEVEL 4: FINAL APPROVER ──────────────────────────────
        l3_summary = self._format_l3_summary(l3_results)
        l4_config = APPROVAL_CHAIN["level_4"]

        final_prompt = L4_PROMPT.format(
            signal=signal, instrument=instrument, grade=grade,
            confidence=confidence,
            regime=tech_result.get("regime", "unknown"),
            num_categories=tech_result.get("num_categories", 0),
            l2_pass=l2_passed, l2_total=l2_total, l2_summary=l2_summary,
            l3_pass=l3_approved, l3_total=l3_total, l3_summary=l3_summary,
        )

        prov = l4_config["provider"]
        model = l4_config["model"]
        logger.info(f"L4 FINAL | Head trader | {prov}:{model}")

        final_result = self._call_reviewer(model, prov, final_prompt, L4_SYSTEM)

        if final_result["success"] and final_result.get("parsed"):
            parsed = final_result["parsed"]
            verdict = parsed.get("verdict", "SKIP")
            adj_conf = parsed.get("adjusted_confidence", confidence * 0.8)
            reasoning = parsed.get("reasoning", "")

            logger.info(f"L4 FINAL | {instrument} | {verdict} | Conf: {adj_conf:.0%} | {reasoning}")

            return {
                "verdict": verdict,
                "adjusted_confidence": adj_conf,
                "reasoning": reasoning,
                "l2_results": l2_results,
                "l3_results": l3_results,
                "level": 4,
            }

        logger.warning("L4 | Final approver failed — using L3 consensus")
        avg_adj = sum(r["confidence_adjustment"] for r in l3_results if r["approved"]) / max(l3_approved, 1)
        return {
            "verdict": signal,
            "adjusted_confidence": max(0, min(confidence + avg_adj, 0.95)),
            "reasoning": f"L4 unavailable — L3 consensus ({l3_approved}/{l3_total} approved)",
            "l2_results": l2_results, "l3_results": l3_results, "level": 3,
        }

    def _haircut_result(self, signal, confidence, reason, level):
        return {
            "verdict": signal,
            "adjusted_confidence": confidence * 0.85,
            "reasoning": f"{reason} — passing with 15% haircut",
            "l2_results": [], "l3_results": [], "level": level,
        }
