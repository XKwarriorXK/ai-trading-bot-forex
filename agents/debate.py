"""
3-Level Trade Approval Chain — institutional-grade validation.

Level 1: Strategy ensemble (11 strategies) produces A+ signal
Level 2: 3 AI sub-approvers (different models) independently review
         - Groq/Llama 3.3 70B: Technical Expert
         - Cerebras/GPT-OSS 120B: Structure Expert
         - Groq/Llama 3.1 8B: Risk Expert
         Need 2/3 to approve.
Level 3: Final Approver (Cerebras/GPT-OSS 120B) reviews everything —
         all expert opinions, why it hit A+, what additional data
         they sourced. Makes the final EXECUTE or REJECT call.
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


REVIEWER_PROMPTS = {
    "technical_expert": {
        "system": (
            "You are a senior technical analyst at a prop firm. You ONLY approve "
            "trades where indicators genuinely confirm the signal direction. "
            "You source additional indicator insights beyond what's provided."
        ),
        "prompt": (
            "ROLE: Technical Expert — Level 2 Reviewer\n\n"
            "{context}\n\n"
            "YOUR JOB: Independently verify the technical case. Check:\n"
            "1. RSI — is it confirming direction? (BUY: RSI 40-65 is ideal entry zone, >75 = overextended. "
            "SELL: RSI 35-60 is ideal, <25 = overextended)\n"
            "2. MACD — is histogram expanding in signal direction? Shrinking = momentum dying\n"
            "3. ADX — is trend strength sufficient? Below 20 = no trend, breakout likely to fail. "
            "Above 25 = confirmed trend\n"
            "4. Stochastic — any hidden divergence between price and stochastic?\n"
            "5. EMA alignment — is price above EMA 50 and 200 for BUY, below for SELL?\n"
            "6. ATR — is volatility sufficient for a meaningful move?\n\n"
            "Based on ALL indicators, give your verdict.\n"
            "Respond in JSON:\n"
            "{{\"approve\": true or false, "
            "\"confidence_adjustment\": -0.10 to +0.05, "
            "\"key_finding\": \"one sentence with specific numbers\", "
            "\"additional_data\": \"any extra indicator insight you derived\"}}"
        ),
    },
    "structure_expert": {
        "system": (
            "You are a market structure analyst specializing in institutional order flow. "
            "You verify that price is at a significant structural level before approving entry. "
            "You think in terms of supply/demand zones, order blocks, and liquidity."
        ),
        "prompt": (
            "ROLE: Structure Expert — Level 2 Reviewer\n\n"
            "{context}\n\n"
            "YOUR JOB: Verify the structural case. Check:\n"
            "1. Is price at or near a key support/resistance level for the signal direction?\n"
            "2. Does the market structure bias (HH/HL for bullish, LH/LL for bearish) MATCH "
            "the signal direction? A BUY in bearish structure = high risk\n"
            "3. Is entry chasing? If price already moved significantly from the key level, "
            "the best entry is gone\n"
            "4. Where is the realistic take-profit based on the next structural level?\n"
            "5. Does the risk-to-reward make sense? Entry to stop vs entry to target — "
            "minimum 1:2 required\n"
            "6. Any liquidity trap risk? (stop hunt below support before reversal)\n\n"
            "Based on structure analysis, give your verdict.\n"
            "Respond in JSON:\n"
            "{{\"approve\": true or false, "
            "\"confidence_adjustment\": -0.10 to +0.05, "
            "\"key_finding\": \"one sentence with specific levels\", "
            "\"additional_data\": \"structural insight you derived\"}}"
        ),
    },
    "risk_expert": {
        "system": (
            "You are the risk manager at a prop firm with a 2% daily loss limit. "
            "Your job is to protect capital. You reject trades that have hidden risks "
            "even if the technical setup looks clean. You think about what could go wrong."
        ),
        "prompt": (
            "ROLE: Risk Expert — Level 2 Reviewer\n\n"
            "{context}\n\n"
            "YOUR JOB: Stress-test this trade for hidden risks. Check:\n"
            "1. Session timing — is this during a high-volume session overlap? "
            "Dead hours = low liquidity = bad fills and false breakouts\n"
            "2. News risk — any high-impact events (FOMC, NFP, CPI) that could "
            "invalidate the entire setup?\n"
            "3. Correlation exposure — would this trade stack risk on top of "
            "existing positions in correlated pairs?\n"
            "4. Volatility regime — is ATR unusually high (potential reversal exhaustion) "
            "or unusually low (potential breakout but also potential chop)?\n"
            "5. Is this the kind of trade that looks good on paper but fails in practice? "
            "(e.g., buying at resistance, selling at support, fading a strong trend)\n"
            "6. Weekend/holiday risk — is a market close approaching that could gap?\n\n"
            "Based on risk analysis, give your verdict.\n"
            "Respond in JSON:\n"
            "{{\"approve\": true or false, "
            "\"confidence_adjustment\": -0.10 to +0.05, "
            "\"key_finding\": \"one sentence on biggest risk or why it's clean\", "
            "\"additional_data\": \"risk factor you identified\"}}"
        ),
    },
}

FINAL_APPROVER_PROMPT = (
    "You are the HEAD TRADER making the final execution decision.\n\n"
    "TRADE: {signal} {instrument} | Grade {grade} | {confidence:.0%}\n"
    "Regime: {regime} | {num_categories} independent confluence categories\n\n"
    "=== LEVEL 2 EXPERT REVIEWS ===\n"
    "{expert_reviews}\n\n"
    "=== APPROVAL STATUS ===\n"
    "Approvals: {approvals}/{total} (need {min_needed})\n\n"
    "YOUR JOB as final approver:\n"
    "1. Review WHY each expert approved or rejected\n"
    "2. Check if rejecting experts found a REAL problem or were being overly cautious\n"
    "3. Weigh the technical, structural, and risk opinions together\n"
    "4. If 2/3 approved and no dealbreaker was found → APPROVE\n"
    "5. If a rejecting expert found a concrete data-backed problem → REJECT even if 2/3 approved\n"
    "6. Set final confidence based on all expert adjustments\n\n"
    "Respond in JSON:\n"
    "{{\"verdict\": \"BUY\" or \"SELL\" or \"SKIP\", "
    "\"adjusted_confidence\": 0.0 to 0.95, "
    "\"reasoning\": \"one sentence citing specific expert findings\"}}"
)


class DebateAgent:
    def __init__(self, router=None, provider: AIProvider = None):
        self.router = router
        self.provider = provider or (router.provider if router else None)

    def debate(self, instrument: str, tech_result: dict) -> dict:
        context = _build_context(instrument, tech_result)
        signal = tech_result.get("signal", "SKIP")
        confidence = tech_result.get("confidence", 0)

        # === LEVEL 2: Expert Panel (3 different AI models) ===
        reviewers = APPROVAL_CHAIN["level_2_reviewers"]
        min_approvals = APPROVAL_CHAIN["min_approvals"]
        expert_results = []

        for reviewer in reviewers:
            role = reviewer["role"]
            provider_name = reviewer["provider"]
            model_key = reviewer["model"]
            prompts = REVIEWER_PROMPTS.get(role, {})

            if not prompts:
                logger.warning(f"No prompt template for role: {role}")
                continue

            prompt = prompts["prompt"].format(context=context)
            system = prompts["system"]

            logger.info(f"LEVEL 2 | {role} | Calling {provider_name}:{model_key}")

            result = self.provider.call_json(
                model_key, prompt, system,
                priority="high",
                provider_name=provider_name,
            )

            if result["success"] and result.get("parsed"):
                parsed = result["parsed"]
                approved = parsed.get("approve", False)
                conf_adj = parsed.get("confidence_adjustment", 0)
                finding = parsed.get("key_finding", "No finding")
                extra = parsed.get("additional_data", "")

                logger.info(
                    f"LEVEL 2 | {role} ({provider_name}) | "
                    f"{'APPROVED' if approved else 'REJECTED'} | "
                    f"Conf adj: {conf_adj:+.0%} | {finding}"
                )

                expert_results.append({
                    "role": role,
                    "provider": provider_name,
                    "approved": approved,
                    "confidence_adjustment": conf_adj,
                    "key_finding": finding,
                    "additional_data": extra,
                })
            else:
                logger.warning(f"LEVEL 2 | {role} ({provider_name}) | FAILED — excluded from vote")

        approvals = sum(1 for e in expert_results if e["approved"])
        rejections = sum(1 for e in expert_results if not e["approved"])
        total = len(expert_results)

        logger.info(f"LEVEL 2 COMPLETE | {approvals} approved, {rejections} rejected, {len(reviewers) - total} unavailable")

        if total == 0:
            logger.warning("LEVEL 2 | No reviewers available — passing with confidence haircut")
            return {
                "verdict": signal,
                "adjusted_confidence": confidence * 0.85,
                "reasoning": "No AI reviewers available — original signal with 15% haircut",
                "expert_results": [],
                "level": 2,
            }

        if rejections > approvals:
            reject_list = [e for e in expert_results if not e["approved"]]
            reject_reasons = "; ".join(e["key_finding"] for e in reject_list)
            logger.info(f"LEVEL 2 REJECTED | {reject_reasons}")
            return {
                "verdict": "SKIP",
                "adjusted_confidence": confidence * 0.5,
                "reasoning": f"Expert panel rejected ({approvals} approve, {rejections} reject): {reject_reasons}",
                "expert_results": expert_results,
                "level": 2,
            }

        if approvals == 0 and rejections == 0:
            logger.warning("LEVEL 2 | All reviewers failed — passing with confidence haircut")
            return {
                "verdict": signal,
                "adjusted_confidence": confidence * 0.85,
                "reasoning": "All AI reviewers unavailable — original signal with 15% haircut",
                "expert_results": [],
                "level": 2,
            }

        # === LEVEL 3: Final Approver ===
        expert_reviews_text = ""
        for e in expert_results:
            status = "APPROVED" if e["approved"] else "REJECTED"
            expert_reviews_text += (
                f"\n{e['role'].upper()} ({e['provider']}) — {status}\n"
                f"  Finding: {e['key_finding']}\n"
                f"  Confidence adjustment: {e['confidence_adjustment']:+.0%}\n"
                f"  Additional data: {e.get('additional_data', 'None')}\n"
            )

        final_prompt = FINAL_APPROVER_PROMPT.format(
            signal=signal,
            instrument=instrument,
            grade=tech_result.get("grade", "?"),
            confidence=confidence,
            regime=tech_result.get("regime", "unknown"),
            num_categories=tech_result.get("num_categories", 0),
            expert_reviews=expert_reviews_text,
            approvals=approvals,
            total=total,
            min_needed=min_approvals,
        )

        level3 = APPROVAL_CHAIN["level_3_approver"]
        logger.info(f"LEVEL 3 | Final approver | Calling {level3['provider']}:{level3['model']}")

        final_result = self.provider.call_json(
            level3["model"], final_prompt,
            system_prompt=(
                "You are the head trader at an institutional prop firm. "
                "You have the final say. Your experts have reviewed this trade — "
                "trust their specific findings but make your own judgment. "
                "If a rejecting expert found a concrete problem, weigh it heavily "
                "even if the majority approved. Protect capital above all."
            ),
            priority="high",
            provider_name=level3["provider"],
        )

        if final_result["success"] and final_result.get("parsed"):
            parsed = final_result["parsed"]
            verdict = parsed.get("verdict", "SKIP")
            adj_conf = parsed.get("adjusted_confidence", confidence * 0.8)
            reasoning = parsed.get("reasoning", "")

            logger.info(
                f"LEVEL 3 FINAL | {instrument} | Verdict: {verdict} | "
                f"Conf: {adj_conf:.0%} | {reasoning}"
            )

            return {
                "verdict": verdict,
                "adjusted_confidence": adj_conf,
                "reasoning": reasoning,
                "expert_results": expert_results,
                "level": 3,
                "approvals": f"{approvals}/{total}",
            }

        logger.warning(f"LEVEL 3 | Final approver failed — using Level 2 consensus")
        avg_adj = sum(e["confidence_adjustment"] for e in expert_results if e["approved"]) / max(approvals, 1)
        return {
            "verdict": signal,
            "adjusted_confidence": max(0, min(confidence + avg_adj, 0.95)),
            "reasoning": f"Final approver unavailable — Level 2 consensus ({approvals}/{total} approved)",
            "expert_results": expert_results,
            "level": 2,
            "approvals": f"{approvals}/{total}",
        }
