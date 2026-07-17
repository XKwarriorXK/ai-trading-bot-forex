"""
Monte Carlo Simulation — tests if the strategy survives random trade ordering.
Answers: "Does this system survive bad luck?"
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


class MonteCarloAnalyzer:
    def __init__(self, simulations: int = 10000):
        self.simulations = simulations

    def analyze(self, trades: list, initial_balance: float = 10000) -> dict:
        if not trades:
            return {"error": "No trades to simulate"}

        pnls = [t["pnl"] for t in trades]

        final_balances = []
        max_drawdowns = []
        ruin_count = 0
        ruin_threshold = initial_balance * 0.5

        for _ in range(self.simulations):
            shuffled = np.random.permutation(pnls)
            balance = initial_balance
            peak = balance
            max_dd = 0

            for pnl in shuffled:
                balance += pnl
                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak * 100
                if dd > max_dd:
                    max_dd = dd
                if balance <= ruin_threshold:
                    ruin_count += 1
                    break

            final_balances.append(balance)
            max_drawdowns.append(max_dd)

        final_balances = np.array(final_balances)
        max_drawdowns = np.array(max_drawdowns)

        return {
            "simulations": self.simulations,
            "initial_balance": initial_balance,
            "mean_final_balance": round(float(np.mean(final_balances)), 2),
            "median_final_balance": round(float(np.median(final_balances)), 2),
            "std_final_balance": round(float(np.std(final_balances)), 2),
            "best_case": round(float(np.max(final_balances)), 2),
            "worst_case": round(float(np.min(final_balances)), 2),
            "confidence_intervals": {
                "95%": {
                    "lower": round(float(np.percentile(final_balances, 2.5)), 2),
                    "upper": round(float(np.percentile(final_balances, 97.5)), 2),
                },
                "99%": {
                    "lower": round(float(np.percentile(final_balances, 0.5)), 2),
                    "upper": round(float(np.percentile(final_balances, 99.5)), 2),
                },
            },
            "probability_of_ruin": round(ruin_count / self.simulations, 4),
            "median_max_drawdown_pct": round(float(np.median(max_drawdowns)), 2),
            "p95_max_drawdown_pct": round(float(np.percentile(max_drawdowns, 95)), 2),
        }
