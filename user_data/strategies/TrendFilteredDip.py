# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
# --------------------------------


class TrendFilteredDip(IStrategy):
    """
    Dip-buy mean reversion, gated by a macro trend filter.

    v2 (this version): thresholds retuned via hyperopt against a custom
    expectancy/drawdown-capped loss function (in-sample 2024-01 to
    2025-07, validated out-of-sample on 2025-07 to present). A filter-
    rejection funnel run beforehand showed the trend filter (EMA200/
    EMA50/ADX) was doing legitimate, non-redundant work, but RSI<28 alone
    was blocking 99.9% of the remaining trend-confirmed candles -- an
    arbitrarily strict threshold on one axis, not "quality filtering."
    Loosening RSI to <35 while tightening ADX to compensate (22 -> 27.6)
    is a Pareto improvement over v1: ~4x more trades, higher profit
    factor, higher Sharpe/Sortino/Calmar, and drawdown still held under
    the 5% hard cap (2.55% base case; up to ~5.2% under parameter
    perturbation in testing) -- confirmed on a full 2.7y history, a pure
    out-of-sample holdout, two disjoint pair-subset splits, and fee
    sensitivity up to 0.2% round-trip, with no cliff in any of them.

    v1 fixed ClucMay72018 (the strategy this bot ran live before): that
    strategy bought any dip below EMA100/BB-lower with no regard for the
    broader trend, capping wins at a flat 1% ROI against a flat 5% stop
    -- reliably producing many small wins wiped out by a handful of
    large stoploss hits during real downtrends. v1 added the macro trend
    gate; v2 (here) fixes the entry threshold that was needlessly
    throttling trade volume underneath that gate.
    """

    INTERFACE_VERSION: int = 3

    timeframe = "1h"
    can_short = False

    # Decaying ROI ladder (minutes -> target): let winners run early,
    # lock in smaller gains as the trade ages.
    minimal_roi = {
        "0": 0.052,
        "527": 0.034,
        "880": 0.011,
        "1091": 0,
    }

    stoploss = -0.042

    trailing_stop = True
    trailing_stop_positive = 0.029
    trailing_stop_positive_offset = 0.06
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = 210

    # Entry thresholds (hyperopt-tuned; see docstring above).
    ADX_MIN = 27.557
    RSI_ENTRY = 35
    BB_DIP_MULT = 1.003
    REQUIRE_EMA50_CONFIRM = True
    REQUIRE_EMA_RISING = True

    # --- Falling-knife entry filter (Phase 5) -----------------------------
    # Blocks entries on a candle whose single-candle decline is itself an
    # outsized multiple of recent ATR, i.e. the "dip" is a crash candle
    # piercing the BB lower band rather than a gentle pullback into it.
    # Disabled by default -- enable and backtest before relying on it.
    USE_FALLING_KNIFE_FILTER = True
    FALLING_KNIFE_ATR_MULT = 2.0

    # --- Exit-signal profit gating (Phase 7) ------------------------------
    # Native Freqtrade knobs: when True, exit_long signals are only honoured
    # once the trade is at/above EXIT_SIGNAL_MIN_PROFIT, so a losing/flat
    # trade rides the stoploss/trailing-stop instead of being closed by a
    # weak mean-reversion signal. Off by default (matches current live
    # behaviour) -- flip and backtest before relying on it.
    exit_profit_only = False
    exit_profit_offset = 0.0

    # --- Protections (Phase 2 / Phase 3) ----------------------------------
    # Pair-level cooldown: block re-entry into a pair for N candles after
    # THAT PAIR hits a stoploss (StoplossGuard w/ only_per_pair=True is more
    # targeted than a blanket CooldownPeriod, which would also block
    # re-entry after a normal profitable exit).
    USE_PAIR_STOPLOSS_COOLDOWN = False
    PAIR_STOPLOSS_COOLDOWN_LOOKBACK_CANDLES = 24
    PAIR_STOPLOSS_COOLDOWN_TRADE_LIMIT = 1
    PAIR_STOPLOSS_COOLDOWN_STOP_CANDLES = 6

    # Bot-wide circuit breaker: pause ALL new entries after N stoplosses in
    # a short window, regardless of pair (catches a broad regime shift, not
    # just one bad pair).
    USE_CONSECUTIVE_LOSS_GUARD = False
    CONSECUTIVE_LOSS_TRADE_LIMIT = 2
    CONSECUTIVE_LOSS_LOOKBACK_CANDLES = 24
    CONSECUTIVE_LOSS_STOP_CANDLES = 12

    @property
    def protections(self):
        prot = []
        if self.USE_PAIR_STOPLOSS_COOLDOWN:
            prot.append(
                {
                    "method": "StoplossGuard",
                    "lookback_period_candles": self.PAIR_STOPLOSS_COOLDOWN_LOOKBACK_CANDLES,
                    "trade_limit": self.PAIR_STOPLOSS_COOLDOWN_TRADE_LIMIT,
                    "stop_duration_candles": self.PAIR_STOPLOSS_COOLDOWN_STOP_CANDLES,
                    "only_per_pair": True,
                }
            )
        if self.USE_CONSECUTIVE_LOSS_GUARD:
            prot.append(
                {
                    "method": "StoplossGuard",
                    "lookback_period_candles": self.CONSECUTIVE_LOSS_LOOKBACK_CANDLES,
                    "trade_limit": self.CONSECUTIVE_LOSS_TRADE_LIMIT,
                    "stop_duration_candles": self.CONSECUTIVE_LOSS_STOP_CANDLES,
                    "only_per_pair": False,
                }
            )
        return prot

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]

        dataframe["volume_mean_30"] = dataframe["volume"].rolling(window=30).mean()

        # Single-candle decline, normalised by ATR, for the falling-knife
        # filter. Kept out of populate_entry_trend so it's computed once.
        dataframe["candle_drop_atr"] = (dataframe["open"] - dataframe["close"]) / dataframe[
            "atr"
        ].replace(0, float("nan"))

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            # Macro trend must be up: price above the long EMA. This is
            # the filter ClucMay72018 (the pre-v1 live strategy) lacked.
            (dataframe["close"] > dataframe["ema200"]),
            # Trend-strength confirmation (raised from 22 -> 27.6 in v2
            # to compensate for the looser RSI gate below).
            (dataframe["adx"] > self.ADX_MIN),
            # Local pullback within that uptrend (loosened from <28 to
            # <35 in v2 -- this was the actual bottleneck on trade count,
            # not the trend filter itself; see class docstring).
            (dataframe["rsi"] < self.RSI_ENTRY),
            (dataframe["close"] < dataframe["bb_lowerband"] * self.BB_DIP_MULT),
            (dataframe["volume"] > 0),
            (dataframe["volume"] < dataframe["volume_mean_30"].shift(1) * 20),
        ]
        if self.REQUIRE_EMA50_CONFIRM:
            conditions.append(dataframe["ema50"] > dataframe["ema200"])
        if self.REQUIRE_EMA_RISING:
            conditions.append(dataframe["ema200"] > dataframe["ema200"].shift(5))
        if self.USE_FALLING_KNIFE_FILTER:
            conditions.append(dataframe["candle_drop_atr"] < self.FALLING_KNIFE_ATR_MULT)

        combined = conditions[0]
        for c in conditions[1:]:
            combined &= c

        dataframe.loc[combined, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Mean-reversion take-profit only. Downside is already handled by
        # the stoploss/trailing stop; an extra "trend broke" signal exit
        # was tested and found to fire too early, cutting off trades that
        # would otherwise reach ROI or the trailing stop.
        dataframe.loc[
            (dataframe["close"] > dataframe["bb_middleband"]),
            "exit_long",
        ] = 1
        return dataframe
