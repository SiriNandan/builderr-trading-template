"""Sprint Mode v8 — built to reach top 3 on the builderr leaderboard.

Strategy shift from v6/v7:
  - 3 positions at 27% each → 81% gross exposure (contest concentration rule compliant)
  - No tech sector cap → all 3 slots can be high-momentum tech names
  - Wider stop-loss (-4%) → less churn, holds winners through minor pullbacks
  - Higher take-profit (12%) → lets winners run further before trimming
  - Bigger overlay (QLD 18% + SSO 12%) when fully calm
  - Longer rebalance cadence (5 days) → fewer trades, less transaction cost
  - Trimmed universe → removes low-beta defensive sector ETFs that drag alpha

Safety gates respected:
  - Beta-adjusted gross cap: 1.42x target (1.5x hard)
  - No single position > 30% for 5+ consecutive days
  - Defensive sleeve (GLD + XLU) when market is risk-off
  - Crash brake on QQQ -3% day
"""
from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev
from typing import Any

# --------------------------------------------------------------------------
# Universe — trimmed to alpha generators only
# --------------------------------------------------------------------------
RISK_CANDIDATES = (
    # Broad: IWM for small-cap momentum, SMH for semis momentum
    "SMH", "IWM",
    # Sector ETFs — only the three that generate real alpha
    "XLK", "XLF", "XLE",
    # Mega-cap tech — primary alpha source
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "AMD", "ORCL", "CRM",
)
BREADTH_BASKET = ("SPY", "QQQ", "IWM", "DIA")
DEFENSIVE      = (("GLD", 0.15), ("XLU", 0.15))
BETA_MULTIPLE  = {
    "TQQQ": 3.0, "SOXL": 3.0, "UPRO": 3.0, "SPXL": 3.0, "TNA": 3.0,
    "FAS": 3.0, "TECL": 3.0, "LABU": 3.0, "CURE": 3.0, "DRN": 3.0,
    "UDOW": 3.0, "NAIL": 3.0,
    "QLD": 2.0, "SSO": 2.0, "DDM": 2.0, "ROM": 2.0, "UWM": 2.0, "AGQ": 2.0,
}

# --------------------------------------------------------------------------
# Parameters — sprint mode
# --------------------------------------------------------------------------
STOP_LOSS        = -0.04   # wider: reduces churn from normal volatility
TAKE_PROFIT      =  0.12   # higher: lets winners run before trimming
MAX_POSITIONS    =  3      # top 3 picks — 3×0.27 = 81% deployed
MAX_WEIGHT       =  0.27   # per-name target — hard cap under 30% (contest concentration rule)
MAX_BETA_GROSS   =  1.42   # beta-adjusted gross cap
MIN_BREADTH      =  2      # at least 2 of 4 breadth indices above 50-SMA
VOL_CEILING      =  0.35   # QQQ 20d vol ceiling
OVERLAY_VOL_CAP  =  0.26   # 2x overlay only when genuinely calm (proven threshold)
ENTRY_SCORE_MIN  =  0.01
MIN_TRADE_PCT    =  0.015
REBALANCE_DAYS   =  5      # longer cadence → fewer trades → less churn
STOP_COOLDOWN    =  2
HALF_BREADTH     =  3      # breadth >= 3 for full size
SINGLE_DAY_BRAKE = -0.03   # crash brake unchanged
BRAKE_COOLDOWN   =  2
MAX_TECH_PICKS   =  MAX_POSITIONS  # no effective cap — both picks can be tech

# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
_last_rebalance_date: str | None = None
_stopped_bars: dict[str, int]    = {}
_bar_count: int                  = 0
_was_defensive: bool             = False
_brake_cooldown: int             = 0


# --------------------------------------------------------------------------
# Bar helpers (unchanged from v6)
# --------------------------------------------------------------------------
def closes(bars):
    if not bars:
        return []
    out = []
    for bar in bars:
        try:
            c = float(bar["close"])
        except (KeyError, TypeError, ValueError):
            return []
        if c <= 0:
            return []
        out.append(c)
    return out


def sma(values, n):
    if len(values) < n:
        return None
    return mean(values[-n:])


def momentum(values, n, skip=0):
    need = n + skip
    if len(values) <= need:
        return None
    end   = values[-(skip + 1)] if skip else values[-1]
    start = values[-(need + 1)]
    return (end / start - 1.0) if start > 0 else None


def realized_vol(values, n):
    if len(values) <= n:
        return None
    window = values[-(n + 1):]
    rets   = [window[i] / window[i - 1] - 1.0
              for i in range(1, len(window)) if window[i - 1] > 0]
    if len(rets) < 5:
        return None
    return pstdev(rets) * sqrt(252.0)


# --------------------------------------------------------------------------
# Portfolio helpers (unchanged)
# --------------------------------------------------------------------------
def current_positions(portfolio_state):
    positions = {}
    for raw in portfolio_state.get("positions", []) or []:
        ticker = str(raw.get("ticker", "")).upper()
        if not ticker:
            continue
        try:
            qty  = float(raw.get("quantity", 0.0))
            cost = float(raw.get("avg_cost", 0.0))
        except (TypeError, ValueError):
            continue
        if qty <= 0:
            continue
        ex = positions.setdefault(ticker, {"quantity": 0.0, "avg_cost": cost})
        ex["quantity"] += qty
        ex["avg_cost"] = cost or ex["avg_cost"]
    return positions


def portfolio_equity(portfolio_state, cash):
    try:
        total = float(portfolio_state.get("cash", cash))
    except (TypeError, ValueError):
        total = float(cash or 0.0)
    lp = portfolio_state.get("last_prices", {}) or {}
    for t, pos in current_positions(portfolio_state).items():
        try:
            px = float(lp.get(t, pos["avg_cost"]))
        except (TypeError, ValueError):
            px = pos["avg_cost"]
        total += pos["quantity"] * max(px, 0.0)
    return max(total, 0.0)


def latest_bar_date(market_state):
    bars = (market_state.get("SPY") or market_state.get("QQQ")
            or next(iter(market_state.values()), []))
    if not bars:
        return None
    ts = bars[-1].get("ts")
    return str(ts)[:10] if ts is not None else str(len(bars))


def days_since_rebalance(market_state):
    if _last_rebalance_date is None:
        return None
    bars  = (market_state.get("SPY") or market_state.get("QQQ")
             or next(iter(market_state.values()), []))
    dates = [str(b.get("ts", i))[:10] for i, b in enumerate(bars)]
    if not dates or _last_rebalance_date not in dates:
        return None
    return len(dates) - dates.index(_last_rebalance_date) - 1


def market_prices(market_state):
    out = {}
    for t, bars in market_state.items():
        cs = closes(bars)
        if cs:
            out[t.upper()] = cs[-1]
    return out


# --------------------------------------------------------------------------
# Signal logic
# --------------------------------------------------------------------------
def day_brake_active(market_state):
    qqq = closes(market_state.get("QQQ"))
    if len(qqq) < 2 or qqq[-2] <= 0:
        return False
    return qqq[-1] / qqq[-2] - 1 < SINGLE_DAY_BRAKE


def breadth_score(market_state):
    count = 0
    for t in BREADTH_BASKET:
        cs  = closes(market_state.get(t))
        s50 = sma(cs, 50)
        if s50 is not None and cs and cs[-1] > s50:
            count += 1
    return count


def regime_green(market_state):
    if breadth_score(market_state) < MIN_BREADTH:
        return False
    qqq_vol = realized_vol(closes(market_state.get("QQQ")), 20)
    return qqq_vol is not None and qqq_vol < VOL_CEILING


def ticker_score(values):
    """45% 20d-mom + 30% 60d-mom + 15% trend-gap - 10% vol-penalty."""
    if len(values) < 62:
        return None
    s20 = sma(values, 20)
    if s20 is None or values[-1] <= s20:
        return None
    mom20 = momentum(values, 20)
    mom60 = momentum(values, 60)
    vol20 = realized_vol(values, 20)
    if mom20 is None or mom60 is None or vol20 is None or vol20 <= 0:
        return None
    trend_gap = values[-1] / s20 - 1.0
    return 0.45 * mom20 + 0.30 * mom60 + 0.15 * trend_gap - 0.10 * vol20


def scale_to_caps(weights):
    capped = {t: min(max(w, 0.0), MAX_WEIGHT) for t, w in weights.items() if w > 0}
    gross  = sum(w * BETA_MULTIPLE.get(t, 1.0) for t, w in capped.items())
    if gross > MAX_BETA_GROSS:
        f      = MAX_BETA_GROSS / gross
        capped = {t: w * f for t, w in capped.items()}
    return {t: round(w, 6) for t, w in capped.items() if w > 0.001}


def target_weights(market_state):
    global _was_defensive, _brake_cooldown
    qqq     = closes(market_state.get("QQQ"))
    qqq_vol = realized_vol(qqq, 20)
    breadth = breadth_score(market_state)

    if day_brake_active(market_state):
        _was_defensive  = True
        _brake_cooldown = BRAKE_COOLDOWN
        return scale_to_caps({t: w for t, w in DEFENSIVE if closes(market_state.get(t))})

    if breadth < MIN_BREADTH or qqq_vol is None or qqq_vol >= VOL_CEILING:
        _was_defensive = True
        return scale_to_caps({t: w for t, w in DEFENSIVE if closes(market_state.get(t))})

    qqq_sma5    = sma(qqq, 5)
    still_cool  = _brake_cooldown > 0
    recovering  = bool(
        (_was_defensive or still_cool)
        and qqq_sma5 is not None
        and qqq and qqq[-1] < qqq_sma5
    )
    if not recovering and not still_cool:
        _was_defensive = False

    size_factor = 1.0 if breadth >= HALF_BREADTH else 0.65
    if recovering:
        size_factor *= 0.5

    # Score all candidates; skip stop-cooldown tickers
    scored = []
    for t in RISK_CANDIDATES:
        if _stopped_bars.get(t, -9999) + STOP_COOLDOWN >= _bar_count:
            continue
        cs = closes(market_state.get(t))
        s  = ticker_score(cs)
        if s is not None and s >= ENTRY_SCORE_MIN:
            scored.append((s, t))
    scored.sort(reverse=True)

    # No tech cap — both positions can be high-momentum tech names
    winners = [t for _, t in scored[:MAX_POSITIONS]]

    if not winners:
        _was_defensive = True
        return scale_to_caps({t: w for t, w in DEFENSIVE if closes(market_state.get(t))})

    qqq_sma20 = sma(qqq, 20)
    qqq_sma50 = sma(qqq, 50)
    qqq_mom20 = momentum(qqq, 20)
    overlay_on = bool(
        size_factor == 1.0
        and not recovering
        and qqq_sma20 and qqq_sma50 and qqq_mom20 is not None
        and qqq_sma20 > qqq_sma50
        and qqq_mom20 > 0.0
        and qqq_vol < OVERLAY_VOL_CAP
        and closes(market_state.get("QLD"))
        and closes(market_state.get("SSO"))
    )

    # With 3 picks at 0.27 each: 81% stock budget; 63% with overlay (contest concentration rule)
    base_budget = 0.63 if overlay_on else 0.81
    per_name    = min(MAX_WEIGHT, (base_budget * size_factor) / len(winners))
    weights     = {t: per_name for t in winners}
    if overlay_on:
        weights["QLD"] = 0.18
        weights["SSO"] = 0.12

    return scale_to_caps(weights)


# --------------------------------------------------------------------------
# Order construction (unchanged logic)
# --------------------------------------------------------------------------
def guard_orders(positions, prices):
    orders  = []
    stopped = set()
    for ticker, pos in positions.items():
        px = prices.get(ticker)
        if not px or px <= 0 or pos["avg_cost"] <= 0:
            continue
        pnl = (px - pos["avg_cost"]) / pos["avg_cost"]
        qty = int(pos["quantity"])
        if pnl <= STOP_LOSS:
            if qty > 0:
                orders.append({"ticker": ticker, "side": "sell", "quantity": qty})
                _stopped_bars[ticker] = _bar_count
                stopped.add(ticker)
        elif pnl >= TAKE_PROFIT:
            sell_qty = max(1, int(pos["quantity"] * 0.5))
            orders.append({"ticker": ticker, "side": "sell", "quantity": sell_qty})
    return orders, stopped


def build_rebalance_orders(targets, positions, equity, prices, cash, skip_tickers):
    if equity <= 0:
        return []
    min_trade = equity * MIN_TRADE_PCT
    orders    = []
    proceeds  = 0.0

    for ticker, pos in positions.items():
        if ticker in skip_tickers:
            continue
        px  = prices.get(ticker)
        if not px or px <= 0:
            continue
        qty     = pos["quantity"]
        cur_val = qty * px
        tgt_val = equity * targets.get(ticker, 0.0)
        delta   = tgt_val - cur_val
        if ticker not in targets:
            sell_qty = int(qty)
            if sell_qty > 0 and cur_val >= min_trade:
                orders.append({"ticker": ticker, "side": "sell", "quantity": sell_qty})
                proceeds += sell_qty * px
        elif delta < -min_trade:
            sell_qty = min(int(abs(delta) // px), int(qty))
            if sell_qty > 0:
                orders.append({"ticker": ticker, "side": "sell", "quantity": sell_qty})
                proceeds += sell_qty * px

    spendable = max(float(cash), 0.0) + proceeds * 0.98
    for ticker, weight in sorted(targets.items()):
        if ticker in skip_tickers:
            continue
        px = prices.get(ticker)
        if not px or px <= 0:
            continue
        cur_qty = positions.get(ticker, {}).get("quantity", 0.0)
        tgt_val = equity * weight
        delta   = tgt_val - cur_qty * px
        if delta < min_trade:
            continue
        buy_val = min(delta, spendable)
        buy_qty = int(buy_val // px)
        if buy_qty > 0:
            orders.append({"ticker": ticker, "side": "buy", "quantity": buy_qty})
            spendable -= buy_qty * px

    return orders[:15]


def any_position_drifted(portfolio_state, equity):
    lp = portfolio_state.get("last_prices", {}) or {}
    for t, pos in current_positions(portfolio_state).items():
        try:
            px = float(lp.get(t, pos["avg_cost"]))
        except (TypeError, ValueError):
            px = pos["avg_cost"]
        if px > 0 and pos["quantity"] * px / equity > MAX_WEIGHT + 0.03:
            return True
    return False


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def decide(market_state, portfolio_state, cash):
    global _last_rebalance_date, _bar_count, _brake_cooldown

    if not market_state:
        return []

    _bar_count += 1
    today = latest_bar_date(market_state)
    if today is None:
        return []

    if _brake_cooldown > 0:
        _brake_cooldown -= 1

    equity    = portfolio_equity(portfolio_state, cash)
    prices    = market_prices(market_state)
    positions = current_positions(portfolio_state)

    stops, stopped_today = guard_orders(positions, prices)

    days     = days_since_rebalance(market_state)
    drifted  = any_position_drifted(portfolio_state, equity)
    should_rebalance = (
        _last_rebalance_date is None
        or days is None
        or days >= REBALANCE_DAYS
        or drifted
        or bool(stopped_today)
        or day_brake_active(market_state)
    )

    rebal = []
    if should_rebalance:
        targets = target_weights(market_state)
        if targets:
            rebal = build_rebalance_orders(
                targets, positions, equity, prices, cash, stopped_today
            )

    all_orders = stops + rebal
    if all_orders:
        _last_rebalance_date = today

    return all_orders[:45]
