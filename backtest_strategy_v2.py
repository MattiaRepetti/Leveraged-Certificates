"""
Backtest v2 — Strategia Certificati a Leva Fissa (OTTIMIZZATO)
================================================================
Modulo 6 del percorso "Certificati a Leva Fissa"

Ottimizzazioni rispetto alla v1:
  1. Filtro Long/Short asimmetrico: Short richiede ADX > 30 (vs 25 per Long)
  2. Uscita parziale: a +7% cert chiude 50% e sposta stop a breakeven
  3. Costi reali: bid-ask spread + funding giornaliero incorporati nel P&L

Confronta automaticamente v1 (base) vs v2 (ottimizzata) sugli stessi dati.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta

np.random.seed(42)

# ============================================================
# CONFIGURAZIONE
# ============================================================
USE_SYNTHETIC = False
TICKER = "FTSEMIB.MI"
LEVERAGE = 7
RISK_PER_TRADE = 0.01
INITIAL_CAPITAL = 10_000
MAX_HOLDING_DAYS = 5
TARGET_MULTIPLIER = 1.5
STOP_LOSS_ATR_MULT = 1.5

# Parametri scanner di regime
ADX_THRESHOLD_LONG = 25      # Soglia ADX per trade Long
ADX_THRESHOLD_SHORT = 30     # OPT 1: soglia più alta per Short
ADX_WEAK = 20
ROC_THRESHOLD_LONG = 0.5     # %
ROC_THRESHOLD_SHORT = 0.8    # OPT 1: ROC più stringente per Short
ATR_VOL_LIMIT = 1.3

# OPT 2: Uscita parziale
PARTIAL_EXIT_ENABLED = True
PARTIAL_EXIT_THRESHOLD = 0.07  # +7% sul certificato → chiudi 50%
PARTIAL_EXIT_FRACTION = 0.5    # Chiudi il 50% della posizione

# OPT 3: Costi reali
COSTS_ENABLED = True
BID_ASK_SPREAD = 0.008        # 0.8% per round-trip (0.4% entry + 0.4% exit)
DAILY_FUNDING_COST = 0.00014  # ~3.5% annuo / 252 giorni

# Walk-forward
WF_TRAIN_DAYS = 150
WF_TEST_DAYS = 50
WF_STEP_DAYS = 50

print("=" * 65)
print("BACKTEST v2 — STRATEGIA OTTIMIZZATA")
print("=" * 65)
print(f"\n  Ottimizzazioni attive:")
print(f"    1. Filtro asimmetrico:  Long ADX>{ADX_THRESHOLD_LONG} "
      f"| Short ADX>{ADX_THRESHOLD_SHORT}")
print(f"    2. Uscita parziale:    {'SI' if PARTIAL_EXIT_ENABLED else 'NO'} "
      f"(soglia +{PARTIAL_EXIT_THRESHOLD*100:.0f}% cert)")
print(f"    3. Costi reali:        {'SI' if COSTS_ENABLED else 'NO'} "
      f"(spread {BID_ASK_SPREAD*100:.1f}% + funding "
      f"{DAILY_FUNDING_COST*100:.3f}%/gg)")

# ============================================================
# DATI
# ============================================================
if USE_SYNTHETIC:
    print("\nGenerando dati sintetici realistici...")
    n_days = 600
    dates = pd.bdate_range(end=datetime.now(), periods=n_days)

    close = [34000.0]
    regime_dur = 0
    current_regime = 'trend_up'

    for i in range(1, n_days):
        regime_dur += 1
        if regime_dur > np.random.randint(12, 45):
            current_regime = np.random.choice(
                ['trend_up', 'trend_down', 'lateral', 'volatile'],
                p=[0.30, 0.20, 0.35, 0.15])
            regime_dur = 0

        if current_regime == 'trend_up':
            mu, sigma = 0.003, 0.008
        elif current_regime == 'trend_down':
            mu, sigma = -0.0025, 0.009
        elif current_regime == 'lateral':
            mu, sigma = 0.0001, 0.007
        else:
            mu, sigma = 0.0, 0.018

        ret = np.random.normal(mu, sigma)
        close.append(close[-1] * (1 + ret))

    close = np.array(close)
    high = close * (1 + np.abs(np.random.normal(0, 0.004, n_days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.004, n_days)))

    df = pd.DataFrame({
        'Open': close * (1 + np.random.normal(0, 0.001, n_days)),
        'High': high, 'Low': low, 'Close': close,
        'Volume': np.random.randint(500e6, 2e9, n_days),
    }, index=dates)
    print(f"Dati sintetici: {len(df)} righe")
else:
    print(f"\nScaricando {TICKER}...")
    end = datetime.now()
    start = end - timedelta(days=900)
    df = yf.download(TICKER, start=start, end=end, progress=False)
    if hasattr(df.columns, 'levels'):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    print(f"Dati scaricati: {len(df)} righe")

# ============================================================
# INDICATORI
# ============================================================
def compute_indicators(df):
    h, l, c = df['High'], df['Low'], df['Close']

    tr = pd.concat([h - l, (h - c.shift(1)).abs(),
                     (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    atr_pct = (atr14 / c) * 100
    atr_pct_ma = atr_pct.rolling(20).mean()

    plus_dm = h.diff()
    minus_dm = -l.diff()
    plus_dm = pd.Series(np.where((plus_dm > minus_dm) & (plus_dm > 0),
                                  plus_dm, 0), index=df.index)
    minus_dm = pd.Series(np.where((minus_dm > plus_dm) & (minus_dm > 0),
                                   minus_dm, 0), index=df.index)
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).abs()
    adx = dx.rolling(14).mean()

    roc5 = ((c - c.shift(5)) / c.shift(5)) * 100

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

    return adx, plus_di, minus_di, atr_pct, atr_pct_ma, roc5, macd_hist, atr14


adx, plus_di, minus_di, atr_pct, atr_pct_ma, roc5, macd_hist, atr14 = \
    compute_indicators(df)

# ============================================================
# REGIME CLASSIFICATION — v1 (base) e v2 (ottimizzata)
# ============================================================
def get_signal_v1(i):
    """Segnale ORIGINALE (v1): soglie simmetriche Long/Short."""
    if i < 1:
        return None
    vals = [adx.iloc[i], atr_pct.iloc[i], atr_pct_ma.iloc[i],
            roc5.iloc[i], macd_hist.iloc[i], macd_hist.iloc[i-1],
            plus_di.iloc[i], minus_di.iloc[i]]
    if any(pd.isna(v) for v in vals):
        return None

    a, atr_v, atr_ma, roc, mh, mh_prev, pdi, mdi = vals

    if a < ADX_WEAK:
        return None
    if atr_v > atr_ma * ATR_VOL_LIMIT:
        return None
    vol_ok = atr_v <= atr_ma

    if pdi > mdi:
        direction = 'LONG'
        macd_exp = mh > mh_prev and mh > 0
        roc_ok = roc > 0.5
    else:
        direction = 'SHORT'
        macd_exp = mh < mh_prev and mh < 0
        roc_ok = roc < -0.5

    if a >= 25 and vol_ok and macd_exp and roc_ok:
        return direction
    return None


def get_signal_v2(i):
    """Segnale OTTIMIZZATO (v2): soglie asimmetriche Long/Short."""
    if i < 1:
        return None
    vals = [adx.iloc[i], atr_pct.iloc[i], atr_pct_ma.iloc[i],
            roc5.iloc[i], macd_hist.iloc[i], macd_hist.iloc[i-1],
            plus_di.iloc[i], minus_di.iloc[i]]
    if any(pd.isna(v) for v in vals):
        return None

    a, atr_v, atr_ma, roc, mh, mh_prev, pdi, mdi = vals

    if a < ADX_WEAK:
        return None
    if atr_v > atr_ma * ATR_VOL_LIMIT:
        return None
    vol_ok = atr_v <= atr_ma

    if pdi > mdi:
        direction = 'LONG'
        adx_ok = a >= ADX_THRESHOLD_LONG
        macd_exp = mh > mh_prev and mh > 0
        roc_ok = roc > ROC_THRESHOLD_LONG
    else:
        direction = 'SHORT'
        adx_ok = a >= ADX_THRESHOLD_SHORT      # OPT 1: più stringente
        macd_exp = mh < mh_prev and mh < 0
        roc_ok = roc < -ROC_THRESHOLD_SHORT     # OPT 1: più stringente

    if adx_ok and vol_ok and macd_exp and roc_ok:
        return direction
    return None


# ============================================================
# BACKTEST ENGINE (con uscita parziale e costi)
# ============================================================
class Trade:
    def __init__(self, entry_date, direction, entry_price, stop_price,
                 target_price, position_size, entry_idx):
        self.entry_date = entry_date
        self.direction = direction
        self.entry_price = entry_price
        self.stop_price = stop_price
        self.target_price = target_price
        self.position_size = position_size
        self.remaining_size = position_size
        self.entry_idx = entry_idx
        self.exit_date = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl_pct = 0
        self.pnl_eur = 0
        self.holding_days = 0
        self.cert_return = 0
        self.partial_exit_done = False
        self.partial_pnl_eur = 0
        self.total_costs = 0


def run_backtest(df, start_idx, end_idx, capital, signal_fn,
                 use_partial=False, use_costs=False):
    """Esegue il backtest su un segmento del dataframe."""
    trades = []
    equity = [capital]
    equity_dates = [df.index[start_idx]]
    current_capital = capital
    in_trade = None

    for i in range(start_idx, min(end_idx, len(df) - 1)):
        date = df.index[i]
        close_today = df['Close'].iloc[i]

        # --- Gestione posizione aperta ---
        if in_trade is not None:
            in_trade.holding_days += 1
            day_return = (df['Close'].iloc[i] - df['Close'].iloc[i-1]) / \
                         df['Close'].iloc[i-1]

            if in_trade.direction == 'LONG':
                lev_return = LEVERAGE * day_return
            else:
                lev_return = -LEVERAGE * day_return

            # OPT 3: Sottrarre funding cost giornaliero
            if use_costs:
                lev_return -= DAILY_FUNDING_COST
                in_trade.total_costs += in_trade.remaining_size * DAILY_FUNDING_COST

            in_trade.cert_return = (1 + in_trade.cert_return) * \
                                   (1 + lev_return) - 1

            # OPT 2: Uscita parziale
            if (use_partial and not in_trade.partial_exit_done and
                    in_trade.cert_return >= PARTIAL_EXIT_THRESHOLD):
                partial_size = in_trade.remaining_size * PARTIAL_EXIT_FRACTION
                partial_pnl = partial_size * in_trade.cert_return

                # Costo spread sulla vendita parziale
                if use_costs:
                    partial_cost = partial_size * BID_ASK_SPREAD / 2
                    partial_pnl -= partial_cost
                    in_trade.total_costs += partial_cost

                in_trade.partial_pnl_eur += partial_pnl
                current_capital += partial_pnl
                in_trade.remaining_size -= partial_size
                in_trade.partial_exit_done = True

                # Sposta stop a breakeven
                in_trade.stop_price = in_trade.entry_price

            exit_now = False
            reason = None

            # Check stop loss
            if in_trade.direction == 'LONG' and close_today <= in_trade.stop_price:
                exit_now, reason = True, 'STOP_LOSS'
                if in_trade.partial_exit_done:
                    reason = 'STOP_BE'  # Stop a breakeven dopo partial
            elif in_trade.direction == 'SHORT' and close_today >= in_trade.stop_price:
                exit_now, reason = True, 'STOP_LOSS'
                if in_trade.partial_exit_done:
                    reason = 'STOP_BE'

            # Check target
            if in_trade.direction == 'LONG' and close_today >= in_trade.target_price:
                exit_now, reason = True, 'TARGET'
            elif in_trade.direction == 'SHORT' and close_today <= in_trade.target_price:
                exit_now, reason = True, 'TARGET'

            # Check max holding
            if in_trade.holding_days >= MAX_HOLDING_DAYS:
                exit_now, reason = True, 'TIME_STOP'

            # Check regime deterioration
            signal = signal_fn(i)
            if in_trade.direction == 'LONG' and signal != 'LONG':
                if in_trade.holding_days >= 2:
                    exit_now, reason = True, 'REGIME_EXIT'
            elif in_trade.direction == 'SHORT' and signal != 'SHORT':
                if in_trade.holding_days >= 2:
                    exit_now, reason = True, 'REGIME_EXIT'

            if exit_now:
                in_trade.exit_date = date
                in_trade.exit_price = close_today

                # P&L sulla porzione rimasta
                remaining_pnl = in_trade.remaining_size * in_trade.cert_return

                # OPT 3: Costo spread sulla chiusura finale
                if use_costs:
                    exit_cost = in_trade.remaining_size * BID_ASK_SPREAD / 2
                    remaining_pnl -= exit_cost
                    in_trade.total_costs += exit_cost

                in_trade.exit_reason = reason
                in_trade.pnl_eur = in_trade.partial_pnl_eur + remaining_pnl
                in_trade.pnl_pct = (in_trade.pnl_eur / in_trade.position_size) * 100
                current_capital += remaining_pnl
                trades.append(in_trade)
                in_trade = None

        # --- Nuovi ingressi ---
        elif in_trade is None:
            signal = signal_fn(i)
            if signal is not None:
                atr_val = atr14.iloc[i]
                if pd.isna(atr_val) or atr_val <= 0:
                    continue

                stop_dist_pct = (STOP_LOSS_ATR_MULT * atr_val) / close_today
                if stop_dist_pct < 0.003:
                    stop_dist_pct = 0.003
                if stop_dist_pct > 0.03:
                    continue

                max_loss = current_capital * RISK_PER_TRADE
                pos_size = max_loss / (LEVERAGE * stop_dist_pct)
                pos_size = min(pos_size, current_capital * 0.25)

                if signal == 'LONG':
                    stop_p = close_today * (1 - stop_dist_pct)
                    target_p = close_today * (1 + stop_dist_pct * TARGET_MULTIPLIER)
                else:
                    stop_p = close_today * (1 + stop_dist_pct)
                    target_p = close_today * (1 - stop_dist_pct * TARGET_MULTIPLIER)

                trade = Trade(
                    entry_date=date, direction=signal,
                    entry_price=close_today, stop_price=stop_p,
                    target_price=target_p, position_size=pos_size,
                    entry_idx=i)
                trade.cert_return = 0

                # OPT 3: Costo spread all'ingresso
                if use_costs:
                    entry_cost = pos_size * BID_ASK_SPREAD / 2
                    current_capital -= entry_cost
                    trade.total_costs = entry_cost

                in_trade = trade

        equity.append(current_capital)
        equity_dates.append(date)

    # Chiudi trade aperto a fine periodo
    if in_trade is not None:
        in_trade.exit_date = df.index[min(end_idx, len(df)-1)]
        in_trade.exit_price = df['Close'].iloc[min(end_idx, len(df)-1)]
        in_trade.exit_reason = 'END_OF_PERIOD'
        remaining_pnl = in_trade.remaining_size * in_trade.cert_return
        if use_costs:
            exit_cost = in_trade.remaining_size * BID_ASK_SPREAD / 2
            remaining_pnl -= exit_cost
            in_trade.total_costs += exit_cost
        in_trade.pnl_eur = in_trade.partial_pnl_eur + remaining_pnl
        in_trade.pnl_pct = (in_trade.pnl_eur / in_trade.position_size) * 100
        current_capital += remaining_pnl
        trades.append(in_trade)
        equity.append(current_capital)
        equity_dates.append(in_trade.exit_date)

    return trades, equity, equity_dates, current_capital


# ============================================================
# METRICHE
# ============================================================
def compute_metrics(trades, initial_cap, final_cap, label=""):
    if not trades:
        print(f"  [{label}] Nessun trade eseguito.")
        return {}

    pnls = [t.pnl_eur for t in trades]
    pnl_pcts = [t.pnl_pct for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    holding = [t.holding_days for t in trades]
    total_costs = sum(t.total_costs for t in trades)
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    n_long = sum(1 for t in trades if t.direction == 'LONG')
    n_short = sum(1 for t in trades if t.direction == 'SHORT')
    n_partial = sum(1 for t in trades if t.partial_exit_done)

    total_return = (final_cap / initial_cap - 1) * 100
    n_trades = len(trades)
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = abs(sum(wins) / sum(losses)) \
        if losses and sum(losses) != 0 else float('inf')
    expectancy = np.mean(pnls)

    eq_arr = np.array([initial_cap] + [initial_cap + sum(pnls[:i+1])
                       for i in range(len(pnls))])
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak * 100
    max_dd = dd.min()

    if len(pnl_pcts) > 1 and np.std(pnl_pcts) > 0:
        trades_per_year = 252 / np.mean(holding) if np.mean(holding) > 0 else 50
        sharpe = (np.mean(pnl_pcts) / np.std(pnl_pcts)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0

    downside = [p for p in pnl_pcts if p < 0]
    if downside and np.std(downside) > 0:
        sortino = (np.mean(pnl_pcts) / np.std(downside)) * np.sqrt(trades_per_year)
    else:
        sortino = 0

    max_consec_loss = 0
    current_streak = 0
    for p in pnls:
        if p <= 0:
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            current_streak = 0

    print(f"\n  [{label}]")
    print(f"  Trades totali:        {n_trades} (L:{n_long} S:{n_short})")
    print(f"  Win rate:             {win_rate:.1f}%")
    if wins:
        print(f"  Avg win:              +{avg_win:.2f} EUR "
              f"({np.mean([p for p in pnl_pcts if p > 0]):.2f}%)")
    if losses:
        print(f"  Avg loss:             {avg_loss:.2f} EUR "
              f"({np.mean([p for p in pnl_pcts if p <= 0]):.2f}%)")
    print(f"  Expectancy:           {expectancy:+.2f} EUR/trade")
    print(f"  Profit factor:        {profit_factor:.2f}")
    print(f"  Rendimento totale:    {total_return:+.2f}%")
    print(f"  Max drawdown:         {max_dd:.2f}%")
    print(f"  Sharpe (ann.):        {sharpe:.2f}")
    print(f"  Sortino (ann.):       {sortino:.2f}")
    print(f"  Holding medio:        {np.mean(holding):.1f} giorni")
    print(f"  Max consec. losses:   {max_consec_loss}")
    if n_partial > 0:
        print(f"  Uscite parziali:      {n_partial}")
    if total_costs > 0:
        print(f"  Costi totali:         {total_costs:.2f} EUR")
    print(f"  Motivi uscita:        {reasons}")

    return {
        'n_trades': n_trades, 'win_rate': win_rate,
        'total_return': total_return, 'max_dd': max_dd,
        'sharpe': sharpe, 'sortino': sortino,
        'profit_factor': profit_factor, 'expectancy': expectancy,
    }


# ============================================================
# ESECUZIONE: CONFRONTO v1 vs v2
# ============================================================
start_bt = 60

# --- v1: Base (nessuna ottimizzazione, nessun costo) ---
print(f"\n{'─' * 65}")
print("RUN v1 — BASE (no costi, soglie simmetriche, no partial)")
print(f"{'─' * 65}")
trades_v1, eq_v1, eqd_v1, final_v1 = run_backtest(
    df, start_bt, len(df), INITIAL_CAPITAL, get_signal_v1,
    use_partial=False, use_costs=False)
m_v1 = compute_metrics(trades_v1, INITIAL_CAPITAL, final_v1, "v1 BASE")

# --- v2: Ottimizzata (costi reali, soglie asimmetriche, partial exit) ---
print(f"\n{'─' * 65}")
print("RUN v2 — OTTIMIZZATA (costi reali, asimmetrica, partial exit)")
print(f"{'─' * 65}")
trades_v2, eq_v2, eqd_v2, final_v2 = run_backtest(
    df, start_bt, len(df), INITIAL_CAPITAL, get_signal_v2,
    use_partial=PARTIAL_EXIT_ENABLED, use_costs=COSTS_ENABLED)
m_v2 = compute_metrics(trades_v2, INITIAL_CAPITAL, final_v2, "v2 OTTIMIZZATA")

# --- Confronto diretto ---
print(f"\n{'═' * 65}")
print("CONFRONTO v1 vs v2")
print(f"{'═' * 65}")
if m_v1 and m_v2:
    metrics_names = [
        ('n_trades', 'Trades'),
        ('win_rate', 'Win rate (%)'),
        ('expectancy', 'Expectancy (EUR)'),
        ('profit_factor', 'Profit factor'),
        ('total_return', 'Return (%)'),
        ('max_dd', 'Max DD (%)'),
        ('sharpe', 'Sharpe'),
        ('sortino', 'Sortino'),
    ]
    print(f"\n  {'Metrica':<22} {'v1 Base':>12} {'v2 Ottim.':>12} {'Delta':>12}")
    print(f"  {'─'*58}")
    for key, name in metrics_names:
        v1_val = m_v1.get(key, 0)
        v2_val = m_v2.get(key, 0)
        delta = v2_val - v1_val
        print(f"  {name:<22} {v1_val:>12.2f} {v2_val:>12.2f} {delta:>+12.2f}")

# ============================================================
# WALK-FORWARD v2
# ============================================================
print(f"\n{'─' * 65}")
print("WALK-FORWARD VALIDATION — v2 OTTIMIZZATA")
print(f"{'─' * 65}")
print(f"  Training: {WF_TRAIN_DAYS}gg | Test: {WF_TEST_DAYS}gg | "
      f"Step: {WF_STEP_DAYS}gg")

wf_results = []
wf_trades_all = []
wf_start = start_bt

while wf_start + WF_TRAIN_DAYS + WF_TEST_DAYS <= len(df):
    train_end = wf_start + WF_TRAIN_DAYS
    test_end = train_end + WF_TEST_DAYS

    test_trades, test_eq, test_eq_dates, test_final = run_backtest(
        df, train_end, test_end, INITIAL_CAPITAL, get_signal_v2,
        use_partial=PARTIAL_EXIT_ENABLED, use_costs=COSTS_ENABLED)

    period_label = (f"{df.index[train_end].strftime('%Y-%m-%d')} -> "
                    f"{df.index[min(test_end-1, len(df)-1)].strftime('%Y-%m-%d')}")

    if test_trades:
        test_pnls = [t.pnl_eur for t in test_trades]
        test_ret = (test_final / INITIAL_CAPITAL - 1) * 100
        test_wr = sum(1 for p in test_pnls if p > 0) / len(test_pnls) * 100
        wf_results.append({
            'period': period_label,
            'n_trades': len(test_trades),
            'return': test_ret,
            'win_rate': test_wr,
        })
        wf_trades_all.extend(test_trades)
        print(f"  {period_label}: {len(test_trades)} trades, "
              f"return {test_ret:+.2f}%, WR {test_wr:.0f}%")
    else:
        wf_results.append({
            'period': period_label, 'n_trades': 0,
            'return': 0, 'win_rate': 0,
        })
        print(f"  {period_label}: 0 trades")

    wf_start += WF_STEP_DAYS

if wf_trades_all:
    wf_final = INITIAL_CAPITAL
    for t in wf_trades_all:
        wf_final += t.pnl_eur
    print(f"\n  Walk-forward aggregato:")
    wf_m = compute_metrics(wf_trades_all, INITIAL_CAPITAL, wf_final,
                            "WF v2 OOS")

# ============================================================
# GRAFICI
# ============================================================
fig = plt.figure(figsize=(16, 24))
fig.patch.set_facecolor('white')
gs = GridSpec(5, 2, figure=fig, hspace=0.35, wspace=0.3)

# 1. Equity curve: v1 vs v2
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(eqd_v1, eq_v1, color='#B4B2A9', linewidth=1.2, label='v1 Base')
ax1.plot(eqd_v2, eq_v2, color='#185FA5', linewidth=1.8, label='v2 Ottimizzata')
ax1.axhline(INITIAL_CAPITAL, color='#888780', linewidth=0.5, linestyle='--')
ax1.set_title('Equity curve — v1 vs v2', fontsize=14, fontweight='500')
ax1.set_ylabel('Capitale (EUR)', fontsize=12)
ax1.legend(fontsize=11)
ax1.tick_params(labelsize=10)

# 2. P&L per trade v2
ax2 = fig.add_subplot(gs[1, 0])
if trades_v2:
    pnl_pcts_v2 = [t.pnl_pct for t in trades_v2]
    colors_pnl = ['#639922' if p > 0 else '#E24B4A' for p in pnl_pcts_v2]
    bars = ax2.bar(range(len(pnl_pcts_v2)), pnl_pcts_v2,
                   color=colors_pnl, alpha=0.7)
    # Evidenzia trade con uscita parziale
    for j, t in enumerate(trades_v2):
        if t.partial_exit_done:
            bars[j].set_edgecolor('#534AB7')
            bars[j].set_linewidth(2)
    ax2.axhline(0, color='#2C2C2A', linewidth=0.5)
    ax2.set_title('P&L per trade v2 (bordo viola = partial exit)',
                  fontsize=13, fontweight='500')
    ax2.set_xlabel('Trade #', fontsize=11)
    ax2.set_ylabel('P&L (%)', fontsize=11)

# 3. Distribuzione P&L v2
ax3 = fig.add_subplot(gs[1, 1])
if trades_v2:
    ax3.hist(pnl_pcts_v2, bins=25, color='#378ADD', alpha=0.7, edgecolor='none')
    ax3.axvline(0, color='#2C2C2A', linewidth=1)
    ax3.axvline(np.mean(pnl_pcts_v2), color='#D85A30', linewidth=2,
                linestyle='--',
                label=f'Media: {np.mean(pnl_pcts_v2):+.2f}%')
    ax3.set_title('Distribuzione P&L v2', fontsize=14, fontweight='500')
    ax3.set_xlabel('P&L (%)', fontsize=11)
    ax3.legend(fontsize=10)

# 4. Motivi di uscita v2
ax4 = fig.add_subplot(gs[2, 0])
if trades_v2:
    reasons_list = [t.exit_reason for t in trades_v2]
    reason_labels, reason_counts = np.unique(reasons_list, return_counts=True)
    reason_colors = {
        'TARGET': '#639922', 'STOP_LOSS': '#E24B4A',
        'TIME_STOP': '#EF9F27', 'REGIME_EXIT': '#378ADD',
        'END_OF_PERIOD': '#888780', 'STOP_BE': '#534AB7',
    }
    bar_colors = [reason_colors.get(r, '#888780') for r in reason_labels]
    ax4.barh(reason_labels, reason_counts, color=bar_colors, alpha=0.8)
    ax4.set_title('Motivi di uscita v2', fontsize=14, fontweight='500')
    ax4.set_xlabel('Numero trades', fontsize=11)

# 5. Long vs Short performance v2
ax5 = fig.add_subplot(gs[2, 1])
if trades_v2:
    long_pnls = [t.pnl_pct for t in trades_v2 if t.direction == 'LONG']
    short_pnls = [t.pnl_pct for t in trades_v2 if t.direction == 'SHORT']
    data_box = []
    labels_box = []
    if long_pnls:
        data_box.append(long_pnls)
        labels_box.append(f'Long (N={len(long_pnls)})')
    if short_pnls:
        data_box.append(short_pnls)
        labels_box.append(f'Short (N={len(short_pnls)})')
    if data_box:
        bp = ax5.boxplot(data_box, labels=labels_box, patch_artist=True)
        colors_bp = ['#639922', '#E24B4A']
        for patch, color in zip(bp['boxes'], colors_bp[:len(data_box)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
        ax5.axhline(0, color='#2C2C2A', linewidth=0.5, linestyle='--')
        ax5.set_title('P&L Long vs Short v2', fontsize=14, fontweight='500')
        ax5.set_ylabel('P&L (%)', fontsize=11)

# 6. Confronto metriche v1 vs v2
ax6 = fig.add_subplot(gs[3, 0])
if m_v1 and m_v2:
    compare_keys = ['win_rate', 'profit_factor', 'sharpe', 'sortino']
    compare_names = ['Win rate', 'Profit\nfactor', 'Sharpe', 'Sortino']
    x = np.arange(len(compare_keys))
    w = 0.35
    vals_v1 = [m_v1.get(k, 0) for k in compare_keys]
    vals_v2 = [m_v2.get(k, 0) for k in compare_keys]
    ax6.bar(x - w/2, vals_v1, w, label='v1 Base', color='#B4B2A9', alpha=0.7)
    ax6.bar(x + w/2, vals_v2, w, label='v2 Ottim.', color='#185FA5', alpha=0.7)
    ax6.set_xticks(x)
    ax6.set_xticklabels(compare_names, fontsize=10)
    ax6.set_title('Confronto metriche v1 vs v2', fontsize=14, fontweight='500')
    ax6.legend(fontsize=10)

# 7. Walk-forward v2
ax7 = fig.add_subplot(gs[3, 1])
if wf_results:
    wf_returns = [w['return'] for w in wf_results]
    wf_labels = [w['period'].split(' -> ')[0] for w in wf_results]
    wf_colors = ['#639922' if r > 0 else '#E24B4A' for r in wf_returns]
    ax7.bar(range(len(wf_returns)), wf_returns, color=wf_colors, alpha=0.7)
    ax7.set_xticks(range(len(wf_labels)))
    ax7.set_xticklabels(wf_labels, rotation=45, ha='right', fontsize=9)
    ax7.axhline(0, color='#2C2C2A', linewidth=0.5)
    ax7.set_title('Walk-forward v2 OOS', fontsize=14, fontweight='500')
    ax7.set_ylabel('Rendimento (%)', fontsize=11)

# 8. Costi cumulati v2
ax8 = fig.add_subplot(gs[4, :])
if trades_v2:
    cum_costs = np.cumsum([t.total_costs for t in trades_v2])
    cum_pnl_gross = np.cumsum([t.pnl_eur + t.total_costs for t in trades_v2])
    cum_pnl_net = np.cumsum([t.pnl_eur for t in trades_v2])
    ax8.plot(range(1, len(trades_v2)+1), cum_pnl_gross,
             color='#639922', linewidth=1.5, label='P&L lordo (no costi)')
    ax8.plot(range(1, len(trades_v2)+1), cum_pnl_net,
             color='#185FA5', linewidth=1.8, label='P&L netto')
    ax8.fill_between(range(1, len(trades_v2)+1), cum_pnl_gross, cum_pnl_net,
                      alpha=0.15, color='#E24B4A', label='Costi cumulati')
    ax8.axhline(0, color='#2C2C2A', linewidth=0.5, linestyle='--')
    ax8.set_title('P&L cumulato: lordo vs netto (impatto costi)',
                  fontsize=14, fontweight='500')
    ax8.set_xlabel('Trade #', fontsize=11)
    ax8.set_ylabel('EUR', fontsize=11)
    ax8.legend(fontsize=10)

fig.suptitle(f'Backtest v2 Ottimizzato — Leva {LEVERAGE}x — {TICKER}\n'
             f'v1: {INITIAL_CAPITAL:,} -> {final_v1:,.0f} EUR '
             f'({(final_v1/INITIAL_CAPITAL-1)*100:+.1f}%) | '
             f'v2: {INITIAL_CAPITAL:,} -> {final_v2:,.0f} EUR '
             f'({(final_v2/INITIAL_CAPITAL-1)*100:+.1f}%)',
             fontsize=14, fontweight='500', y=0.995)

plt.savefig('backtest_results.png', dpi=150,
            bbox_inches='tight', facecolor='white')
print(f"\n[Grafico salvato: backtest_results.png]")

# ============================================================
# TRADE LOG v2
# ============================================================
print(f"\n{'─' * 65}")
print("TRADE LOG v2 (ultimi 15 trades)")
print(f"{'─' * 65}")
print(f"{'Data':>12} {'Dir':>5} {'Entry':>10} {'Exit':>10} "
      f"{'P&L%':>8} {'P&L EUR':>10} {'Gg':>4} {'Partial':>7} "
      f"{'Costi':>7} {'Motivo':>14}")
print("─" * 95)

for t in trades_v2[-15:]:
    partial_flag = "SI" if t.partial_exit_done else "—"
    print(f"{t.entry_date.strftime('%Y-%m-%d'):>12} {t.direction:>5} "
          f"{t.entry_price:>10.2f} {t.exit_price:>10.2f} "
          f"{t.pnl_pct:>+8.2f} {t.pnl_eur:>+10.2f} "
          f"{t.holding_days:>4} {partial_flag:>7} "
          f"{t.total_costs:>7.2f} {t.exit_reason:>14}")

print(f"\n{'═' * 65}")
print("BACKTEST v2 COMPLETATO")
print(f"{'═' * 65}")
