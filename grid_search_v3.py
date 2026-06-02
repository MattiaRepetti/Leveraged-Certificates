"""
Grid Search — parametri strategia backtest_strategy_v3
=======================================================
Ottimizza i parametri chiave della strategia su walk-forward OOS
per evitare overfitting in-sample.

Parametri ottimizzati:
  - ADX_THRESHOLD_LONG / ADX_THRESHOLD_SHORT
  - ROC_THRESHOLD_LONG / ROC_THRESHOLD_SHORT
  - TARGET_MULTIPLIER
  - STOP_LOSS_ATR_MULT
  - MAX_HOLDING_DAYS
  - PARTIAL_EXIT_THRESHOLD

Metrica principale: Sharpe OOS (media su tutti gli asset e leve).
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# CONFIGURAZIONE FISSA (non ottimizzata)
# ============================================================
TICKERS = {
    'FTSE MIB':    'FTSEMIB.MI',
    'DAX':         '^GDAXI',
    'EuroStoxx50': '^STOXX50E',
    'S&P 500':     '^GSPC',
    'CAC 40':      '^FCHI',
}
LEVERAGE_LEVELS = [3, 5, 7]
RISK_PER_TRADE    = 0.01
INITIAL_CAPITAL   = 10_000
BASE_BID_ASK      = 0.008
ANNUAL_FUNDING_RATE = 0.035
ADX_WEAK          = 20
ATR_VOL_LIMIT     = 1.3
PARTIAL_EXIT_FRACTION = 0.5

# Walk-forward
WF_TRAIN_DAYS = 150
WF_TEST_DAYS  = 50
WF_STEP_DAYS  = 50

# ============================================================
# GRIGLIA DEI PARAMETRI
# ============================================================
PARAM_GRID = {
    'ADX_THRESHOLD_LONG':   [20, 25, 30],
    'ADX_THRESHOLD_SHORT':  [25, 30, 35],
    'ROC_THRESHOLD_LONG':   [0.3, 0.5, 0.8],
    'ROC_THRESHOLD_SHORT':  [0.5, 0.8, 1.2],
    'TARGET_MULTIPLIER':    [1.2, 1.5, 2.0],
    'STOP_LOSS_ATR_MULT':   [1.0, 1.5, 2.0],
    'MAX_HOLDING_DAYS':     [3, 5, 7],
    'PARTIAL_EXIT_THRESHOLD': [0.05, 0.07, 0.10],
}

# Numero totale di combinazioni
total_combos = 1
for v in PARAM_GRID.values():
    total_combos *= len(v)

print("=" * 75)
print("GRID SEARCH — OTTIMIZZAZIONE PARAMETRI BACKTEST v3")
print("=" * 75)
print(f"\nParametri nella griglia:")
for k, v in PARAM_GRID.items():
    print(f"  {k:<28}: {v}")
print(f"\nCombinazioni totali: {total_combos}")
print(f"Asset x Leve:        {len(TICKERS)} x {len(LEVERAGE_LEVELS)}")
print(f"Metrica principale:  Sharpe OOS (walk-forward)")

# ============================================================
# FUNZIONI CORE
# ============================================================
def compute_indicators(df):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift(1)).abs(),
                    (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    atr_pct = (atr14 / c) * 100
    atr_pct_ma = atr_pct.rolling(20).mean()

    plus_dm  = h.diff()
    minus_dm = -l.diff()
    plus_dm  = pd.Series(np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0), index=df.index)
    minus_dm = pd.Series(np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0), index=df.index)
    plus_di  = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).abs()
    adx = dx.rolling(14).mean()

    roc5      = ((c - c.shift(5)) / c.shift(5)) * 100
    ema12     = c.ewm(span=12, adjust=False).mean()
    ema26     = c.ewm(span=26, adjust=False).mean()
    macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    return adx, plus_di, minus_di, atr_pct, atr_pct_ma, roc5, macd_hist, atr14


def make_signal_fn(adx, atr_pct, atr_pct_ma, roc5, macd_hist, plus_di, minus_di, p):
    adx_long  = p['ADX_THRESHOLD_LONG']
    adx_short = p['ADX_THRESHOLD_SHORT']
    roc_long  = p['ROC_THRESHOLD_LONG']
    roc_short = p['ROC_THRESHOLD_SHORT']

    def get_signal(i):
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
            adx_ok   = a >= adx_long
            macd_exp = mh > mh_prev and mh > 0
            roc_ok   = roc > roc_long
        else:
            direction = 'SHORT'
            adx_ok   = a >= adx_short
            macd_exp = mh < mh_prev and mh < 0
            roc_ok   = roc < -roc_short
        if adx_ok and vol_ok and macd_exp and roc_ok:
            return direction
        return None
    return get_signal


class Trade:
    __slots__ = ('entry_date', 'direction', 'entry_price', 'stop_price',
                 'target_price', 'position_size', 'remaining_size',
                 'entry_idx', 'exit_date', 'exit_price', 'exit_reason',
                 'pnl_pct', 'pnl_eur', 'holding_days', 'cert_return',
                 'partial_exit_done', 'partial_pnl_eur', 'total_costs')

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
        self.pnl_pct = 0.0
        self.pnl_eur = 0.0
        self.holding_days = 0
        self.cert_return = 0.0
        self.partial_exit_done = False
        self.partial_pnl_eur = 0.0
        self.total_costs = 0.0


def run_backtest(df, start_idx, end_idx, capital, signal_fn, leverage,
                 bid_ask, daily_funding, p):
    target_mult  = p['TARGET_MULTIPLIER']
    sl_atr_mult  = p['STOP_LOSS_ATR_MULT']
    max_hold     = p['MAX_HOLDING_DAYS']
    partial_thr  = p['PARTIAL_EXIT_THRESHOLD']

    trades = []
    current_capital = capital
    in_trade = None
    indicators = compute_indicators(df)
    atr14_local = indicators[7]

    for i in range(start_idx, min(end_idx, len(df) - 1)):
        date = df.index[i]
        close_today = df['Close'].iloc[i]

        if in_trade is not None:
            in_trade.holding_days += 1
            day_return = (df['Close'].iloc[i] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1]
            lev_return = leverage * day_return if in_trade.direction == 'LONG' else -leverage * day_return
            lev_return -= daily_funding
            in_trade.total_costs += in_trade.remaining_size * daily_funding
            in_trade.cert_return = (1 + in_trade.cert_return) * (1 + lev_return) - 1

            if not in_trade.partial_exit_done and in_trade.cert_return >= partial_thr:
                partial_size = in_trade.remaining_size * PARTIAL_EXIT_FRACTION
                partial_pnl  = partial_size * in_trade.cert_return
                partial_cost = partial_size * bid_ask / 2
                partial_pnl -= partial_cost
                in_trade.total_costs += partial_cost
                in_trade.partial_pnl_eur += partial_pnl
                current_capital += partial_pnl
                in_trade.remaining_size -= partial_size
                in_trade.partial_exit_done = True
                in_trade.stop_price = in_trade.entry_price

            exit_now, reason = False, None

            if in_trade.direction == 'LONG' and close_today <= in_trade.stop_price:
                exit_now = True
                reason = 'STOP_BE' if in_trade.partial_exit_done else 'STOP_LOSS'
            elif in_trade.direction == 'SHORT' and close_today >= in_trade.stop_price:
                exit_now = True
                reason = 'STOP_BE' if in_trade.partial_exit_done else 'STOP_LOSS'

            if in_trade.direction == 'LONG' and close_today >= in_trade.target_price:
                exit_now, reason = True, 'TARGET'
            elif in_trade.direction == 'SHORT' and close_today <= in_trade.target_price:
                exit_now, reason = True, 'TARGET'

            if in_trade.holding_days >= max_hold:
                exit_now, reason = True, 'TIME_STOP'

            signal = signal_fn(i)
            if in_trade.direction == 'LONG' and signal != 'LONG' and in_trade.holding_days >= 2:
                exit_now, reason = True, 'REGIME_EXIT'
            elif in_trade.direction == 'SHORT' and signal != 'SHORT' and in_trade.holding_days >= 2:
                exit_now, reason = True, 'REGIME_EXIT'

            if exit_now:
                in_trade.exit_date  = date
                in_trade.exit_price = close_today
                remaining_pnl = in_trade.remaining_size * in_trade.cert_return
                exit_cost     = in_trade.remaining_size * bid_ask / 2
                remaining_pnl -= exit_cost
                in_trade.total_costs += exit_cost
                in_trade.exit_reason = reason
                in_trade.pnl_eur = in_trade.partial_pnl_eur + remaining_pnl
                in_trade.pnl_pct = (in_trade.pnl_eur / in_trade.position_size) * 100
                current_capital += remaining_pnl
                trades.append(in_trade)
                in_trade = None

        elif in_trade is None:
            signal = signal_fn(i)
            if signal is not None:
                atr_val = atr14_local.iloc[i]
                if pd.isna(atr_val) or atr_val <= 0:
                    continue
                stop_dist_pct = (sl_atr_mult * atr_val) / close_today
                stop_dist_pct = max(stop_dist_pct, 0.003)
                if stop_dist_pct > 0.03:
                    continue

                max_loss = current_capital * RISK_PER_TRADE
                pos_size = max_loss / (leverage * stop_dist_pct)
                pos_size = min(pos_size, current_capital * 0.25)

                if signal == 'LONG':
                    stop_p   = close_today * (1 - stop_dist_pct)
                    target_p = close_today * (1 + stop_dist_pct * target_mult)
                else:
                    stop_p   = close_today * (1 + stop_dist_pct)
                    target_p = close_today * (1 - stop_dist_pct * target_mult)

                trade = Trade(
                    entry_date=date, direction=signal,
                    entry_price=close_today, stop_price=stop_p,
                    target_price=target_p, position_size=pos_size,
                    entry_idx=i)
                entry_cost = pos_size * bid_ask / 2
                current_capital -= entry_cost
                trade.total_costs = entry_cost
                in_trade = trade

    if in_trade is not None:
        in_trade.exit_date  = df.index[min(end_idx, len(df)-1)]
        in_trade.exit_price = df['Close'].iloc[min(end_idx, len(df)-1)]
        in_trade.exit_reason = 'END_OF_PERIOD'
        remaining_pnl = in_trade.remaining_size * in_trade.cert_return
        exit_cost     = in_trade.remaining_size * bid_ask / 2
        remaining_pnl -= exit_cost
        in_trade.total_costs += exit_cost
        in_trade.pnl_eur = in_trade.partial_pnl_eur + remaining_pnl
        in_trade.pnl_pct = (in_trade.pnl_eur / in_trade.position_size) * 100
        current_capital += remaining_pnl
        trades.append(in_trade)

    return trades, current_capital


def calc_metrics(trades, initial_cap, final_cap):
    if not trades:
        return {'n_trades': 0, 'win_rate': 0, 'expectancy': 0,
                'profit_factor': 0, 'total_return': 0, 'max_dd': 0,
                'sharpe': 0, 'sortino': 0, 'holding': 0}

    pnls     = [t.pnl_eur for t in trades]
    pnl_pcts = [t.pnl_pct for t in trades]
    wins     = [p for p in pnls if p > 0]
    losses   = [p for p in pnls if p <= 0]
    holding  = [t.holding_days for t in trades]

    n  = len(trades)
    wr = len(wins) / n * 100
    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 99.0
    ret = (final_cap / initial_cap - 1) * 100

    eq_arr = np.array([initial_cap] + [initial_cap + sum(pnls[:i+1]) for i in range(n)])
    peak   = np.maximum.accumulate(eq_arr)
    dd     = (eq_arr - peak) / peak * 100
    max_dd = dd.min()

    avg_hold = np.mean(holding) if holding else 1
    tpy = 252 / avg_hold
    sharpe  = (np.mean(pnl_pcts) / np.std(pnl_pcts)) * np.sqrt(tpy) \
              if len(pnl_pcts) > 1 and np.std(pnl_pcts) > 0 else 0
    down    = [p for p in pnl_pcts if p < 0]
    sortino = (np.mean(pnl_pcts) / np.std(down)) * np.sqrt(tpy) \
              if down and np.std(down) > 0 else 0

    return {'n_trades': n, 'win_rate': wr, 'expectancy': np.mean(pnls),
            'profit_factor': pf, 'total_return': ret, 'max_dd': max_dd,
            'sharpe': sharpe, 'sortino': sortino, 'holding': avg_hold}


def score_params(p, datasets):
    """Calcola Sharpe OOS medio su tutti gli asset e leve per un set di parametri."""
    sharpe_scores = []
    n_trades_total = 0

    for asset_name, df in datasets.items():
        adx, plus_di, minus_di, atr_pct, atr_pct_ma, roc5, macd_hist, atr14 = \
            compute_indicators(df)
        signal_fn = make_signal_fn(adx, atr_pct, atr_pct_ma, roc5,
                                   macd_hist, plus_di, minus_di, p)
        start_bt = 60

        for lev in LEVERAGE_LEVELS:
            bid_ask       = BASE_BID_ASK * (lev / 7)
            daily_funding = ANNUAL_FUNDING_RATE / 252

            wf_trades = []
            wf_start  = start_bt
            while wf_start + WF_TRAIN_DAYS + WF_TEST_DAYS <= len(df):
                train_end = wf_start + WF_TRAIN_DAYS
                test_end  = train_end + WF_TEST_DAYS
                t_trades, _ = run_backtest(
                    df, train_end, test_end, INITIAL_CAPITAL, signal_fn,
                    lev, bid_ask, daily_funding, p)
                wf_trades.extend(t_trades)
                wf_start += WF_STEP_DAYS

            if len(wf_trades) >= 5:
                wf_final = INITIAL_CAPITAL + sum(t.pnl_eur for t in wf_trades)
                m = calc_metrics(wf_trades, INITIAL_CAPITAL, wf_final)
                sharpe_scores.append(m['sharpe'])
                n_trades_total += m['n_trades']

    if not sharpe_scores:
        return -99.0, 0

    avg_trades_per_combo = n_trades_total / len(sharpe_scores)
    penalty = max(0, (10 - avg_trades_per_combo) * 0.1)
    avg_sharpe = np.mean(sharpe_scores) - penalty
    return avg_sharpe, n_trades_total


def _worker(args):
    """Worker top-level per ProcessPoolExecutor (picklable su Windows)."""
    combo, keys, datasets_dict = args
    p = dict(zip(keys, combo))
    score, n_trades = score_params(p, datasets_dict)
    return {**p, 'sharpe_oos': score, 'n_trades': n_trades}


if __name__ == '__main__':
    # ============================================================
    # SCARICA DATI
    # ============================================================
    print(f"\nScaricando dati per {len(TICKERS)} sottostanti...")
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=1200)
    datasets = {}

    

    for name, ticker in TICKERS.items():
        try:
            df = yf.download(ticker, start=start_dt, end=end_dt, progress=False)
            if hasattr(df.columns, 'levels'):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            if len(df) > 100:
                datasets[name] = df
                print(f"  {name:15s} ({ticker:12s}): {len(df)} righe")
            else:
                print(f"  {name:15s} ({ticker:12s}): SKIP ({len(df)} righe)")
        except Exception as e:
            print(f"  {name:15s} ({ticker:12s}): ERRORE — {e}")

    # ============================================================
    # GRID SEARCH (parallela)
    # ============================================================
    keys       = list(PARAM_GRID.keys())
    values     = list(PARAM_GRID.values())
    all_combos = list(product(*values))

    n_workers = max(1, multiprocessing.cpu_count() - 1)

    start_time = datetime.now()

    print(f"\n{'═' * 75}")
    print(f"AVVIO GRID SEARCH — {total_combos} combinazioni | {n_workers} worker")
    print(f"{'═' * 75}\n")

    results     = []
    best_score  = -np.inf
    best_params = None
    completed   = 0

    worker_args = [(combo, keys, datasets) for combo in all_combos]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, arg): arg for arg in worker_args}
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            completed += 1
            score = row['sharpe_oos']
            if score > best_score:
                best_score  = score
                best_params = {k: row[k] for k in keys}
                marker = " *** NUOVO BEST ***"
            else:
                marker = ""
            if completed % 50 == 0 or completed == total_combos or completed <= 5:
                pct = completed / total_combos * 100
                print(f"[{completed:>4}/{total_combos}] ({pct:4.1f}%) "
                      f"Sharpe={score:+.3f}  trades={row['n_trades']}"
                      f"{marker}")

    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"\nTempo totale: {elapsed}")
    # ============================================================
    # RISULTATI
    # ============================================================
    results_df = pd.DataFrame(results).sort_values('sharpe_oos', ascending=False)
    results_df.to_csv('grid_search_results.csv', index=False)

    print(f"\n{'═' * 75}")
    print("TOP 15 COMBINAZIONI (per Sharpe OOS medio)")
    print(f"{'═' * 75}\n")
    print(results_df.head(15).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n{'═' * 75}")
    print("PARAMETRI OTTIMALI")
    print(f"{'═' * 75}\n")
    for k, v in best_params.items():
        print(f"  {k:<28}: {v}  (range: {PARAM_GRID[k]})")
    print(f"\n  Sharpe OOS medio: {best_score:+.4f}")

    # ============================================================
    # GRAFICI
    # ============================================================
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('white')
    gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])
    valid = results_df['sharpe_oos'].replace(-99, np.nan).dropna()
    ax1.hist(valid, bins=40, color='#378ADD', edgecolor='white', alpha=0.85)
    ax1.axvline(valid.mean(), color='#D85A30', linewidth=1.5, linestyle='--',
                label=f'Media: {valid.mean():.3f}')
    ax1.axvline(best_score, color='#639922', linewidth=2, linestyle='-',
                label=f'Best: {best_score:.3f}')
    ax1.set_xlabel('Sharpe OOS medio', fontsize=12)
    ax1.set_ylabel('Frequenza', fontsize=12)
    ax1.set_title('Distribuzione Sharpe OOS — tutte le combinazioni', fontsize=13, fontweight='500')
    ax1.legend(fontsize=11)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(results_df['n_trades'], results_df['sharpe_oos'],
                alpha=0.4, s=15, color='#378ADD')
    ax2.axhline(0, color='#888780', linewidth=0.5, linestyle='--')
    ax2.set_xlabel('N. trade totali (OOS)', fontsize=11)
    ax2.set_ylabel('Sharpe OOS', fontsize=11)
    ax2.set_title('Trade vs Sharpe', fontsize=12, fontweight='500')

    param_axes = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    for param_name, (r, c) in zip(keys[:6], param_axes):
        ax = fig.add_subplot(gs[r, c])
        grp  = results_df.groupby(param_name)['sharpe_oos'].mean().reset_index()
        bars = ax.bar(grp[param_name].astype(str), grp['sharpe_oos'],
                      color='#378ADD', edgecolor='white', alpha=0.85)
        opt_val = best_params[param_name]
        for bar, val in zip(bars, grp[param_name]):
            if val == opt_val:
                bar.set_color('#639922')
        ax.axhline(0, color='#888780', linewidth=0.5, linestyle='--')
        ax.set_title(param_name, fontsize=10, fontweight='500')
        ax.set_xlabel('Valore', fontsize=9)
        ax.set_ylabel('Sharpe medio', fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle('Grid Search — Ottimizzazione parametri strategia v3\n'
                 f'{total_combos} combinazioni | Walk-forward OOS | Metrica: Sharpe medio',
                 fontsize=14, fontweight='500', y=0.998)

    plt.savefig('grid_search_results.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n[Risultati salvati: grid_search_results.csv, grid_search_results.png]")

    print(f"\n{'═' * 75}")
    print("GRID SEARCH COMPLETATA")
    print(f"{'═' * 75}")
