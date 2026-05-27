"""
Backtest Completo — Strategia Certificati a Leva Fissa
=======================================================
Modulo 6 del percorso "Certificati a Leva Fissa"

Implementa:
- Scanner di regime (ADX, ATR%, ROC, MACD)
- Position sizing con rischio 1%
- Stop loss tecnico + temporale + di regime
- Target di profitto
- Walk-forward validation
- Metriche complete (Sharpe, Sortino, max drawdown, win rate, ecc.)

Quando esegui sul tuo PC con accesso a internet, cambia USE_SYNTHETIC = False
per usare dati reali da Yahoo Finance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta

np.random.seed(42)

# ============================================================
# CONFIGURAZIONE
# ============================================================
USE_SYNTHETIC = True  # Cambia a False per dati reali (serve internet)
TICKER = "FTSEMIB.MI"
LEVERAGE = 7
RISK_PER_TRADE = 0.01       # 1% del capitale per trade
INITIAL_CAPITAL = 10_000     # Capitale iniziale
MAX_HOLDING_DAYS = 5         # Max giorni in posizione
TARGET_MULTIPLIER = 1.5      # Target = 1.5x lo stop loss (R:R = 1:1.5)
STOP_LOSS_ATR_MULT = 1.5     # Stop = 1.5 x ATR giornaliero

# Parametri scanner di regime
ADX_THRESHOLD = 25
ADX_WEAK = 20
ROC_THRESHOLD = 0.5  # %
ATR_VOL_LIMIT = 1.3  # ATR% > 1.3x media = troppo volatile

# Walk-forward
WF_TRAIN_DAYS = 150   # Giorni di training
WF_TEST_DAYS = 50     # Giorni di test
WF_STEP_DAYS = 50     # Avanzamento della finestra

print("=" * 65)
print("BACKTEST — STRATEGIA CERTIFICATI A LEVA FISSA")
print("=" * 65)

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
    import yfinance as yf
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

    # True Range e ATR
    tr = pd.concat([h - l, (h - c.shift(1)).abs(),
                     (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    atr_pct = (atr14 / c) * 100
    atr_pct_ma = atr_pct.rolling(20).mean()

    # ADX
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

    # ROC(5)
    roc5 = ((c - c.shift(5)) / c.shift(5)) * 100

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

    return adx, plus_di, minus_di, atr_pct, atr_pct_ma, roc5, macd_hist, atr14


adx, plus_di, minus_di, atr_pct, atr_pct_ma, roc5, macd_hist, atr14 = \
    compute_indicators(df)

# ============================================================
# REGIME CLASSIFICATION
# ============================================================
def get_signal(i):
    """Restituisce 'LONG', 'SHORT', o None"""
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
        roc_ok = roc > ROC_THRESHOLD
    else:
        direction = 'SHORT'
        macd_exp = mh < mh_prev and mh < 0
        roc_ok = roc < -ROC_THRESHOLD

    if a >= ADX_THRESHOLD and vol_ok and macd_exp and roc_ok:
        return direction
    return None


# ============================================================
# BACKTEST ENGINE
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
        self.entry_idx = entry_idx
        self.exit_date = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl_pct = 0
        self.pnl_eur = 0
        self.holding_days = 0
        self.cert_return = 0


def run_backtest(df, start_idx, end_idx, capital):
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

            in_trade.cert_return = (1 + in_trade.cert_return) * \
                                   (1 + lev_return) - 1

            exit_now = False
            reason = None

            # Check stop loss
            if in_trade.direction == 'LONG' and close_today <= in_trade.stop_price:
                exit_now, reason = True, 'STOP_LOSS'
            elif in_trade.direction == 'SHORT' and close_today >= in_trade.stop_price:
                exit_now, reason = True, 'STOP_LOSS'

            # Check target
            if in_trade.direction == 'LONG' and close_today >= in_trade.target_price:
                exit_now, reason = True, 'TARGET'
            elif in_trade.direction == 'SHORT' and close_today <= in_trade.target_price:
                exit_now, reason = True, 'TARGET'

            # Check max holding
            if in_trade.holding_days >= MAX_HOLDING_DAYS:
                exit_now, reason = True, 'TIME_STOP'

            # Check regime deterioration
            signal = get_signal(i)
            if in_trade.direction == 'LONG' and signal != 'LONG':
                if in_trade.holding_days >= 2:
                    exit_now, reason = True, 'REGIME_EXIT'
            elif in_trade.direction == 'SHORT' and signal != 'SHORT':
                if in_trade.holding_days >= 2:
                    exit_now, reason = True, 'REGIME_EXIT'

            if exit_now:
                in_trade.exit_date = date
                in_trade.exit_price = close_today
                in_trade.exit_reason = reason
                in_trade.pnl_pct = in_trade.cert_return * 100
                in_trade.pnl_eur = in_trade.position_size * in_trade.cert_return
                current_capital += in_trade.pnl_eur
                trades.append(in_trade)
                in_trade = None

        # --- Nuovi ingressi ---
        elif in_trade is None:
            signal = get_signal(i)
            if signal is not None:
                atr_val = atr14.iloc[i]
                if pd.isna(atr_val) or atr_val <= 0:
                    continue

                stop_dist_pct = (STOP_LOSS_ATR_MULT * atr_val) / close_today
                if stop_dist_pct < 0.003:
                    stop_dist_pct = 0.003
                if stop_dist_pct > 0.03:
                    continue  # Troppo largo

                max_loss = current_capital * RISK_PER_TRADE
                pos_size = max_loss / (LEVERAGE * stop_dist_pct)
                pos_size = min(pos_size, current_capital * 0.25)

                if signal == 'LONG':
                    stop_p = close_today * (1 - stop_dist_pct)
                    target_p = close_today * (1 + stop_dist_pct * TARGET_MULTIPLIER)
                else:
                    stop_p = close_today * (1 + stop_dist_pct)
                    target_p = close_today * (1 - stop_dist_pct * TARGET_MULTIPLIER)

                in_trade = Trade(
                    entry_date=date, direction=signal,
                    entry_price=close_today, stop_price=stop_p,
                    target_price=target_p, position_size=pos_size,
                    entry_idx=i)
                in_trade.cert_return = 0

        equity.append(current_capital)
        equity_dates.append(date)

    # Chiudi trade aperto a fine periodo
    if in_trade is not None:
        in_trade.exit_date = df.index[min(end_idx, len(df)-1)]
        in_trade.exit_price = df['Close'].iloc[min(end_idx, len(df)-1)]
        in_trade.exit_reason = 'END_OF_PERIOD'
        in_trade.pnl_pct = in_trade.cert_return * 100
        in_trade.pnl_eur = in_trade.position_size * in_trade.cert_return
        current_capital += in_trade.pnl_eur
        trades.append(in_trade)
        equity.append(current_capital)
        equity_dates.append(in_trade.exit_date)

    return trades, equity, equity_dates, current_capital


# ============================================================
# ESECUZIONE: FULL BACKTEST + WALK-FORWARD
# ============================================================
print(f"\n{'─' * 65}")
print("FULL BACKTEST (in-sample — tutto il dataset)")
print(f"{'─' * 65}")

start_bt = 60  # Skip primi 60 giorni per warm-up indicatori
all_trades, equity, eq_dates, final_cap = run_backtest(
    df, start_bt, len(df), INITIAL_CAPITAL)

# --- Metriche ---
def compute_metrics(trades, initial_cap, final_cap, label=""):
    if not trades:
        print(f"  [{label}] Nessun trade eseguito.")
        return {}

    pnls = [t.pnl_eur for t in trades]
    pnl_pcts = [t.pnl_pct for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    holding = [t.holding_days for t in trades]
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    total_return = (final_cap / initial_cap - 1) * 100
    n_trades = len(trades)
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
    expectancy = np.mean(pnls)

    # Max drawdown sull'equity
    eq_arr = np.array([initial_cap] + [initial_cap + sum(pnls[:i+1]) for i in range(len(pnls))])
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak * 100
    max_dd = dd.min()

    # Sharpe (annualizzato, assumendo ~100 trade/anno)
    if len(pnl_pcts) > 1 and np.std(pnl_pcts) > 0:
        trades_per_year = 252 / np.mean(holding) if np.mean(holding) > 0 else 50
        sharpe = (np.mean(pnl_pcts) / np.std(pnl_pcts)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0

    # Sortino
    downside = [p for p in pnl_pcts if p < 0]
    if downside and np.std(downside) > 0:
        sortino = (np.mean(pnl_pcts) / np.std(downside)) * np.sqrt(trades_per_year)
    else:
        sortino = 0

    # Max consecutive losses
    max_consec_loss = 0
    current_streak = 0
    for p in pnls:
        if p <= 0:
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            current_streak = 0

    print(f"\n  [{label}]")
    print(f"  Trades totali:        {n_trades}")
    print(f"  Win rate:             {win_rate:.1f}%")
    print(f"  Avg win:              +{avg_win:.2f} EUR ({np.mean([p for p in pnl_pcts if p > 0]):.2f}%)" if wins else "  Avg win:              N/A")
    print(f"  Avg loss:             {avg_loss:.2f} EUR ({np.mean([p for p in pnl_pcts if p <= 0]):.2f}%)" if losses else "  Avg loss:             N/A")
    print(f"  Expectancy:           {expectancy:+.2f} EUR/trade")
    print(f"  Profit factor:        {profit_factor:.2f}")
    print(f"  Rendimento totale:    {total_return:+.2f}%")
    print(f"  Max drawdown:         {max_dd:.2f}%")
    print(f"  Sharpe (ann.):        {sharpe:.2f}")
    print(f"  Sortino (ann.):       {sortino:.2f}")
    print(f"  Holding medio:        {np.mean(holding):.1f} giorni")
    print(f"  Max consec. losses:   {max_consec_loss}")
    print(f"  Motivi uscita:        {reasons}")

    return {
        'n_trades': n_trades, 'win_rate': win_rate,
        'total_return': total_return, 'max_dd': max_dd,
        'sharpe': sharpe, 'sortino': sortino,
        'profit_factor': profit_factor, 'expectancy': expectancy,
    }

metrics_full = compute_metrics(all_trades, INITIAL_CAPITAL, final_cap,
                                "FULL BACKTEST")

# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================
print(f"\n{'─' * 65}")
print("WALK-FORWARD VALIDATION")
print(f"{'─' * 65}")
print(f"  Training: {WF_TRAIN_DAYS}gg | Test: {WF_TEST_DAYS}gg | "
      f"Step: {WF_STEP_DAYS}gg")

wf_results = []
wf_trades_all = []
wf_start = start_bt

while wf_start + WF_TRAIN_DAYS + WF_TEST_DAYS <= len(df):
    train_end = wf_start + WF_TRAIN_DAYS
    test_end = train_end + WF_TEST_DAYS

    # Training: calcoliamo le metriche in-sample (non ottimizziamo nulla
    # per ora — usiamo parametri fissi, ma la struttura è pronta per
    # ottimizzazione futura)
    train_trades, _, _, _ = run_backtest(
        df, wf_start, train_end, INITIAL_CAPITAL)

    # Test: out-of-sample
    test_trades, test_eq, test_eq_dates, test_final = run_backtest(
        df, train_end, test_end, INITIAL_CAPITAL)

    period_label = (f"{df.index[train_end].strftime('%Y-%m-%d')} → "
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
            'period': period_label,
            'n_trades': 0,
            'return': 0,
            'win_rate': 0,
        })
        print(f"  {period_label}: 0 trades")

    wf_start += WF_STEP_DAYS

if wf_trades_all:
    wf_final = INITIAL_CAPITAL
    for t in wf_trades_all:
        wf_final += t.pnl_eur
    print(f"\n  Walk-forward aggregato:")
    wf_metrics = compute_metrics(wf_trades_all, INITIAL_CAPITAL, wf_final,
                                  "WALK-FORWARD OOS")

# ============================================================
# GRAFICI
# ============================================================
fig = plt.figure(figsize=(16, 22))
fig.patch.set_facecolor('white')
gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)

# 1. Equity curve
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(eq_dates, equity, color='#185FA5', linewidth=1.5)
ax1.axhline(INITIAL_CAPITAL, color='#888780', linewidth=0.5, linestyle='--')
ax1.fill_between(eq_dates, INITIAL_CAPITAL, equity,
                  where=[e >= INITIAL_CAPITAL for e in equity],
                  alpha=0.15, color='#639922')
ax1.fill_between(eq_dates, INITIAL_CAPITAL, equity,
                  where=[e < INITIAL_CAPITAL for e in equity],
                  alpha=0.15, color='#E24B4A')
ax1.set_title('Equity curve', fontsize=14, fontweight='500')
ax1.set_ylabel('Capitale (EUR)', fontsize=12)
ax1.tick_params(labelsize=10)

# 2. P&L per trade
ax2 = fig.add_subplot(gs[1, 0])
if all_trades:
    pnl_pcts = [t.pnl_pct for t in all_trades]
    colors_pnl = ['#639922' if p > 0 else '#E24B4A' for p in pnl_pcts]
    ax2.bar(range(len(pnl_pcts)), pnl_pcts, color=colors_pnl, alpha=0.7)
    ax2.axhline(0, color='#2C2C2A', linewidth=0.5)
    ax2.set_title('P&L per trade (%)', fontsize=14, fontweight='500')
    ax2.set_xlabel('Trade #', fontsize=11)
    ax2.set_ylabel('P&L (%)', fontsize=11)
    ax2.tick_params(labelsize=10)

# 3. Distribuzione P&L
ax3 = fig.add_subplot(gs[1, 1])
if all_trades:
    ax3.hist(pnl_pcts, bins=30, color='#378ADD', alpha=0.7, edgecolor='none')
    ax3.axvline(0, color='#2C2C2A', linewidth=1)
    ax3.axvline(np.mean(pnl_pcts), color='#D85A30', linewidth=2,
                linestyle='--', label=f'Media: {np.mean(pnl_pcts):+.2f}%')
    ax3.set_title('Distribuzione P&L', fontsize=14, fontweight='500')
    ax3.set_xlabel('P&L (%)', fontsize=11)
    ax3.legend(fontsize=10)
    ax3.tick_params(labelsize=10)

# 4. Motivi di uscita
ax4 = fig.add_subplot(gs[2, 0])
if all_trades:
    reasons_list = [t.exit_reason for t in all_trades]
    reason_labels, reason_counts = np.unique(reasons_list, return_counts=True)
    reason_colors = {
        'TARGET': '#639922', 'STOP_LOSS': '#E24B4A',
        'TIME_STOP': '#EF9F27', 'REGIME_EXIT': '#378ADD',
        'END_OF_PERIOD': '#888780'
    }
    bar_colors = [reason_colors.get(r, '#888780') for r in reason_labels]
    ax4.barh(reason_labels, reason_counts, color=bar_colors, alpha=0.8)
    ax4.set_title('Motivi di uscita', fontsize=14, fontweight='500')
    ax4.set_xlabel('Numero trades', fontsize=11)
    ax4.tick_params(labelsize=10)

# 5. Holding period distribution
ax5 = fig.add_subplot(gs[2, 1])
if all_trades:
    holdings = [t.holding_days for t in all_trades]
    ax5.hist(holdings, bins=range(1, MAX_HOLDING_DAYS + 3),
             color='#534AB7', alpha=0.7, edgecolor='none', align='left')
    ax5.set_title('Distribuzione holding period', fontsize=14, fontweight='500')
    ax5.set_xlabel('Giorni', fontsize=11)
    ax5.set_ylabel('Frequenza', fontsize=11)
    ax5.tick_params(labelsize=10)

# 6. Walk-forward results
ax6 = fig.add_subplot(gs[3, :])
if wf_results:
    wf_returns = [w['return'] for w in wf_results]
    wf_labels = [w['period'].split(' → ')[0] for w in wf_results]
    wf_colors = ['#639922' if r > 0 else '#E24B4A' for r in wf_returns]
    ax6.bar(range(len(wf_returns)), wf_returns, color=wf_colors, alpha=0.7)
    ax6.set_xticks(range(len(wf_labels)))
    ax6.set_xticklabels(wf_labels, rotation=45, ha='right', fontsize=9)
    ax6.axhline(0, color='#2C2C2A', linewidth=0.5)
    ax6.set_title('Walk-forward: rendimento per finestra OOS',
                  fontsize=14, fontweight='500')
    ax6.set_ylabel('Rendimento (%)', fontsize=11)
    ax6.tick_params(labelsize=10)

fig.suptitle(f'Backtest Certificato Leva {LEVERAGE}x — '
             f'{len(all_trades)} trades\n'
             f'Capitale: {INITIAL_CAPITAL:,} EUR → {final_cap:,.0f} EUR '
             f'({(final_cap/INITIAL_CAPITAL-1)*100:+.1f}%)',
             fontsize=15, fontweight='500', y=0.98)

plt.savefig('./backtest_results.png', dpi=150,
            bbox_inches='tight', facecolor='white')
print(f"\n[Grafico salvato: backtest_results.png]")

# ============================================================
# TRADE LOG
# ============================================================
print(f"\n{'─' * 65}")
print("TRADE LOG (ultimi 15 trades)")
print(f"{'─' * 65}")
print(f"{'Data':>12} {'Dir':>5} {'Entry':>10} {'Exit':>10} "
      f"{'P&L%':>8} {'P&L EUR':>10} {'Giorni':>6} {'Motivo':>14}")
print("─" * 80)

for t in all_trades[-15:]:
    print(f"{t.entry_date.strftime('%Y-%m-%d'):>12} {t.direction:>5} "
          f"{t.entry_price:>10.2f} {t.exit_price:>10.2f} "
          f"{t.pnl_pct:>+8.2f} {t.pnl_eur:>+10.2f} "
          f"{t.holding_days:>6} {t.exit_reason:>14}")

print(f"\n{'=' * 65}")
print("BACKTEST COMPLETATO")
print(f"{'=' * 65}")
