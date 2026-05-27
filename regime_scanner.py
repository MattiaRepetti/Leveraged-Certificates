"""
Scanner di Regime per Certificati a Leva Fissa
===============================================
Modulo 3 del percorso "Certificati a Leva Fissa"

Calcola il semaforo operativo su dati reali del FTSE MIB:
- ADX (forza del trend)
- ATR% normalizzato (volatilità relativa)
- ROC a 5 giorni (momentum direzionale)
- MACD histogram (timing)

Produce un grafico storico dei regimi e segnala le finestre operative.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta

# ============================================================
# SCARICA DATI (o genera sintetici se la rete non è disponibile)
# ============================================================
ticker = "FTSEMIB.MI"
end_date = datetime.now()
start_date = end_date - timedelta(days=400)

try:
    print("Scaricando dati FTSE MIB...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        raise ValueError("Dati vuoti")
    if hasattr(df.columns, 'levels'):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    print(f"Dati scaricati: {len(df)} righe")
except Exception:
    print("Rete non disponibile — genero dati sintetici realistici (FTSE MIB)")
    ticker = "FTSE MIB (sintetico)"
    np.random.seed(123)
    n_days = 300
    dates = pd.bdate_range(end=end_date, periods=n_days)

    # Simula regime-switching: periodi di trend + periodi laterali
    close = [34000.0]
    regimes_sim = []
    current_regime = 'trend_up'
    regime_dur = 0
    for i in range(1, n_days):
        regime_dur += 1
        if regime_dur > np.random.randint(15, 40):
            current_regime = np.random.choice(
                ['trend_up', 'trend_down', 'lateral', 'volatile'])
            regime_dur = 0

        if current_regime == 'trend_up':
            mu, sigma = 0.004, 0.008
        elif current_regime == 'trend_down':
            mu, sigma = -0.003, 0.009
        elif current_regime == 'lateral':
            mu, sigma = 0.0001, 0.007
        else:  # volatile
            mu, sigma = 0.0, 0.018

        ret = np.random.normal(mu, sigma)
        close.append(close[-1] * (1 + ret))
        regimes_sim.append(current_regime)

    close = np.array(close)
    high = close * (1 + np.abs(np.random.normal(0, 0.004, n_days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.004, n_days)))
    volume = np.random.randint(500_000_000, 2_000_000_000, n_days)

    df = pd.DataFrame({
        'Open': close * (1 + np.random.normal(0, 0.001, n_days)),
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
    }, index=dates)

print(f"Dati: {len(df)} righe, da {df.index[0].date()} "
      f"a {df.index[-1].date()}")
print(f"Ticker: {ticker}\n")

# ============================================================
# CALCOLO INDICATORI
# ============================================================

# --- ADX (14 periodi) ---
def calc_adx(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']

    plus_dm = high.diff()
    minus_dm = low.diff().abs() * -1
    minus_dm = -low.diff()

    plus_dm = pd.Series(
        np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0),
        index=df.index)
    minus_dm = pd.Series(
        np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0),
        index=df.index)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).abs()
    adx = dx.rolling(window=period).mean()

    return adx, plus_di, minus_di


# --- ATR% normalizzato ---
def calc_atr_pct(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    atr_pct = (atr / close) * 100  # in percentuale
    atr_pct_ma = atr_pct.rolling(window=20).mean()  # media 20gg

    return atr_pct, atr_pct_ma


# --- ROC (Rate of Change) ---
def calc_roc(df, period=5):
    close = df['Close']
    roc = ((close - close.shift(period)) / close.shift(period)) * 100
    return roc


# --- MACD ---
def calc_macd(df, fast=12, slow=26, signal=9):
    close = df['Close']
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# Calcola tutto
adx, plus_di, minus_di = calc_adx(df)
atr_pct, atr_pct_ma = calc_atr_pct(df)
roc5 = calc_roc(df, 5)
macd_line, signal_line, macd_hist = calc_macd(df)

# ============================================================
# CLASSIFICAZIONE DEL REGIME
# ============================================================
def classify_regime(adx_val, atr_pct_val, atr_ma_val, roc_val,
                    macd_h, macd_h_prev, plus_di_val, minus_di_val):
    """
    Restituisce:
      'GREEN_LONG'  — trend rialzista, condizioni favorevoli
      'GREEN_SHORT' — trend ribassista, condizioni favorevoli
      'AMBER'       — cautela, condizioni miste
      'RED'         — stare fuori
    """
    # Filtro 1: ADX
    if pd.isna(adx_val) or adx_val < 20:
        return 'RED'

    # Filtro 2: ATR% vs media
    vol_ok = True
    if pd.isna(atr_pct_val) or pd.isna(atr_ma_val):
        vol_ok = False
    elif atr_pct_val > atr_ma_val * 1.3:  # vol 30% sopra media
        return 'RED'
    elif atr_pct_val > atr_ma_val:
        vol_ok = False  # cautela

    # Filtro 3: ROC e MACD
    if pd.isna(roc_val) or pd.isna(macd_h) or pd.isna(macd_h_prev):
        return 'RED'

    # Determina direzione
    if plus_di_val > minus_di_val:
        direction = 'LONG'
    else:
        direction = 'SHORT'

    # MACD in espansione nella direzione giusta?
    if direction == 'LONG':
        macd_expanding = macd_h > macd_h_prev and macd_h > 0
        roc_confirm = roc_val > 0.5
    else:
        macd_expanding = macd_h < macd_h_prev and macd_h < 0
        roc_confirm = roc_val < -0.5

    if adx_val >= 25 and vol_ok and macd_expanding and roc_confirm:
        return f'GREEN_{direction}'
    elif adx_val >= 25 and (macd_expanding or roc_confirm):
        return 'AMBER'
    elif adx_val >= 20:
        return 'AMBER'
    else:
        return 'RED'


# Applica classificazione
regimes = []
for i in range(len(df)):
    if i < 1:
        regimes.append('RED')
        continue

    regime = classify_regime(
        adx.iloc[i] if i < len(adx) else np.nan,
        atr_pct.iloc[i] if i < len(atr_pct) else np.nan,
        atr_pct_ma.iloc[i] if i < len(atr_pct_ma) else np.nan,
        roc5.iloc[i] if i < len(roc5) else np.nan,
        macd_hist.iloc[i] if i < len(macd_hist) else np.nan,
        macd_hist.iloc[i - 1] if i - 1 < len(macd_hist) else np.nan,
        plus_di.iloc[i] if i < len(plus_di) else np.nan,
        minus_di.iloc[i] if i < len(minus_di) else np.nan,
    )
    regimes.append(regime)

df['regime'] = regimes

# ============================================================
# STATISTICHE
# ============================================================
print("=" * 60)
print("DISTRIBUZIONE DEI REGIMI")
print("=" * 60)

regime_counts = df['regime'].value_counts()
for regime, count in regime_counts.items():
    pct = count / len(df) * 100
    print(f"  {regime:15s}: {count:4d} giorni ({pct:.1f}%)")

# Performance del sottostante nei diversi regimi
print(f"\n{'=' * 60}")
print("RENDIMENTO MEDIO GIORNALIERO PER REGIME")
print(f"{'=' * 60}")

daily_ret = df['Close'].pct_change() * 100
for regime in ['GREEN_LONG', 'GREEN_SHORT', 'AMBER', 'RED']:
    mask = df['regime'] == regime
    if mask.sum() > 0:
        ret_in_regime = daily_ret[mask]
        mu = ret_in_regime.mean()
        sigma = ret_in_regime.std()
        sharpe_d = mu / sigma if sigma > 0 else 0
        print(f"  {regime:15s}: μ={mu:+.3f}%  σ={sigma:.3f}%  "
              f"Sharpe_d={sharpe_d:.3f}  (N={mask.sum()})")

# Stato attuale
print(f"\n{'=' * 60}")
print("STATO ATTUALE DEL SEMAFORO")
print(f"{'=' * 60}")
latest = df.index[-1].date()
print(f"  Data:        {latest}")
print(f"  Prezzo:      {df['Close'].iloc[-1]:.2f}")
print(f"  Regime:      {df['regime'].iloc[-1]}")
print(f"  ADX:         {adx.iloc[-1]:.1f}")
print(f"  ATR%:        {atr_pct.iloc[-1]:.3f}%  "
      f"(media 20gg: {atr_pct_ma.iloc[-1]:.3f}%)")
print(f"  ROC(5):      {roc5.iloc[-1]:+.2f}%")
print(f"  MACD hist:   {macd_hist.iloc[-1]:.2f}  "
      f"(prev: {macd_hist.iloc[-2]:.2f})")
expanding = "SI" if (
    (macd_hist.iloc[-1] > macd_hist.iloc[-2] and macd_hist.iloc[-1] > 0) or
    (macd_hist.iloc[-1] < macd_hist.iloc[-2] and macd_hist.iloc[-1] < 0)
) else "NO"
print(f"  MACD expand: {expanding}")

# ============================================================
# GRAFICO
# ============================================================
fig = plt.figure(figsize=(16, 20))
fig.patch.set_facecolor('white')
gs = GridSpec(5, 1, figure=fig, height_ratios=[3, 1.2, 1.2, 1.2, 1.2],
              hspace=0.15)

# Colori regime per background
regime_colors = {
    'GREEN_LONG': '#97C459',
    'GREEN_SHORT': '#ED93B1',
    'AMBER': '#FAC775',
    'RED': '#F7C1C1',
}

# --- Panel 1: Prezzo + regime overlay ---
ax1 = fig.add_subplot(gs[0])
ax1.plot(df.index, df['Close'], color='#2C2C2A', linewidth=1.2)

# Colora lo sfondo per regime
for i in range(1, len(df)):
    color = regime_colors.get(df['regime'].iloc[i], '#F1EFE8')
    ax1.axvspan(df.index[i - 1], df.index[i], alpha=0.25, color=color,
                linewidth=0)

ax1.set_ylabel('Prezzo', fontsize=12)
ax1.set_title(f'{ticker} — Prezzo e regimi operativi', fontsize=14,
              fontweight='500')
ax1.tick_params(labelsize=10)
ax1.set_xlim(df.index[50], df.index[-1])

# Legenda
patches = [
    mpatches.Patch(color='#97C459', alpha=0.4, label='Verde Long'),
    mpatches.Patch(color='#ED93B1', alpha=0.4, label='Verde Short'),
    mpatches.Patch(color='#FAC775', alpha=0.4, label='Ambra (cautela)'),
    mpatches.Patch(color='#F7C1C1', alpha=0.4, label='Rosso (stop)'),
]
ax1.legend(handles=patches, fontsize=10, loc='upper left')

# --- Panel 2: ADX ---
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.plot(df.index, adx, color='#534AB7', linewidth=1.2, label='ADX(14)')
ax2.axhline(25, color='#639922', linewidth=1, linestyle='--', alpha=0.6)
ax2.axhline(20, color='#E24B4A', linewidth=1, linestyle='--', alpha=0.6)
ax2.fill_between(df.index, 0, adx,
                 where=adx >= 25, alpha=0.15, color='#639922')
ax2.fill_between(df.index, 0, adx,
                 where=adx < 20, alpha=0.15, color='#E24B4A')
ax2.set_ylabel('ADX', fontsize=11)
ax2.set_ylim(0, 60)
ax2.tick_params(labelsize=10)
ax2.legend(fontsize=9)

# --- Panel 3: ATR% ---
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax3.plot(df.index, atr_pct, color='#D85A30', linewidth=1, label='ATR%')
ax3.plot(df.index, atr_pct_ma, color='#378ADD', linewidth=1.5,
         linestyle='--', label='Media 20gg')
ax3.set_ylabel('ATR%', fontsize=11)
ax3.tick_params(labelsize=10)
ax3.legend(fontsize=9)

# --- Panel 4: ROC(5) ---
ax4 = fig.add_subplot(gs[3], sharex=ax1)
ax4.bar(df.index, roc5, width=1, color=np.where(roc5 >= 0, '#639922', '#E24B4A'),
        alpha=0.6)
ax4.axhline(0, color='#2C2C2A', linewidth=0.5, alpha=0.3)
ax4.axhline(0.5, color='#639922', linewidth=0.5, linestyle=':', alpha=0.5)
ax4.axhline(-0.5, color='#E24B4A', linewidth=0.5, linestyle=':', alpha=0.5)
ax4.set_ylabel('ROC(5) %', fontsize=11)
ax4.tick_params(labelsize=10)

# --- Panel 5: MACD Histogram ---
ax5 = fig.add_subplot(gs[4], sharex=ax1)
colors_macd = []
for i in range(len(macd_hist)):
    if i == 0:
        colors_macd.append('#888780')
        continue
    val = macd_hist.iloc[i]
    prev = macd_hist.iloc[i - 1]
    if val > 0 and val > prev:
        colors_macd.append('#639922')
    elif val > 0:
        colors_macd.append('#C0DD97')
    elif val < 0 and val < prev:
        colors_macd.append('#E24B4A')
    else:
        colors_macd.append('#F7C1C1')

ax5.bar(df.index, macd_hist, width=1, color=colors_macd, alpha=0.8)
ax5.axhline(0, color='#2C2C2A', linewidth=0.5, alpha=0.3)
ax5.set_ylabel('MACD Hist', fontsize=11)
ax5.set_xlabel('Data', fontsize=12)
ax5.tick_params(labelsize=10)

plt.savefig('/home/claude/regime_scanner.png', dpi=150,
            bbox_inches='tight', facecolor='white')
print(f"\n[Grafico salvato: regime_scanner.png]")

# ============================================================
# BACKTEST SEMPLIFICATO: entry su GREEN, exit dopo N giorni
# ============================================================
print(f"\n{'=' * 60}")
print("BACKTEST SEMPLIFICATO — Ingresso su GREEN, holding 1-5gg")
print(f"{'=' * 60}")

leverage = 7
daily_ret_arr = daily_ret.values

for hold_days in [1, 2, 3, 5]:
    trades_long = []
    trades_short = []

    for i in range(50, len(df) - hold_days):
        regime = df['regime'].iloc[i]
        if regime == 'GREEN_LONG':
            cert_ret = 1.0
            for d in range(1, hold_days + 1):
                r = daily_ret_arr[i + d] / 100
                cert_ret *= (1 + leverage * r)
            trades_long.append((cert_ret - 1) * 100)
        elif regime == 'GREEN_SHORT':
            cert_ret = 1.0
            for d in range(1, hold_days + 1):
                r = daily_ret_arr[i + d] / 100
                cert_ret *= (1 + (-leverage) * r)
            trades_short.append((cert_ret - 1) * 100)

    all_trades = trades_long + trades_short
    if all_trades:
        wins = sum(1 for t in all_trades if t > 0)
        avg_win = np.mean([t for t in all_trades if t > 0]) if wins > 0 else 0
        losses = sum(1 for t in all_trades if t <= 0)
        avg_loss = np.mean([t for t in all_trades if t <= 0]) if losses > 0 else 0

        print(f"\n  Holding: {hold_days}gg | Trades: {len(all_trades)} "
              f"(L:{len(trades_long)} S:{len(trades_short)})")
        print(f"    Win rate:   {wins/len(all_trades)*100:.1f}%")
        print(f"    Avg win:    {avg_win:+.2f}%")
        print(f"    Avg loss:   {avg_loss:+.2f}%")
        print(f"    Avg P&L:    {np.mean(all_trades):+.2f}%")
        print(f"    Mediana:    {np.median(all_trades):+.2f}%")
        print(f"    Sharpe:     {np.mean(all_trades)/np.std(all_trades):.3f}"
              if np.std(all_trades) > 0 else "    Sharpe:     N/A")

print(f"\n{'=' * 60}")
print("SCANNER COMPLETATO")
print(f"{'=' * 60}")
