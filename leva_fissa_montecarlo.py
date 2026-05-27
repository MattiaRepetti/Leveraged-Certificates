"""
Simulatore Monte Carlo per Certificati a Leva Fissa
====================================================
Modulo 2 del percorso "Certificati a Leva Fissa"

Questo script simula 10.000 percorsi di prezzo per un indice (es. FTSE MIB)
e calcola il valore corrispondente di un certificato a leva fissa,
mostrando la distribuzione dei rendimenti e l'effetto del volatility decay.

Autore: Mattia (percorso formativo)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

np.random.seed(42)

# ============================================================
# PARAMETRI — modifica qui per sperimentare
# ============================================================
LEVERAGE = 7                    # Leva del certificato
N_SIMULATIONS = 10_000          # Numero di percorsi simulati
HOLDING_PERIODS = [1, 5, 10, 20, 60]  # Giorni di holding da analizzare

# Parametri del sottostante (FTSE MIB valori tipici)
ANNUAL_RETURN = 0.08            # Rendimento atteso annuo (8%)
ANNUAL_VOLATILITY = 0.18        # Volatilità annua (18%)

# Derivati giornalieri (252 giorni di trading)
DAILY_MU = ANNUAL_RETURN / 252
DAILY_SIGMA = ANNUAL_VOLATILITY / np.sqrt(252)

# Costo di finanziamento annuo (€STR + spread emittente)
ANNUAL_FUNDING_COST = 0.035     # 3.5% annuo
DAILY_FUNDING = ANNUAL_FUNDING_COST / 252

print("=" * 65)
print("SIMULATORE MONTE CARLO — CERTIFICATI A LEVA FISSA")
print("=" * 65)
print(f"\nParametri del sottostante:")
print(f"  Rendimento annuo atteso:  {ANNUAL_RETURN:.1%}")
print(f"  Volatilità annua:         {ANNUAL_VOLATILITY:.1%}")
print(f"  μ giornaliero:            {DAILY_MU:.4%}")
print(f"  σ giornaliero:            {DAILY_SIGMA:.2%}")
print(f"\nParametri del certificato:")
print(f"  Leva:                     {LEVERAGE}x")
print(f"  Costo funding giornaliero:{DAILY_FUNDING:.4%}")
print(f"\nDecay teorico giornaliero:  "
      f"{0.5 * LEVERAGE * (LEVERAGE - 1) * DAILY_SIGMA**2:.4%}")
print(f"Breakeven μ giornaliero:    "
      f"{0.5 * (LEVERAGE - 1) * DAILY_SIGMA**2:.4%}")

# ============================================================
# SIMULAZIONE
# ============================================================
max_days = max(HOLDING_PERIODS)

# Genera rendimenti giornalieri: (N_SIMULATIONS, max_days)
daily_returns = np.random.normal(DAILY_MU, DAILY_SIGMA,
                                 size=(N_SIMULATIONS, max_days))

# Percorso dell'indice: prodotto cumulato di (1 + r)
index_cumulative = np.cumprod(1 + daily_returns, axis=1)

# Percorso del certificato: prodotto cumulato di (1 + L*r - funding)
cert_daily = 1 + LEVERAGE * daily_returns - DAILY_FUNDING
cert_cumulative = np.cumprod(cert_daily, axis=1)

# Rendimento "naive" atteso: L * rendimento indice
naive_leverage = 1 + LEVERAGE * (index_cumulative - 1)

# ============================================================
# ANALISI PER HOLDING PERIOD
# ============================================================
print("\n" + "=" * 65)
print("RISULTATI PER HOLDING PERIOD")
print("=" * 65)

results = {}
for hp in HOLDING_PERIODS:
    idx_ret = (index_cumulative[:, hp - 1] - 1) * 100
    cert_ret = (cert_cumulative[:, hp - 1] - 1) * 100
    naive_ret = (naive_leverage[:, hp - 1] - 1) * 100
    decay = cert_ret - naive_ret  # differenza = compounding effect

    results[hp] = {
        'idx_ret': idx_ret,
        'cert_ret': cert_ret,
        'naive_ret': naive_ret,
        'decay': decay,
    }

    pct_positive = np.mean(cert_ret > 0) * 100
    pct_beats_naive = np.mean(cert_ret > naive_ret) * 100

    print(f"\n{'─' * 50}")
    print(f"  Holding period: {hp} giorni")
    print(f"{'─' * 50}")
    print(f"  INDICE:")
    print(f"    Media:    {np.mean(idx_ret):+.2f}%")
    print(f"    Mediana:  {np.median(idx_ret):+.2f}%")
    print(f"    Std dev:  {np.std(idx_ret):.2f}%")
    print(f"  CERTIFICATO (leva {LEVERAGE}x):")
    print(f"    Media:    {np.mean(cert_ret):+.2f}%")
    print(f"    Mediana:  {np.median(cert_ret):+.2f}%")
    print(f"    Std dev:  {np.std(cert_ret):.2f}%")
    print(f"    % positivi:         {pct_positive:.1f}%")
    print(f"  DECAY (cert - naive):")
    print(f"    Media:    {np.mean(decay):+.3f}%")
    print(f"    Mediana:  {np.median(decay):+.3f}%")
    print(f"    % cert > naive:     {pct_beats_naive:.1f}%")

# ============================================================
# GRAFICO 1: Distribuzione rendimenti per holding period
# ============================================================
fig = plt.figure(figsize=(14, 16))
fig.patch.set_facecolor('white')
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

for i, hp in enumerate(HOLDING_PERIODS[:5]):
    row, col = divmod(i, 2)
    ax = fig.add_subplot(gs[row, col])

    cert_ret = results[hp]['cert_ret']
    naive_ret = results[hp]['naive_ret']

    # Limita i bin per leggibilità
    pct1, pct99 = np.percentile(cert_ret, [1, 99])
    bins = np.linspace(pct1, pct99, 80)

    ax.hist(cert_ret, bins=bins, alpha=0.7, color='#D85A30',
            label='Certificato', density=True, edgecolor='none')
    ax.axvline(0, color='#2C2C2A', linewidth=1, linestyle='-', alpha=0.5)
    ax.axvline(np.mean(cert_ret), color='#D85A30', linewidth=2,
               linestyle='--', label=f'Media: {np.mean(cert_ret):+.2f}%')
    ax.axvline(np.mean(naive_ret), color='#378ADD', linewidth=2,
               linestyle='--', label=f'Naive {LEVERAGE}x: {np.mean(naive_ret):+.2f}%')

    ax.set_title(f'{hp} giorni', fontsize=14, fontweight='500')
    ax.set_xlabel('Rendimento (%)', fontsize=11)
    ax.set_ylabel('Densità', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.tick_params(labelsize=10)

# Sesto pannello: decay medio vs holding period
ax6 = fig.add_subplot(gs[2, 1])
decay_means = [np.mean(results[hp]['decay']) for hp in HOLDING_PERIODS]
decay_theory = [-0.5 * LEVERAGE * (LEVERAGE - 1) * DAILY_SIGMA**2 * hp * 100
                for hp in HOLDING_PERIODS]
ax6.plot(HOLDING_PERIODS, decay_means, 'o-', color='#D85A30',
         linewidth=2, markersize=8, label='Decay simulato')
ax6.plot(HOLDING_PERIODS, decay_theory, 's--', color='#378ADD',
         linewidth=1.5, markersize=6, label='Decay teorico (formula)')
ax6.axhline(0, color='#2C2C2A', linewidth=0.5, alpha=0.3)
ax6.set_xlabel('Holding period (giorni)', fontsize=11)
ax6.set_ylabel('Decay medio (%)', fontsize=11)
ax6.set_title('Volatility decay vs holding period', fontsize=14,
              fontweight='500')
ax6.legend(fontsize=10)
ax6.tick_params(labelsize=10)

fig.suptitle(f'Distribuzione rendimenti — Certificato Leva {LEVERAGE}x\n'
             f'(μ={ANNUAL_RETURN:.0%} annuo, σ={ANNUAL_VOLATILITY:.0%} annuo, '
             f'{N_SIMULATIONS:,} simulazioni)',
             fontsize=15, fontweight='500', y=0.98)

plt.savefig('./montecarlo_distributions.png', dpi=150,
            bbox_inches='tight', facecolor='white')
print("\n[Grafico 1 salvato: montecarlo_distributions.png]")

# ============================================================
# GRAFICO 2: Percorsi campione (50 traiettorie)
# ============================================================
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
fig2.patch.set_facecolor('white')

n_sample = 50
for j in range(n_sample):
    ax1.plot(range(1, max_days + 1),
             (index_cumulative[j] - 1) * 100,
             alpha=0.15, color='#378ADD', linewidth=0.7)
    ax2.plot(range(1, max_days + 1),
             (cert_cumulative[j] - 1) * 100,
             alpha=0.15, color='#D85A30', linewidth=0.7)

# Mediana e percentili
idx_median = np.median(index_cumulative, axis=0)
idx_p10 = np.percentile(index_cumulative, 10, axis=0)
idx_p90 = np.percentile(index_cumulative, 90, axis=0)
cert_median = np.median(cert_cumulative, axis=0)
cert_p10 = np.percentile(cert_cumulative, 10, axis=0)
cert_p90 = np.percentile(cert_cumulative, 90, axis=0)

days_arr = range(1, max_days + 1)
ax1.plot(days_arr, (idx_median - 1) * 100, color='#185FA5',
         linewidth=2.5, label='Mediana')
ax1.fill_between(days_arr, (idx_p10 - 1) * 100, (idx_p90 - 1) * 100,
                 alpha=0.15, color='#378ADD', label='10°-90° percentile')
ax1.axhline(0, color='#2C2C2A', linewidth=0.5, alpha=0.3)
ax1.set_ylabel('Rendimento indice (%)', fontsize=12)
ax1.set_title('Indice sottostante', fontsize=14, fontweight='500')
ax1.legend(fontsize=10)
ax1.tick_params(labelsize=10)

ax2.plot(days_arr, (cert_median - 1) * 100, color='#993C1D',
         linewidth=2.5, label='Mediana')
ax2.fill_between(days_arr, (cert_p10 - 1) * 100, (cert_p90 - 1) * 100,
                 alpha=0.15, color='#D85A30', label='10°-90° percentile')
ax2.axhline(0, color='#2C2C2A', linewidth=0.5, alpha=0.3)
ax2.set_xlabel('Giorni', fontsize=12)
ax2.set_ylabel(f'Rendimento certificato {LEVERAGE}x (%)', fontsize=12)
ax2.set_title(f'Certificato Leva {LEVERAGE}x', fontsize=14, fontweight='500')
ax2.legend(fontsize=10)
ax2.tick_params(labelsize=10)

fig2.suptitle(f'{n_sample} percorsi campione + mediana e banda 10-90°\n'
              f'(Holding max {max_days} giorni)',
              fontsize=15, fontweight='500', y=1.01)
plt.savefig('./montecarlo_paths.png', dpi=150,
            bbox_inches='tight', facecolor='white')
print("[Grafico 2 salvato: montecarlo_paths.png]")

# ============================================================
# GRAFICO 3: Heatmap — P&L medio per (trend, volatilità)
# ============================================================
mu_range = np.linspace(-0.005, 0.005, 25)    # da -0.5% a +0.5% giornaliero
sigma_range = np.linspace(0.005, 0.025, 25)  # da 0.5% a 2.5% giornaliero
hp_heatmap = 10  # holding period per la heatmap

pnl_matrix = np.zeros((len(sigma_range), len(mu_range)))

for i, sigma in enumerate(sigma_range):
    for j, mu in enumerate(mu_range):
        rets = np.random.normal(mu, sigma, size=(2000, hp_heatmap))
        cert_paths = np.cumprod(1 + LEVERAGE * rets - DAILY_FUNDING, axis=1)
        pnl_matrix[i, j] = np.mean((cert_paths[:, -1] - 1) * 100)

fig3, ax = plt.subplots(figsize=(12, 9))
fig3.patch.set_facecolor('white')

im = ax.imshow(pnl_matrix, aspect='auto', origin='lower',
               cmap='RdYlGn', vmin=-30, vmax=30,
               extent=[mu_range[0]*100, mu_range[-1]*100,
                       sigma_range[0]*100, sigma_range[-1]*100])

# Linea di breakeven (dove P&L ≈ 0)
cs = ax.contour(mu_range * 100, sigma_range * 100, pnl_matrix,
                levels=[0], colors='black', linewidths=2, linestyles='--')
ax.clabel(cs, fmt='breakeven', fontsize=11)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label(f'P&L medio certificato {LEVERAGE}x (%)', fontsize=12)

ax.set_xlabel('Trend giornaliero μ (%)', fontsize=13)
ax.set_ylabel('Volatilità giornaliera σ (%)', fontsize=13)
ax.set_title(f'P&L medio su {hp_heatmap} giorni — Leva {LEVERAGE}x\n'
             f'(zona verde = profitto, zona rossa = perdita)',
             fontsize=14, fontweight='500')
ax.tick_params(labelsize=11)

# Punto "FTSE MIB medio"
ax.plot(DAILY_MU * 100, DAILY_SIGMA * 100, 'ko', markersize=10)
ax.annotate('FTSE MIB\n(media storica)',
            xy=(DAILY_MU * 100, DAILY_SIGMA * 100),
            xytext=(DAILY_MU * 100 + 0.12, DAILY_SIGMA * 100 + 0.15),
            fontsize=11, fontweight='500',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

plt.savefig('./montecarlo_heatmap.png', dpi=150,
            bbox_inches='tight', facecolor='white')
print("[Grafico 3 salvato: montecarlo_heatmap.png]")

# ============================================================
# GRAFICO 4: Decay per livello di leva
# ============================================================
fig4, ax = plt.subplots(figsize=(12, 7))
fig4.patch.set_facecolor('white')

levas = [2, 3, 5, 7, 10]
colors_lev = ['#378ADD', '#1D9E75', '#EF9F27', '#D85A30', '#E24B4A']
hp_range = np.arange(1, 61)

for lev, col in zip(levas, colors_lev):
    decay_curve = [-0.5 * lev * (lev - 1) * DAILY_SIGMA**2 * d * 100
                   for d in hp_range]
    ax.plot(hp_range, decay_curve, linewidth=2.5, color=col,
            label=f'Leva {lev}x')

ax.axhline(0, color='#2C2C2A', linewidth=0.5, alpha=0.3)
ax.set_xlabel('Holding period (giorni)', fontsize=13)
ax.set_ylabel('Decay teorico cumulato (%)', fontsize=13)
ax.set_title('Volatility decay per livello di leva\n'
             f'(σ giornaliero = {DAILY_SIGMA:.2%})',
             fontsize=14, fontweight='500')
ax.legend(fontsize=11)
ax.tick_params(labelsize=11)
ax.grid(alpha=0.15)

plt.savefig('./montecarlo_decay_by_leverage.png', dpi=150,
            bbox_inches='tight', facecolor='white')
print("[Grafico 4 salvato: montecarlo_decay_by_leverage.png]")

print("\n" + "=" * 65)
print("SIMULAZIONE COMPLETATA")
print("=" * 65)
print(f"\nFile generati:")
print(f"  1. montecarlo_distributions.png  — Distribuzione P&L per HP")
print(f"  2. montecarlo_paths.png          — Percorsi campione")
print(f"  3. montecarlo_heatmap.png        — Heatmap trend vs vol")
print(f"  4. montecarlo_decay_by_leverage.png — Decay per leva")
