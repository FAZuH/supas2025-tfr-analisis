# %% [markdown]
# # Determinan dan Disparitas Angka Kelahiran Total Antarprovinsi di Indonesia
#
# Data: BPS, Berita Resmi Statistik Hasil SUPAS 2025 (Lampiran 1-6) tanggal 5 Mei 2026 (IKG hlm. 62).

# %%
"""Analisis statistika SUPAS 2025 tingkat provinsi.

Angka berasal dari data/dataset.csv: BRS SUPAS 2025 Lampiran 1-6,
IKG dari bahan tayang BRS hlm. 62.

uv run --with pandas --with numpy --with scipy --with statsmodels --with matplotlib python3 analisis.py
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.ticker import FuncFormatter
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

matplotlib.use("Agg")

# %% [markdown]
# ## Pemeriksaan data dan nilai jangkar
# Memuat `data/dataset.csv`, memeriksa 38 baris provinsi dan nilai jangkar tiap
# indikator. Tidak menulis berkas keluaran; skrip berhenti dengan
# `AssertionError` bila angka mentah berubah.

# %%
BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

df = pd.read_csv(os.path.join(BASE, "data", "dataset.csv"))

assert len(df) == 38, "harus 38 provinsi"
assert "Indonesia" not in set(df["provinsi"]), "Indonesia tidak termasuk baris provinsi"
assert set(df["region"]) == {
    "Sumatera", "Jawa", "Bali & Nusa Tenggara",
    "Kalimantan", "Sulawesi", "Maluku & Papua",
}, "region tidak sesuai"


def _prov(col):
    return df.loc[df[col].idxmin(), "provinsi"], df.loc[df[col].idxmax(), "provinsi"]


assert np.isclose(df["tfr"].min(), 1.79), "tfr min"
assert _prov("tfr")[0] == "DKI Jakarta"
assert np.isclose(df["tfr"].max(), 2.78), "tfr max"
assert _prov("tfr")[1] == "Papua Pegunungan"
assert np.isclose(df["imr"].min(), 9.26), "imr min"
assert _prov("imr")[0] == "DKI Jakarta"
assert np.isclose(df["imr"].max(), 37.04), "imr max"
assert _prov("imr")[1] == "Papua Pegunungan"
assert np.isclose(df["pct_lansia"].max(), 17.83), "pct_lansia max"
assert _prov("pct_lansia")[1] == "DI Yogyakarta"
assert np.isclose(df["ikg"].min(), 0.144), "ikg min"
assert _prov("ikg")[0] == "DKI Jakarta"
assert np.isclose(df["ikg"].max(), 0.584), "ikg max"
assert _prov("ikg")[1] == "Papua Pegunungan"
print("[anchor] semua sanity check lolos")


# %% [markdown]
# ## Statistik deskriptif TFR
# n, rerata, median, simpangan baku, CV, kuartil, IQR, dan rasio maks/min TFR
# 38 provinsi. Tidak menulis berkas keluaran.

# %%
t = df["tfr"]
q1, q3 = t.quantile(0.25), t.quantile(0.75)
desc = {
    "n": int(t.count()),
    "mean": float(t.mean()),
    "median": float(t.median()),
    "sd": float(t.std(ddof=1)),
    "cv": float(t.std(ddof=1) / t.mean()),
    "min": float(t.min()),
    "q1": float(q1),
    "q3": float(q3),
    "iqr": float(q3 - q1),
    "max": float(t.max()),
    "rasio_max_min": float(t.max() / t.min()),
}

# %% [markdown]
# ## Korelasi TFR dengan indikator demografi lain
# Korelasi Pearson dan Spearman TFR terhadap 13 indikator; menulis
# `results/korelasi.csv` dan `results/korelasi.txt`.

# %%
VARS = ["imr", "cmr", "u5mr", "cdr", "pct_lansia", "dependency_ratio",
        "migran_neto_risen", "migran_keluar_seumur", "migran_masuk_seumur",
        "ikg", "sex_ratio", "cbr", "asfr15_19"]

kor = []
for v in VARS:
    sub = df[["tfr", v]].dropna()
    pr, pp = stats.pearsonr(sub["tfr"], sub[v])
    sr, sp = stats.spearmanr(sub["tfr"], sub[v])
    kor.append({
        "variabel": v, "n": len(sub),
        "pearson_r": pr, "pearson_p": pp,
        "spearman_rho": sr, "spearman_p": sp,
    })
kor_df = pd.DataFrame(kor)
kor_df.to_csv(os.path.join(RES, "korelasi.csv"), index=False)


# %%
def _sig(p):
    return "signifikan (p<0,05)" if p < 0.05 else "tidak signifikan"


kor_txt = ["KORELASI TFR DENGAN INDIKATOR DEMOGRAFI LAIN (38 PROVINSI)",
           "",
           f"{'variabel':<22}{'n':>4}{'r':>9}{'p_r':>10}{'rho':>9}"
           f"{'p_rho':>10}  status (Pearson)",
           "-" * 90]
for _, r in kor_df.iterrows():
    kor_txt.append(f"{r['variabel']:<22}{r['n']:>4}{r['pearson_r']:>9.3f}"
                   f"{r['pearson_p']:>10.4f}{r['spearman_rho']:>9.3f}"
                   f"{r['spearman_p']:>10.4f}  {_sig(r['pearson_p'])}")
with open(os.path.join(RES, "korelasi.txt"), "w") as f:
    f.write("\n".join(kor_txt) + "\n")

# %% [markdown]
# ## Regresi linear berganda
# Model M1 (IMR, persentase lansia, migrasi risen neto) dan M2 (ditambah IKG),
# dengan VIF, uji Breusch-Pagan, dan Shapiro-Wilk; menulis `results/regresi.txt`
# dan `results/regresi.json`.

# %%
MODELS = {
    "M1": "tfr ~ imr + pct_lansia + migran_neto_risen",
    "M2": "tfr ~ imr + pct_lansia + migran_neto_risen + ikg",
}
reg_json = {}
reg_txt = []


def _fmt(v, d=4):
    return f"{v:.{d}f}"


# %%
for name, formula in MODELS.items():
    m = smf.ols(formula, data=df, missing="drop").fit()
    exog, names = m.model.exog, m.model.exog_names
    vif = [(n, float(variance_inflation_factor(exog, i)))
           for i, n in enumerate(names) if n.lower() != "intercept"]
    bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(m.resid, exog)
    sw_w, sw_p = stats.shapiro(m.resid)

    reg_json[name] = {
        "formula": formula,
        "n": int(m.nobs),
        "params": {k: float(v) for k, v in m.params.items()},
        "bse": {k: float(v) for k, v in m.bse.items()},
        "tvalues": {k: float(v) for k, v in m.tvalues.items()},
        "pvalues": {k: float(v) for k, v in m.pvalues.items()},
        "conf_int_95": {k: [float(a), float(b)]
                        for k, (a, b) in m.conf_int().iterrows()},
        "r2": float(m.rsquared),
        "adj_r2": float(m.rsquared_adj),
        "f": float(m.fvalue),
        "f_p": float(m.f_pvalue),
        "vif": {k: v for k, v in vif},
        "bp_lm": float(bp_lm), "bp_p": float(bp_p),
        "shapiro_w": float(sw_w), "shapiro_p": float(sw_p),
    }

    lines = [f"=== {name}: {formula} ===",
             f"n = {int(m.nobs)}", "",
             f"{'prediktor':<20}{'coef':>10}{'std err':>10}{'t':>9}"
             f"{'p':>10}   {'CI 95% bawah':>12}{'CI 95% atas':>12}{'VIF':>8}"]
    for k in m.params.index:
        ci = m.conf_int().loc[k]
        v = dict(vif).get(k, float("nan"))
        lines.append(f"{k:<20}{m.params[k]:>10.4f}{m.bse[k]:>10.4f}"
                     f"{m.tvalues[k]:>9.3f}{m.pvalues[k]:>10.4f}   "
                     f"{ci[0]:>12.4f}{ci[1]:>12.4f}{v:>8.3f}")
    lines += ["",
              f"R2 = {m.rsquared:.4f} | adj R2 = {m.rsquared_adj:.4f}",
              f"F = {m.fvalue:.3f} (p = {m.f_pvalue:.4g})",
              f"Breusch-Pagan LM = {bp_lm:.3f} (p = {bp_p:.4f})",
              f"Shapiro-Wilk W = {sw_w:.4f} (p = {sw_p:.4f})",
              f"Observasi dipakai = {int(m.nobs)}", ""]
    reg_txt.append("\n".join(lines))

# %%
with open(os.path.join(RES, "regresi.txt"), "w") as f:
    f.write("\n".join(reg_txt) + "\n")

with open(os.path.join(RES, "regresi.json"), "w") as f:
    json.dump(reg_json, f, indent=2, ensure_ascii=False)

# %% [markdown]
# ## Tipologi provinsi
# Empat kuadran menurut median TFR dan median IMR; menulis
# `results/tipe_provinsi.csv`.

# %%
med_tfr, med_imr = t.median(), df["imr"].median()


def tipologi(r):
    tfr_r = "rendah" if r["tfr"] <= med_tfr else "tinggi"
    imr_r = "rendah" if r["imr"] <= med_imr else "tinggi"
    return f"TFR {tfr_r}-IMR {imr_r}"


tip = df[["provinsi", "region", "tfr", "imr"]].copy()
tip["tipe"] = df.apply(tipologi, axis=1)
tip.to_csv(os.path.join(RES, "tipe_provinsi.csv"), index=False)

# %% [markdown]
# ## Figura
# fig1 tren nasional, fig2 sebaran TFR-IMR, fig3 bar TFR per provinsi, dan fig4
# sebaran lansia-TFR. Menulis `figures/fig1-fig4` sebagai PNG 300 dpi dan PDF.

# %%
REG_COLORS = {
    "Sumatera": "#1f77b4",
    "Jawa": "#d62728",
    "Bali & Nusa Tenggara": "#2ca02c",
    "Kalimantan": "#ff7f0e",
    "Sulawesi": "#9467bd",
    "Maluku & Papua": "#8c564b",
}
EXTREMES = ["DKI Jakarta", "DI Yogyakarta", "Papua Pegunungan", "Papua Selatan",
            "Gorontalo", "Sulawesi Barat", "Nusa Tenggara Timur"]
OFFSET = {
    "DKI Jakarta": (6, -10),
    "DI Yogyakarta": (6, 4),
    "Papua Pegunungan": (-6, 6),
    "Papua Selatan": (-6, 9),
    "Gorontalo": (-6, -4),
    "Sulawesi Barat": (8, 0),
    "Nusa Tenggara Timur": (-12, 4),
}
koma1 = FuncFormatter(lambda x, _: f"{x:.1f}".replace(".", ","))
koma2 = lambda x: f"{x:.2f}".replace(".", ",")


def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


# %%
# fig1: tren nasional
tahun = ["SP2010", "SUPAS 2015", "Long Form SP2020", "SUPAS 2025"]
tfr_nas = [2.41, 2.28, 2.18, 2.13]
imr_nas = [26.09, 21.80, 16.85, 14.12]
x = np.arange(len(tahun))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].plot(x, tfr_nas, "-o", color="#d62728")
ax[0].axhline(2.10, ls="--", color="gray", lw=1)
ax[0].text(0.02, 2.10, "Angka penggantian 2,10", va="bottom", ha="left",
           fontsize=8, color="gray")
for xi, yi in zip(x, tfr_nas):
    ax[0].annotate(koma2(yi), (xi, yi), textcoords="offset points",
                   xytext=(0, 8), ha="center", fontsize=9)
ax[0].set_xticks(x, tahun, fontsize=8)
ax[0].set_ylabel("TFR (anak per perempuan)")
ax[0].set_title("Tren Angka Kelahiran Total (TFR) Nasional, 2010-2025")
ax[0].set_ylim(1.95, 2.55)
ax[0].yaxis.set_major_formatter(koma1)

ax[1].plot(x, imr_nas, "-o", color="#1f77b4")
for xi, yi in zip(x, imr_nas):
    ax[1].annotate(koma2(yi), (xi, yi), textcoords="offset points",
                   xytext=(0, 8), ha="center", fontsize=9)
ax[1].set_xticks(x, tahun, fontsize=8)
ax[1].set_ylabel("IMR (kematian bayi per 1.000 kelahiran hidup)")
ax[1].set_title("Tren Angka Kematian Bayi (IMR) Nasional, 2010-2025")
ax[1].set_ylim(10, 29)
ax[1].yaxis.set_major_formatter(koma1)
fig.tight_layout()
save(fig, "fig1_tren_nasional")


# %%
def scatter_extreme(ax, xcol, ycol, ylabel, offsets=None, median_lines=True):
    offsets = OFFSET if offsets is None else offsets
    for reg, g in df.groupby("region"):
        ax.scatter(g[xcol], g[ycol], s=42, color=REG_COLORS[reg],
                   label=reg, edgecolor="white", linewidth=0.5, zorder=3)
    sub = df[[xcol, ycol]].dropna()
    b, a = np.polyfit(sub[xcol], sub[ycol], 1)
    xs = np.linspace(sub[xcol].min(), sub[xcol].max(), 50)
    ax.plot(xs, a + b * xs, color="black", lw=1.3, zorder=2)
    if median_lines:
        ax.axvline(med_tfr, ls=":", color="gray", lw=1)
        ax.axhline(med_imr, ls=":", color="gray", lw=1)
    for p in EXTREMES:
        r = df.loc[df["provinsi"] == p]
        if r[ycol].isna().iloc[0] or r[xcol].isna().iloc[0]:
            continue
        dx, dy = offsets[p]
        ax.annotate(p, (r[xcol].iloc[0], r[ycol].iloc[0]),
                    textcoords="offset points", xytext=(dx, dy),
                    ha="right" if dx < 0 else "left", fontsize=7.5,
                    annotation_clip=True)
    ax.set_xlabel("TFR (anak per perempuan)")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(koma1)
    ax.yaxis.set_major_formatter(koma1)


# %%
fig, ax = plt.subplots(figsize=(8.5, 6))
scatter_extreme(ax, "tfr", "imr", "IMR (kematian bayi per 1.000 kelahiran hidup)")
ax.set_title("Sebaran TFR dan IMR Menurut Provinsi, SUPAS 2025")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8,
          frameon=False, title="Wilayah")
fig.tight_layout()
save(fig, "fig2_scatter_tfr_imr")

# %%
# fig3: bar TFR per provinsi
d3 = df.sort_values("tfr", ascending=True)
fig, ax = plt.subplots(figsize=(7.5, 11))
ax.barh(d3["provinsi"], d3["tfr"],
        color=[REG_COLORS[r] for r in d3["region"]])
ax.axvline(2.10, ls="--", color="gray", lw=1.2)
ax.axvline(2.13, ls="-", color="black", lw=1.2)
ax.set_ylim(-0.6, 46.5)
tr = ax.get_xaxis_transform()
ax.text(2.07, 0.99, "Angka penggantian 2,10", transform=tr, rotation=90,
        rotation_mode="anchor", va="center", ha="right", fontsize=8,
        color="gray")
ax.text(2.16, 0.99, "TFR nasional 2,13", transform=tr, rotation=90,
        rotation_mode="anchor", va="center", ha="right", fontsize=8,
        color="black")
ax.set_xlabel("TFR (anak per perempuan)")
ax.set_title("Angka Kelahiran Total (TFR) Menurut Provinsi, SUPAS 2025")
ax.set_xlim(1.6, 2.9)
ax.xaxis.set_major_formatter(koma1)
handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=r)
           for r, c in REG_COLORS.items()]
ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False,
          title="Wilayah")
fig.tight_layout()
save(fig, "fig3_tfr_provinsi")

# %%
# fig4: lansia vs TFR
OFFSET4 = {
    "DKI Jakarta": (6, -10),
    "DI Yogyakarta": (-6, 5),
    "Papua Pegunungan": (-6, 6),
    "Papua Selatan": (6, -2),
    "Gorontalo": (-10, 20),
    "Sulawesi Barat": (6, 2),
    "Nusa Tenggara Timur": (8, 4),
}
fig, ax = plt.subplots(figsize=(8.5, 6))
scatter_extreme(ax, "pct_lansia", "tfr",
                "TFR (anak per perempuan)",
                offsets=OFFSET4, median_lines=False)
ax.set_xlabel("Persentase penduduk lansia (60 tahun ke atas)")
ax.set_title("Sebaran Persentase Penduduk Lansia dan TFR Menurut Provinsi, SUPAS 2025")
ax.set_ylim(1.7, 2.9)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8,
          frameon=False, title="Wilayah")
fig.tight_layout()
save(fig, "fig4_scatter_lansia_tfr")

# %% [markdown]
# ## Ringkasan
# Merangkai hasil tiap tahap menjadi teks ringkasan.

# %%
s = []
s.append("=" * 72)
s.append("RINGKASAN ANALISIS STATISTIKA SUPAS 2025 (38 PROVINSI)")
s.append("=" * 72)
s.append("")
s.append("Catatan: seluruh rerata provinsi bersifat TIDAK TERTIMBANG (unweighted),")
s.append("sehingga tidak sama dengan angka nasional.")
s.append("")

s.append("--- 1. STATISTIK DESKRIPTIF TFR ---")
s.append(f"n           : {desc['n']}")
s.append(f"mean        : {desc['mean']:.4f}")
s.append(f"median      : {desc['median']:.4f}")
s.append(f"sd          : {desc['sd']:.4f}")
s.append(f"cv          : {desc['cv']:.4f}")
s.append(f"min         : {desc['min']:.4f} (DKI Jakarta)")
s.append(f"q1          : {desc['q1']:.4f}")
s.append(f"q3          : {desc['q3']:.4f}")
s.append(f"iqr         : {desc['iqr']:.4f}")
s.append(f"max         : {desc['max']:.4f} (Papua Pegunungan)")
s.append(f"max/min     : {desc['rasio_max_min']:.4f}")
s.append("")

# %%
s.append("--- 2. KORELASI TFR (Pearson r / Spearman rho) ---")
s.extend(kor_txt)
s.append("")

s.append("--- 3. REGRESI OLS ---")
s.extend(reg_txt)

s.append("--- 4. TIPOLOGI PROVINSI (median TFR = "
         f"{med_tfr:.4f}, median IMR = {med_imr:.4f}) ---")
for tipe, g in tip.groupby("tipe"):
    s.append(f"{tipe} (n={len(g)}): {', '.join(sorted(g['provinsi']))}")
s.append("")
s.append("--- 5. FIGURE ---")
s.append("figures/fig1_tren_nasional.{png,pdf}")
s.append("figures/fig2_scatter_tfr_imr.{png,pdf}")
s.append("figures/fig3_tfr_provinsi.{png,pdf}")
s.append("figures/fig4_scatter_lansia_tfr.{png,pdf}")
s.append("")

with open(os.path.join(RES, "summary.txt"), "w") as f:
    f.write("\n".join(s) + "\n")
print("[selesai] results/ dan figures/ terisi")
