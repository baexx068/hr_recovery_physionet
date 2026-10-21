# -*- coding: utf-8 -*-
"""v1 추가 분석 X1 임상 기준 재분류 · X2 SEM/MDC · X3 HRR–VO2peak 상관 · Table 1
사전지정: analysis.md '사전지정 v1 추가'(2026-10-13, 실행 전). 기술 분석 — 판정 없음. 성별 하위분석 없음."""
import os, json, hashlib
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); V1 = os.path.join(HERE, '..', 'results_hrr_resilience_v1')
V2 = os.path.join(HERE, '..', 'results_hrr_resilience_v2'); OUT = os.path.join(HERE, '..', 'results_hrr_resilience_v1_extra')
RAW = 'D:/treadmil/raw' if os.path.isdir('D:/treadmil/raw') else os.path.expanduser('~/mnt/treadmil/raw')
os.makedirs(OUT, exist_ok=True); NB, SEED = 2000, 42
CELLS = [f'{d}{w}' for d in 'AB' for w in (60, 120, 180)]

sums = dict(l.split()[::-1] for l in open(f'{V1}/SHA256SUMS.txt'))
for f in ('test_level.csv', 'pairs_level.csv'):
    assert hashlib.sha256(open(f'{V1}/{f}', 'rb').read()).hexdigest() == sums[f], f

# ── 엔진 ────────────────────────────────────────────────────────────────
def anova2(Y):
    n, k = Y.shape; gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1); msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return msr, msc, mse
def icc21(Y):
    n, k = Y.shape; msr, msc, mse = anova2(Y); return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
SF = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float)
assert round(icc21(SF), 2) == 0.29
def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool); po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()); return (po - pe) / (1 - pe) if pe < 1 else np.nan
assert abs(kappa([1, 1, 0, 0], [1, 0, 0, 0]) - 0.5) < 1e-12        # 손계산: po .75, pe .5 → .5
def partial_r(y, x, Z):
    Zc = np.column_stack([np.ones(len(y)), Z]); ry = y - Zc @ np.linalg.lstsq(Zc, y, rcond=None)[0]; rx = x - Zc @ np.linalg.lstsq(Zc, x, rcond=None)[0]
    return np.corrcoef(ry, rx)[0, 1]

T = pd.read_csv(f'{V1}/test_level.csv'); E = T[T.eligible].copy()
s = pd.read_csv(f'{RAW}/subject-info.csv').set_index('ID_test')
E['height'] = E.id_test.map(s.Height); E['bmi'] = E.weight / (E.height / 100) ** 2
P = pd.read_csv(f'{V1}/pairs_level.csv'); rng = np.random.default_rng(SEED)
assert len(E) == 596 and E.subj.nunique() == 529 and P.subj.nunique() == 57  # 218_1 사후 제외(OQ-09, 2026-10-13) 뒤 표본

# ── X1 임상 기준 재분류 ─────────────────────────────────────────────────
# Shetler 2001 초록은 "a decrease of <22 beats/min" — 사전지정(≤ 22)과 원문 표기(< 22)를 둘 다 낸다(2026-10-13 초록 대조 후 추가)
crit = [('60 s ≤ 12 bpm (Cole 1999)', 'A60', 'B60', 12, 'le'), ('120 s ≤ 22 bpm (사전지정)', 'A120', 'B120', 22, 'le'),
        ('120 s < 22 bpm (Shetler 2001 원문 표기)', 'A120', 'B120', 22, 'lt')]
groups = {k: g.index.to_numpy() for k, g in E.reset_index(drop=True).groupby('subj')}; keys = list(groups); Er = E.reset_index(drop=True)
x1 = []
for lab, ca, cb, thr, op in crit:
    f = (lambda v: v <= thr) if op == 'le' else (lambda v: v < thr)
    a = f(Er[ca]).to_numpy(); b = f(Er[cb]).to_numpy()
    def stat(i): return np.array([a[i].mean(), b[i].mean(), b[i].mean() - a[i].mean(), np.mean(a[i] != b[i]), kappa(a[i], b[i])])
    est = stat(np.arange(len(a))); B = np.empty((NB, 5))
    for k in range(NB):
        pick = rng.choice(len(keys), len(keys)); B[k] = stat(np.concatenate([groups[keys[j]] for j in pick]))
    lo, hi = np.nanpercentile(B, 2.5, axis=0), np.nanpercentile(B, 97.5, axis=0)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    mc = stats.binomtest(min(n01, n10), n01 + n10, .5).pvalue if n01 + n10 else np.nan
    for j, nm in enumerate(['prev_A', 'prev_B', 'prev_diff_B_minus_A', 'reclassified', 'kappa']):
        x1.append(dict(criterion=lab, metric=nm, estimate=est[j], CI95_lo=lo[j], CI95_hi=hi[j]))
    x1.append(dict(criterion=lab, metric='A_only_abnormal_n', estimate=n10)); x1.append(dict(criterion=lab, metric='B_only_abnormal_n', estimate=n01))
    x1.append(dict(criterion=lab, metric='mcnemar_exact_p_reference', estimate=mc)); x1.append(dict(criterion=lab, metric='n_tests', estimate=len(a)))
pd.DataFrame(x1).to_csv(f'{OUT}/X1_reclassification.csv', index=False)

# ── X2 SEM · MDC95 ─────────────────────────────────────────────────────
x2 = []
for c in CELLS:
    for kind, col in (('raw HRR', c), ('residual', f'res_{c}')):
        Y = P.pivot(index='subj', columns='visit', values=col)[[1, 2]].to_numpy(float)
        def st(Yb): sem = np.sqrt(anova2(Yb)[2]); return np.array([sem, 1.96 * np.sqrt(2) * sem])
        est = st(Y); B = np.array([st(Y[rng.integers(0, len(Y), len(Y))]) for _ in range(NB)])
        mean = P[c].mean()
        x2.append(dict(cell=c, measure=kind, n_pairs=len(Y), SEM=est[0], SEM_lo=np.percentile(B[:, 0], 2.5), SEM_hi=np.percentile(B[:, 0], 97.5),
                       MDC95=est[1], MDC95_lo=np.percentile(B[:, 1], 2.5), MDC95_hi=np.percentile(B[:, 1], 97.5),
                       mean_HRR=mean, SEM_pct_mean=est[0] / mean * 100 if kind == 'raw HRR' else np.nan))
pd.DataFrame(x2).to_csv(f'{OUT}/X2_sem_mdc.csv', index=False)

# ── X3 HRR–VO2peak 상관 (참가자당 첫 검사) ──────────────────────────────
F1 = E.sort_values(['subj', 'age', 'k']).groupby('subj').head(1); assert len(F1) == 529
x3 = []
for c in CELLS:
    y, x = F1[c].to_numpy(float), F1.vo2peak_kg.to_numpy(float); r = np.corrcoef(x, y)[0, 1]; n = len(y)
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    x3.append(dict(cell=c, n=n, pearson_r=r, r_lo=np.tanh(z - 1.96 * se), r_hi=np.tanh(z + 1.96 * se),
                   spearman_rho=stats.spearmanr(x, y).statistic,
                   partial_r_age_sex=partial_r(y, x, F1[['age', 'sex']].to_numpy(float))))
pd.DataFrame(x3).to_csv(f'{OUT}/X3_hrr_vo2peak_correlation.csv', index=False)

# ── Table 1 ────────────────────────────────────────────────────────────
E['height'] = E['height']
grp = {'Repeat participants, test 1': E[E.id_test.isin(P[P.visit == 1].id_test)],
       'Repeat participants, test 2': E[E.id_test.isin(P[P.visit == 2].id_test)],
       'Single-test participants': E[~E.subj.isin(P.subj)], 'All eligible tests': E}
vars_ = [('age', 'Age (y)'), ('height', 'Height (cm)'), ('weight', 'Body mass (kg)'), ('bmi', 'BMI (kg/m²)'),
         ('vo2peak_kg', 'VO2peak (mL/kg/min)'), ('hr_peak', 'Peak HR (bpm)'), ('rer_peak', 'Peak RER'), ('t0_B', 'Exercise duration to end of load (s)')] + [(c, f'HRR {c[0]} {c[1:]} s (bpm)') for c in CELLS]
rows = [dict(variable='n tests', **{k: len(g) for k, g in grp.items()}), dict(variable='n participants', **{k: g.subj.nunique() for k, g in grp.items()}),
        dict(variable='Female, n (%)', **{k: f'{int(g.sex.sum())} ({g.sex.mean() * 100:.1f})' for k, g in grp.items()})]
for v, lab in vars_:
    rows.append(dict(variable=lab, **{k: f'{g[v].mean():.1f} ± {g[v].std():.1f}' if v != 'rer_peak' else f'{g[v].mean():.2f} ± {g[v].std():.2f}' for k, g in grp.items()}))
gap = P.groupby('subj').gap_y.first()
rows.append(dict(variable='Interval between tests (y), median [IQR]', **{'Repeat participants, test 2': f'{gap.median():.1f} [{gap.quantile(.25):.1f}–{gap.quantile(.75):.1f}]'}))
pd.DataFrame(rows).to_csv(f'{OUT}/Table1_characteristics.csv', index=False)

pd.set_option('display.width', 250)
print(pd.DataFrame(x1).round(3).to_string(index=False)); print(pd.DataFrame(x2).round(2).to_string(index=False))
print(pd.DataFrame(x3).round(3).to_string(index=False)); print(pd.DataFrame(rows).to_string(index=False))
