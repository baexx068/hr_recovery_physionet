# -*- coding: utf-8 -*-
"""⑧ 사후 분석 r2 (2026-10-13, 2회차 자기 심사 뒤 사용자 결정) — 사전지정 아님, 원고에 '사후'로 표시한다.
P1 몬테카를로 민감도: 일차 회복 잔차 ICC의 Bonferroni 99.17% 하한을 (a) 재표본 20,000회(seed 42), (b) seed 1–50 × 2,000회로 다시 구한다.
P2 기점 차(Δt₀ = t₀B − t₀A)로 나눈 쌍: 두 검사 모두 |Δt₀| ≤ 22 s(전체 검사 Δt₀의 3사분위) 대 한 검사라도 > 22 s.
   층마다 잔차 ICC(95% CI)와 B − A 차(95% CI). 예측식은 일차와 같이 단일 검사 표본에서 한 번 적합하고 고정.
P3 1분 ≤ 12 bpm 재분류를 검사 단위 |Δt₀| ≤ 22 s 대 > 22 s로 나눔.
입력: results_hrr_resilience_v1/test_level.csv (218_1 사후 제외 반영본) · 출력: results_hrr_resilience_v1_posthoc_r2/"""
import os, json
import numpy as np, pandas as pd

H = os.path.dirname(os.path.abspath(__file__)); V1 = os.path.join(H, '..', 'results_hrr_resilience_v1')
OUT = os.path.join(H, '..', 'results_hrr_resilience_v1_posthoc_r2'); os.makedirs(OUT, exist_ok=True)
CELLS = ['A60', 'A120', 'A180', 'B60', 'B120', 'B180']; THR = 0.40; CUT = 22.0

T = pd.read_csv(f'{V1}/test_level.csv'); E = T[T.eligible].sort_values(['subj', 'age', 'k']).copy()
E['dt'] = E.t0_B - E.t0_A
n_by = E.groupby('subj').id_test.transform('count')
P = E[n_by >= 2].groupby('subj').head(2).copy(); P['visit'] = P.groupby('subj').cumcount() + 1; F = E[n_by == 1]
assert P.subj.nunique() == 57 and len(F) == 472 and np.isclose(E.dt.quantile(.75), CUT)
X = lambda d: np.column_stack([np.ones(len(d)), d.vo2peak_kg, d.age, d.sex, d.hr_peak])
for c in CELLS:
    b = np.linalg.lstsq(X(F), F[c].to_numpy(float), rcond=None)[0]; P['res_' + c] = P[c].to_numpy(float) - X(P) @ b

def icc21(Y):
    n, k = Y.shape; gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1); msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)

def mats(Pp): return {c: Pp.pivot(index='subj', columns='visit', values='res_' + c)[[1, 2]].to_numpy(float) for c in CELLS}

def boot(M, nb, seed):
    rng = np.random.default_rng(seed); n = len(next(iter(M.values()))); B = {c: np.empty(nb) for c in M}
    for i in range(nb):
        idx = rng.integers(0, n, n)
        for c, v in M.items(): B[c][i] = icc21(v[idx])
    return B

M = mats(P); est = {c: icc21(v) for c, v in M.items()}; a = 0.05 / 6
# 일차 재현(seed 42, 2,000회)
B0 = boot(M, 2000, 42); prim = pd.read_csv(f'{V1}/primary_family.csv').set_index('key')
for c in CELLS: assert np.isclose(np.percentile(B0[c], 100 * a / 2), prim.at['res_' + c, 'CIbonf_lo']), c

# P1
B20 = boot(M, 20000, 42)
seeds = {c: [] for c in CELLS}
for s in range(1, 51):
    Bs = boot(M, 2000, s)
    for c in CELLS: seeds[c].append(np.percentile(Bs[c], 100 * a / 2))
P1 = pd.DataFrame([dict(cell=c, ICC=est[c], lo_seed42_2000=np.percentile(B0[c], 100 * a / 2), lo_seed42_20000=np.percentile(B20[c], 100 * a / 2),
                        lo_seeds_median=np.median(seeds[c]), lo_seeds_min=np.min(seeds[c]), lo_seeds_max=np.max(seeds[c]),
                        seeds_pass_of_50=int((np.array(seeds[c]) > THR).sum())) for c in CELLS])
P1.to_csv(f'{OUT}/P1_montecarlo.csv', index=False)

# P2
mx = P.groupby('subj').dt.apply(lambda s: s.abs().max())
rows, rng = [], np.random.default_rng(42)
for lab, ids in (('all', mx.index), (f'max|dt|<={CUT:.0f}s', mx[mx <= CUT].index), (f'max|dt|>{CUT:.0f}s', mx[mx > CUT].index)):
    Ms = mats(P[P.subj.isin(ids)]); n = len(next(iter(Ms.values())))
    Bs = {c: np.empty(2000) for c in CELLS}
    for i in range(2000):
        idx = rng.integers(0, n, n)
        for c, v in Ms.items(): Bs[c][i] = icc21(v[idx])
    for c in CELLS:
        rows.append(dict(stratum=lab, n_pairs=n, cell=c, ICC=icc21(Ms[c]), CI95_lo=np.percentile(Bs[c], 2.5), CI95_hi=np.percentile(Bs[c], 97.5)))
    for w in (60, 120, 180):
        d = Bs[f'B{w}'] - Bs[f'A{w}']
        rows.append(dict(stratum=lab, n_pairs=n, cell=f'B-A {w}', ICC=icc21(Ms[f'B{w}']) - icc21(Ms[f'A{w}']), CI95_lo=np.percentile(d, 2.5), CI95_hi=np.percentile(d, 97.5)))
P2 = pd.DataFrame(rows); P2.to_csv(f'{OUT}/P2_dt_strata.csv', index=False)
# P2b 절단값 견고성(점추정, 3회차 심사 반영 2026-10-13) · P2c 개인 안 기점 차 재현성
rb = []
for cut in (15, 22, 30, 40):
    for lab, ids in ('close', mx[mx <= cut].index), ('long', mx[mx > cut].index):
        Ms = mats(P[P.subj.isin(ids)])
        rb.append(dict(cut_s=cut, group=lab, n_pairs=len(ids), **{c: icc21(Ms[c]) for c in CELLS}, **{f'B-A {w}': icc21(Ms[f'B{w}']) - icc21(Ms[f'A{w}']) for w in (60, 120, 180)}))
pd.DataFrame(rb).to_csv(f'{OUT}/P2b_cutoffs.csv', index=False)
from scipy.stats import spearmanr
wd = P.pivot(index='subj', columns='visit', values='dt'); lg = wd.abs() > CUT
json.dump(dict(spearman_dt_test1_test2=float(spearmanr(wd[1], wd[2])[0]), pairs_long_any=int(lg.any(axis=1).sum()), pairs_long_both=int(lg.all(axis=1).sum()),
               dt_pct2_5=float(np.percentile(E.dt, 2.5)), dt_pct97_5=float(np.percentile(E.dt, 97.5)), dt_min=float(E.dt.min()), dt_max=float(E.dt.max())),
          open(f'{OUT}/P2c_lag.json', 'w'), indent=1)

# P3
E['abnA'] = E.A60 <= 12; E['abnB'] = E.B60 <= 12
P3 = pd.DataFrame([dict(stratum=lab, n_tests=int(len(g)), abnormal_A_pct=100 * g.abnA.mean(), abnormal_B_pct=100 * g.abnB.mean(),
                        A_only_n=int((g.abnA & ~g.abnB).sum()))
                   for lab, g in (('all', E), (f'|dt|<={CUT:.0f}s', E[E.dt.abs() <= CUT]), (f'|dt|>{CUT:.0f}s', E[E.dt.abs() > CUT]))])
P3.to_csv(f'{OUT}/P3_reclass_by_dt.csv', index=False)
json.dump(dict(cut_s=CUT, cut_rule='75th percentile of dt in eligible tests', n_tests=int(len(E)), seed=42), open(f'{OUT}/run_info.json', 'w'), indent=1)
pd.set_option('display.width', 200)
for d in (P1, P2, P3): print(d.round(3).to_string(index=False))
