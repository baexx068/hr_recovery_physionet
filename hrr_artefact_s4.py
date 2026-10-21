# -*- coding: utf-8 -*-
"""⑧ 사후 민감도 S4 — HR 신호 이상 의심 검사 제외 (규칙은 analysis.md '사후 민감도 S4'에 실행 전 기록, 2026-10-13).
규칙: 평활 HR 최고값 − 부하 종료(t₀ B ± 5 s) 평활 HR 평균 > 10 bpm(견고성 5 · 15 bpm)이면 표시하고 적격 표본에서 뺀다.
그 뒤 적합 표본 · 반복 쌍(남은 검사 중 첫 두 검사)을 다시 만들어 일차와 같은 6셀 잔차 ICC(99.17% 부트스트랩, seed 42) ·
B − A 대비(95%) · 1분 ≤ 12 bpm 재분류를 낸다.
입력: results_hrr_resilience_v1/test_level.csv · 원자료 test_measure.csv  출력: results_hrr_resilience_v1_posthoc_r2/S4_*.csv"""
import os, json
import numpy as np, pandas as pd

H = os.path.dirname(os.path.abspath(__file__)); V1 = os.path.join(H, '..', 'results_hrr_resilience_v1')
OUT = os.path.join(H, '..', 'results_hrr_resilience_v1_posthoc_r2')
RAW = 'D:/treadmil/raw' if os.path.isdir('D:/treadmil/raw') else os.path.expanduser('~/mnt/treadmil/raw')
CELLS = ['A60', 'A120', 'A180', 'B60', 'B120', 'B180']; THR = 0.40; A = 0.05 / 6

def smooth(y, w=10): return pd.Series(y).rolling(w, min_periods=3, center=True).mean().to_numpy()  # hrr_resilience.py와 같다

T = pd.read_csv(f'{V1}/test_level.csv'); E = T[T.eligible].copy()
m = pd.read_csv(f'{RAW}/test_measure.csv'); m = m[m.ID_test.isin(E.id_test)]
m = m[m.HR.between(30, 230) | m.HR.isna()]
gap = {}
for tid, g in m.groupby('ID_test'):
    t = (g.time - g.time.min()).to_numpy(float); hs = smooth(g.HR.to_numpy(float))
    r = E.set_index('id_test').loc[tid]
    assert abs(np.nanmax(hs) - r.hr_peak) < 1e-6, tid          # 최고 HR 재현
    k = np.abs(t - r.t0_B) <= 5
    gap[tid] = r.hr_peak - np.nanmean(hs[k])
E['gap_end'] = E.id_test.map(gap); assert E.gap_end.notna().all()

def icc21(Y):
    n, k = Y.shape; gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1); msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)

def run(Es):
    Es = Es.sort_values(['subj', 'age', 'k']); nb = Es.groupby('subj').id_test.transform('count')
    P = Es[nb >= 2].groupby('subj').head(2).copy(); P['visit'] = P.groupby('subj').cumcount() + 1; F = Es[nb == 1]
    X = lambda d: np.column_stack([np.ones(len(d)), d.vo2peak_kg, d.age, d.sex, d.hr_peak])
    M = {}
    for c in CELLS:
        b = np.linalg.lstsq(X(F), F[c].to_numpy(float), rcond=None)[0]
        P['r'] = P[c].to_numpy(float) - X(P) @ b
        M[c] = P.pivot(index='subj', columns='visit', values='r')[[1, 2]].to_numpy(float)
    rng = np.random.default_rng(42); n = len(M['A60']); B = {c: np.empty(2000) for c in CELLS}
    for i in range(2000):
        idx = rng.integers(0, n, n)
        for c in CELLS: B[c][i] = icc21(M[c][idx])
    rows = [dict(cell=c, ICC=icc21(M[c]), CIbonf_lo=np.percentile(B[c], 100 * A / 2), CIbonf_hi=np.percentile(B[c], 100 * (1 - A / 2))) for c in CELLS]
    for w in (60, 120, 180):
        d = B[f'B{w}'] - B[f'A{w}']
        rows.append(dict(cell=f'B-A {w}', ICC=icc21(M[f'B{w}']) - icc21(M[f'A{w}']), CIbonf_lo=np.percentile(d, 2.5), CIbonf_hi=np.percentile(d, 97.5)))
    R = pd.DataFrame(rows); R['n_pairs'] = n; R['n_fit'] = len(F); R['n_tests'] = len(Es)
    R['abn1_A_pct'] = 100 * (Es.A60 <= 12).mean(); R['abn1_B_pct'] = 100 * (Es.B60 <= 12).mean()
    R['pass'] = np.where(R.cell.isin(CELLS), R.CIbonf_lo > THR, np.nan)
    return R

prim = pd.read_csv(f'{V1}/primary_family.csv').set_index('key')
full = run(E)
for c in CELLS: assert np.isclose(full.set_index('cell').at[c, 'CIbonf_lo'], prim.at[f'res_{c}', 'CIbonf_lo']), c   # 일차 재현
res = []
for cut in (5, 10, 15):
    flag = E.gap_end > cut
    R = run(E[~flag]); R.insert(0, 'gap_cut_bpm', cut); R['n_flagged'] = int(flag.sum())
    R['n_flagged_dt_gt22'] = int((flag & ((E.t0_B - E.t0_A).abs() > 22)).sum()); res.append(R)
S4 = pd.concat(res); S4.to_csv(f'{OUT}/S4_artefact.csv', index=False)
E[['id_test', 'subj', 'gap_end']].assign(dt=E.t0_B - E.t0_A).to_csv(f'{OUT}/S4_gap_by_test.csv', index=False)
dt = E.t0_B - E.t0_A
json.dump(dict(rule='peak smoothed HR minus mean smoothed HR within t0_B ± 5 s > cut', main_cut_bpm=10,
               gap_median_dt_le22=float(E.gap_end[dt.abs() <= 22].median()), gap_median_dt_gt22=float(E.gap_end[dt.abs() > 22].median()),
               gap_median_dt_gt60=float(E.gap_end[dt > 60].median()), n_dt_gt22=int((dt.abs() > 22).sum())), open(f'{OUT}/S4_info.json', 'w'), indent=1)
pd.set_option('display.width', 200); print(S4.round(3).to_string(index=False)); print(json.load(open(f'{OUT}/S4_info.json')))
