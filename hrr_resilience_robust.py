# -*- coding: utf-8 -*-
"""
hr_recovery_resilience — 조건부 판정의 견고성 점검 (v1 보충, 사후)
계획(실행 전 기록): 05_RUNS/hr_recovery_resilience/2026-10-12_conditional_plan.md
R1 전 검사 ICC(1) · R2 예측식 재적합 부트스트랩 · R3 회복≥125 s 전 검사+재적합 · R4 jackknife
v1 일차 판정은 바꾸지 않는다.
실행: python hrr_resilience_robust.py [V1_DIR] [OUT_DIR]
"""
import os, sys, json, hashlib, warnings
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'results_hrr_resilience_v1')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, '..', 'results_hrr_resilience_v1_robust')
os.makedirs(OUT, exist_ok=True)
SEED, NB, THR = 42, 2000, 0.40
C6 = [f'{d}{w}' for d in 'AB' for w in (60, 120, 180)]
C4 = [f'{d}{w}' for d in 'AB' for w in (60, 120)]
COV = ['vo2peak_kg', 'age', 'sex', 'hr_peak']

# ── 입력 무결성: v1 test_level.csv ↔ v1 SHA256SUMS ─────────────────────
sums = dict(l.split()[::-1] for l in open(f'{V1}/SHA256SUMS.txt'))
h = hashlib.sha256(open(f'{V1}/test_level.csv', 'rb').read()).hexdigest()
assert h == sums['test_level.csv'], 'v1 test_level.csv SHA-256 불일치'
T = pd.read_csv(f'{V1}/test_level.csv')
ok = T.fail.isna() | (T.fail == '')

# ── ICC 함수 · 엔진 검증 ────────────────────────────────────────────────
def icc21(Y):
    Y = Y[~np.isnan(Y).any(1)]; n, k = Y.shape; gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1); msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)

def icc1_unbal(y, g):
    """일원 무선 ICC(1), 불균형 ANOVA 추정."""
    d = pd.DataFrame({'y': y, 'g': g}).dropna(); N = len(d)
    grp = d.groupby('g').y; ni = grp.size().to_numpy(); k = len(ni); mi = grp.mean().to_numpy()
    msb = (ni * (mi - d.y.mean()) ** 2).sum() / (k - 1)
    msw = ((d.y - d.g.map(grp.mean())) ** 2).sum() / (N - k)
    n0 = (N - (ni ** 2).sum() / N) / (k - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw)

SF = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float)
assert round(icc21(SF), 2) == 0.29
assert round(icc1_unbal(SF.ravel(), np.repeat(np.arange(6), 4)), 2) == 0.17  # Shrout & Fleiss ICC(1,1)=0.17

def fit(F, y, P):
    X = lambda d: np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in COV])
    b, *_ = np.linalg.lstsq(X(F), F[y].to_numpy(float), rcond=None)
    return P[y].to_numpy(float) - X(P) @ b

def sets(cells, rec):
    E = T[T.adult & ok & (T.rec_dur >= rec) & T.std_load & T[cells + ['vo2peak_kg']].notna().all(1)].sort_values(['subj', 'age', 'k'])
    c = E.groupby('subj').id_test.transform('count')
    R = E[c >= 2].copy(); F = E[c == 1].copy()
    R['visit'] = R.groupby('subj').cumcount() + 1
    return R, F

def first_two(R):
    return R[R.visit <= 2]

def wide(R, col):
    return R.pivot(index='subj', columns='visit', values=col)[[1, 2]].to_numpy(float)

def ci(x, a): return np.nanpercentile(x, 100 * a / 2), np.nanpercentile(x, 100 * (1 - a / 2))

rng = np.random.default_rng(SEED)
rows = []

def run(label, cells, rec, all_visits, refit):
    R, F = sets(cells, rec)
    if not all_visits: R = first_two(R)
    subs = R.subj.unique(); fsub = F.subj.to_numpy()
    Rg = {s: g for s, g in R.groupby('subj')}
    def stat(Rb, Fb, y):
        res = fit(Fb, y, Rb)
        if all_visits: return icc1_unbal(res, Rb['_g'].to_numpy())
        tmp = Rb.assign(_r=res)
        return icc21(tmp.pivot(index='_g', columns='visit', values='_r')[[1, 2]].to_numpy(float))
    base = R.assign(_g=R.subj)
    est = {y: stat(base, F, y) for y in cells}
    B = {y: np.empty(NB) for y in cells}
    for b in range(NB):
        pick = rng.choice(subs, len(subs))
        Rb = pd.concat([Rg[s].assign(_g=i) for i, s in enumerate(pick)])
        Fb = F.iloc[rng.integers(0, len(F), len(F))] if refit else F
        for y in cells: B[y][b] = stat(Rb, Fb, y)
    ab = 0.05 / len(cells)
    for y in cells:
        lo, hi = ci(B[y], .05); blo, bhi = ci(B[y], ab)
        rows.append(dict(scenario=label, cell=y, rec_min_s=rec, n_subj=len(subs), n_tests=len(R), n_fit=len(F),
                         icc_type='ICC(1) 전 검사' if all_visits else 'ICC(2,1) 첫 두 검사', refit=refit,
                         ICC=est[y], CI95_lo=lo, CI95_hi=hi, CIbonf_lo=blo, CIbonf_hi=bhi,
                         verdict='통과' if blo > THR else '미달'))

# 엔진 대조: 불균형 ICC(1) vs MixedLM (R1 표본, B120 잔차)
R1s, F1s = sets(C6, 180)
r = fit(F1s, 'B120', R1s)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import statsmodels.formula.api as smf
    mm = smf.mixedlm('r ~ 1', pd.DataFrame({'r': r, 'g': R1s.subj.to_numpy()}), groups='g').fit(reml=True)
icc_mlm = float(mm.cov_re.iloc[0, 0] / (mm.cov_re.iloc[0, 0] + mm.scale))
icc_anova = icc1_unbal(r, R1s.subj.to_numpy())
assert abs(icc_mlm - icc_anova) < 0.03, (icc_mlm, icc_anova)

run('R1 전 검사', C6, 180, all_visits=True, refit=False)
run('R2 예측식 재적합', C6, 180, all_visits=False, refit=True)
run('R3 회복≥125 s 전 검사+재적합', C4, 125, all_visits=True, refit=True)
S = pd.DataFrame(rows); S.to_csv(f'{OUT}/robust_summary.csv', index=False)

# R4 jackknife — v1 57쌍, v1 잔차(고정 예측식)
P = pd.read_csv(f'{V1}/pairs_level.csv')
jk = []
for y in ('res_B120', 'res_B180'):
    W = P.pivot(index='subj', columns='visit', values=y)[[1, 2]]
    full = icc21(W.to_numpy(float))
    loo = pd.Series({s: icc21(W.drop(s).to_numpy(float)) for s in W.index})
    jk.append(dict(cell=y, ICC_full=full, loo_min=loo.min(), loo_max=loo.max(), most_influential_subj=int((loo - full).abs().idxmax()),
                   loo_at_most_influential=float(loo[(loo - full).abs().idxmax()]), stays_ge_060=bool(loo.min() >= .60)))
J = pd.DataFrame(jk); J.to_csv(f'{OUT}/jackknife.csv', index=False)

# 종합 판정 (계획의 규칙)
def passed(sc, c): return bool((S.scenario.eq(sc) & S.cell.eq(c) & S.verdict.eq('통과')).any())
robust = {c: passed('R1 전 검사', c) and passed('R2 예측식 재적합', c) and bool(J.set_index('cell').loc[f'res_{c}', 'stays_ge_060']) for c in ('B120', 'B180')}
summary = dict(engine_icc1_mixedlm=icc_mlm, engine_icc1_anova=icc_anova, robust=robust,
               r3_any_pass=bool(S[S.scenario.str.startswith('R3')].verdict.eq('통과').any()),
               plan='05_RUNS/hr_recovery_resilience/2026-10-12_conditional_plan.md', seed=SEED, nboot=NB,
               input_sha256_test_level=h)
json.dump(summary, open(f'{OUT}/robust_verdict.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
pd.set_option('display.width', 220)
print(S[['scenario', 'cell', 'n_subj', 'n_tests', 'ICC', 'CI95_lo', 'CIbonf_lo', 'verdict']].round(3).to_string(index=False))
print(J.round(3).to_string(index=False)); print(json.dumps(summary, ensure_ascii=False, default=str))

# ── 탐색(판정에 쓰지 않음): 회복 125–180 s 검사가 ≥180 s 검사와 다른가 — 측정 품질 차이 점검 ──────
RAW = 'D:/treadmil/raw' if os.path.isdir('D:/treadmil/raw') else os.path.expanduser('~/mnt/treadmil/raw')
m = pd.read_csv(os.path.join(RAW, 'test_measure.csv'))
E = T[T.adult & ok & (T.rec_dur >= 125) & T.std_load & T[C4 + ['vo2peak_kg']].notna().all(1)].copy()
E['grp'] = np.where(E.rec_dur >= 180, 'rec>=180', 'rec125-180')
mg = {k: g for k, g in m[m.ID_test.isin(E.id_test)].groupby('ID_test')}
q = []
for _, r in E.iterrows():
    g = mg[r.id_test]; t = g.time - g.time.min(); tr = t - r.t0_B
    q.append(dict(grp=r.grp, speed_110_130=g.Speed[(tr >= 110) & (tr <= 130)].mean(), speed_last10=g.Speed[t >= t.max() - 10].mean()))
Q = pd.concat([E[['grp', 'rec_dur', 'age', 'vo2peak_kg', 'hr_peak', 'B60', 'B120']].reset_index(drop=True), pd.DataFrame(q).drop(columns='grp')], axis=1)
Q.groupby('grp').agg(['count', 'mean', 'std']).round(2).to_csv(f'{OUT}/explore_recovery_length.csv')
print(Q.groupby('grp').mean(numeric_only=True).round(2).to_string())
