# -*- coding: utf-8 -*-
"""
hr_recovery_resilience v1 — 운동 후 심박 회복 탄력성 표현형
사전지정: 03_RESEARCH/projects/hr_recovery_resilience/analysis.md '사전지정' (2026-10-12 21:30 KST)

질문: 같은 체력(VO2peak/kg)·나이·성별·HR최고점에서 기대되는 HRR보다 빠르거나 느린 정도(회복 잔차)가
      같은 사람을 다시 쟀을 때도 유지되는가?  t0 정의 A/B × 회복 구간 60/120/180 s, 6조합.
판정: 조합별 잔차 ICC(2,1)의 Bonferroni(6) 보정 부트스트랩 CI 하한 > 0.40

실행: python hrr_resilience.py [RAW_DIR] [OUT_DIR]
  RAW_DIR 기본: D:/treadmil/raw 가 있으면 그것, 없으면 ~/mnt/treadmil/raw
  OUT_DIR 기본: 이 파일 옆 ../results_hrr_resilience_v1
"""
import os, sys, json, hashlib, platform, datetime
import numpy as np, pandas as pd
from scipy import stats

SEED, NBOOT = 42, 2000
WINDOWS, DEFS = (60, 120, 180), ('A', 'B')
CELLS = [f'{d}{w}' for d in DEFS for w in WINDOWS]
ALPHA_BONF = 0.05 / len(CELLS)
THRESH = 0.40
SHA = {'subject-info.csv': 'e9ca5ba20eb1272df56c11d3e584ec3997d5484454812bd94966e58082415f16',
       'test_measure.csv': '3b5ed5586dc7412ccdfb26f804e29c06482a2a1df2060399c2731ebc68d4613d'}

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = sys.argv[1] if len(sys.argv) > 1 else ('D:/treadmil/raw' if os.path.isdir('D:/treadmil/raw')
                                             else os.path.expanduser('~/mnt/treadmil/raw'))
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, '..', 'results_hrr_resilience_v1')
os.makedirs(OUT, exist_ok=True)

# ── 0. 엔진 검증 · 원자료 무결성 ─────────────────────────────────────────
def _ms(Y):
    Y = np.asarray(Y, float); Y = Y[~np.isnan(Y).any(1)]
    n, k = Y.shape
    gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return n, k, msr, msc, mse

def icc21(Y):
    """ICC(2,1): 이원 무선, 단일 측정, 절대 일치 (Shrout & Fleiss 1979; McGraw & Wong 1996)."""
    if np.asarray(Y, float)[~np.isnan(np.asarray(Y, float)).any(1)].shape[0] < 3: return np.nan
    n, k, msr, msc, mse = _ms(Y)
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)

def icc21_ci_F(Y, alpha):
    """[사후 점검] McGraw & Wong 1996 ICC(A,1) F 근사 CI — 부트스트랩 백분위 CI의 소표본 피복 점검용."""
    n, k, msr, msc, mse = _ms(Y); r = icc21(Y)
    a = k * r / (n * (1 - r)); b = 1 + k * r * (n - 1) / (n * (1 - r))
    v = (a * msc + b * mse) ** 2 / ((a * msc) ** 2 / (k - 1) + (b * mse) ** 2 / ((n - 1) * (k - 1)))
    fl, fu = stats.f.ppf(1 - alpha / 2, n - 1, v), stats.f.ppf(1 - alpha / 2, v, n - 1)
    lo = n * (msr - fl * mse) / (fl * (k * msc + (k * n - k - n) * mse) + n * msr)
    hi = n * (fu * msr - mse) / (k * msc + (k * n - k - n) * mse + n * fu * msr)
    return lo, hi

SF = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]
assert round(icc21(SF), 2) == 0.29, 'ICC(2,1) 엔진 검증 실패 (Shrout & Fleiss 1979 Table 2 = 0.29)'
assert [round(x, 2) for x in icc21_ci_F(SF, 0.05)] == [0.02, 0.76], 'F 근사 CI 검증 실패 (pingouin 0.6.1 ICC(A,1) CI95 = [0.02, 0.76])'

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
for fn, exp in SHA.items():
    got = sha256(os.path.join(RAW, fn))
    assert got == exp, f'SHA-256 불일치 {fn}: {got}'

s = pd.read_csv(os.path.join(RAW, 'subject-info.csv'))
m = pd.read_csv(os.path.join(RAW, 'test_measure.csv'))
assert s.ID_test.is_unique and set(s.Sex.unique()) <= {0, 1} and (s.Age > 0).all()
assert set(m.ID_test.unique()) <= set(s.ID_test)
m = m.sort_values(['ID_test', 'time'], kind='stable')
# 호흡별 기록이라 같은 초에 두 호흡이 찍힌 행이 있다(3,021행/509검사) — 감소만 금지한다
assert (m.groupby('ID_test').time.diff().dropna() >= 0).all(), 'time이 검사 안에서 감소함'

# ── 1. 검사별 지표 ──────────────────────────────────────────────────────
def smooth(y, w=10):
    return pd.Series(y).rolling(w, min_periods=3, center=True).mean().to_numpy()

def roll30(t, y):
    k = ~np.isnan(y)
    if k.sum() < 3: return np.full(len(t), np.nan)
    sr = pd.Series(y[k], index=pd.to_timedelta(t[k], unit='s'))
    r = sr.rolling('30s', min_periods=3).mean().to_numpy()
    out = np.full(len(t), np.nan); out[k] = r
    return out

flow = dict(tests=int(s.shape[0]))
info = s.set_index('ID_test')
rows = []
for tid, g in m.groupby('ID_test'):
    g = g[g.HR.between(30, 230) | g.HR.isna()]
    t = (g.time - g.time.min()).to_numpy(float)
    hr, sp = g.HR.to_numpy(float), g.Speed.to_numpy(float)
    vo2, vco2 = g.VO2.to_numpy(float), g.VCO2.to_numpy(float)
    r = dict(id_test=tid, subj=int(info.at[tid, 'ID']), age=float(info.at[tid, 'Age']),
             sex=int(info.at[tid, 'Sex']), weight=float(info.at[tid, 'Weight']),
             k=int(str(tid).split('_')[1]))
    if np.all(np.isnan(hr)) or np.all(np.isnan(sp)):
        r['fail'] = 'no_signal'; rows.append(r); continue
    hrs = smooth(hr)
    r['hr_peak'] = float(np.nanmax(hrs))
    t0 = {'A': t[int(np.nanargmax(hrs))]}
    ip = int(np.nanargmax(sp)); aft = np.where((t > t[ip]) & (sp < np.nanmax(sp) - 2.0))[0]
    if len(aft) == 0:
        r['fail'] = 'no_onset_B'; rows.append(r); continue
    t0['B'] = t[aft[0]]
    r['t0_A'], r['t0_B'] = float(t0['A']), float(t0['B'])
    r['rec_dur'] = float(t.max() - max(t0.values()))
    if r['rec_dur'] < 60:
        r['fail'] = 'short_recovery'; rows.append(r); continue
    trB = t - t0['B']; kB = (trB >= 0) & (trB <= 180)
    r['rec_speed'], r['rec_speed_sd'] = float(np.nanmean(sp[kB])), float(np.nanstd(sp[kB]))
    for d in DEFS:
        rec = t >= t0[d]; tr = t[rec] - t0[d]; hs = smooth(hr[rec])
        for w in WINDOWS:
            kk = (np.abs(tr - w) <= 5) & ~np.isnan(hs)
            r[f'{d}{w}'] = r['hr_peak'] - float(np.mean(hs[kk])) if (kk.sum() >= 2 and r['rec_dur'] >= w) else np.nan
    v30 = roll30(t, vo2)
    r['vo2peak_kg'] = float(np.nanmax(v30)) / r['weight'] if np.any(~np.isnan(v30)) else np.nan
    ex = t <= t0['B']
    c30, o30 = roll30(t[ex], vco2[ex]), roll30(t[ex], vo2[ex])
    with np.errstate(invalid='ignore', divide='ignore'):
        rer = c30 / o30
    r['rer_peak'] = float(np.nanmax(rer)) if np.any(np.isfinite(rer)) else np.nan
    r['fail'] = ''
    rows.append(r)

T = pd.DataFrame(rows)
# 사후 제외(사용자 결정 2026-10-13, OQ-09 · 심사 r2 A2-02/D2-05): 속도 채널이 실제 부하를 반영하지 않은 기록 —
# 5 km/h 구간(700–1,200 s)에 HR 173 · VO₂ 2,777 mL/min, t₀ B가 잘못 잡힘. 원고에 사전지정과의 차이로 공개한다.
EXCLUDE = {'218_1': 'speed_channel_anomaly'}
T.loc[T.id_test.isin(list(EXCLUDE)), 'fail'] = T.id_test.map(EXCLUDE)
T['adult'] = T.age >= 18
T['std_load'] = T.rec_speed.between(4.5, 5.5) & (T.rec_speed_sd < 0.5)
T['eligible'] = T.adult & (T.fail == '') & (T.rec_dur >= 180) & T.std_load & T[CELLS + ['vo2peak_kg']].notna().all(1)
flow.update(adults=int(T.adult.sum()),
            adult_fail=T[T.adult].fail.replace('', 'ok').value_counts().to_dict(),
            adult_rec_lt180=int((T.adult & (T.fail == '') & (T.rec_dur < 180)).sum()),
            adult_ok_not_std=int((T.adult & (T.fail == '') & ~T.std_load).sum()),
            eligible_tests=int(T.eligible.sum()), eligible_subjects=int(T[T.eligible].subj.nunique()))
T.to_csv(f'{OUT}/test_level.csv', index=False)

# 2026-10-12 독립 검증: 첫 실행은 회복 ≥185 s로 잘라 사전지정(≥180 s)과 달랐다 → 사전지정대로 고침(기록 전, 같은 회차)
# ── 2. 반복 쌍 · 적합 표본 ──────────────────────────────────────────────
E = T[T.eligible].sort_values(['subj', 'age', 'k'])
cnt = E.groupby('subj').id_test.transform('count')
P = E[cnt >= 2].groupby('subj').head(2).copy()
P['visit'] = P.groupby('subj').cumcount() + 1
F = E[cnt == 1].copy()                       # 예측식 적합 표본: 반복 쌍에 들지 않은 참가자
gap = P.pivot(index='subj', columns='visit', values='age')
P['gap_y'] = P.subj.map(gap[2] - gap[1])
flow.update(pairs=int(P.subj.nunique()), fit_n=int(len(F)),
            gap_median_y=float((gap[2] - gap[1]).median()))
assert not set(F.subj) & set(P.subj)

def fit_predict(fit, app, y, covars):
    X = lambda d: np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in covars])
    b, *_ = np.linalg.lstsq(X(fit), fit[y].to_numpy(float), rcond=None)
    pf = X(fit) @ b; yf = fit[y].to_numpy(float)
    r2 = 1 - ((yf - pf) ** 2).sum() / ((yf - yf.mean()) ** 2).sum()
    return app[y].to_numpy(float) - X(app) @ b, r2, b

COV = ['vo2peak_kg', 'age', 'sex', 'hr_peak']

def build(P, F, covars, cells=CELLS):
    """참가자 × (visit1, visit2) 행렬들: 잔차 + 원 HRR + VO2peak/kg."""
    r2s = {}
    for c in cells:
        P[f'res_{c}'], r2s[c], _ = fit_predict(F, P, c, covars)
    keys = [f'res_{c}' for c in cells] + list(cells) + ['vo2peak_kg']
    W = P.pivot(index='subj', columns='visit', values=keys)
    mats = {k: W[k][[1, 2]].to_numpy(float) for k in keys}
    return mats, r2s

def boot_all(mats, nb=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = next(iter(mats.values())).shape[0]
    est = {k: icc21(v) for k, v in mats.items()}
    B = {k: np.empty(nb) for k in mats}
    for b in range(nb):
        i = rng.integers(0, n, n)
        for k, v in mats.items(): B[k][b] = icc21(v[i])
    return est, B

def ci(x, a):
    return np.nanpercentile(x, 100 * a / 2), np.nanpercentile(x, 100 * (1 - a / 2))

def icc_table(mats, label, r2s=None, alpha_b=ALPHA_BONF):
    est, B = boot_all(mats)
    out = []
    for k in mats:
        lo, hi = ci(B[k], 0.05); blo, bhi = ci(B[k], alpha_b)
        prim = k.startswith('res_')
        flo, fhi = icc21_ci_F(mats[k], alpha_b) if prim else (np.nan, np.nan)
        out.append(dict(analysis=label, key=k, n_pairs=int((~np.isnan(mats[k]).any(1)).sum()),
                        ICC=est[k], CI95_lo=lo, CI95_hi=hi,
                        CIbonf_lo=blo if prim else np.nan, CIbonf_hi=bhi if prim else np.nan,
                        posthoc_F_CIbonf_lo=flo, posthoc_F_CIbonf_hi=fhi,
                        posthoc_F_verdict=('특성 있음' if flo > THRESH else '말할 수 없음') if prim else '',
                        r2_fit=(r2s or {}).get(k.replace('res_', ''), np.nan) if prim else np.nan,
                        verdict=('특성 있음' if blo > THRESH else '말할 수 없음') if prim else ''))
    return pd.DataFrame(out), est, B

# ── 3. 일차 family + 이차 E1·E2·E4 ──────────────────────────────────────
mats, r2s = build(P, F, COV)
prim, est, B = icc_table(mats, 'main', r2s)
prim.to_csv(f'{OUT}/primary_family.csv', index=False)

diffs = []
def add_diff(a, b, lab):
    d = B[a] - B[b]; lo, hi = ci(d, 0.05)
    diffs.append(dict(contrast=lab, a=a, b=b, dICC=est[a] - est[b], CI95_lo=lo, CI95_hi=hi))
for w in WINDOWS: add_diff(f'res_B{w}', f'res_A{w}', f'B−A @{w}s')
for d in DEFS:
    add_diff(f'res_{d}120', f'res_{d}60', f'{d}: 120−60')
    add_diff(f'res_{d}180', f'res_{d}60', f'{d}: 180−60')
    add_diff(f'res_{d}180', f'res_{d}120', f'{d}: 180−120')
pd.DataFrame(diffs).to_csv(f'{OUT}/secondary_icc_differences.csv', index=False)

# E3 일치도 (적격 성인 검사 전체)
ag = [dict(metric='t0 (s)', n=len(E), mean=(E.t0_B - E.t0_A).mean(), sd=(E.t0_B - E.t0_A).std(),
           median=(E.t0_B - E.t0_A).median(), q1=(E.t0_B - E.t0_A).quantile(.25), q3=(E.t0_B - E.t0_A).quantile(.75))]
for w in WINDOWS:
    d = E[f'B{w}'] - E[f'A{w}']
    ag.append(dict(metric=f'HRR {w}s (bpm)', n=int(d.notna().sum()), mean=d.mean(), sd=d.std(),
                   median=d.median(), q1=d.quantile(.25), q3=d.quantile(.75)))
AG = pd.DataFrame(ag); AG['loa_lo'] = AG['mean'] - 1.96 * AG.sd; AG['loa_hi'] = AG['mean'] + 1.96 * AG.sd
AG.to_csv(f'{OUT}/secondary_agreement_AB.csv', index=False)

# ── 4. 민감도 ───────────────────────────────────────────────────────────
sens = []
okP = P.groupby('subj').rer_peak.transform(lambda x: (x >= 1.10).all())
P1, F1 = P[okP].copy(), F[F.rer_peak >= 1.10]
m1, r1 = build(P1, F1, COV); sens.append(icc_table({k: v for k, v in m1.items() if k.startswith('res_')}, 'S1 RER≥1.10', r1)[0])
P2 = P.copy(); m2, r2 = build(P2, F, ['vo2peak_kg', 'age', 'sex'])
sens.append(icc_table({k: v for k, v in m2.items() if k.startswith('res_')}, 'S2 HR최고점 제외', r2)[0])
for lab, sel in (('S3 간격<0.5y', P.gap_y < 0.5), ('S3 간격≥0.5y', P.gap_y >= 0.5)):
    P3 = P[sel].copy(); m3, r3 = build(P3, F, COV)
    sens.append(icc_table({k: v for k, v in m3.items() if k.startswith('res_')}, lab, r3)[0])
pd.concat(sens).to_csv(f'{OUT}/sensitivity.csv', index=False)

# ── 4b. [사후 탐색 — 사전지정 아님] 회복 ≥125 s 표본, 60/120 s 4조합 ────────────
# 이유: 사전지정 ≥185 s 조건에서 성인 검사의 상당수가 빠져 반복 쌍이 줄었다(flow 참조). 판정에 쓰지 않는다.
C4 = [f'{d}{w}' for d in DEFS for w in (60, 120)]
T['eligible125'] = T.adult & (T.fail == '') & (T.rec_dur >= 125) & T.std_load & T[C4 + ['vo2peak_kg']].notna().all(1)
E4 = T[T.eligible125].sort_values(['subj', 'age', 'k'])
c4 = E4.groupby('subj').id_test.transform('count')
P4 = E4[c4 >= 2].groupby('subj').head(2).copy(); P4['visit'] = P4.groupby('subj').cumcount() + 1
F4 = E4[c4 == 1].copy()
m4, r4 = build(P4, F4, COV, C4)
ph = icc_table(m4, 'posthoc rec≥125s', r4, 0.05 / len(C4))[0]
ph.to_csv(f'{OUT}/posthoc_rec125.csv', index=False)
flow.update(posthoc_eligible_tests=int(len(E4)), posthoc_pairs=int(P4.subj.nunique()), posthoc_fit_n=int(len(F4)))

# ── 5. 기술 통계 · 그림용 ───────────────────────────────────────────────
desc = []
for lab, d in (('반복 쌍 visit1', P[P.visit == 1]), ('반복 쌍 visit2', P[P.visit == 2]), ('적합 표본', F), ('적격 전체', E)):
    desc.append(dict(group=lab, n=len(d), age_mean=d.age.mean(), age_sd=d.age.std(), female_pct=d.sex.mean() * 100,
                     weight_mean=d.weight.mean(), vo2kg_mean=d.vo2peak_kg.mean(), vo2kg_sd=d.vo2peak_kg.std(),
                     hrpeak_mean=d.hr_peak.mean(), hrpeak_sd=d.hr_peak.std(), rer_ge110_pct=(d.rer_peak >= 1.10).mean() * 100,
                     **{f'{c}_mean': d[c].mean() for c in CELLS}))
pd.DataFrame(desc).to_csv(f'{OUT}/descriptives.csv', index=False)
P.to_csv(f'{OUT}/pairs_level.csv', index=False)

# 예시 곡선: 반복 쌍 visit1 중 t0 차이가 중앙값에 가장 가까운 검사
dt = (P[P.visit == 1].t0_B - P[P.visit == 1].t0_A)
ex_id = P[P.visit == 1].loc[(dt - dt.median()).abs().idxmin(), 'id_test']
g = m[m.ID_test == ex_id]; g = g[g.HR.between(30, 230)]
pd.DataFrame(dict(t=g.time - g.time.min(), speed=g.Speed, hr=g.HR, hr_smooth=smooth(g.HR.to_numpy(float)))) \
  .assign(id_test=ex_id).to_csv(f'{OUT}/example_curve.csv', index=False)

json.dump(dict(flow=flow, example_test=str(ex_id)), open(f'{OUT}/sample_flow.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1, default=str)
json.dump(dict(run=datetime.datetime.now().isoformat(timespec='seconds'), python=platform.python_version(),
               numpy=np.__version__, pandas=pd.__version__, seed=SEED, nboot=NBOOT, raw_dir=RAW, sha256=SHA,
               prespec='03_RESEARCH/projects/hr_recovery_resilience/analysis.md 사전지정 2026-10-12 21:30 KST'),
          open(f'{OUT}/run_info.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

pd.set_option('display.width', 200)
print(json.dumps(flow, ensure_ascii=False, default=str))
print(prim.round(3).to_string(index=False))
print(pd.DataFrame(diffs).round(3).to_string(index=False))
print(AG.round(2).to_string(index=False))
print(pd.concat(sens)[['analysis', 'key', 'n_pairs', 'ICC', 'CIbonf_lo', 'verdict', 'posthoc_F_CIbonf_lo', 'posthoc_F_verdict']].round(3).to_string(index=False))
print(ph[['key', 'n_pairs', 'ICC', 'CI95_lo', 'CIbonf_lo', 'verdict', 'posthoc_F_CIbonf_lo', 'posthoc_F_verdict', 'r2_fit']].round(3).to_string(index=False))
