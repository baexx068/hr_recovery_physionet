# -*- coding: utf-8 -*-
"""
hr_recovery_resilience v2 — 개인 기대 회복 범위(A, 일차) · 운동 중 신호 예측(C, 이차)
사전지정: 03_RESEARCH/projects/hr_recovery_resilience/analysis.md '사전지정 v2' (승인 1 사용자 2026-10-13, C → 이차)
계획:     05_RUNS/hr_recovery_resilience/2026-10-13_v2_plan.md

실행 순서 (호출당 시간 제한 때문에 나눈다)
  python hrr_predict_v2.py features            # 특성 표 → features.csv
  python hrr_predict_v2.py run A60 ... B180     # 조합별 OOF 예측 → oof_<cell>.csv
  python hrr_predict_v2.py ext A60 A120 B60 B120  # S3 회복 ≥125 s 확장 표본
  python hrr_predict_v2.py combine             # 일차·이차·민감도 표
"""
import os, sys, json, hashlib, platform, datetime, warnings
import numpy as np, pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(HERE, '..', 'results_hrr_resilience_v1')
OUT = os.path.join(HERE, '..', 'results_hrr_resilience_v2'); os.makedirs(OUT, exist_ok=True)
RAW = 'D:/treadmil/raw' if os.path.isdir('D:/treadmil/raw') else os.path.expanduser('~/mnt/treadmil/raw')
CELLS = [f'{d}{w}' for d in 'AB' for w in (60, 120, 180)]
A_SET = ['vo2peak_kg', 'age', 'sex', 'hr_peak']
C_ADD = ['weight', 'hr_warmup', 'hr_at10', 'rer_peak', 've_peak', 'ex_dur']
C_SET = A_SET + C_ADD
ENV = ['temperature', 'humidity']
DEPTHS, LR, NITER = (1, 2, 3), 0.05, 200
REPEATS, K, ALPHA, NB = (42, 43, 44, 45, 46), 5, 0.10, 2000

# ── 엔진 검증 · 입력 무결성 ─────────────────────────────────────────────
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
SUMS = dict(l.split()[::-1] for l in open(f'{V1}/SHA256SUMS.txt'))
assert sha(f'{V1}/test_level.csv') == SUMS['test_level.csv'], 'v1 test_level.csv SHA-256 불일치'
assert sha(f'{RAW}/test_measure.csv') == '3b5ed5586dc7412ccdfb26f804e29c06482a2a1df2060399c2731ebc68d4613d'
assert sha(f'{RAW}/subject-info.csv') == 'e9ca5ba20eb1272df56c11d3e584ec3997d5484454812bd94966e58082415f16'
_r = np.random.default_rng(0); _X = _r.normal(size=(200, 3)); _y = _X @ [1, -2, .5] + 3 + _r.normal(size=200)
_b = np.linalg.lstsq(np.column_stack([np.ones(200), _X]), _y, rcond=None)[0]
_m = LinearRegression().fit(_X, _y)
assert np.allclose(_b, np.r_[_m.intercept_, _m.coef_]), '선형회귀 엔진 불일치'

def cq(res, a=ALPHA):
    """유한표본 conformal 분위수 ⌈(n+1)(1−a)⌉/n (① code_jsams/re3.py와 같은 정의)."""
    r = np.sort(np.abs(res)); n = len(r); k = int(np.ceil((n + 1) * (1 - a))) - 1
    return r[min(k, n - 1)]

_cov = []
for s in range(200):  # 모의자료 피복 점검 — 정규오차, 선형 참모형
    g = np.random.default_rng(1000 + s); Xs = g.normal(size=(600, 2)); ys = Xs @ [1, 1] + g.normal(size=600)
    tr, te = slice(0, 400), slice(400, 600)
    mm = LinearRegression().fit(Xs[:200], ys[:200]); q = cq(ys[200:400] - mm.predict(Xs[200:400]))
    _cov.append(np.mean(np.abs(ys[te] - mm.predict(Xs[te])) <= q))
assert 0.88 <= np.mean(_cov) <= 0.92, f'conformal 피복 엔진 점검 실패 {np.mean(_cov):.3f}'

# ── 특성 ────────────────────────────────────────────────────────────────
def build_features():
    T = pd.read_csv(f'{V1}/test_level.csv'); ok = T.fail.isna()
    E = T[T.adult & ok & (T.rec_dur >= 125) & T.std_load & T.vo2peak_kg.notna()].copy()
    s = pd.read_csv(f'{RAW}/subject-info.csv').set_index('ID_test')
    m = pd.read_csv(f'{RAW}/test_measure.csv'); m = m[m.ID_test.isin(E.id_test)]
    m = m[m.HR.between(30, 230) | m.HR.isna()]
    rows = []
    for tid, g in m.groupby('ID_test'):
        r = E.set_index('id_test').loc[tid]
        t = (g.time - g.time.min()).to_numpy(float); sp = g.Speed.to_numpy(float); hr = g.HR.to_numpy(float); ve = g.VE.to_numpy(float)
        ex = t <= r.t0_B; asc = t <= t[np.nanargmax(sp)]
        k10 = asc & (np.abs(sp - 10) <= .5) & ~np.isnan(hr); kw = (t < 60) & ~np.isnan(hr)
        kv = ex & ~np.isnan(ve)
        vroll = pd.Series(ve[kv], index=pd.to_timedelta(t[kv], unit='s')).rolling('30s', min_periods=3).mean().max() if kv.sum() >= 3 else np.nan
        rows.append(dict(id_test=tid, hr_warmup=hr[kw].mean() if kw.sum() >= 3 else np.nan,
                         hr_at10=hr[k10].mean() if k10.sum() >= 3 else np.nan, ve_peak=vroll, ex_dur=r.t0_B,
                         temperature=s.at[tid, 'Temperature'], humidity=s.at[tid, 'Humidity']))
    F = E.merge(pd.DataFrame(rows), on='id_test', how='left')
    F['main'] = F.eligible.astype(bool)                       # v1 주 분석 표본(회복 ≥180 s)
    F.to_csv(f'{OUT}/features.csv', index=False)
    print('features', len(F), 'main', int(F.main.sum()), F[C_ADD + ENV].isna().mean().round(3).to_dict())

# ── 모형 ────────────────────────────────────────────────────────────────
def impute(Xtr, Xte):
    med = np.nanmedian(Xtr, axis=0)
    f = lambda X: np.where(np.isnan(X), med, X)
    return f(Xtr), f(Xte)

def fit_pred(kind, Xtr, ytr, Xte, depth=None, seed=0):
    Xtr, Xte = impute(Xtr, Xte)
    if kind == 'lin': mdl = LinearRegression()
    else: mdl = HistGradientBoostingRegressor(max_depth=depth, learning_rate=LR, max_iter=NITER, random_state=seed)
    return mdl.fit(Xtr, ytr).predict(Xte)

def inner(kind, X, y, g, seed):
    """학습 fold 안 내부 CV: 깊이 선택(1-SE, 가장 얕은 것) + 선택 설정의 내부 OOF 잔차(conformal용)."""
    cfgs = [None] if kind == 'lin' else list(DEPTHS)
    gk = GroupKFold(n_splits=K, shuffle=True, random_state=seed + 100)
    oof = {c: np.full(len(y), np.nan) for c in cfgs}; fold_mae = {c: [] for c in cfgs}
    for tr, te in gk.split(X, y, g):
        for c in cfgs:
            p = fit_pred(kind, X[tr], y[tr], X[te], c, seed); oof[c][te] = p
            fold_mae[c].append(np.mean(np.abs(y[te] - p)))
    mean = {c: np.mean(v) for c, v in fold_mae.items()}; se = {c: np.std(v, ddof=1) / np.sqrt(K) for c, v in fold_mae.items()}
    best = min(cfgs, key=lambda c: mean[c]) if kind != 'lin' else None
    pick = None if kind == 'lin' else min(c for c in cfgs if mean[c] <= mean[best] + se[best])
    return pick, cq(y - oof[pick])

def oof_run(D, y_col, feats, kind, label, perm_cols=None):
    out = []
    X0 = D[feats].to_numpy(float); y = D[y_col].to_numpy(float); g = D.subj.to_numpy()
    for seed in REPEATS:
        X = X0.copy()
        if perm_cols:  # 순열 대조군: 해당 열들을 행 단위로 섞는다(같은 순서로 함께)
            idx = np.random.default_rng(seed + 7).permutation(len(X))
            cols = [feats.index(c) for c in perm_cols]; X[:, cols] = X[idx][:, cols]
        gk = GroupKFold(n_splits=K, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(gk.split(X, y, g)):
            pick, q = inner(kind, X[tr], y[tr], g[tr], seed)
            p = fit_pred(kind, X[tr], y[tr], X[te], pick, seed)
            out.append(pd.DataFrame(dict(id_test=D.id_test.to_numpy()[te], subj=g[te], y=y[te], pred=p, q=q,
                                         repeat=seed, fold=fold, model=label, depth=pick if pick else 0)))
    return pd.concat(out)

def run_cell(cell, ext=False):
    F = pd.read_csv(f'{OUT}/features.csv')
    if ext:
        D = F[F[[cell] + A_SET].notna().all(1)].copy()
        res = [oof_run(D, cell, A_SET, 'lin', 'S3_A_lin'), oof_run(D, cell, A_SET, 'gbm', 'S3_A_gbm')]
        pd.concat(res).to_csv(f'{OUT}/oof_ext_{cell}.csv', index=False); return
    D = F[F.main].copy(); assert len(D) == 597 and D[cell].notna().all()
    S1 = D[D.rer_peak >= 1.10].copy()
    res = [oof_run(D, cell, A_SET, 'lin', 'A_lin'), oof_run(D, cell, A_SET, 'gbm', 'A_gbm'),
           oof_run(D, cell, C_SET, 'lin', 'C_lin'), oof_run(D, cell, C_SET, 'gbm', 'C_gbm'),
           oof_run(D, cell, C_SET, 'lin', 'C_perm_lin', perm_cols=C_ADD),
           oof_run(D, cell, C_SET + ENV, 'lin', 'S2_Cenv_lin'),
           oof_run(S1, cell, A_SET, 'lin', 'S1_A_lin'), oof_run(S1, cell, A_SET, 'gbm', 'S1_A_gbm')]
    pd.concat(res).assign(cell=cell).to_csv(f'{OUT}/oof_{cell}.csv', index=False)

# ── 종합 ────────────────────────────────────────────────────────────────
def icc21(Y):
    Y = Y[~np.isnan(Y).any(1)]; n, k = Y.shape; gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1); msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)

def wkappa(a, b, cats=(-1, 0, 1)):
    """선형 가중 κ (3범주)."""
    n = len(a); kk = len(cats); O = np.zeros((kk, kk))
    for x, y_ in zip(a, b): O[cats.index(x), cats.index(y_)] += 1
    W = 1 - np.abs(np.subtract.outer(range(kk), range(kk))) / (kk - 1)
    Eo = np.outer(O.sum(1), O.sum(0)) / n
    return ((W * O).sum() - (W * Eo).sum()) / (n - (W * Eo).sum())
assert abs(wkappa([1, 0, -1, 1], [1, 0, -1, 1]) - 1) < 1e-12

def per_test(O, model):
    """검사별: 5회 반복 평균 예측·구간·절대오차."""
    d = O[O.model == model]
    return d.groupby(['id_test', 'subj']).agg(y=('y', 'first'), pred=('pred', 'mean'), q=('q', 'mean'),
                                              ae=('pred', lambda p: np.nan)).reset_index().assign(
        ae=lambda x: d.assign(ae=np.abs(d.y - d.pred)).groupby('id_test').ae.mean().reindex(x.id_test).to_numpy())

def boot_diff(Pa, Pb, rng):
    """ΔMAE(a − b), 참가자 군집 부트스트랩. 반환: 점추정, 95% CI, 양측 P."""
    d = Pa[['id_test', 'subj', 'ae']].merge(Pb[['id_test', 'ae']], on='id_test', suffixes=('_a', '_b'))
    d['dd'] = d.ae_a - d.ae_b
    G = d.groupby('subj').dd.agg(['sum', 'size']); s, n = G['sum'].to_numpy(), G['size'].to_numpy()
    est = s.sum() / n.sum(); B = np.empty(NB)
    for b in range(NB):
        i = rng.integers(0, len(s), len(s)); B[b] = s[i].sum() / n[i].sum()
    p = min(1.0, 2 * min((B >= 0).mean(), (B <= 0).mean())); p = max(p, 1 / NB)
    return est, *np.percentile(B, [2.5, 97.5]), p

def perf(O, model):
    d = O[O.model == model]; rows = []
    for rpt, g in d.groupby('repeat'):
        e = g.y - g.pred
        rows.append(dict(mae=np.mean(np.abs(e)), r2=1 - (e ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum(),
                         slope=np.polyfit(g.pred, g.y, 1)[0], sd_pred=g.pred.std(), sd_obs=g.y.std(),
                         cover=np.mean(np.abs(e) <= g.q), width=2 * g.q.mean()))
    R = pd.DataFrame(rows)
    out = {f'{c}_mean': R[c].mean() for c in R}; out.update({f'{c}_sd': R[c].std() for c in ('mae', 'r2')})
    return out

def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o): run = max(run, (m - i) * p[j]); adj[j] = min(1, run)
    return adj
assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

def combine():
    rng = np.random.default_rng(42); F = pd.read_csv(f'{OUT}/features.csv')
    prim, sec, perfs, rep_rows, sens = [], [], [], [], []
    v1pairs = pd.read_csv(f'{V1}/pairs_level.csv')[['subj', 'id_test', 'visit']]
    for c in CELLS:
        O = pd.read_csv(f'{OUT}/oof_{c}.csv'); P = {m: per_test(O, m) for m in O.model.unique()}
        e, lo, hi, p = boot_diff(P['A_gbm'], P['A_lin'], rng)
        prim.append(dict(contrast=f'A{CELLS.index(c) + 1}', cell=c, comparison='부스팅 − 선형 (A 세트)', dMAE=e, CI95_lo=lo, CI95_hi=hi, p=p))
        for lab, a, b in (('C: 선형 C 세트 − 선형 A 세트', 'C_lin', 'A_lin'), ('E3: 부스팅 − 선형 (C 세트)', 'C_gbm', 'C_lin'),
                          ('E5: 실제 C 세트 − 순열 C 세트 (선형)', 'C_lin', 'C_perm_lin'), ('S2: 온도·습도 추가 − C 세트 (선형)', 'S2_Cenv_lin', 'C_lin'),
                          ('S1: 부스팅 − 선형 (RER ≥ 1.10)', 'S1_A_gbm', 'S1_A_lin')):
            e2, l2, h2, p2 = boot_diff(P[a], P[b], rng)
            (sens if lab.startswith('S') else sec).append(dict(cell=c, comparison=lab, dMAE=e2, CI95_lo=l2, CI95_hi=h2, p_unadj=p2))
        for m in O.model.unique():
            perfs.append(dict(cell=c, model=m, n_tests=O[O.model == m].id_test.nunique(),
                              depth_mode=int(O[O.model == m].depth.mode()[0]), **perf(O, m)))
            sx = P[m].merge(F[['id_test', 'sex']], on='id_test')
            perfs[-1].update({f'cover_sex{k}': np.mean(np.abs(g.y - g.pred) <= g.q) for k, g in sx.groupby('sex')})
        for m in ('A_lin', 'A_gbm'):   # E4 회복 탄력성 판정 재현성 — v1과 같은 57쌍(첫 두 검사)
            d = P[m].merge(v1pairs, on=['id_test', 'subj'])
            d['side'] = np.where(d.y > d.pred + d.q, 1, np.where(d.y < d.pred - d.q, -1, 0))
            d['z'] = (d.y - d.pred) / d.q
            w = d.pivot(index='subj', columns='visit', values=['side', 'z'])
            a, b = w['side'][1].astype(int).tolist(), w['side'][2].astype(int).tolist()
            rep_rows.append(dict(cell=c, model=m, n_pairs=len(a), wkappa=wkappa(a, b), same_side=np.mean(np.array(a) == np.array(b)),
                                 outside_v1=np.mean(np.array(a) != 0), outside_v2=np.mean(np.array(b) != 0),
                                 icc_z=icc21(w['z'][[1, 2]].to_numpy(float))))
    PR = pd.DataFrame(prim); PR['p_holm'] = holm(PR.p)
    PR['verdict'] = np.where((PR.p_holm < .05) & (PR.dMAE < 0), 'AI 이득 있음', '이 자료로는 말할 수 없음')
    PR.to_csv(f'{OUT}/primary_family.csv', index=False)
    pd.DataFrame(sec).to_csv(f'{OUT}/secondary_contrasts.csv', index=False)
    pd.DataFrame(perfs).to_csv(f'{OUT}/performance.csv', index=False)
    pd.DataFrame(rep_rows).to_csv(f'{OUT}/secondary_resilience_repeatability.csv', index=False)
    for c in ('A60', 'A120', 'B60', 'B120'):   # S3
        O = pd.read_csv(f'{OUT}/oof_ext_{c}.csv'); P = {m: per_test(O, m) for m in O.model.unique()}
        e, lo, hi, p = boot_diff(P['S3_A_gbm'], P['S3_A_lin'], rng)
        sens.append(dict(cell=c, comparison='S3: 부스팅 − 선형 (회복 ≥125 s 확장)', dMAE=e, CI95_lo=lo, CI95_hi=hi, p_unadj=p,
                         n_tests=O.id_test.nunique()))
    pd.DataFrame(sens).to_csv(f'{OUT}/sensitivity.csv', index=False)
    json.dump(dict(run=datetime.datetime.now().isoformat(timespec='seconds'), python=platform.python_version(),
                   sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__, repeats=REPEATS, k=K,
                   depths=DEPTHS, lr=LR, n_iter=NITER, alpha=ALPHA, nboot=NB, conformal_engine_coverage=float(np.mean(_cov)),
                   prespec='analysis.md 사전지정 v2 (승인 1 2026-10-13)'), open(f'{OUT}/run_info.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    pd.set_option('display.width', 250)
    print(PR.round(3).to_string(index=False))

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'features': build_features()
    elif mode == 'run': [run_cell(c) for c in sys.argv[2:]]
    elif mode == 'ext': [run_cell(c, ext=True) for c in sys.argv[2:]]
    elif mode == 'combine': combine()
