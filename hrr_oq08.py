# -*- coding: utf-8 -*-
"""
OQ-08 재계산 — HR최고점·VO2peak를 운동 구간(t ≤ t0 B)으로 제한
사전지정: 03_RESEARCH/projects/hr_recovery_resilience/analysis.md '사전지정 OQ-08' (2026-10-13, 실행 전)
실행: python hrr_oq08.py feat → v1 → v2 A60 … B180 → combine
"""
import os, sys, json
import numpy as np, pandas as pd
import hrr_predict_v2 as v2          # 엔진 검증(선형·conformal·Holm·κ)과 원자료·v1 SHA-256 대조가 import 때 실행된다

OUT = os.path.join(v2.HERE, '..', 'results_hrr_resilience_oq08'); os.makedirs(OUT, exist_ok=True)
V1 = v2.V1; CELLS = v2.CELLS
A_OLD = ['vo2peak_kg', 'age', 'sex', 'hr_peak']
A_NEW = ['vo2peak_ex', 'age', 'sex', 'hr_peak_ex']
C_NEW = A_NEW + v2.C_ADD
SEED, NB = 42, 2000

def smooth(y, w=10): return pd.Series(y).rolling(w, min_periods=3, center=True).mean().to_numpy()

def feat():
    F = pd.read_csv(f'{v2.OUT}/features.csv'); F = F[F.main & ~F.id_test.isin(['218_1'])].copy()  # v1과 같은 사후 제외(OQ-09, 2026-10-13)
    m = pd.read_csv(f'{v2.RAW}/test_measure.csv'); m = m[m.ID_test.isin(F.id_test)]
    m = m[m.HR.between(30, 230) | m.HR.isna()]
    rows = []
    for tid, g in m.groupby('ID_test'):
        r = F.set_index('id_test').loc[tid]
        t = (g.time - g.time.min()).to_numpy(float); hr = g.HR.to_numpy(float); vo2 = g.VO2.to_numpy(float); sp = g.Speed.to_numpy(float)
        ex = t <= r.t0_B; hs = smooth(hr)
        kv = ex & ~np.isnan(vo2)
        v30 = pd.Series(vo2[kv], index=pd.to_timedelta(t[kv], unit='s')).rolling('30s', min_periods=3).mean().max()
        after = t > r.t0_B
        rows.append(dict(id_test=tid, hr_peak_ex=np.nanmax(hs[ex]), vo2peak_ex=v30 / r.weight,
                         reaccel=bool(np.nanmax(sp[after]) >= np.nanmax(sp) - 2.0) if after.any() else False))
    F = F.merge(pd.DataFrame(rows), on='id_test')
    assert (F.hr_peak_ex <= F.hr_peak + 1e-9).all() and (F.vo2peak_ex <= F.vo2peak_kg + 1e-9).all()
    F.to_csv(f'{OUT}/features_oq08.csv', index=False)
    print('n', len(F), '| hr_peak 변화 검사', int((F.hr_peak - F.hr_peak_ex > 1e-9).sum()), '중앙', round((F.hr_peak - F.hr_peak_ex)[F.hr_peak > F.hr_peak_ex].median(), 2),
          '| vo2 변화 검사', int((F.vo2peak_kg - F.vo2peak_ex > 1e-9).sum()), '| 재가속(S-a 제외)', int(F.reaccel.sum()), list(F.id_test[F.reaccel]))

# ── v1 일차: 회복 잔차 ICC ────────────────────────────────────────────────
def resid_icc(F, covs, cells=CELLS):
    E = F.sort_values(['subj', 'age', 'k']); c = E.groupby('subj').id_test.transform('count')
    P = E[c >= 2].groupby('subj').head(2).copy(); P['visit'] = P.groupby('subj').cumcount() + 1; Fi = E[c == 1]
    X = lambda d: np.column_stack([np.ones(len(d))] + [d[v].to_numpy(float) for v in covs])
    mats = {}
    for cell in cells:
        b = np.linalg.lstsq(X(Fi), Fi[cell].to_numpy(float), rcond=None)[0]
        P['r'] = P[cell].to_numpy(float) - X(P) @ b
        mats[cell] = P.pivot(index='subj', columns='visit', values='r')[[1, 2]].to_numpy(float)
    rng = np.random.default_rng(SEED); n = len(next(iter(mats.values()))); B = {k: np.empty(NB) for k in mats}
    for i in range(NB):
        idx = rng.integers(0, n, n)
        for k, v in mats.items(): B[k][i] = v2.icc21(v[idx])
    a = 0.05 / 6
    return pd.DataFrame([dict(cell=k, n_pairs=n, n_fit=len(Fi), ICC=v2.icc21(v), CIbonf_lo=np.percentile(B[k], 100 * a / 2),
                              CIbonf_hi=np.percentile(B[k], 100 * (1 - a / 2)), verdict='특성 있음' if np.percentile(B[k], 100 * a / 2) > .40 else '말할 수 없음')
                         for k, v in mats.items()])

def run_v1():
    F = pd.read_csv(f'{OUT}/features_oq08.csv')
    old = resid_icc(F, A_OLD)
    v1p = pd.read_csv(f'{V1}/primary_family.csv').set_index('key')
    assert np.allclose(old.ICC.to_numpy(), v1p.loc[[f'res_{c}' for c in CELLS], 'ICC'].to_numpy(), atol=1e-9), 'v1 재현 실패'
    new = resid_icc(F, A_NEW); sa = resid_icc(F[~F.reaccel], A_NEW)
    R = pd.concat([old.assign(analysis='v1 원래 (재현)'), new.assign(analysis='OQ-08 운동 구간 변수'), sa.assign(analysis='OQ-08 + S-a 재가속 제외')])
    R['v1_verdict'] = R.cell.map(dict(zip(CELLS, v1p.loc[[f'res_{c}' for c in CELLS], 'verdict'])))
    R['changed'] = R.verdict != R.v1_verdict
    R.to_csv(f'{OUT}/v1_primary_recomputed.csv', index=False); print(R.round(3).to_string(index=False))

# ── v2 일차·이차 ──────────────────────────────────────────────────────────
def run_v2(cell):
    F = pd.read_csv(f'{OUT}/features_oq08.csv'); S = F[~F.reaccel]
    res = [v2.oof_run(F, cell, A_NEW, 'lin', 'A_lin'), v2.oof_run(F, cell, A_NEW, 'gbm', 'A_gbm'), v2.oof_run(F, cell, C_NEW, 'lin', 'C_lin'),
           v2.oof_run(S, cell, A_NEW, 'lin', 'Sa_A_lin'), v2.oof_run(S, cell, A_NEW, 'gbm', 'Sa_A_gbm')]
    pd.concat(res).assign(cell=cell).to_csv(f'{OUT}/oof_{cell}.csv', index=False)

def combine():
    rng = np.random.default_rng(42); v2p = pd.read_csv(f'{v2.OUT}/primary_family.csv').set_index('cell'); rows, sec = [], []
    for lab, (a, b) in (('OQ-08 운동 구간 변수', ('A_gbm', 'A_lin')), ('OQ-08 + S-a 재가속 제외', ('Sa_A_gbm', 'Sa_A_lin'))):
        part = []
        for c in CELLS:
            O = pd.read_csv(f'{OUT}/oof_{c}.csv'); P = {m: v2.per_test(O, m) for m in O.model.unique()}
            e, lo, hi, p = v2.boot_diff(P[a], P[b], rng)
            part.append(dict(analysis=lab, cell=c, dMAE=e, CI95_lo=lo, CI95_hi=hi, p=p, n_tests=len(P[a]),
                             v2_dMAE=v2p.at[c, 'dMAE'], v2_verdict=v2p.at[c, 'verdict']))
            if lab.startswith('OQ-08 운동'):
                e2, l2, h2, p2 = v2.boot_diff(P['C_lin'], P['A_lin'], rng)
                sec.append(dict(cell=c, comparison='C: 선형 C 세트 − 선형 A 세트 (운동 구간 변수)', dMAE=e2, CI95_lo=l2, CI95_hi=h2, p_unadj=p2,
                                r2_A_lin=v2.perf(O, 'A_lin')['r2_mean'], r2_C_lin=v2.perf(O, 'C_lin')['r2_mean'],
                                cover_A_lin=v2.perf(O, 'A_lin')['cover_mean']))
        D = pd.DataFrame(part); D['p_holm'] = v2.holm(D.p)
        D['verdict'] = np.where((D.p_holm < .05) & (D.dMAE < 0), 'AI 이득 있음', '이 자료로는 말할 수 없음'); rows.append(D)
    R = pd.concat(rows); R['changed'] = R.verdict != R.v2_verdict
    R.to_csv(f'{OUT}/v2_primary_recomputed.csv', index=False); pd.DataFrame(sec).to_csv(f'{OUT}/v2_secondary_recomputed.csv', index=False)
    v1r = pd.read_csv(f'{OUT}/v1_primary_recomputed.csv')
    json.dump(dict(v1_changed=v1r[v1r.analysis != 'v1 원래 (재현)'][['analysis', 'cell']][v1r.changed].values.tolist(),
                   v2_changed=R[R.changed][['analysis', 'cell']].values.tolist(),
                   prespec='analysis.md 사전지정 OQ-08 (2026-10-13)'), open(f'{OUT}/oq08_verdict.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    pd.set_option('display.width', 220); print(R.round(3).to_string(index=False)); print(pd.DataFrame(sec).round(3).to_string(index=False))

if __name__ == '__main__':
    md = sys.argv[1]
    if md == 'feat': feat()
    elif md == 'v1': run_v1()
    elif md == 'v2': [run_v2(c) for c in sys.argv[2:]]
    elif md == 'combine': combine()
