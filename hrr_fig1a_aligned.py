# -*- coding: utf-8 -*-
"""그림 1a 자료(2026-10-13, 사용자: 예시 1명 대신 분석한 모든 검사를 보이게) — 기술 통계, 새 분석 아님.
적격 검사 596개의 평활 HR(hrr_resilience.py와 같은 10-호흡 중심 이동평균)을 부하 종료(t₀ B) 기준 시간 −180…+195 s,
1 s 격자에 선형 보간해 저장한다. 최고 HR 재현을 assert로 확인.
입력: results_hrr_resilience_v1/test_level.csv · 원자료 test_measure.csv  출력: results_hrr_resilience_v1_extra/F1a_hr_aligned_endofload.csv"""
import os
import numpy as np, pandas as pd

H = os.path.dirname(os.path.abspath(__file__)); V1 = os.path.join(H, '..', 'results_hrr_resilience_v1')
OUT = os.path.join(H, '..', 'results_hrr_resilience_v1_extra')
RAW = 'D:/treadmil/raw' if os.path.isdir('D:/treadmil/raw') else os.path.expanduser('~/mnt/treadmil/raw')
G = np.arange(-180, 196)

def smooth(y, w=10): return pd.Series(y).rolling(w, min_periods=3, center=True).mean().to_numpy()  # hrr_resilience.py와 같다

T = pd.read_csv(f'{V1}/test_level.csv'); E = T[T.eligible].set_index('id_test')
m = pd.read_csv(f'{RAW}/test_measure.csv'); m = m[m.ID_test.isin(E.index)]
m = m[m.HR.between(30, 230) | m.HR.isna()]
rows = {}
for tid, g in m.groupby('ID_test'):
    t = (g.time - g.time.min()).to_numpy(float); hs = smooth(g.HR.to_numpy(float)); r = E.loc[tid]
    assert abs(np.nanmax(hs) - r.hr_peak) < 1e-6, tid
    ok = ~np.isnan(hs); rows[tid] = np.interp(r.t0_B + G, t[ok], hs[ok], left=np.nan, right=np.nan)
F = pd.DataFrame.from_dict(rows, orient='index', columns=[str(s) for s in G]).round(2); F.index.name = 'id_test'
assert len(F) == len(E) == 596
F.to_csv(f'{OUT}/F1a_hr_aligned_endofload.csv')
print(F.shape, 'median at 0 s', F['0'].median(), 'n at 180 s', F['180'].notna().sum())
