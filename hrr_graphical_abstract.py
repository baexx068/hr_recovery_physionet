# -*- coding: utf-8 -*-
"""EJAP 그래픽 초록 v4 (사용자: 글로 된 부분을 그림으로) ← v3 (2026-10-13, 사용자: 아이콘판이 장난스러움 → 학술적 개념도). 지침: 한 패널, 1200 × 1200 px · 300 dpi,
Arial 12–16 pt, 원고 그림과 다를 것, 통계 세부 넣지 않음. 위 곡선은 개념도(실제 기록 아님). 아래 재검사 도식은 사후 분석
(최고 HR과 부하 종료의 간격 Δt가 한 사람 안에서 반복되지 않음, Spearman ρ 0.03)을 옮긴 것. 수치는 primary_family.csv(120 s 잔차 ICC).
출력: submission_ejap_hrr/Graphical_abstract.tif(RGB) · .png"""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from PIL import Image

H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, '..', 'results_hrr_resilience_v1')
OUT = os.path.join(H, '..', 'submission_ejap_hrr'); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 12, 'axes.linewidth': 1})  # Arial과 글자 폭 동일
CA, CB, INK, MUTE, BAND = '#B54D00', '#0072B2', '#222222', '#595959', '#EDEDED'
P = pd.read_csv(f'{R}/primary_family.csv').set_index('key')
A, B = P.loc['res_A120'], P.loc['res_B120']
assert A.CIbonf_lo <= 0.40 < B.CIbonf_lo           # 일차 판정과 같은 그림인지 확인

def hr(t, p):  # 개념 곡선: 최고 HR 시각 p(s, 부하 종료 = 0)
    return PchipInterpolator([-130, p - 45, p, 8, 40, 80, 150, 200], [150, 174, 186, 183, 168, 152, 136, 128])(t)

fig = plt.figure(figsize=(4, 4), dpi=300)
fig.text(.5, .962, 'Where does heart rate recovery begin?', ha='center', va='center', fontsize=14, fontweight='bold', color=INK)

# 1. 개념도: 운동 · 회복 구간, 두 시작점, 120 s 창, 두 시점의 간격 Δt
pA = -40; ax = fig.add_axes([.1, .615, .86, .305]); t = np.linspace(-130, 190, 400)
ax.axvspan(-130, 0, color=BAND, lw=0)
ax.fill_between(t, 90, hr(t, pA), where=(t >= 0) & (t <= 120), color=CB, alpha=.15, lw=0)
ax.plot(t, hr(t, pA), color=INK, lw=2)
ax.plot([pA, pA], [90, 207], color=CA, lw=1.4, ls='--'); ax.plot(pA, 186, '^', color=CA, ms=7, zorder=3)
ax.plot([0, 0], [90, 207], color=CB, lw=1.4); ax.plot(0, hr(0, pA), 'o', color=CB, ms=6.5, zorder=3)
ax.text(pA - 4, 214, 'A  peak HR', ha='right', va='center', fontsize=12, color=CA, fontweight='bold')
ax.text(4, 214, 'B  end of load', ha='left', va='center', fontsize=12, color=CB, fontweight='bold')
ax.annotate('', xy=(0, 198), xytext=(pA, 198), arrowprops=dict(arrowstyle='<->', color=INK, lw=1, shrinkA=0, shrinkB=0))
ax.text(pA / 2, 201, 'Δt', ha='center', va='bottom', fontsize=12, color=INK)
for y0, x0, col, ls in ((136, pA, CA, '--'), (118, 0, CB, '-')):   # 120 s 창(괄호)
    ax.plot([x0, x0, x0 + 120, x0 + 120], [y0 + 4, y0, y0, y0 + 4], color=col, lw=1.4, ls=ls)
ax.text(60, 127, '120 s', ha='center', va='center', fontsize=12, color=MUTE)
ax.plot(t, np.where(t < 0, 96 + 4 * np.floor((t + 130) / 33), 94), color='#8C8C8C', lw=1.6, drawstyle='steps-post')
ax.text(-126, 112, 'Load', ha='left', va='bottom', fontsize=12, color=MUTE)
ax.set_xlim(-130, 190); ax.set_ylim(90, 222); ax.set_xticks([]); ax.set_yticks([])
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.set_ylabel('Heart rate', fontsize=12, color=INK, labelpad=2); ax.set_xlabel('Time', fontsize=12, color=INK, labelpad=2, loc='right')

# 2. 같은 사람의 두 검사: Δt가 검사마다 달라지면 A의 창은 다른 구간을 잰다
fig.text(.285, .548, 'Test 1', ha='center', va='center', fontsize=12, color=INK, fontweight='bold')
fig.text(.525, .548, 'Test 2', ha='center', va='center', fontsize=12, color=INK, fontweight='bold')
for yc, d, col, t0s, ok, lab, mk in ((.478, B, CB, (0, 0), True, 'B', 'o'), (.364, A, CA, (-40, -5), False, 'A', '^')):
    fig.text(.09, yc, lab, ha='center', va='center', fontsize=14, color=col, fontweight='bold')
    for x0, p, t0 in ((.17, -40, t0s[0]), (.41, -5, t0s[1])):
        m = fig.add_axes([x0, yc - .045, .23, .09]); tt = np.linspace(-130, 190, 300); yy = hr(tt, p)
        m.axvspan(-130, 0, color=BAND, lw=0)
        m.fill_between(tt, 120, yy, where=(tt >= t0) & (tt <= t0 + 120), color=col, alpha=.3, lw=0)
        m.plot(tt, yy, color=INK, lw=1.3); m.plot([t0, t0], [120, 196], color=col, lw=1.3, ls='-' if ok else '--')
        m.set_xlim(-130, 190); m.set_ylim(120, 196); m.set_xticks([]); m.set_yticks([])
        for s in ('top', 'right', 'left'): m.spines[s].set_visible(False)
fp = fig.add_axes([.69, .307, .27, .228])
for yy, d, col, mk in ((1, B, CB, 'o'), (0, A, CA, '^')):
    ok = d.CIbonf_lo > 0.40
    fp.plot([d.CIbonf_lo, d.CIbonf_hi], [yy, yy], color=col, lw=2, solid_capstyle='butt')
    fp.plot(d.ICC, yy, mk, ms=8, color=col if ok else 'white', mec=col, mew=1.6, zorder=3)
fp.axvline(.40, color=INK, ls=':', lw=1.2)
fp.set_xlim(0, 1); fp.set_ylim(-.5, 1.5); fp.set_yticks([]); fp.set_xticks([0, .4, 1], ['0', '0.4', '1'])
fp.tick_params(axis='x', labelsize=12, length=3)
for sp in ('top', 'right', 'left'): fp.spines[sp].set_visible(False)
fp.set_xlabel('Adjusted ICC\n(120 s)', fontsize=12, color=INK, labelpad=2)
fig.text(.5, .127, 'Recovery timed from the end of the load,', ha='center', va='center', fontsize=12, fontweight='bold', color=CB)
fig.text(.5, .089, 'not from peak HR, is stable on retest', ha='center', va='center', fontsize=12, fontweight='bold', color=CB)
fig.text(.5, .032, '596 maximal treadmill tests · 57 adults retested', ha='center', va='center', fontsize=12, color=MUTE)

fig.savefig(f'{OUT}/Graphical_abstract.png', dpi=300)
fig.savefig(f'{OUT}/Graphical_abstract.tif', dpi=300, pil_kwargs={'compression': 'tiff_lzw'})
im = Image.open(f'{OUT}/Graphical_abstract.tif').convert('RGB'); im.save(f'{OUT}/Graphical_abstract.tif', compression='tiff_lzw', dpi=(300, 300))
assert im.size == (1200, 1200), im.size
print('ok', im.size, im.mode, f'A {A.ICC:.2f} B {B.ICC:.2f}')
