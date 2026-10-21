# -*- coding: utf-8 -*-
"""원고용 핵심 그림 — 회복 잔차 재검사 ICC, 6조합 전부 (보고 원칙: 통과·미달 같은 표기 규칙).
입력: results_hrr_resilience_v1/primary_family.csv · secondary_icc_differences.csv (원자료를 다시 읽지 않는다)
출력: results_hrr_resilience_v1/figures_manuscript/Fig_core_resilience_icc.{tif,png}
그림 안에 제목을 넣지 않는다(JSAMS). 색은 Okabe–Ito(색각 이상 대응)."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, '..', 'results_hrr_resilience_v1')
OUT = os.path.join(R, 'figures_manuscript'); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': 1.0, 'xtick.major.width': 1.0, 'ytick.major.width': 1.0, 'xtick.major.size': 4, 'ytick.major.size': 4,  # 축 선 굵게(사용자 2026-10-13, .6 → 1.0)
                     'savefig.dpi': 600,
                     'axes.labelweight': 'bold',
                     'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans', 'mathtext.it': 'Liberation Sans:italic',
                     'mathtext.bf': 'Liberation Sans:bold', 'mathtext.bfit': 'Liberation Sans:italic:bold'})  # 수식 글자도 본문 글꼴(P 이탤릭, 사용자 2026-10-13)  # 축 이름 굵게(사용자 2026-10-13, 모든 그림)
CA, CB, GR, THR = '#B54D00', '#0072B2', '#7F7F7F', 0.40  # CA 대비 5.2:1 (EJAP 글자 대비 ≥ 4.5:1)
MK = {'A': '^', 'B': 'o'}
def wtick(ax_, ws):  # x 눈금 이름을 회복 창 색 · 굵게(그림 1c와 같은 형식)
    for tl, w_ in zip(ax_.get_xticklabels(), ws): tl.set_color(CW[w_]); tl.set_fontweight('bold')
CW = {60: '#117A5E', 120: '#5E57A6', 180: '#C2185B'}  # 회복 창 색(그림 1, 대비 ≥ 5:1)  # 색 외 모양으로도 정의 구분
P = pd.read_csv(f'{R}/primary_family.csv').set_index('key'); D = pd.read_csv(f'{R}/secondary_icc_differences.csv').set_index('contrast')
W = (60, 120, 180); x = np.arange(3)

# 그림 2: 위 a–c = 부트스트랩 분포(슬랩) — 오차막대 대신(사용자 2026-10-13), 아래 d = 57명 전원의 검사 1 · 검사 2 잔차
from scipy.stats import gaussian_kde
def icc21(Y):
    n, k = Y.shape; gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1); msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
TL = pd.read_csv(f'{R}/test_level.csv'); EE = TL[TL.eligible].sort_values(['subj', 'age', 'k']); nb = EE.groupby('subj').id_test.transform('count')
PR = EE[nb >= 2].groupby('subj').head(2).copy(); PR['visit'] = PR.groupby('subj').cumcount() + 1; FT = EE[nb == 1]
XM = lambda d_: np.column_stack([np.ones(len(d_)), d_.vo2peak_kg, d_.age, d_.sex, d_.hr_peak])
CELLS = ('A60', 'A120', 'A180', 'B60', 'B120', 'B180'); RES = {}
for cell in CELLS:
    bb = np.linalg.lstsq(XM(FT), FT[cell].to_numpy(float), rcond=None)[0]; PR['r'] = PR[cell].to_numpy(float) - XM(PR) @ bb
    RES[cell] = PR.pivot(index='subj', columns='visit', values='r')[[1, 2]].to_numpy(float)
    assert np.isclose(icc21(RES[cell]), P.loc[f'res_{cell}'].ICC), cell
rng_ = np.random.default_rng(42); BS = {c_: np.empty(2000) for c_ in CELLS}   # 일차와 같은 재표본(seed 42, 2,000회, 참가자 단위)
for i_ in range(2000):
    ix = rng_.integers(0, 57, 57)
    for c_ in CELLS: BS[c_][i_] = icc21(RES[c_][ix])
AB = .05 / 6
for c_ in CELLS: assert np.isclose(np.percentile(BS[c_], 100 * AB / 2), P.loc[f'res_{c_}'].CIbonf_lo), c_   # 일차 하한 재현
DIF = {f'B−A @{w}s': BS[f'B{w}'] - BS[f'A{w}'] for w in W}
for d in 'AB': DIF.update({f'{d}: 120−60': BS[f'{d}120'] - BS[f'{d}60'], f'{d}: 180−60': BS[f'{d}180'] - BS[f'{d}60'], f'{d}: 180−120': BS[f'{d}180'] - BS[f'{d}120']})
for k_, v_ in DIF.items(): assert np.isclose(np.percentile(v_, 2.5), D.loc[k_, 'CI95_lo']), k_   # 이차 CI 재현

def pboot(v, ref, m=1):   # 사후 표시용(사용자 2026-10-13): 양측 부트스트랩 P(H0: 값 = ref), Bonferroni ×m — CI 판정과 같은 결론
    k = min((v <= ref).sum(), (v >= ref).sum()); nb_ = len(v)
    return (min(1., 2 * m * k / nb_), k == 0, 2 * m / nb_)
def pfmt(pp, bold=False):   # 통계 기호 P는 이탤릭
    p_, zero, res = pp; P_ = r'$\mathbfit{P}$' if bold else r'$\mathit{P}$'
    if zero: return f'{P_} < {res:.3f}'
    return f'{P_} = ' + (f'{p_:.3f}' if p_ < .01 else f'{p_:.2f}')
for c_ in CELLS: assert (pboot(BS[c_], THR, 6)[0] < .05) == (P.loc[f'res_{c_}'].CIbonf_lo > THR), c_
for k_, v_ in DIF.items(): assert (pboot(v_, 0)[0] < .05) == (D.loc[k_, 'CI95_lo'] > 0 or D.loc[k_, 'CI95_hi'] < 0), k_

def slab(ax_, y0, v, lo, hi, col, h=.78):   # 가로 슬랩: 연한 = 부트스트랩 분포, 진한 = CI 구간
    gx_ = np.linspace(v.min(), v.max(), 300); dn = gaussian_kde(v)(gx_); dn = dn / dn.max() * h
    ax_.fill_between(gx_, y0, y0 + dn, color=col, alpha=.22, lw=0, zorder=1)
    k_ = (gx_ >= lo) & (gx_ <= hi); ax_.fill_between(gx_[k_], y0, y0 + dn[k_], color=col, alpha=.55, lw=0, zorder=2)
    ax_.plot(gx_, y0 + dn, color=col, lw=.6, zorder=2)

fig = plt.figure(figsize=(174 / 25.4, 205 / 25.4))
gt = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.15, 1.25], left=.085, right=.99, top=.965, bottom=.625, wspace=.45)
a, b, c2 = (fig.add_subplot(gt[0, i]) for i in range(3))
YA = {'A60': 6.4, 'A120': 5.4, 'A180': 4.4, 'B60': 2.8, 'B120': 1.8, 'B180': .8}
a.axvline(THR, color='#444444', ls='--', lw=.8, zorder=0); vo = P.loc['vo2peak_kg']; a.axvline(vo.ICC, color=GR, ls=':', lw=.9, zorder=0)
for c_ in CELLS:
    d, w = c_[0], int(c_[1:]); col = CA if d == 'A' else CB; r = P.loc[f'res_{c_}']; ok = r.CIbonf_lo > THR; y0 = YA[c_]
    slab(a, y0, BS[c_], r.CIbonf_lo, r.CIbonf_hi, col)
    a.plot(r.ICC, y0, MK[d], ms=5.5, color=col if ok else 'white', mec=col, mew=1.2, zorder=4, clip_on=False)
    a.plot(P.loc[c_].ICC, y0 - .2, 'D', ms=3, color=GR, zorder=3)
    a.text(1.03, y0 + .3, f'{r.ICC:.2f} ({r.CIbonf_lo:.2f})', fontsize=8, va='center', ha='left', color=col, fontweight='bold' if ok else 'normal')
    a.text(1.03, y0 - .12, pfmt(pboot(BS[c_], THR, 6), ok), fontsize=8, va='center', ha='left', color=col, fontweight='bold' if ok else 'normal')
a.set_yticks(list(YA.values()), [f'{c_[0]} · {c_[1:]} s' for c_ in YA]); a.set_ylim(.3, 7.4)
for tl, c_ in zip(a.get_yticklabels(), YA): tl.set_color(CW[int(c_[1:])]); tl.set_fontweight('bold')
a.set_xlim(0, 1.33); a.set_xticks([0, .2, .4, .6, .8, 1]); a.spines['bottom'].set_bounds(0, 1); a.set_xlabel('Test–retest ICC(2,1), residual', x=.38)
a.text(-.27, 1.02, 'a', transform=a.transAxes, fontsize=10, fontweight='bold', va='bottom')

b.axvline(0, color='#444444', lw=.6, zorder=0)
for i, w in enumerate(W):
    r = D.loc[f'B−A @{w}s']; y0 = 2.4 - i; v = DIF[f'B−A @{w}s']
    slab(b, y0, v, r.CI95_lo, r.CI95_hi, '#333333', h=.38)   # a와 같은 실제 높이
    b.plot(r.dICC, y0, 's', ms=4.5, color='#333333', zorder=4)
    b.text(.5, y0 + .3, f'{r.dICC:+.2f}'.replace('-', '−'), fontsize=8, va='center', ha='left', fontweight='bold' if r.CI95_lo > 0 else 'normal')
    b.text(.5, y0 - .02, pfmt(pboot(v, 0), r.CI95_lo > 0), fontsize=8, va='center', ha='left', fontweight='bold' if r.CI95_lo > 0 else 'normal')
b.set_yticks([2.4, 1.4, .4], [f'{w} s' for w in W]); b.set_ylim(-.05, 3.4)
for tl, w in zip(b.get_yticklabels(), W): tl.set_color(CW[w]); tl.set_fontweight('bold')
b.set_xlim(-.3, .85); b.set_xticks([-.2, 0, .2, .4]); b.spines['bottom'].set_bounds(-.3, .5); b.set_xlabel('ICC difference, B − A', x=.4)
b.text(-.3, 1.02, 'b', transform=b.transAxes, fontsize=10, fontweight='bold', va='bottom')

c2.axvline(0, color='#444444', lw=.6, zorder=0)
CON = ['A: 120−60', 'A: 180−60', 'A: 180−120', 'B: 120−60', 'B: 180−60', 'B: 180−120']; YC = [6.4, 5.4, 4.4, 2.8, 1.8, .8]
for k_, y0 in zip(CON, YC):
    r = D.loc[k_]; col = CA if k_[0] == 'A' else CB
    slab(c2, y0, DIF[k_], r.CI95_lo, r.CI95_hi, col)
    c2.plot(r.dICC, y0, 's', ms=4, color=col, zorder=4)
    c2.text(.37, y0 + .3, f'{r.dICC:+.2f}'.replace('-', '−'), fontsize=8, va='center', ha='left', color=col)
    c2.text(.37, y0 - .12, pfmt(pboot(DIF[k_], 0)), fontsize=8, va='center', ha='left', color=col)
c2.set_yticks(YC, [k_[3:] + ' s' for k_ in CON])  # 정의는 색으로(주황 A · 파랑 B); c2.set_ylim(.3, 7.4)
for tl, k_ in zip(c2.get_yticklabels(), CON): tl.set_color(CA if k_[0] == 'A' else CB)
c2.set_xlim(-.45, .68); c2.set_xticks([-.4, -.2, 0, .2]); c2.spines['bottom'].set_bounds(-.45, .35); c2.set_xlabel('ICC difference between windows', x=.42)
c2.text(-.5, 1.02, 'c', transform=c2.transAxes, fontsize=10, fontweight='bold', va='bottom')
for ax_ in (a, b, c2): ax_.spines['left'].set_visible(False); ax_.tick_params(axis='y', length=0)
from matplotlib.patches import Patch
H = [Line2D([], [], color=CA, marker='^', lw=0, ms=5, label=r'Definition A: t$_0$ = peak heart rate'),
     Line2D([], [], color=CB, marker='o', lw=0, ms=5, label=r'Definition B: t$_0$ = end of load'),
     Line2D([], [], color='#444444', marker='o', mfc='white', lw=0, ms=5, label='Open: lower bound ≤ 0.40'),
     Patch(fc='#777777', alpha=.22, label='Bootstrap distribution'), Patch(fc='#777777', alpha=.55, label='99.17% CI (a); 95% CI (b, c)'),
     Line2D([], [], color=GR, marker='D', lw=0, ms=3, label='Unadjusted HRR'),
     Line2D([], [], color='#444444', ls='--', lw=.8, label='Decision threshold (0.40)'),
     Line2D([], [], color=GR, ls=':', lw=.9, label=f'VO$_2$peak/kg ICC ({vo.ICC:.2f})')]
fig.legend(handles=H, frameon=False, fontsize=8, loc='lower center', bbox_to_anchor=(.5, .005), ncol=3, columnspacing=1.2, handletextpad=.4)

# d: 반복 참가자 57명의 회복 잔차, 검사 1 대 검사 2 (위에서 계산한 RES — ICC를 primary_family와 assert로 대조함)
from scipy import stats
X2R = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(R)), 'results_hrr_resilience_v1_extra', 'X2_sem_mdc.csv')); X2R = X2R[X2R.measure == 'residual'].set_index('cell')
lim = np.ceil(max(np.abs(v).max() for v in RES.values()) / 5) * 5 + 13   # 아래 통계 글자가 점을 가리지 않게 여유
gd = fig.add_gridspec(2, 3, left=.13, right=.985, top=.535, bottom=.135, wspace=.12, hspace=.12)
for ri, (d, col) in enumerate((('A', CA), ('B', CB))):
    for ci, w in enumerate(W):
        ax_ = fig.add_subplot(gd[ri, ci]); M = RES[f'{d}{w}']; r = P.loc[f'res_{d}{w}']; ok = r.CIbonf_lo > THR
        ax_.plot([-lim, lim], [-lim, lim], color='#888888', ls='--', lw=.8, zorder=1)
        ax_.axhline(0, color='#DDDDDD', lw=.6, zorder=0); ax_.axvline(0, color='#DDDDDD', lw=.6, zorder=0)
        ax_.scatter(M[:, 0], M[:, 1], marker=MK[d], s=14, color=col if ok else 'white', edgecolors=col, linewidths=.8, zorder=2)
        x2r = X2R.loc[f'{d}{w}']; dd_ = M[:, 1] - M[:, 0]; tp = stats.ttest_rel(M[:, 1], M[:, 0]).pvalue
        assert np.isclose(np.sqrt(((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()) ** 2).sum() / (len(M) - 1)), x2r.SEM)   # SEM 재현
        ax_.text(.04, .96, f'ICC {r.ICC:.2f} [{r.CIbonf_lo:.2f}, {r.CIbonf_hi:.2f}]\n{pfmt(pboot(BS[f"{d}{w}"], THR, 6), ok)}', transform=ax_.transAxes, ha='left', va='top', fontsize=8, color=col, fontweight='bold' if ok else 'normal', linespacing=1.15, bbox=dict(fc='white', ec='none', alpha=.85, pad=1), zorder=5)
        ax_.text(.96, .04, f'SEM {x2r.SEM:.1f} · MDC$_{{95}}$ {x2r.MDC95:.1f} bpm\nBias {dd_.mean():+.1f} bpm, {pfmt((tp, False, 0))}'.replace('-', '−'), transform=ax_.transAxes, ha='right', va='bottom', fontsize=8, color='#333333', linespacing=1.15, bbox=dict(fc='white', ec='none', alpha=.85, pad=1), zorder=5)
        ax_.set_xlim(-lim, lim); ax_.set_ylim(-lim, lim); ax_.set_aspect('equal', adjustable='box')
        if ri == 0: ax_.set_title(f'{w} s', fontsize=8, fontweight='bold', color=CW[w], pad=3)
        if ci == 0: ax_.set_ylabel(f'Definition {d}\nTest 2 (bpm)', color=col)
        else: ax_.tick_params(labelleft=False)
        if ri == 1: ax_.set_xlabel('Test 1 (bpm)')
        else: ax_.tick_params(labelbottom=False)
        if ri == 0 and ci == 0: ax_.text(-.42, 1.04, 'd', transform=ax_.transAxes, fontsize=10, fontweight='bold', va='bottom')
for ext in ('tif', 'png'):
    fig.savefig(f'{OUT}/Fig_core_resilience_icc.{ext}', **({'pil_kwargs': {'compression': 'tiff_lzw'}} if ext == 'tif' else {}))
print('ok', sorted(os.listdir(OUT)))

# ── Figure 1 (영문): 두 회복 시작점 정의와 일치도 ─────────────────────────────
T = pd.read_csv(f'{R}/test_level.csv'); E = T[T.eligible]; O_X = os.path.join(os.path.dirname(os.path.abspath(R)), 'results_hrr_resilience_v1_extra')
AG = pd.read_csv(f'{R}/secondary_agreement_AB.csv')
fig = plt.figure(figsize=(174 / 25.4, 128 / 25.4)); gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=.55, wspace=.42)
MED = '#5E5E5E'   # 중앙값 선: 검정 → 중간 회색, 덜 튀게(사용자 2026-10-13)
a = fig.add_subplot(gs[0, :])   # 예시 1명 → 적격 검사 596개 전부(사용자 2026-10-13), 부하 종료(t₀ B) = 0 정렬
F1 = pd.read_csv(f'{O_X}/F1a_hr_aligned_endofload.csv', index_col='id_test'); gx = F1.columns.astype(int).to_numpy()
assert len(F1) == len(E)
for row in F1.to_numpy(): a.plot(gx, row, color='#9A9A9A', lw=.25, alpha=.12, zorder=1)
q = F1.quantile([.25, .5, .75]).to_numpy()
a.fill_between(gx, q[0], q[2], color='#555555', alpha=.25, lw=0, zorder=2); a.plot(gx, q[1], color=MED, lw=1.2, zorder=4)
dA = (E.t0_A - E.t0_B).to_numpy(); inn = dA >= gx[0]
a.scatter(dA[inn], E.hr_peak.to_numpy()[inn], marker='^', s=5, color=CA, alpha=.45, lw=0, zorder=3)
ytop = np.nanmax(F1.to_numpy()) + 4; ybot = np.nanpercentile(F1.to_numpy(), .5) - 6
for s_ in W: a.text(s_, ybot + 2, f'{s_} s', ha='center', va='bottom', fontsize=8, color=CW[s_], fontweight='bold', bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'), zorder=6)
from matplotlib.patches import Patch   # 회색이 무엇인지 그림 안에 명시(사용자 2026-10-13: 속도로 오해)
a.legend(handles=[Line2D([], [], color='#9A9A9A', lw=.8, label=f'Heart rate, individual tests (n = {len(F1)})'),
                  Line2D([], [], color=MED, lw=1.2, label='Median heart rate'), Patch(fc='#555555', alpha=.25, label='Interquartile range'),
                  Line2D([], [], color=CA, marker='^', lw=0, ms=4, label=r'A: t$_0$ = peak HR of each test'),
                  Line2D([], [], color=CB, lw=1.2, label=r'B: t$_0$ = end of load'),
                  Patch(fc='#BBBBBB', ec='#777777', lw=.8, label='60, 120, 180 s windows')],
         loc='upper center', ncol=3, frameon=False, fontsize=8, handlelength=1.6, columnspacing=1.2, borderaxespad=.1)
yd = ytop; ytop = ybot + (yd - ybot) / .78; fr = (yd - ybot) / (ytop - ybot)  # 위 22%는 범례 두 줄 자리
for s_ in W: a.axvspan(s_ - 5, s_ + 5, ymax=fr, fc=CW[s_], alpha=.4, ec=CW[s_], lw=1, zorder=0)  # 창 띠 진하게(사용자 2026-10-13)
a.axvline(0, ymax=fr, color=CB, lw=1.2, zorder=5)
if (~inn).sum(): a.text(gx[0] + 2, ybot + 2, f'{(~inn).sum()} peaks before −180 s not shown', ha='left', va='bottom', fontsize=8, color='#555555')
a.set_xlim(gx[0], gx[-1]); a.set_ylim(ybot, ytop); a.set_xlabel('Time from end of load (s)'); a.set_ylabel('Heart rate (bpm)')
a.text(-.07, 1.03, 'a', transform=a.transAxes, fontsize=10, fontweight='bold', va='bottom')
d = E.t0_B - E.t0_A; r0 = AG.iloc[0]   # 잘라내지 않은 실제 값(사용자 2026-10-13) — 끊긴 x축: 본체 + 먼 값
assert np.isclose(d.median(), r0['median']) and len(d) == r0.n
BRK = 180; far = d[d > BRK]; assert len(far) >= 1
sg = gs[1, 0].subgridspec(1, 2, width_ratios=[6, 1], wspace=.08); b = fig.add_subplot(sg[0]); b2 = fig.add_subplot(sg[1], sharey=b)
bins = np.arange(5 * np.floor(d.min() / 5), d.max() + 5, 5)
for ax_ in (b, b2): ax_.hist(d, bins=bins, color=CB, alpha=.85, zorder=2)
b.axvspan(r0.q1, r0.q3, color='#DDDDDD', lw=0, zorder=0); b.axvline(r0['median'], color='#222222', lw=1, zorder=3)
b.axvline(0, color='#444444', lw=.6, ls=':', zorder=3)
b.set_xlim(bins[0] - 5, BRK); b2.set_xlim(5 * np.floor(far.min() / 5) - 10, 5 * np.ceil(far.max() / 5) + 10)
b2.spines['left'].set_visible(False); b2.tick_params(axis='y', left=False, labelleft=False); b2.set_xticks([5 * round(far.max() / 5)])
for ax_, xs in ((b, 1), (b2, 0)):   # 끊김 표시 //
    ax_.plot([xs - .015 * (6 if xs == 0 else 1), xs + .015 * (6 if xs == 0 else 1)], [-.03, .03], transform=ax_.transAxes, color='#222222', lw=.6, clip_on=False)
b.text(.97, .95, f'Median {r0["median"]:.0f} s\nIQR {r0.q1:.0f}–{r0.q3:.0f} s (grey band)\nRange {d.min():.0f} to {d.max():.0f} s\nn = {int(r0.n)} tests'.replace('-', '−'), transform=b.transAxes, ha='right', va='top', fontsize=8)
b.set_xlabel(r't$_0$(B) − t$_0$(A) (s)', x=.58); b.set_ylabel('Tests')
b.text(-.2, 1.03, 'b', transform=b.transAxes, fontsize=10, fontweight='bold', va='bottom')
c = fig.add_subplot(gs[1, 1]); g = AG.iloc[1:].reset_index(drop=True)
c.axhline(0, color='#444444', lw=.6, zorder=1); rng = np.random.default_rng(42)
for i, r in g.iterrows():
    w_ = W[i]; v = (E[f'B{w_}'] - E[f'A{w_}']).dropna().to_numpy(); assert np.isclose(v.mean(), r['mean'])
    vp = c.violinplot(v, positions=[i], widths=.5, showextrema=False)
    for b_ in vp['bodies']: b_.set_facecolor(CW[w_]); b_.set_edgecolor('none'); b_.set_alpha(.28)
    c.scatter(i + rng.uniform(-.1, .1, len(v)), v, s=1.5, color=CW[w_], alpha=.45, lw=0, zorder=2)
    for yy, ls in ((r.loa_lo, '--'), (r.loa_hi, '--'), (r['mean'], '-')): c.plot([i - .22, i + .22], [yy, yy], color='#222222', lw=1 if ls == '-' else .8, ls=ls, zorder=3)
    c.text(i + .25, r['mean'], f'{r["mean"]:+.1f}', fontsize=8, va='center', color='#222222', fontweight='bold')
    for yy in (r.loa_lo, r.loa_hi): c.text(i + .25, yy, f'{yy:+.1f}'.replace('-', '−'), fontsize=8, va='center', color='#555555')
c.set_xticks(x, [f'{w_} s' for w_ in W]); wtick(c, W); c.set_xlim(-.5, 2.75); c.set_xlabel(r'Recovery window after t$_0$')
c.set_ylabel('HRR difference, B − A (bpm)')
# 그림 안 설명 문구 삭제 — 범례에 'mean with 95% LoA'(EJAP, 3회차 심사)
c.text(-.24, 1.03, 'c', transform=c.transAxes, fontsize=10, fontweight='bold', va='bottom')
fig.subplots_adjust(left=.09, right=.92, top=.95, bottom=.12)
for ext in ('tif', 'png'):
    fig.savefig(f'{OUT}/Fig1_onset_definitions.{ext}', **({'pil_kwargs': {'compression': 'tiff_lzw'}} if ext == 'tif' else {}))
print('fig1 ok')

# ── Figure 3: 견고성·민감도 행렬 (6조합 × 분석 전부, 통과·미달 같은 표기) ───────────
O_ = os.path.dirname(os.path.abspath(R)); CELLS6 = ['A60', 'A120', 'A180', 'B60', 'B120', 'B180']
S_ = pd.read_csv(f'{R}/sensitivity.csv'); RB_ = pd.read_csv(f'{O_}/results_hrr_resilience_v1_robust/robust_summary.csv')
Q_ = pd.read_csv(f'{O_}/results_hrr_resilience_oq08/v1_primary_recomputed.csv'); J_ = pd.read_csv(f'{O_}/results_hrr_resilience_v1_robust/jackknife.csv')
def from_sens(lab): d = S_[S_.analysis == lab].set_index('key'); return {c: (d.at[f'res_{c}', 'ICC'], d.at[f'res_{c}', 'CIbonf_lo'], d.at[f'res_{c}', 'CIbonf_hi'], int(d.n_pairs.iloc[0])) for c in CELLS6}
def from_rb(lab): d = RB_[RB_.scenario == lab].set_index('cell'); return {c: (d.at[c, 'ICC'], d.at[c, 'CIbonf_lo'], d.at[c, 'CIbonf_hi'], int(d.n_subj.iloc[0])) for c in d.index}
_S4F = pd.read_csv(f'{O_}/results_hrr_resilience_v1_posthoc_r2/S4_artefact.csv'); _S4F = _S4F[_S4F.gap_cut_bpm == 10]
# 그림 3: 행렬(표 4와 중복) → 숲 그림(사용자 2026-10-13). 사전지정 · 사후 묶음, 칸 = 조합, 판정선 0.40
GRP = [('Pre-specified', [('Primary analysis', {c: (P.loc[f'res_{c}', 'ICC'], P.loc[f'res_{c}', 'CIbonf_lo'], P.loc[f'res_{c}', 'CIbonf_hi'], 57) for c in CELLS6}),
                          ('Peak RER ≥ 1.10', from_sens('S1 RER≥1.10')),
                          ('Model without peak HR', from_sens('S2 HR최고점 제외')),
                          ('Test interval < 0.5 y', from_sens('S3 간격<0.5y')), ('Test interval ≥ 0.5 y', from_sens('S3 간격≥0.5y'))]),
       ('Post hoc', [('All repeat tests (ICC1)', from_rb('R1 전 검사')),
                     ('Model refitted per resample', from_rb('R2 예측식 재적합')),
                     ('Recovery ≥ 125 s (extended)', from_rb('R3 회복≥125 s 전 검사+재적합')),
                     ('Exercise-phase peak HR, VO$_2$peak', {r.cell: (r.ICC, r.CIbonf_lo, r.CIbonf_hi, 57) for r in Q_[Q_.analysis == 'OQ-08 운동 구간 변수'].itertuples()}),
                     ('Suspect HR signal excluded', {r.cell: (r.ICC, r.CIbonf_lo, r.CIbonf_hi, int(r.n_pairs)) for r in _S4F.itertuples() if r.cell in CELLS6})])]
for c in CELLS6: assert np.isclose(GRP[0][1][0][1][c][1], P.loc[f'res_{c}', 'CIbonf_lo'])
YS, labs, y = [], [], 0.0
for g, rows in GRP:
    y -= .9; labs.append((y, g, True))
    for lab, d in rows: y -= 1; YS.append((y, lab, d)); labs.append((y, lab, False))
ND = {lab: min(v[3] for v in d.values()) for _, lab, d in YS}
fig = plt.figure(figsize=(174 / 25.4, 180 / 25.4))
GL, GR_ = .36, .995
gf = fig.add_gridspec(1, 6, left=GL, right=GR_, top=.915, bottom=.2, wspace=.1)
for j, c in enumerate(CELLS6):
    ax = fig.add_subplot(gf[0, j]); d_, col = c[0], (CA if c[0] == 'A' else CB)
    ax.axvline(THR, color='#444444', ls='--', lw=.8, zorder=0)
    ax.axhspan(YS[0][0] - .48, YS[0][0] + .42, color='#EDEDED', lw=0, zorder=0)   # 일차 분석 줄
    for y0, lab, d in YS:
        if c not in d: ax.text(.45, y0, 'n/a', ha='center', va='center', fontsize=8, color='#595959', bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none'), zorder=4); continue
        icc, lo, hi, n = d[c]; ok = lo > THR
        ax.plot([lo, hi], [y0 + .14, y0 + .14], color=col, lw=1.3, solid_capstyle='butt', zorder=2)
        ax.plot(icc, y0 + .14, MK[d_], ms=5, color=col if ok else 'white', mec=col, mew=1.1, zorder=3)
        lb = (f'{lo:.3f}' if abs(lo - THR) < .01 else f'{lo:.2f}').replace('-', '−')   # 표(부록 1)와 같은 자릿수 규칙
        ax.text(.45, y0 - .26, f'{icc:.2f} ({lb})' + ('*' if ok else ''), ha='center', va='center', fontsize=8, color=col, fontweight='bold' if ok else 'normal', zorder=4,
                bbox=dict(boxstyle='square,pad=0.05', fc='#EDEDED' if y0 == YS[0][0] else 'white', ec='none'))   # ICC (하한), 사용자 2026-10-13 · 점선 위 흰 바탕
    ax.set_xlim(-.1, 1); ax.set_xticks([0, .4, 1], ['0', '0.4', '1']); ax.set_ylim(y - .6, .1); ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_title(f'{c[1:]} s', fontsize=8, fontweight='bold', color=CW[int(c[1:])], pad=3)
    if j == 0:
        for y0, lab, hdr in labs:
            ax.text(-.16, y0, lab if hdr else lab + f'  (n = {ND[lab]})',
                    transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=8, fontweight='bold' if hdr else 'normal')
for j0, d_ in ((0, 'A'), (3, 'B')):
    x0 = GL + (GR_ - GL) * (j0 / 6) ; fig.text(x0 + (GR_ - GL) / 4, .965, f'Definition {d_}', ha='center', va='center', fontsize=8, fontweight='bold', color=CA if d_ == 'A' else CB)
fig.text(GL + (GR_ - GL) / 2, .148, 'Residual ICC with Bonferroni-adjusted CI', ha='center', va='center', fontsize=8, fontweight='bold')
from matplotlib.patches import Patch   # 기호 설명을 그림 안에(사용자 2026-10-13)
from matplotlib.legend_handler import HandlerTuple
mk_ = lambda d_, f: Line2D([], [], color=CA if d_ == 'A' else CB, marker=MK[d_], mfc=None if f else 'white', lw=0, ms=5)
blank = Line2D([], [], lw=0)
fig.legend(handles=[Line2D([], [], color=CA, marker='^', lw=1.3, ms=5), Line2D([], [], color=CB, marker='o', lw=1.3, ms=5),
                    (mk_('A', True), mk_('B', True)), (mk_('A', False), mk_('B', False)),
                    blank, Line2D([], [], color='#444444', ls='--', lw=.8), Patch(fc='#EDEDED'), blank],
           labels=[r'Definition A: t$_0$ = peak heart rate', r'Definition B: t$_0$ = end of load',
                   'Filled: lower bound > 0.40', 'Open: lower bound ≤ 0.40',
                   '0.71 (0.47): ICC (lower bound of the CI)', 'Decision threshold (0.40)', 'Primary analysis',
                   r'*: CI excludes 0.40 (adjusted $\mathit{P}$ < 0.05)'],
           handler_map={tuple: HandlerTuple(ndivide=None, pad=.6)},
           loc='lower center', bbox_to_anchor=(.5, .005), ncol=2, frameon=False, fontsize=8, columnspacing=2.5, handletextpad=.5, handlelength=2.2)
for ext in ('tif', 'png'):
    fig.savefig(f'{OUT}/Fig3_robustness_matrix.{ext}', **({'pil_kwargs': {'compression': 'tiff_lzw'}} if ext == 'tif' else {}))
plt.close(fig); print('fig3 ok')

# ── Figure 4: 측정 실무 — 임상 기준 재분류 · MDC95 · 체력 상관 ────────────────────
X_ = os.path.join(O_, 'results_hrr_resilience_v1_extra')
X1 = pd.read_csv(f'{X_}/X1_reclassification.csv'); X2 = pd.read_csv(f'{X_}/X2_sem_mdc.csv'); X3 = pd.read_csv(f'{X_}/X3_hrr_vo2peak_correlation.csv')
# 그림 4 v3(사용자 2026-10-13: a–c도 그림 1c처럼 개인 값 전부 — 바이올린 + 점) · d 2×2 판정표
from scipy import stats as st_
from matplotlib.patches import Patch
E4 = T[T.eligible].copy(); assert len(E4) == 596
F4 = E4.sort_values(['subj', 'age', 'k']).groupby('subj').head(1); assert len(F4) == 529
crits = [('60 s ≤ 12 bpm (Cole 1999)', '1-min HRR ≤ 12 bpm', 60, 12, lambda v: v <= 12), ('120 s < 22 bpm (Shetler 2001 원문 표기)', '2-min HRR < 22 bpm', 120, 22, lambda v: v < 22)]
rng4 = np.random.default_rng(42)
def rain(ax_, xpos, v, col, w=.5, jit=.11, s_=1.6):   # 그림 1c와 같은 형식: 바이올린 + 흩뿌린 점
    vp = ax_.violinplot(v, positions=[xpos], widths=w, showextrema=False)
    for b_ in vp['bodies']: b_.set_facecolor(col); b_.set_edgecolor('none'); b_.set_alpha(.25)
    ax_.scatter(xpos + rng4.uniform(-jit, jit, len(v)), v, s=s_, color=col, alpha=.45, lw=0, zorder=2)
fig = plt.figure(figsize=(174 / 25.4, 215 / 25.4))
gt4 = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], left=.085, right=.985, top=.955, bottom=.715, wspace=.3)
a, b = fig.add_subplot(gt4[0, 0]), fig.add_subplot(gt4[0, 1])
# a: 검사 596회의 HRR 분포와 기준선 — 기준 이하 = 비정상
for g, (key, lab, w_, cut, fn) in enumerate(crits):
    d = X1[X1.criterion == key].set_index('metric'); x0 = g * 2.3
    for dd, off in (('A', 0), ('B', .95)):
        col = CA if dd == 'A' else CB; v = E4[f'{dd}{w_}'].to_numpy(); ab_ = fn(v)
        assert np.isclose(ab_.mean(), d.at[f'prev_{dd}', 'estimate'])
        rain(a, x0 + off, v, col)
        a.scatter(x0 + off + rng4.uniform(-.11, .11, ab_.sum()), v[ab_], s=4, color='#222222', lw=0, zorder=3)
        pv, lo, hi = (d.at[f'prev_{dd}', m] * 100 for m in ('estimate', 'CI95_lo', 'CI95_hi'))
        a.text(x0 + off, 104, f'{pv:.1f}%\n[{lo:.1f}, {hi:.1f}]', ha='center', va='bottom', fontsize=8, color=col, linespacing=1.1)
    a.plot([x0 - .38, x0 + 1.33], [cut, cut], color=CW[w_], ls='--', lw=1.1, zorder=4)
    kp = d.at['kappa', 'estimate']; deg = d.at['prev_B', 'estimate'] == 0
    a.text(x0 + .475, 128, (f'κ = {kp:.2f}' if not deg else 'κ = 0†'), ha='center', va='bottom', fontsize=8)
a.set_xticks([0, .95, 2.3, 3.25], ['A', 'B', 'A', 'B']); [tl.set_color(cc) or tl.set_fontweight('bold') for tl, cc in zip(a.get_xticklabels(), (CA, CB, CA, CB))]
for g, (key, lab, w_, cut, fn) in enumerate(crits): a.text(g * 2.3 + .475, -.11, lab.replace(' HRR ', ' HRR\n'), transform=a.get_xaxis_transform(), ha='center', va='top', fontsize=8, fontweight='bold', color=CW[w_])
a.set_xlim(-.55, 3.8); a.set_ylim(-15, 140); a.set_yticks([0, 25, 50, 75, 100]); a.set_ylabel('HRR (bpm)')
a.text(-.25, 1.02, 'a', transform=a.transAxes, fontsize=10, fontweight='bold', va='bottom')
# b: 57명의 검사 2 − 검사 1 차이와 ±MDC95 (원 HRR)
d2 = X2[X2.measure == 'raw HRR'].set_index('cell')
XB = {'A60': 0, 'A120': 1, 'A180': 2, 'B60': 3.4, 'B120': 4.4, 'B180': 5.4}
for c_, xb in XB.items():
    dd = c_[0]; col = CA if dd == 'A' else CB; M_ = PR.pivot(index='subj', columns='visit', values=c_)[[1, 2]].to_numpy(float)
    sem_ = np.sqrt(((M_ - M_.mean(1, keepdims=True) - M_.mean(0, keepdims=True) + M_.mean()) ** 2).sum() / (len(M_) - 1)); assert np.isclose(sem_, d2.loc[c_, 'SEM'])
    dif = M_[:, 1] - M_[:, 0]; mdc = d2.loc[c_, 'MDC95']; out_ = np.abs(dif) > mdc
    rain(b, xb, dif, col, w=.6, jit=.12, s_=5)
    for yy in (-mdc, mdc): b.plot([xb - .32, xb + .32], [yy, yy], color='#222222', ls='--', lw=.9, zorder=4)
    b.text(xb, mdc + 2, f'±{mdc:.1f}', ha='center', va='bottom', fontsize=8, color=col, fontweight='bold')
    b.text(xb, -mdc - 2, f'{out_.sum()}/{len(dif)}', ha='center', va='top', fontsize=8, color='#444444')
b.axhline(0, color='#444444', lw=.6, zorder=1)
b.set_xticks(list(XB.values()), [f'{c_[1:]} s' for c_ in XB]); [tl.set_color(CW[int(c_[1:])]) or tl.set_fontweight('bold') for tl, c_ in zip(b.get_xticklabels(), XB)]
for xg, dd in ((1, 'A'), (4.4, 'B')): b.text(xg, -.11, f'Definition {dd}', transform=b.get_xaxis_transform(), ha='center', va='top', fontsize=8, fontweight='bold', color=CA if dd == 'A' else CB)
b.set_xlim(-.6, 6); b.set_ylim(-55, 55); b.set_ylabel('Test 2 − test 1, HRR (bpm)')
b.text(-.17, 1.02, 'b', transform=b.transAxes, fontsize=10, fontweight='bold', va='bottom')
# c: 529명 HRR 대 VO2peak/kg, 정의(행) × 창(열)
d3 = X3.set_index('cell')
gc4 = fig.add_gridspec(2, 3, left=.085, right=.985, top=.62, bottom=.335, wspace=.24, hspace=.12)
for ci, w_ in enumerate(W):
    ylo, yhi = min(F4[f'A{w_}'].min(), F4[f'B{w_}'].min()) - 3, max(F4[f'A{w_}'].max(), F4[f'B{w_}'].max()) + 14
    for ri, dd in enumerate('AB'):
        ax_ = fig.add_subplot(gc4[ri, ci]); col = CA if dd == 'A' else CB; r = d3.loc[f'{dd}{w_}']
        rr, pp = st_.pearsonr(F4.vo2peak_kg, F4[f'{dd}{w_}']); assert np.isclose(rr, r.pearson_r)
        ax_.scatter(F4.vo2peak_kg, F4[f'{dd}{w_}'], marker=MK[dd], s=4, color=col, alpha=.4, lw=0)
        bb = np.polyfit(F4.vo2peak_kg, F4[f'{dd}{w_}'], 1); xx = np.array([F4.vo2peak_kg.min(), F4.vo2peak_kg.max()]); ax_.plot(xx, np.polyval(bb, xx), color='#222222', lw=1.1)
        ax_.text(.03, .97, f'r = {r.pearson_r:.2f} [{r.r_lo:.2f}, {r.r_hi:.2f}], ' + (r'$\mathit{P}$ < 0.001' if pp < .001 else rf'$\mathit{{P}}$ = {pp:.3f}'),
                 transform=ax_.transAxes, ha='left', va='top', fontsize=8, color=col, fontweight='bold' if dd == 'B' else 'normal')
        ax_.set_ylim(ylo, yhi)
        if ri == 0: ax_.set_title(f'{w_} s', fontsize=8, fontweight='bold', color=CW[w_], pad=3); ax_.tick_params(labelbottom=False)
        if ci == 0: ax_.set_ylabel(f'Definition {dd}\nHRR (bpm)', color=col)
        if ri == 0 and ci == 0: ax_.text(-.3, 1.06, 'c', transform=ax_.transAxes, fontsize=10, fontweight='bold', va='bottom')
fig.text(.535, .305, r'VO$_2$peak/kg (mL·kg$^{-1}$·min$^{-1}$); n = 529 participants, first test of each', ha='center', va='center', fontsize=8, fontweight='bold')
# d: 2×2 판정표
gd4 = fig.add_gridspec(1, 2, left=.09, right=.5, top=.2, bottom=.05, wspace=.75)
for k, (key, lab, w_, cut, fn) in enumerate(crits):
    ax_ = fig.add_subplot(gd4[0, k]); d = X1[X1.criterion == key].set_index('metric')
    ab, bb_ = fn(E4[f'A{w_}']).to_numpy(), fn(E4[f'B{w_}']).to_numpy()
    M2 = np.array([[(ab & bb_).sum(), (ab & ~bb_).sum()], [(~ab & bb_).sum(), (~ab & ~bb_).sum()]])
    assert M2[0, 1] == d.at['A_only_abnormal_n', 'estimate'] and M2[1, 0] == d.at['B_only_abnormal_n', 'estimate'] and M2.sum() == 596
    FC = [['#E3E3E3', '#F3D9C8'], ['#D6E6F2', '#E3E3E3']]
    for ri_ in range(2):
        for cj in range(2):
            ax_.add_patch(plt.Rectangle((cj, 1 - ri_), .96, .96, color=FC[ri_][cj], lw=0))
            ax_.text(cj + .48, 1 - ri_ + .48, f'{M2[ri_, cj]}', ha='center', va='center', fontsize=8, fontweight='bold')
    ax_.set_xlim(0, 2); ax_.set_ylim(0, 2); ax_.set_xticks([.48, 1.48], ['Abn.', 'Norm.']); ax_.set_yticks([1.48, .48], ['Abn.', 'Norm.'])
    ax_.tick_params(length=0); [sp.set_visible(False) for sp in ax_.spines.values()]
    ax_.xaxis.set_label_position('top'); ax_.xaxis.tick_top(); ax_.set_xlabel('Definition B', color=CB, labelpad=2); ax_.set_ylabel('Definition A', color=CA, labelpad=1)
    ax_.set_title(lab, fontsize=8, fontweight='bold', color=CW[w_], pad=3)
    mp = d.at['mcnemar_exact_p_reference', 'estimate']
    ax_.text(1, -.12, r'McNemar $\mathit{P}$ < 0.001' if mp < .001 else rf'McNemar $\mathit{{P}}$ = {mp:.3f}', ha='center', va='top', fontsize=8)
    if k == 0: ax_.text(-.85, 2.6, 'd', fontsize=10, fontweight='bold', va='bottom')
blank = Line2D([], [], lw=0)
fig.legend(handles=[Line2D([], [], color=CA, marker='^', lw=0, ms=5), Line2D([], [], color=CB, marker='o', lw=0, ms=5),
                    Line2D([], [], color='#222222', marker='o', lw=0, ms=3), Line2D([], [], color='#666666', ls='--', lw=1),
                    Line2D([], [], color='#222222', ls='--', lw=.9), Line2D([], [], color='#222222', lw=1.1), Patch(fc='#F3D9C8'), Patch(fc='#D6E6F2'), blank],
           labels=[r'Definition A: t$_0$ = peak heart rate', r'Definition B: t$_0$ = end of load', 'Abnormal test (a)', 'Criterion cut-off (a)',
                   '±MDC$_{95}$ (b); n outside below', 'Least-squares line (c)', 'Abnormal under A only (d)', 'Abnormal under B only (d)',
                   '† κ = 0 by construction: no test abnormal under B'],
           loc='center left', bbox_to_anchor=(.56, .125), ncol=1, frameon=False, fontsize=8, handletextpad=.5, labelspacing=.35)
for ext in ('tif', 'png'):
    fig.savefig(f'{OUT}/Fig4_measurement_practice.{ext}', **({'pil_kwargs': {'compression': 'tiff_lzw'}} if ext == 'tif' else {}))
plt.close(fig); print('fig4 ok')
# Supplementary Figure S1(v2 예측 모형)은 원고에서 제외 — 사용자 결정 2026-10-13. 설명용 한글 그림 6은 results_hrr_resilience_v2/figures/에 남는다.

# EJAP 제출용: TIFF를 RGB로(지침 'RGB'), 파일 이름 Fig1–Fig4 (지침 'Fig1.eps' 형식) — 2026-10-13
from PIL import Image
SUB = os.path.join(R, '..', 'submission_ejap_hrr', 'figures'); os.makedirs(SUB, exist_ok=True)
for n, f in ((1, 'Fig1_onset_definitions'), (2, 'Fig_core_resilience_icc'), (3, 'Fig3_robustness_matrix'), (4, 'Fig4_measurement_practice')):
    im = Image.open(f'{OUT}/{f}.tif').convert('RGB')
    im.save(f'{OUT}/{f}.tif', compression='tiff_lzw', dpi=(600, 600)); im.save(f'{SUB}/Fig{n}.tif', compression='tiff_lzw', dpi=(600, 600))
print('tiff rgb ok')
