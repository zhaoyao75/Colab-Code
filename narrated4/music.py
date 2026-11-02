"""《后悔按钮》配乐：16 秒，纯 numpy 合成。A 小调，稀疏钢琴 + 低沉铺底 + 钟表滴答。
结构（对齐 shots.py 的时间点）：
  0–5.3   钢琴单音 + 滴答；3.8 起上升扫频
  5.3     按下：低频重击 + 亮色泛音；5.45–7.2 下行琶音（倒带感），落在 A4
  8–10.35 沉下去的铺底，滴答继续
  10.35   滴答骤停 + 下滑音（按钮熄灭）
  12.4    一个悬而未决的 E4，长混响，淡出到 16 秒
输出 bgm.wav（立体声 44.1k）
"""
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve, butter, lfilter

SR, DUR = 44100, 16.0
N = int(SR * DUR)
rng = np.random.default_rng(7)
T = np.arange(N) / SR


def mtof(m): return 440 * 2 ** ((m - 69) / 12)


def place(buf, sig, t0):
    i = int(t0 * SR)
    j = min(N, i + len(sig))
    if i < N: buf[i:j] += sig[:j - i]


def piano(m, dur=4.0, vel=1.0):
    t = np.arange(int(SR * dur)) / SR
    f = mtof(m)
    s = np.zeros_like(t)
    for h in range(1, 9):
        s += (1 / h ** 1.3) * np.sin(2 * np.pi * f * h * (1 + 0.0004 * h * h) * t) * np.exp(-t * (1.1 + 0.5 * h) * (0.5 + 80 / (f + 100)))
    s *= np.minimum(1, t / 0.004)
    return s * vel * 0.35


def pad(m, dur, vol=0.12):
    t = np.arange(int(SR * dur)) / SR
    s = sum(np.sin(2 * np.pi * mtof(m) * (1 + d) * t + p) for d, p in ((0, 0), (0.003, 1.3), (-0.003, 2.1)))
    s += 0.4 * np.sin(2 * np.pi * mtof(m) * 2 * t)
    env = np.minimum(1, t / 1.5) * np.minimum(1, (dur - t) / 1.5)
    return s * env * vol / 3


def lp(x, fc):
    b, a = butter(2, fc / (SR / 2))
    return lfilter(b, a, x)


L = np.zeros(N); Rr = np.zeros(N)
dry = np.zeros(N)

# ---- 铺底 ----
for m, t0, d in ((45, 0, 5.6), (41, 4.0, 4.4), (45, 7.6, 3.2)):
    place(dry, pad(m, d, 0.16), t0)
place(dry, pad(57, 4.0, 0.07), 0.5)
place(dry, pad(48, 4.5, 0.12), 8.0)           # C3 沉下去
place(dry, pad(52, 3.7, 0.08), 12.3)          # E3 悬着

# ---- 钢琴 ----
for t0, m, v in ((0.6, 64, .8), (1.6, 69, .8), (2.6, 72, .85), (3.4, 71, .7),
                 (4.6, 72, .8), (5.9 - 0.2, 65, .0), (8.6, 67, .55), (9.4, 60, .5)):
    if v: place(dry, piano(m, 4, v), t0)
# 上升扫频（倒带前的吸气）
n = int((5.3 - 3.8) * SR)
x = rng.normal(0, 1, n)
sw = lp(x, 3000) * np.linspace(0, 1, n) ** 3 * 0.18
place(dry, sw, 3.8)
# 按下：低频重击 + 亮泛音
tt = np.arange(int(1.6 * SR)) / SR
thud = np.sin(2 * np.pi * (48 + 60 * np.exp(-tt * 18)) * tt) * np.exp(-tt * 3.2) * 0.9
place(dry, thud, 5.3)
place(dry, piano(81, 3.5, .5), 5.3)
place(dry, piano(88, 3.5, .35), 5.34)
# 下行琶音（倒带），间隔逐渐变慢
notes = [84, 81, 79, 76, 74, 72, 69]
ts = np.cumsum([0] + list(np.linspace(0.15, 0.32, len(notes) - 1)))
ts = 5.45 + ts / ts[-1] * (7.2 - 5.45 - 0.0)
for t0, m in zip(ts, notes):
    place(dry, piano(m, 3.0, .55), t0)
place(dry, piano(57, 4, .6), 7.2)
place(dry, piano(69, 4, .7), 7.2)
# ---- 滴答（到 10.35 骤停） ----
for t0 in np.arange(0.5, 10.35, 0.5):
    if abs(t0 - 5.5) < 0.01: continue
    k = np.arange(int(0.05 * SR)) / SR
    tick = lp(rng.normal(0, 1, len(k)), 4500) * np.exp(-k * 140) * (0.10 if int(t0 * 2) % 2 == 0 else 0.06)
    place(dry, tick, t0)
# ---- 熄灭：下滑音 ----
k = np.arange(int(1.8 * SR)) / SR
f = 220 * np.exp(-k * 1.1)
fall = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-k * 2.2) * 0.35 * np.minimum(1, k / 0.02)
place(dry, fall, 10.35)
# ---- 问句：悬着的 E4（不解决） ----
place(dry, piano(64, 4, .85), 12.4)
place(dry, piano(71, 4, .35), 12.9)

# ---- 混响 + 立体声 ----
rl = int(2.6 * SR)
tr = np.arange(rl) / SR
def ir(): return lp(rng.normal(0, 1, rl), 5000) * np.exp(-tr * 2.4)
wet_l = fftconvolve(dry, ir())[:N] * 0.045
wet_r = fftconvolve(dry, ir())[:N] * 0.045
L = dry + wet_l; Rr = dry + wet_r
st = np.stack([L, Rr])
fade = np.minimum(1, T / 0.5) * np.minimum(1, (DUR - T) / 1.2)
st *= fade
st /= np.abs(st).max() / 0.8
wavfile.write("bgm.wav", SR, (st.T * 32767).astype(np.int16))
for a in range(0, 16, 2):
    seg = st[:, a * SR:(a + 2) * SR]
    print(f"{a:2d}-{a+2:2d}s RMS {np.sqrt((seg**2).mean()):.3f}")
