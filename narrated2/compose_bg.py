"""《青城》背景音乐：128 秒，古朴、草原风格，原创旋律。

D 羽调式（D F G A C，五声音阶），60 BPM，4/4，32 小节 = 128 秒，每 4 小节（16 秒）对应一章。
配器：持续低音（五度）、拉弦乐（模仿马头琴的滑音和揉弦）、拨弦（模仿古筝/托布秀尔）、
      手鼓、驼铃、风声、弦乐长音铺底。整体刻意做得很淡，给旁白让位。

运行：python3 compose_bg.py   ->  .cache/music.wav、music.mp3
"""
import os, subprocess
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 44100
BEAT = 1.0
BAR = 4.0
LENGTH = 128.0
N = int((LENGTH + 4) * SR)
rng = np.random.default_rng(11)


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


D3, F3, G3, A3, C4 = 50, 53, 55, 57, 60
D4, F4, G4, A4, C5, D5, F5 = 62, 65, 67, 69, 72, 74, 77


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], "band", fs=SR, output="sos"), x)


def lp(x, f, order=2):
    return sosfilt(butter(order, f, "low", fs=SR, output="sos"), x)


def add(buf, sig, t, gain=1.0, pan=0.0):
    i = int(t * SR)
    j = min(i + len(sig), buf.shape[1])
    l, r = np.sqrt((1 - pan) / 2), np.sqrt((1 + pan) / 2)
    buf[0, i:j] += sig[:j - i] * gain * l
    buf[1, i:j] += sig[:j - i] * gain * r


# ---------------------------------------------------------------- 乐器
def bowed(phrase, t0, octave=0, glide=0.09):
    """拉弦乐：连奏一整句，音间滑音，延迟揉弦，带弓噪。phrase = [(midi, 拍数), ...]"""
    dur = sum(b for _, b in phrase) * BEAT
    n = int((dur + 1.2) * SR)
    t = np.arange(n) / SR
    f = np.zeros(n)
    amp = np.zeros(n)
    pos = 0.0
    prev = hz(phrase[0][0] + 12 * octave)
    for k, (m, b) in enumerate(phrase):
        fm = hz(m + 12 * octave)
        a, e = int(pos * SR), int((pos + b * BEAT) * SR)
        seg = np.arange(e - a) / SR
        g = np.clip(seg / glide, 0, 1)
        g = g * g * (3 - 2 * g)
        f[a:e] = prev * (fm / prev) ** g
        # 每个音：起弓渐强、句中呼吸
        env = np.clip(seg / 0.18, 0, 1) * (1 - 0.25 * np.clip((seg - (e - a) / SR + 0.3) / 0.3, 0, 1))
        amp[a:e] = 0.75 + 0.25 * env
        prev = fm
        pos += b * BEAT
    e = int(dur * SR)
    f[e:] = prev
    amp[e:] = np.exp(-(t[e:] - dur) / 0.25)
    amp[:e] *= np.clip(t[:e] / 0.4, 0, 1)
    # 揉弦：每个音起来后才出现
    vib = 1 + 0.006 * np.sin(2 * np.pi * 5.2 * t) * np.clip((t % 4) / 0.8, 0.2, 1)
    ph = 2 * np.pi * np.cumsum(f * vib) / SR
    sig = sum(np.sin(h * ph) / h ** 1.1 for h in range(1, 14) if True)
    sig += 0.06 * rng.standard_normal(n)                   # 弓噪
    sig = bp(sig, 180, 3200) + 0.6 * bp(sig, 700, 1300)     # 共鸣箱的鼻音感
    return sig * amp * 0.22


def pluck(m, dur=2.5):
    """拨弦：衰减泛音，起音后轻微下滑（古筝的味道）"""
    f = hz(m)
    n = int(dur * SR)
    t = np.arange(n) / SR
    bend = 1 - 0.004 * (1 - np.exp(-t / 0.15))
    ph = 2 * np.pi * f * np.cumsum(bend) / SR
    sig = np.zeros(n)
    for h in range(1, 9):
        sig += np.sin(h * ph) * (0.8 ** h) * np.exp(-t * (1.6 + 0.9 * h))
    sig += 0.3 * rng.standard_normal(n) * np.exp(-t / 0.004)
    return lp(sig, 5000) * 0.35


def drum(soft=1.0):
    n = int(1.2 * SR)
    t = np.arange(n) / SR
    f = 70 + 50 * np.exp(-t / 0.03)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.28)
    skin = lp(rng.standard_normal(n), 900) * np.exp(-t / 0.05) * 0.6
    return (body + skin) * 0.5 * soft


def bell():
    """驼铃：非谐泛音，短促"""
    f = rng.uniform(1300, 1700)
    n = int(1.5 * SR)
    t = np.arange(n) / SR
    sig = sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t * d)
              for r, a, d in [(1, 1, 4), (2.76, 0.5, 7), (5.4, 0.25, 12)])
    return sig * 0.12


def pad(notes, t0, dur):
    n = int((dur + 2) * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for m in notes:
        for det in (0.997, 1.0, 1.003):
            ph = 2 * np.pi * hz(m) * det * t + rng.random() * 6.28
            sig += sum(np.sin(h * ph) / h for h in range(1, 6))
    sig = lp(sig, 1400)
    env = np.clip(t / 2.0, 0, 1) * np.clip((dur + 1.5 - t) / 2.0, 0, 1)
    return sig * env * 0.03


# ---------------------------------------------------------------- 编排
L = np.zeros((2, N))   # 拉弦乐
P = np.zeros((2, N))   # 拨弦
DR = np.zeros((2, N))
BG = np.zeros((2, N))  # 低音、铺底、风

t = np.arange(N) / SR
# 风声：慢慢起伏的带通噪声
wind = np.zeros((2, N))
for c in range(2):
    w = bp(rng.standard_normal(N), 250, 1800)
    w *= 0.5 + 0.5 * np.sin(2 * np.pi * (0.07 + 0.02 * c) * t + c) ** 2
    wind[c] = w * 0.05
wind_level = np.interp(t, [0, 6, 20, 100, 116, 128, 132], [0.0, 1.0, 0.6, 0.5, 0.9, 1.0, 0.0])
BG += wind * wind_level

# 持续低音 D–A，随段落起伏
drone_lv = np.interp(t, [0, 8, 64, 80, 112, 126, 130], [0, 0.6, 0.9, 1.0, 0.8, 0.5, 0])
ph = 2 * np.pi * t
dr = (np.sin(ph * hz(38)) + 0.6 * np.sin(ph * hz(45)) + 0.3 * np.sin(ph * hz(50))
      + 0.15 * np.sin(ph * hz(57))) * (1 + 0.1 * np.sin(2 * np.pi * 0.11 * t))
BG += lp(dr, 600) * drone_lv * 0.07

# 铺底和弦：每 4 小节换一个
CH = [(D3, A3, F4), (F3, C4, A4), (G3, D4, C5), (D3, A3, F4)]
for sec in range(2, 7):
    for k in range(4):
        tb = sec * 16 + k * 4
        s = pad(CH[k], tb, 4.0)
        add(BG, s, tb, 1.0, -0.3)
        add(BG, s, tb + 0.03, 1.0, 0.3)

# 主旋律（原创）
A = [(A4, 3), (G4, 1), (A4, 2), (C5, 2), (D5, 4), (C5, 2), (A4, 2)]
A2 = [(G4, 3), (F4, 1), (D4, 4), (F4, 2), (G4, 2), (A4, 4)]
B = [(D5, 3), (C5, 1), (A4, 2), (G4, 2), (A4, 4), (F4, 2), (G4, 2)]
END = [(F4, 2), (G4, 2), (A4, 4), (D4, 8)]
# 每章一句，(起始秒, 句子, 八度, 音量)
MEL = [(18, A, 0, 0.8), (34, A2, 0, 0.85), (50, B, 0, 0.9), (66, A, 0, 1.0),
       (82, B, 0, 1.0), (98, A, 0, 0.95), (114, END, 0, 0.8)]
for t0, ph_, o, g in MEL:
    s = bowed(ph_, t0, o)
    add(L, s, t0, g, -0.15)

# 拨弦：每拍或每两拍一个，音取自当前和弦 + 五声音阶
SCALE = [D4, F4, G4, A4, C5, D5]
for bar in range(32):
    sec = bar // 4
    dens = [1, 2, 2, 2, 4, 4, 2, 1][sec]          # 每小节几个音
    chord = CH[bar % 4]
    for k in range(dens):
        tb = bar * 4 + k * 4 / dens + (0.5 if sec == 0 else 0)
        m = chord[k % 3] + 12 if k % 2 == 0 else SCALE[(bar * 3 + k * 2) % len(SCALE)]
        add(P, pluck(m), tb, 0.55 + 0.15 * rng.random(), 0.35 * (-1) ** k)

# 手鼓：第 3 章起心跳式，第 5、6 章每拍
for beat in range(128):
    sec = beat // 16
    if sec in (2, 3, 6) and beat % 2 == 0:
        add(DR, drum(0.6 if beat % 4 else 0.9), beat, 1.0)
    elif sec in (4, 5):
        add(DR, drum(0.9 if beat % 2 == 0 else 0.5), beat, 1.0)
        if sec == 5 and beat % 2 == 1:
            add(DR, drum(0.35), beat + 0.5, 1.0)

# 驼铃：清代驼队那一章最多
for sec, cnt in [(5, 14), (6, 4), (7, 3)]:
    for _ in range(cnt):
        add(BG, bell(), sec * 16 + rng.uniform(0, 15), 1.0, rng.uniform(-0.6, 0.6))

# ---------------------------------------------------------------- 混音
def reverb(x, secs=3.2, wet=0.35):
    n = int(secs * SR)
    tt = np.arange(n) / SR
    out = np.zeros_like(x)
    for c in range(2):
        ir = rng.standard_normal(n) * np.exp(-tt / (secs / 6.9))
        ir = lp(ir, 4000)
        ir /= np.sqrt((ir ** 2).sum())
        out[c] = x[c] + wet * fftconvolve(x[c], ir)[:x.shape[1]]
    return out


mix = reverb(L * 1.0, wet=0.5) + reverb(P * 0.8, wet=0.45) + reverb(DR * 0.7, 2.0, 0.25) + reverb(BG, wet=0.3)
mix = mix[:, :int(LENGTH * SR)]
fade = np.clip((LENGTH - t[:mix.shape[1]]) / 4.0, 0, 1) * np.clip(t[:mix.shape[1]] / 2.0, 0, 1)
mix *= fade
mix = np.tanh(mix * 1.5) / 1.5
mix *= 0.9 / np.abs(mix).max()

wav = os.path.join(HERE, ".cache", "music.wav")
os.makedirs(os.path.dirname(wav), exist_ok=True)
wavfile.write(wav, SR, (mix.T * 32767).astype(np.int16))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", wav, "-b:a", "192k",
                os.path.join(HERE, "music.mp3")], check=True)

# 自检：每 8 秒的 RMS
rms = [np.sqrt((mix[:, int(a * SR):int((a + 8) * SR)] ** 2).mean()) for a in range(0, 128, 8)]
print("RMS/8s:", " ".join(f"{r:.3f}" for r in rms))
print(f"时长 {mix.shape[1] / SR:.2f}s，峰值 {np.abs(mix).max():.3f}")
