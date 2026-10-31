"""
《霓虹雨夜》(Neon Rain) —— 128 秒原创 Lo-fi 配乐
风格参考：lo-fi hip-hop（电钢琴七和弦、摇摆鼓点、黑胶噪声、磁带抖动、雨声）
主旋律为原创，只借鉴风格手法。

90 BPM，4/4 拍，每小节 8/3 秒，48 小节 = 128 秒。F 大调。
运行：python3 compose.py   ->  neon_rain.wav / neon_rain.mp3
"""
import subprocess
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.io import wavfile

SR = 44100
BPM = 90
BEAT = 60 / BPM          # 0.667 s
BAR = 4 * BEAT           # 2.667 s
LENGTH = 128.0
BARS = 48
SWING = 0.09             # 反拍八分音符推后的拍数（约 59% 摇摆）
N = int(LENGTH * SR) + SR * 4
OUT = "neon_rain"
rng = np.random.default_rng(2026)

NOTE = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6,
        "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}


def m(name):
    return NOTE[name[:-1]] + 12 * (int(name[-1]) + 1)


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def swing(beat):
    """把反拍八分音符推后一点，产生 lo-fi 的慵懒律动"""
    frac = beat % 1
    return beat + SWING if abs(frac - 0.5) < 1e-6 else beat


def human():
    return rng.normal(0, 0.006)   # 约 ±6ms 的人性化误差


def env_ar(n, att, dur, rel):
    t = np.arange(n) / SR
    e = np.clip(t / att, 0, 1)
    ri = int(dur * SR)
    e[ri:] *= np.exp(-(t[ri:] - dur) / (rel / 4))
    return e


# ---------------------------------------------------------------- 乐器
def epiano(f, dur, vel=1.0, rel=0.35):
    """FM 电钢琴（Rhodes 感）：调制指数随时间衰减 + 高频「金属齿」敲击声"""
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    idx = (0.4 + 1.6 * vel) * np.exp(-t / 0.35) + 0.25
    sig = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * t))
    sig += 0.12 * vel * np.sin(2 * np.pi * f * 14 * t) * np.exp(-t / 0.012)   # tine
    sig *= np.exp(-t / (2.2 * (260 / f) ** 0.4))
    sig *= 1 + 0.12 * np.sin(2 * np.pi * 4.5 * t)                               # 颤音
    return sig * env_ar(n, 0.002, dur, rel) * 0.5


def lead(f, dur, rel=0.25):
    """柔和的正弦 + 三角波主奏，慢颤音"""
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    vib = 0.005 * np.sin(2 * np.pi * 5.2 * t) * np.clip((t - 0.2) / 0.3, 0, 1)
    ph = np.cumsum(f * (1 + vib)) / SR
    tri = 2 * np.abs(2 * (ph % 1) - 1) - 1
    sig = 0.7 * np.sin(2 * np.pi * ph) + 0.3 * tri
    return sig * env_ar(n, 0.03, dur, rel) * 0.55


def flute(f, dur, att=0.06, rel=0.3):
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    vib = 0.004 * np.sin(2 * np.pi * 5 * t) * np.clip((t - 0.25) / 0.3, 0, 1)
    ph = 2 * np.pi * f * np.cumsum(1 + vib) / SR
    sig = np.sin(ph) + 0.3 * np.sin(2 * ph) + 0.06 * np.sin(3 * ph)
    breath = rng.standard_normal(n) * 0.03
    sig += breath * (0.4 + 0.6 * np.exp(-t / 0.1))
    return sig * env_ar(n, att, dur, rel) / 1.5


def pad(f, dur, att=0.8, rel=1.0):
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for det in (0.996, 1.0, 1.004):
        ph = f * det * t + rng.random()
        sig += 2 * (ph % 1) - 1
    return sig * env_ar(n, att, dur, rel) / 3


def bass(f, dur, rel=0.12):
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    sig = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t) + 0.1 * np.sin(6 * np.pi * f * t)
    sig = np.tanh(1.5 * sig) / np.tanh(1.5)
    e = env_ar(n, 0.008, dur, rel) * (0.75 + 0.25 * np.exp(-t / 0.2))
    return sig * e


def kick():
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    f = 48 + 85 * np.exp(-t / 0.035)
    sig = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.22)
    sig += 0.25 * rng.standard_normal(n) * np.exp(-t / 0.003)
    return np.tanh(1.6 * sig)


def snare():
    n = int(0.4 * SR)
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.05)
    noise = sosfilt(butter(2, [900 / (SR / 2), 5500 / (SR / 2)], btype="band", output="sos"),
                    rng.standard_normal(n)) * np.exp(-t / 0.11)
    return 0.6 * tone + 1.3 * noise


def hat(open_=False):
    n = int((0.3 if open_ else 0.08) * SR)
    t = np.arange(n) / SR
    noise = sosfilt(butter(2, 7000 / (SR / 2), btype="high", output="sos"), rng.standard_normal(n))
    return noise * np.exp(-t / (0.11 if open_ else 0.018))


# ---------------------------------------------------------------- 混音轨道
tracks = {}
SEND = {"ep": 0.3, "mel": 0.32, "lead": 0.35, "flute": 0.38, "pad": 0.45, "bass": 0.03,
        "kick": 0.04, "snare": 0.18, "hat": 0.06, "rain": 0.0, "vinyl": 0.0}


def add(track, start, sig, gain=1.0, pan=0.0):
    if track not in tracks:
        tracks[track] = np.zeros((2, N))
    i = max(0, int(start * SR))
    j = min(N, i + len(sig))
    sig = sig[: j - i] * gain
    a = (pan + 1) * np.pi / 4
    tracks[track][0, i:j] += sig * np.cos(a)
    tracks[track][1, i:j] += sig * np.sin(a)


# ---------------------------------------------------------------- 和声（爵士七和弦 / 九和弦）
CH = {   # 低音, 和弦音（无根音的紧凑排列）
    "Bbmaj7": ("Bb1", ["D4", "F4", "A4", "C5"]),
    "Am7":    ("A1",  ["C4", "E4", "G4", "B4"]),
    "Gm7":    ("G1",  ["Bb3", "D4", "F4", "A4"]),
    "C7":     ("C2",  ["Bb3", "D4", "E4", "A4"]),
    "Fmaj7":  ("F1",  ["A3", "C4", "E4", "G4"]),
    "Dm7":    ("D2",  ["C4", "E4", "F4", "A4"]),
}
LOOP = ["Bbmaj7", "Am7", "Gm7", "C7"]
A_PROG = ["Bbmaj7", "Am7", "Gm7", "C7", "Bbmaj7", "Am7", ("Gm7", "C7"), "Fmaj7"]
B_PROG = ["Dm7", "Bbmaj7", "Gm7", "C7"] * 2
PROG = LOOP * 2 + A_PROG * 2 + B_PROG + A_PROG + A_PROG
assert len(PROG) == BARS


def section(b):
    for name, lo, hi in [("intro", 0, 8), ("A1", 8, 16), ("A2", 16, 24), ("B", 24, 32),
                         ("A3", 32, 40), ("outro", 40, 48)]:
        if lo <= b < hi:
            return name


# ---------------------------------------------------------------- 旋律（原创）
THEME = [
    [("D5", .5), ("F5", .5), ("A5", 1), ("G5", .5), ("F5", 1.5)],
    [("E5", 1), ("C5", .5), ("D5", .5), ("E5", 2)],
    [("D5", .5), ("F5", .5), ("A5", 1), ("G5", .5), ("F5", .5), ("D5", 1)],
    [("E5", 1.5), ("G4", .5), ("C5", 2)],
    [("D5", .5), ("F5", .5), ("A5", 1), ("G5", .5), ("F5", 1.5)],
    [("E5", 1), ("G5", .5), ("A5", .5), ("C6", 2)],
    [("A5", .5), ("G5", .5), ("F5", 1), ("D5", .5), ("E5", 1.5)],
    [("F5", 3), (None, 1)],
]
B_THEME = [
    [("A5", 2), ("F5", 1), ("D5", 1)],
    [("F5", 1.5), ("D5", .5), ("C5", 2)],
    [("Bb4", 1), ("D5", 1), ("F5", 1), ("A5", 1)],
    [("G5", 3), (None, 1)],
    [("A5", 2), ("C6", 1), ("A5", 1)],
    [("F5", 1.5), ("G5", .5), ("A5", 2)],
    [("Bb5", 1), ("A5", 1), ("G5", 1), ("F5", 1)],
    [("E5", 2), ("G5", 2)],
]


def melody_notes(phrase, first_bar):
    out = []
    for k, bar in enumerate(phrase):
        beat = 0
        for name, length in bar:
            if name:
                st = swing(beat)
                out.append(((first_bar + k) * BAR + st * BEAT + human(), m(name), length * BEAT))
            beat += length
    return out


# ---------------------------------------------------------------- 编排
kick_times = []
for b in range(BARS):
    sec = section(b)
    t0 = b * BAR
    chords = PROG[b] if isinstance(PROG[b], tuple) else (PROG[b],)

    # 电钢琴和弦：第 1 拍 + 第 2 拍反拍（切分），略微分解
    eg = {"intro": .30, "A1": .30, "A2": .28, "B": .30, "A3": .27, "outro": .30}[sec]
    if sec == "intro":
        eg *= 0.75 + 0.25 * b / 7
    if sec == "outro" and b >= 44:
        eg *= max(0.35, 1 - (b - 44) / 5)
    for ci, ch in enumerate(chords):
        low, tones = CH[ch]
        span = 4 / len(chords)
        hits = [(0, 1.4, 1.0), (1.5, 2.4, 0.75)] if span == 4 else [(0, 1.9, 1.0)]
        if b == BARS - 1:
            hits = [(0, 3.6, 0.9)]
        for hb, hl, hv in hits:
            for k, nt in enumerate(tones):
                st = t0 + (ci * span + swing(hb)) * BEAT + k * 0.012 + human() * 0.5
                add("ep", st, epiano(hz(m(nt)), hl * BEAT, vel=0.55 * hv), eg * hv, -0.25 + 0.17 * k)

        # 低音：第 1 拍长音 + 第 3 拍前的切分
        if sec != "intro" or b >= 4:
            bg = {"intro": .3, "A1": .42, "A2": .46, "B": .46, "A3": .5, "outro": .4}[sec]
            if sec == "outro" and b >= 44:
                bg *= max(0.0, 1 - (b - 44) / 4)
            f = hz(m(low))
            if span == 4:
                add("bass", t0 + human(), bass(f, BEAT * 1.4), bg)
                add("bass", t0 + swing(1.5) * BEAT + human(), bass(f, BEAT * 0.4), bg * 0.6)
                add("bass", t0 + 2.5 * BEAT + SWING * BEAT, bass(f * (1.5 if b % 2 else 1), BEAT * 1.3), bg * 0.85)
            else:
                add("bass", t0 + ci * span * BEAT, bass(f, BEAT * 1.8), bg)

    # 铺底合成垫音
    if sec in ("A2", "B", "A3"):
        pg = {"A2": .07, "B": .1, "A3": .1}[sec]
        low, tones = CH[chords[0]]
        for nt in tones:
            add("pad", t0, pad(hz(m(nt)), BAR * 0.98), pg, 0)

    # 鼓：boom-bap 摇摆律动
    drums = (sec in ("A1", "A2", "B", "A3")) or (sec == "intro" and b >= 4) or (sec == "outro" and b < 44)
    if drums:
        lite = sec == "intro"
        kpos = [0, 1.75, 2.5] if b % 2 == 0 else [0, 2.5, 3.5]
        if lite:
            kpos = [0, 2.5]
        for kb in kpos:
            st = t0 + swing(kb) * BEAT + human()
            add("kick", st, kick(), 0.55 if kb == 0 else 0.45)
            kick_times.append(st)
        if not lite:
            for sb in (1, 3):
                add("snare", t0 + sb * BEAT + 0.012 + human(), snare(), 0.32)
            if b % 4 == 3:   # 每四小节加一个轻的鬼音
                add("snare", t0 + swing(3.5) * BEAT + human(), snare(), 0.1)
        for k in range(8):
            hb = k * 0.5
            open_ = sec == "A3" and k == 7 and b % 2 == 1
            v = (0.16 if k % 2 == 0 else 0.10) * (0.7 if lite else 1) * rng.uniform(0.8, 1.15)
            add("hat", t0 + swing(hb) * BEAT + human(), hat(open_), v, 0.3)
    # B 段最后两拍停顿，营造「回到主歌」的期待
    if b == 31:
        for tr in ("kick", "snare", "hat"):
            if tr in tracks:
                i0, i1 = int((t0 + 2 * BEAT) * SR), int((t0 + BAR) * SR)
                tracks[tr][:, i0:i1] *= np.linspace(1, 0, i1 - i0) ** 4

# 主旋律：前奏预告 4 小节 + A1/A2/A3 完整 + 尾声
for t, nt, ln in melody_notes(THEME[:4], 4):
    add("mel", t, epiano(hz(nt), ln * 0.95, vel=0.7), 0.22, 0.1)
for t, nt, ln in melody_notes(THEME, 8):
    add("mel", t, epiano(hz(nt), ln * 0.95, vel=0.9), 0.4, 0.1)
    add("mel", t, epiano(hz(nt + 12), ln * 0.6, vel=0.5), 0.1, 0.25)
for t, nt, ln in melody_notes(THEME, 16):
    add("lead", t, lead(hz(nt), ln * 0.92), 0.34, 0.05)
    add("mel", t, epiano(hz(nt), ln * 0.9, vel=0.7), 0.2, 0.15)
for t, nt, ln in melody_notes(B_THEME, 24):
    add("flute", t, flute(hz(nt), ln * 0.95), 0.32, 0.05)
for t, nt, ln in melody_notes(THEME, 32):
    add("lead", t, lead(hz(nt), ln * 0.92), 0.34, 0.05)
    add("mel", t, epiano(hz(nt + 12), ln * 0.7, vel=0.6), 0.14, 0.3)
    add("flute", t, flute(hz(nt - 12), ln * 0.9), 0.1, -0.2)
for t, nt, ln in melody_notes(THEME, 40):
    fade = max(0.4, 1 - max(0, t / BAR - 44) / 5)
    add("mel", t, epiano(hz(nt), ln * 0.95, vel=0.8), 0.38 * fade, 0.1)

# 雨声 + 黑胶噪声：贯穿全曲，前奏和尾声更明显
n_all = N
rain = rng.standard_normal((2, n_all))
rain = sosfilt(butter(2, [400 / (SR / 2), 5000 / (SR / 2)], btype="band", output="sos"), rain, axis=1)
drops = np.zeros((2, n_all))
for c in range(2):
    pos = rng.integers(0, n_all - 400, int(LENGTH * 25))
    drops[c, pos] = rng.uniform(0.2, 1, len(pos)) ** 3
drops = sosfilt(butter(2, 2500 / (SR / 2), btype="high", output="sos"), drops, axis=1)
tt = np.arange(n_all) / SR
rain_lv = np.interp(tt, [0, 4, 10, 21, 106, 120, 132], [0.0, .1, .1, .045, .045, .1, .1])
tracks["rain"] = (rain * 0.9 + drops * 4) * rain_lv

vinyl = np.zeros((2, n_all))
pos = rng.integers(0, n_all - 400, int(LENGTH * 7))
amp = rng.uniform(0.1, 1, len(pos)) ** 4
vinyl[0, pos] = amp
vinyl[1, pos] = amp * rng.uniform(0.6, 1, len(pos))
vinyl = sosfilt(butter(2, [800 / (SR / 2), 7000 / (SR / 2)], btype="band", output="sos"), vinyl, axis=1)
hiss = sosfilt(butter(2, 6000 / (SR / 2), output="sos"), rng.standard_normal((2, n_all)), axis=1)
tracks["vinyl"] = vinyl * 0.6 + hiss * 0.004

# ---------------------------------------------------------------- 后期
# 侧链：鼓点一响，和弦/垫音稍稍让位（lo-fi 的「呼吸感」）
duck = np.ones(N)
for kt in kick_times:
    i = int(kt * SR)
    n = int(0.35 * SR)
    j = min(N, i + n)
    duck[i:j] = np.minimum(duck[i:j], 1 - 0.3 * np.exp(-np.arange(j - i) / SR / 0.12))
for k in ("ep", "pad", "bass"):
    tracks[k] *= duck

# 各轨电平平衡（主旋律要压过伴奏）
for k, g in {"ep": 0.75, "bass": 0.62, "kick": 0.9, "hat": 1.35, "snare": 1.1,
             "mel": 1.6, "lead": 1.7, "flute": 1.5}.items():
    tracks[k] *= g

tracks["pad"] = sosfilt(butter(2, 1400 / (SR / 2), output="sos"), tracks["pad"], axis=1)
tracks["snare"] = sosfilt(butter(2, 6000 / (SR / 2), output="sos"), tracks["snare"], axis=1)
# 前奏电钢琴：低通滤波器慢慢打开
lp = sosfilt(butter(2, 900 / (SR / 2), output="sos"), tracks["ep"], axis=1)
open_ = np.clip(tt / (4 * BAR), 0, 1)[None, :]
tracks["ep"] = lp * (1 - open_) + tracks["ep"] * open_

music = [k for k in tracks if k not in ("rain", "vinyl")]
dry = sum(tracks[k] for k in music)
send = sum(tracks[k] * SEND[k] for k in music)
ir_len = int(2.0 * SR)
ti = np.arange(ir_len) / SR
ir = rng.standard_normal((2, ir_len)) * np.exp(-ti / 0.45)
ir = sosfilt(butter(1, 4000 / (SR / 2), output="sos"), ir, axis=1)
ir[:, : int(0.015 * SR)] = 0
ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True))
wet = np.stack([fftconvolve(send[c], ir[c])[:N] for c in range(2)])
mus = dry + 0.8 * wet

# 磁带抖动（wow & flutter）：缓慢的音高漂移
d = 0.0022 + 0.0016 * np.sin(2 * np.pi * 0.45 * tt) + 0.0002 * np.sin(2 * np.pi * 6.3 * tt)
src = np.arange(N) - d * SR
mus = np.stack([np.interp(src, np.arange(N), mus[c]) for c in range(2)])
# lo-fi 音色：高频滚降 + 低频收紧
mus = sosfilt(butter(2, 9500 / (SR / 2), output="sos"), mus, axis=1)
mus = sosfilt(butter(2, 32 / (SR / 2), btype="high", output="sos"), mus, axis=1)
# 尾声结尾也让「音乐」先淡出，雨声留到最后
mfade = np.interp(tt, [0, 124, 128.5], [1, 1, 0.0]) ** 1.5
mix = mus * mfade + tracks["rain"] + tracks["vinyl"]

mix = mix[:, : int(LENGTH * SR)]
mix /= np.max(np.abs(mix)) + 1e-9
mix = np.tanh(1.5 * mix) / np.tanh(1.5)
fade = np.ones(mix.shape[1])
fi = int(0.5 * SR)
fade[:fi] = np.linspace(0, 1, fi)
fl = int(3.0 * SR)
fade[-fl:] = np.linspace(1, 0, fl) ** 2
mix *= fade * 0.89

wavfile.write(OUT + ".wav", SR, (mix.T * 32767).astype(np.int16))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", OUT + ".wav",
                "-codec:a", "libmp3lame", "-b:a", "256k",
                "-metadata", "title=Neon Rain 霓虹雨夜",
                "-metadata", "comment=Original lo-fi theme, generated with code",
                OUT + ".mp3"], check=True)
print("done: %.2f s" % (mix.shape[1] / SR))
# 自检：每 8 秒的 RMS 响度
x = mix.mean(axis=0)
seg = int(8 * SR)
print("RMS/8s:", " ".join("%.3f" % np.sqrt((x[i:i + seg] ** 2).mean()) for i in range(0, len(x) - seg + 1, seg)))
print("peak: %.3f" % np.abs(mix).max())
for k in tracks:
    print("  %-6s rms %.4f" % (k, np.sqrt((tracks[k] ** 2).mean())))
