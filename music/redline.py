"""
《红线》(Redline) —— 128 秒原创赛车配乐
风格参考：Hans Zimmer《F1：狂飙飞车》配乐的手法（脉冲合成器低音、十六分音符弦乐 ostinato、
重型鼓组、失真吉他、引擎呼啸、上升音效、最后一段升调）。主旋律为原创。

150 BPM，4/4 拍，每小节 1.6 秒，80 小节 = 128 秒；D 大调（高潮升到 E 大调）。
运行：python3 redline.py          ->  完整版 redline.wav / .mp3
     python3 redline.py demo     ->  16 秒 demo redline_demo.wav / .mp3（2 小节引入 + 8 小节副歌）
"""
import subprocess
import sys
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.io import wavfile

SR = 44100
DEMO = len(sys.argv) > 1 and sys.argv[1] == "demo"
OUT = "redline_demo" if DEMO else "redline"
BPM = 150
BEAT = 60 / BPM          # 0.4 s
BAR = 4 * BEAT           # 1.6 s
S16 = BEAT / 4           # 0.1 s
rng = np.random.default_rng(11)

NOTE = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7,
        "G#": 8, "A": 9, "A#": 10, "B": 11}


def m(name):
    return NOTE[name[:-1]] + 12 * (int(name[-1]) + 1)


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


# ---------------------------------------------------------------- 段落表
# 每 8 小节一段；demo 只取「引入 2 小节 + 副歌 8 小节」= 10 小节 = 16 秒
if DEMO:
    PLAN = [("lead_in", 2), ("chorus", 8)]
else:
    PLAN = [("intro", 8), ("verse", 8), ("verse2", 8), ("build", 8),
            ("chorus", 8), ("chorus2", 8), ("breakdown", 8), ("build2", 8),
            ("climax", 8), ("outro", 8)]
SECS = []
for name, n in PLAN:
    SECS += [(name, k) for k in range(n)]
BARS = len(SECS)
LENGTH = BARS * BAR
N = int(LENGTH * SR) + SR * 4
TRANS = {"climax": 2}                       # 高潮升一个全音


# ---------------------------------------------------------------- 合成工具
def env_adsr(n, att, dur, rel, t):
    e = np.clip(t / max(att, 1e-4), 0, 1)
    ri = int(dur * SR)
    if ri < n:
        e[ri:] *= np.exp(-(t[ri:] - dur) / (rel / 4))
    return e


def saw_lp(f, t, fc, max_h=60, phase=0.0):
    """限带锯齿波 + 随时间变化的低通（逐泛音加权，fc 可以是数组）"""
    sig = np.zeros_like(t)
    for h in range(1, max_h + 1):
        fh = f * h
        if fh > 14000:
            break
        w = 1.0 / (1 + (fh / fc) ** 4)
        sig += w * np.sin(2 * np.pi * fh * t + phase * h) / h
    return sig


def synth_bass(f, dur):
    n = int((dur + 0.06) * SR)
    t = np.arange(n) / SR
    fc = 180 + 1600 * np.exp(-t / 0.035)
    sig = saw_lp(f, t, fc, 40) + 0.6 * np.sin(2 * np.pi * f * t)
    return np.tanh(1.5 * sig) * env_adsr(n, 0.003, dur, 0.05, t)


def lead(f, dur):
    """亮的超级锯齿主奏（3 路失谐 + 滤波包络 + 延迟颤音）"""
    n = int((dur + 0.18) * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.006 * np.sin(2 * np.pi * 5.5 * t) * np.clip((t - 0.18) / 0.2, 0, 1)
    fc = 2200 + 4500 * np.exp(-t / 0.12)
    sig = np.zeros(n)
    for det in (0.994, 1.0, 1.006):
        tt = np.cumsum(vib) / SR
        sig += saw_lp(f * det, tt, fc, 30, rng.random() * 6.28)
    sig += 0.5 * np.sin(2 * np.pi * f * np.cumsum(vib) / SR)       # 加实音头
    return sig * env_adsr(n, 0.008, dur, 0.15, t) * (0.85 + 0.15 * np.exp(-t / 0.1)) / 2.2


def pluck(f, dur):
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    fc = 600 + 5000 * np.exp(-t / 0.05)
    return saw_lp(f, t, fc, 25) * np.exp(-t / 0.22) * np.clip(t / 0.002, 0, 1)


def pad(f, dur, att=0.3):
    n = int((dur + 0.4) * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for det in (0.995, 1.0, 1.005):
        sig += saw_lp(f * det, t, 900 + 700 * np.clip(t / dur, 0, 1), 20, rng.random() * 6.28)
    return sig * env_adsr(n, att, dur, 0.4, t) / 3


def spiccato(f):
    """短促弦乐（十六分音符 ostinato）"""
    n = int(0.16 * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for det in (0.997, 1.003):
        ph = f * det * t + rng.random()
        sig += 2 * (ph % 1) - 1
    return sig * np.clip(t / 0.006, 0, 1) * np.exp(-t / 0.045) / 2


def guitar(root, dur):
    """失真吉他强力和弦（根音 + 五度 + 八度），闷音"""
    n = int((dur + 0.05) * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for iv in (0, 7, 12):
        for det in (0.998, 1.002):
            ph = hz(root + iv) * det * t + rng.random()
            sig += 2 * (ph % 1) - 1
    sig = np.tanh(4 * sig)
    return sig * env_adsr(n, 0.002, dur, 0.04, t) * np.exp(-t / 0.12)


def braam(root, dur):
    n = int((dur + 0.8) * SR)
    t = np.arange(n) / SR
    fc = 150 + 1400 * np.clip(t / 0.25, 0, 1) * np.exp(-t / 1.2)
    sig = sum(saw_lp(hz(root + iv) * d, t, fc, 40, rng.random() * 6) for iv in (0, 7, 12)
              for d in (0.997, 1.003))
    return np.tanh(1.2 * sig) * env_adsr(n, 0.02, dur, 0.8, t)


def kick():
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    f = 48 + 110 * np.exp(-t / 0.03)
    sig = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.22)
    sig += 0.5 * rng.standard_normal(n) * np.exp(-t / 0.003)
    return np.tanh(1.8 * sig)


def snare():
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    noise = sosfilt(butter(2, [1500 / (SR / 2), 9000 / (SR / 2)], btype="band", output="sos"),
                    rng.standard_normal(n))
    tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.05)
    return 0.9 * noise * np.exp(-t / 0.09) + 0.7 * tone


def hat(open_=False):
    n = int((0.25 if open_ else 0.05) * SR)
    t = np.arange(n) / SR
    noise = sosfilt(butter(2, 7000 / (SR / 2), btype="high", output="sos"), rng.standard_normal(n))
    return noise * np.exp(-t / (0.08 if open_ else 0.012))


def tom(f):
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    ff = f * (1 + 0.6 * np.exp(-t / 0.04))
    sig = np.sin(2 * np.pi * np.cumsum(ff) / SR) * np.exp(-t / 0.25)
    return sig + 0.3 * rng.standard_normal(n) * np.exp(-t / 0.01)


def crash():
    n = int(2.2 * SR)
    t = np.arange(n) / SR
    noise = sosfilt(butter(2, 4000 / (SR / 2), btype="high", output="sos"), rng.standard_normal(n))
    return noise * np.exp(-t / 0.6) * np.clip(t / 0.002, 0, 1)


def boom():
    n = int(2.5 * SR)
    t = np.arange(n) / SR
    f = 38 + 60 * np.exp(-t / 0.1)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.8)


def riser(dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = t / dur
    noise = sosfilt(butter(2, 1200 / (SR / 2), btype="high", output="sos"), rng.standard_normal(n))
    f = 180 * 2 ** (3 * x)
    sweep = np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.5 * np.sin(4 * np.pi * np.cumsum(f) / SR)
    return (0.5 * noise + 0.35 * sweep) * x ** 2.2


def engine_pass(dur, f0=210):
    """赛车从左到右呼啸而过（多普勒）"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    u = (t - dur * 0.55) / 0.22
    f = f0 * (1 + 0.2 * np.tanh(-u)) * (1 + 0.15 * t / dur)
    ph = np.cumsum(f) / SR
    sig = sum(np.sin(2 * np.pi * h * ph) / h ** 0.8 for h in range(1, 14))
    sig = np.tanh(1.6 * sig) + 0.4 * rng.standard_normal(n)
    sig = sosfilt(butter(2, 5000 / (SR / 2), output="sos"), sig)
    amp = 1 / (1 + u ** 2) ** 0.8
    pan = np.tanh(u * 0.7)
    a = (pan + 1) * np.pi / 4
    return np.stack([sig * amp * np.cos(a), sig * amp * np.sin(a)])


# ---------------------------------------------------------------- 轨道 / 混音
tracks = {}
SEND = {"lead": .3, "pluck": .3, "pad": .4, "strings": .3, "guitar": .08, "bass": .02,
        "kick": .04, "snare": .25, "hat": .1, "tom": .2, "crash": .2, "boom": .25,
        "braam": .35, "fx": .3, "engine": .15}
PUMP = {"bass", "pad", "guitar", "strings", "braam"}   # 跟着底鼓「呼吸」的轨道
kick_times = []


def add(track, start, sig, gain=1.0, pan=0.0):
    if track not in tracks:
        tracks[track] = np.zeros((2, N))
    i = int(round(start * SR))
    if i >= N:
        return
    if sig.ndim == 2:
        j = min(N, i + sig.shape[1])
        tracks[track][:, i:j] += sig[:, : j - i] * gain
        return
    j = min(N, i + len(sig))
    a = (pan + 1) * np.pi / 4
    tracks[track][0, i:j] += sig[: j - i] * gain * np.cos(a)
    tracks[track][1, i:j] += sig[: j - i] * gain * np.sin(a)


# ---------------------------------------------------------------- 和声（D 大调）
CHORDS = {  # 低音, 和弦音
    "D":  (m("D2"), [m("D4"), m("F#4"), m("A4")]),
    "A":  (m("A1"), [m("C#4"), m("E4"), m("A4")]),
    "Bm": (m("B1"), [m("B3"), m("D4"), m("F#4")]),
    "G":  (m("G1"), [m("G3"), m("B3"), m("D4")]),
    "Em": (m("E2"), [m("G3"), m("B3"), m("E4")]),
}
HOOK_CH = ["D", "D", "A", "A", "Bm", "Bm", "G", "A"]      # 两小节一换和弦，不催
HOOK_CH_END = HOOK_CH[:7] + ["D"]
BUILD_CH = ["Bm", "G", "D", "A", "Bm", "G", "Em", "A"]

# ---------------------------------------------------------------- 主旋律（原创）
# 动机「哒 哒 哒——」放宽到两小节一句（以四分、二分音符为主），句尾留出换气。
# 下面伴奏仍是十六分音符飞奔，上面旋律舒展——快慢对比让它不着急。
HOOK = [
    [("F#5", 1), ("F#5", 1), ("A5", 2)],
    [("F#5", 1), ("E5", 2), ("D5", 1)],
    [("E5", 1), ("E5", 1), ("F#5", 1), ("E5", 1)],
    [("C#5", 3), (None, 1)],
    [("F#5", 1), ("F#5", 1), ("A5", 2)],
    [("B5", 1), ("A5", 2), ("F#5", 1)],
    [("G5", 3), ("F#5", 1)],
    [("E5", 3), (None, 1)],
]
HOOK_END = HOOK[:6] + [[("G5", 2), ("F#5", 1), ("E5", 1)], [("D5", 4)]]
OUTRO_TAIL = [  # 尾声后半：动机轻声回响一次，落回主音
    [("F#5", 1), ("F#5", 1), ("A5", 2)],
    [("F#5", 1), ("E5", 2), ("D5", 1)],
    [("D5", 4)],
    [(None, 4)],
]
BUILD_MEL = [
    [("B4", 1.5), ("D5", 1.5), ("F#5", 1)],
    [("G5", 3), ("F#5", 1)],
    [("F#5", 1.5), ("E5", 1.5), ("D5", 1)],
    [("E5", 4)],
    [("B4", 1.5), ("D5", 1.5), ("F#5", 1)],
    [("B5", 3), ("A5", 1)],
    [("G5", 1.5), ("F#5", 1.5), ("E5", 1)],
    [("A5", 2), ("C#6", 2)],
]


def chord_of(sec, k):
    last = sec == "outro" or (DEMO and sec == "chorus")
    if sec in ("build", "build2"):
        return BUILD_CH[k]
    if sec == "intro":
        return ["D", "D", "Bm", "Bm", "G", "G", "A", "A"][k]
    if sec == "lead_in":
        return "A"
    if sec == "outro":
        return HOOK_CH_END[4 + k] if k < 4 else "D"
    return (HOOK_CH_END if last else HOOK_CH)[k]


def bar_notes(phrase_bar, t0, tr):
    out, beat = [], 0
    for name, length in phrase_bar:
        if name:
            out.append((t0 + beat * BEAT, m(name) + tr, length * BEAT))
        beat += length
    return out


# ---------------------------------------------------------------- 编排
INTENS = {"lead_in": .6, "intro": .35, "verse": .5, "verse2": .7, "build": .75,
          "chorus": 1.0, "chorus2": 1.0, "breakdown": .35, "build2": .7, "climax": 1.1, "outro": .6}

for b, (sec, k) in enumerate(SECS):
    t0 = b * BAR
    tr = TRANS.get(sec, 0)
    ch = chord_of(sec, k)
    low, tones = CHORDS[ch]
    low += tr
    tones = [x + tr for x in tones]
    I = INTENS[sec]
    sec_len = dict(PLAN)[sec]
    last_bar = k == sec_len - 1
    big = sec in ("chorus", "chorus2", "climax")
    out_fade = max(0.0, 1 - k / 8) if sec == "outro" else 1.0

    # ---- 引擎呼啸 / 上升音效 / 冲击
    if (sec == "intro" and k in (0, 6)) or (sec == "lead_in" and k == 0) or (sec == "outro" and k == 5):
        add("engine", t0 - (0.2 if b else 0), engine_pass(2.6, 200 if k == 0 else 240), 0.5)
    if sec in ("intro", "build", "build2") and k == 6:
        add("fx", t0, riser(2 * BAR), 0.35)
    if sec == "lead_in" and k == 0:
        add("fx", t0, riser(2 * BAR), 0.3)
    if (k == 0 and sec in ("verse2", "chorus", "chorus2", "climax", "build2")) or (sec == "outro" and k == 0):
        add("crash", t0, crash(), 0.3, 0.2)
        add("boom", t0, boom(), 0.55)
        add("braam", t0, braam(low + 12, BAR * 1.8), 0.16)

    # ---- 脉冲低音（十六分音符，八度跳动）
    if sec not in ("breakdown",) or k >= 6:
        pat = [0, 0, 12, 0, 0, 12, 0, 12, 0, 0, 12, 0, 0, 12, 0, 12]
        acc = [1, .55, .8, .55] * 4
        bg = 0.3 * min(1, I + 0.25) * out_fade
        if sec == "intro" and k < 2:
            bg *= 0.5
        for s in range(16):
            add("bass", t0 + s * S16, synth_bass(hz(low + pat[s]), S16 * 0.8), bg * acc[s])
    else:
        add("bass", t0, synth_bass(hz(low), BAR * 0.95), 0.18)

    # ---- 弦乐 ostinato（十六分音符，和弦分解）
    if sec not in ("intro",) or k >= 4:
        r = tones[0]
        ext = [r, tones[1], tones[2], r + 12, tones[1] + 12, tones[2]]
        pat = [0, 3, 2, 3, 1, 3, 2, 4, 0, 3, 2, 3, 1, 5, 4, 3]
        sg = 0.18 * (0.6 + 0.4 * I) * out_fade
        for s in range(16):
            add("strings", t0 + s * S16, spiccato(hz(ext[pat[s]])), sg * (1.25 if s % 4 == 0 else .85),
                -0.35 + 0.7 * (pat[s] / 5))

    # ---- 铺底和弦
    pgain = {"intro": .1, "breakdown": .16}.get(sec, .08 if not big else .12) * out_fade
    for x in tones:
        add("pad", t0, pad(hz(x), BAR * 0.98, att=0.3 if big else 0.6), pgain, 0.4 if x == tones[0] else -0.4)

    # ---- 失真吉他（八分音符闷音 + 每小节首拍放开）
    if sec in ("verse2", "build", "chorus", "chorus2", "build2", "climax", "lead_in") and not (sec == "lead_in" and k == 0):
        gr = low + 12 if low + 12 < m("A2") + 3 else low
        for s in range(8):
            dur = BEAT * 0.45 if s else BEAT * 0.9
            gg = 0.09 * (1.3 if s == 0 else 0.8) * (1.1 if big else 0.8)
            add("guitar", t0 + s * BEAT / 2, guitar(gr, dur), gg, -0.55)
            add("guitar", t0 + s * BEAT / 2 + 0.012, guitar(gr, dur), gg, 0.55)

    # ---- 鼓组
    drums = sec not in ("intro", "breakdown", "lead_in") or (sec == "intro" and k >= 4) or (sec == "lead_in")
    if drums and out_fade > 0.2:
        g = I * out_fade
        if sec in ("build", "build2") and k < 6:
            kpat, spat = [0, 2], [1, 3]                        # 半速推进
        elif sec == "intro" or sec == "lead_in":
            kpat, spat = [0, 1, 2, 3], []
        else:
            kpat, spat = [0, 1.5, 2, 3.5] if big else [0, 1.5, 2], [1, 3]
        if sec == "lead_in" and k == 1:
            kpat, spat = [0, 1, 2, 3], []
        for x in kpat:
            add("kick", t0 + x * BEAT, kick(), 0.5 * min(1, g + 0.3))
            kick_times.append(t0 + x * BEAT)
        for x in spat:
            add("snare", t0 + x * BEAT, snare(), 0.45 * g, 0.05)
        for s in range(8 if not big else 16):
            step = BEAT / (2 if not big else 4)
            add("hat", t0 + s * step, hat(), (0.09 if s % 2 else 0.05) * g, 0.4)
        if big:
            add("hat", t0 + 3.5 * BEAT, hat(True), 0.1, 0.4)
    # 过门：段落最后一小节 / build 末两小节小军鼓滚奏
    if (sec in ("build", "build2") and k >= 6) or (sec == "lead_in" and k == 1):
        start = 0 if (sec != "lead_in" and k == 7) or sec == "lead_in" else 2
        for s in range(start * 4, 16):
            add("snare", t0 + s * S16, snare(), 0.12 + 0.3 * s / 16, 0.05)
    elif last_bar and sec in ("verse", "verse2", "chorus", "chorus2", "climax"):
        for s, f in zip(range(8, 16, 2), (180, 150, 120, 95)):
            add("tom", t0 + s * S16, tom(f), 0.45, -0.4 + 0.25 * (s - 8) / 2)

    # ---- 主旋律
    if sec in ("verse", "verse2", "chorus", "chorus2", "breakdown", "climax", "outro"):
        phrase = HOOK_END if (sec == "outro" or (DEMO and sec == "chorus")) else HOOK
        pb = phrase[k]
        if sec == "outro":
            pb = HOOK_END[4 + k] if k < 4 else OUTRO_TAIL[k - 4]
        if pb:
            for t, note, length in bar_notes(pb, t0, tr):
                f = hz(note)
                if sec == "verse":
                    add("pluck", t, pluck(f, length), 0.32, 0.1)
                    add("pad", t, pad(f, length * 0.95, 0.05), 0.07, -0.1)
                elif sec == "breakdown":
                    add("pluck", t, pluck(f, length), 0.3, 0.1)
                    add("pad", t, pad(f, length * 0.95, 0.05), 0.06, -0.1)
                elif sec == "outro":
                    add("pluck", t, pluck(f, length), 0.3 * out_fade + 0.05, 0.1)
                    add("pad", t, pad(f, length * 0.95, 0.05), 0.06 * out_fade + 0.02, -0.1)
                else:
                    gl = {"verse2": .3, "chorus": .36, "chorus2": .36, "climax": .38}[sec]
                    add("lead", t, lead(f, length * 0.92), gl, 0.0)
                    if big:
                        add("lead", t, lead(f / 2, length * 0.92), gl * 0.45, 0.0)
                        add("pluck", t, pluck(f * 2, length), 0.1, 0.3)
    if sec in ("build", "build2"):
        for t, note, length in bar_notes(BUILD_MEL[k], t0, tr):
            f = hz(note)
            add("lead" if sec == "build" else "pad", t,
                (lead(f, length * 0.95) if sec == "build" else pad(f, length * 0.95, 0.08)),
                0.2 if sec == "build" else 0.14, 0)

    if sec == "outro" and k == 7:     # 结尾重击
        pass
if not DEMO:
    t_end = (BARS - 4) * BAR
    add("boom", t_end, boom(), 0.6)
    add("braam", t_end, braam(m("D2"), BAR * 3), 0.2)
    add("crash", t_end, crash(), 0.3)
else:   # demo 结尾落在主和弦
    t_end = (BARS - 1) * BAR
    add("braam", t_end, braam(m("D2"), BAR * 0.9), 0.12)

# ---------------------------------------------------------------- 后期
for name, (kind, fc) in {"strings": ("low", 5500), "guitar": ("low", 3800), "pad": ("low", 4000),
                         "hat": ("high", 6000), "engine": ("high", 60)}.items():
    if name in tracks:
        tracks[name] = sosfilt(butter(2, fc / (SR / 2), btype=kind, output="sos"), tracks[name], axis=1)

# 附点八分延迟，让主奏更「宽」
if "lead" in tracks:
    d = int(0.75 * BEAT * SR)
    x = tracks["lead"].copy()
    y = x.copy()
    y[0, d:] += 0.3 * x[1, :-d]
    y[1, 2 * d:] += 0.18 * x[0, :-2 * d]
    tracks["lead"] = y

# 侧链：底鼓一响，低音/铺底让开，形成「泵」感
duck = np.ones(N)
tt = np.arange(int(0.3 * SR)) / SR
shape = 1 - 0.6 * np.exp(-tt / 0.07)
for kt in kick_times:
    i = int(kt * SR)
    j = min(N, i + len(shape))
    duck[i:j] = np.minimum(duck[i:j], shape[: j - i])
for name in PUMP & tracks.keys():
    tracks[name] *= duck

dry = sum(tracks.values())
send = sum(tracks[k] * SEND[k] for k in tracks)
ir_len = int(2.4 * SR)
ti = np.arange(ir_len) / SR
ir = rng.standard_normal((2, ir_len)) * np.exp(-ti / 0.45)
ir = sosfilt(butter(1, 5000 / (SR / 2), output="sos"), ir, axis=1)
ir[:, : int(0.015 * SR)] = 0
ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True))
wet = np.stack([fftconvolve(send[c], ir[c])[:N] for c in range(2)])

mix = dry + 0.8 * wet
mix = sosfilt(butter(2, 30 / (SR / 2), btype="high", output="sos"), mix, axis=1)
mix = mix[:, : int(LENGTH * SR)]
mix /= np.max(np.abs(mix)) + 1e-9
mix = np.tanh(1.8 * mix) / np.tanh(1.8)
fade = np.ones(mix.shape[1])
fade[: int(0.01 * SR)] = np.linspace(0, 1, int(0.01 * SR))
fl = int((0.6 if DEMO else 3.0) * SR)
fade[-fl:] = np.linspace(1, 0, fl) ** 2
mix *= fade * 0.89

wavfile.write(OUT + ".wav", SR, (mix.T * 32767).astype(np.int16))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", OUT + ".wav",
                "-codec:a", "libmp3lame", "-b:a", "256k",
                "-metadata", "title=Redline 红线" + (" (demo)" if DEMO else ""),
                "-metadata", "comment=Original racing theme inspired by the style of Hans Zimmer's F1 score",
                OUT + ".mp3"], check=True)
print("done: %.2f s" % (mix.shape[1] / SR))
for s in range(0, int(LENGTH), 2):
    seg = mix[:, s * SR:(s + 2) * SR]
    print("%3d-%3ds rms %.3f peak %.3f" % (s, s + 2, np.sqrt((seg ** 2).mean()), np.abs(seg).max()))
