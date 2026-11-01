"""
《小燃 16 秒居家挑战》音频：Piper 离线中文旁白 + 纯 numpy 合成的 120BPM 背景音乐。
运行：python3 make_audio.py      -> .tmp/audio.wav（旁白+配乐，16 秒）、.tmp/timings.json、.tmp/voice_env.npy
      python3 make_audio.py --voice-only   -> 只出旁白试听 preview/narration_preview.mp3
"""
import json, os, subprocess, sys, wave
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(HERE, ".tmp"); os.makedirs(TMP, exist_ok=True)
MODEL = os.path.join(HERE, ".models", "zh_CN-huayan-medium.onnx")
SR, DUR = 44100, 16.0

# (开始秒, 旁白, 字幕, length_scale)  length_scale<1 更快
LINES = [
    (0.15, "久坐的你，敢接十六秒挑战吗？", "久坐的你，敢接16秒挑战吗？", 0.92),
    (3.05, "第一个，开合跳，跟上节奏！", "第一个，开合跳，跟上节奏！", 0.92),
    (7.05, "第二个，深蹲，背挺直。", "第二个，深蹲，背挺直", 0.95),
    (11.05, "第三个，高抬腿，加油！", "第三个，高抬腿，加油！", 0.92),
    (13.7, "量力而行，做完来评论区打卡！", "量力而行，做完来评论区打卡！", 0.8),
]


def read_wav(p):
    with wave.open(p) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        return x, w.getframerate()


def resample(x, sr0, sr1):
    n = int(len(x) * sr1 / sr0)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def synth_line(i, text, ls):
    out = os.path.join(TMP, f"line{i}.wav")
    subprocess.run([sys.executable, "-m", "piper", "-m", MODEL, "--length-scale", str(ls),
                    "--noise-scale", "0.5", "--noise-w-scale", "0.6", "-f", out],
                   input=text.encode(), check=True, capture_output=True)
    x, sr = read_wav(out)
    x = resample(x, sr, SR)
    nz = np.where(np.abs(x) > 0.01)[0]          # 去掉首尾静音
    x = x[nz[0]:nz[-1] + 1] if len(nz) else x
    return x / max(np.abs(x).max(), 1e-6) * 0.9


def voice_track():
    v = np.zeros(int(SR * DUR), np.float32); timings = []
    for i, (t0, text, sub, ls) in enumerate(LINES):
        x = synth_line(i, text, ls); a = int(t0 * SR); b = min(a + len(x), len(v))
        fade = np.minimum(1, np.minimum(np.arange(b - a), np.arange(b - a)[::-1]) / 300)
        v[a:b] += x[:b - a] * fade
        timings.append(dict(start=t0, end=round(t0 + len(x) / SR, 3), text=sub))
    return v, timings


def tone_env(n, dec):
    return np.exp(-np.arange(n) / SR * dec)


def music():
    m = np.zeros(int(SR * DUR), np.float32); beat = 0.5; rng = np.random.default_rng(3)
    roots = [48, 45, 41, 43, 48, 45, 41, 43]            # C A F G ...（MIDI）
    chords = [(0, 4, 7), (0, 3, 7), (0, 4, 7), (0, 4, 7)] * 2
    chords = [(0, 4, 7), (0, 3, 7), (0, 4, 7), (0, 4, 7)] * 2
    f = lambda n: 440 * 2 ** ((n - 69) / 12)

    def add(t, x, g=1.0):
        a = int(t * SR)
        if a < len(m): m[a:a + len(x)] += (x * g)[:len(m) - a]

    def kick():
        n = int(0.22 * SR); tt = np.arange(n) / SR
        return np.sin(2 * np.pi * (45 * tt + 75 * (1 - np.exp(-tt * 30)) / 30)) * tone_env(n, 14)

    def snare():
        n = int(0.16 * SR); return (rng.standard_normal(n) * 0.5 + np.sin(2 * np.pi * 190 * np.arange(n) / SR) * 0.4) * tone_env(n, 24)

    def hat():
        n = int(0.05 * SR); x = np.diff(rng.standard_normal(n + 1)); return x * tone_env(n, 70)

    def pluck(fr, d=0.25, dec=9):
        n = int(d * SR); tt = np.arange(n) / SR
        return (np.sin(2 * np.pi * fr * tt) + 0.4 * np.sin(4 * np.pi * fr * tt) + 0.2 * np.sin(6 * np.pi * fr * tt)) * tone_env(n, dec)

    for k in range(int(DUR / beat)):
        t = k * beat; bar = k // 4; r = roots[bar % 8]
        if bar >= 1: add(t, kick(), 0.9)                       # 第 1 小节只留铺垫
        if k % 2 == 1 and bar >= 1: add(t, snare(), 0.5)
        add(t, hat(), 0.35); add(t + beat / 2, hat(), 0.25)
        if bar >= 1:
            add(t, pluck(f(r), 0.45, 6), 0.5); add(t + beat / 2, pluck(f(r), 0.2, 12), 0.35)  # 低音
        for j, iv in enumerate((0, 4, 7, 12)):                 # 16 分琶音
            if bar >= 2 or (bar == 0 and k >= 4):
                add(t + j * beat / 4, pluck(f(r + 24 + iv), 0.18, 16), 0.16)
        if k % 2 == 1: add(t, sum(pluck(f(r + 12 + iv), 0.3, 10) for iv in chords[bar % 8]) / 3, 0.22)
    # 开头 riser（0~0.6s 的“扫入”）与结尾淡出
    n = int(0.6 * SR); m[:n] += (rng.standard_normal(n) * np.linspace(0, 0.08, n)).astype(np.float32)
    m *= np.minimum(1, np.linspace(DUR, 0, len(m)) / 0.8)
    return m / np.abs(m).max() * 0.8


def smooth_env(x, win=0.03):
    k = int(win * SR); return np.convolve(np.abs(x), np.ones(k) / k, "same")


def write_wav(p, x):
    with wave.open(p, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def main():
    v, timings = voice_track()
    # 旁白提亮 + 压缩（ffmpeg）
    write_wav(os.path.join(TMP, "voice_raw.wav"), v)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(TMP, "voice_raw.wav"), "-af",
                    "highpass=f=90,equalizer=f=3000:t=q:w=1:g=3,acompressor=threshold=-20dB:ratio=3:makeup=4,aecho=0.8:0.6:35:0.12",
                    os.path.join(TMP, "voice_fx.wav")], check=True)
    v, _ = read_wav(os.path.join(TMP, "voice_fx.wav")); v = np.pad(v, (0, max(0, int(SR * DUR) - len(v))))[:int(SR * DUR)]
    v = v / np.abs(v).max() * 0.92
    ve = smooth_env(v, 0.25); ve = np.clip(ve / ve.max() * 2.5, 0, 1)     # 闪避包络
    mus = music() * (1 - 0.62 * ve)                                       # 旁白出现时压低配乐
    mix = v + mus * 0.55
    mix = np.tanh(mix * 1.1) / np.tanh(1.1); mix = mix / np.abs(mix).max() * 0.92
    write_wav(os.path.join(TMP, "audio.wav"), mix)
    write_wav(os.path.join(TMP, "music.wav"), mus)
    write_wav(os.path.join(TMP, "voice.wav"), v)
    json.dump(timings, open(os.path.join(TMP, "timings.json"), "w"), ensure_ascii=False, indent=1)
    np.save(os.path.join(TMP, "voice_env.npy"), smooth_env(v, 0.04).astype(np.float32))
    for i, t in enumerate(timings):
        nxt = timings[i + 1]["start"] if i + 1 < len(timings) else DUR
        print(f"  {t['start']:5.2f}-{t['end']:5.2f}s  {t['text']}  {'OK' if t['end'] <= nxt - 0.05 else '!! 超过下一句开头'}")
    pk = np.abs(mix).max(); print("峰值", round(float(pk), 3), "时长", len(mix) / SR)
    os.makedirs(os.path.join(HERE, "preview"), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(TMP, "audio.wav"), "-b:a", "160k",
                    os.path.join(HERE, "preview", "audio_mix_preview.mp3")], check=True)


if __name__ == "__main__":
    main()
