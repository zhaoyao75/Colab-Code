"""旁白合成：逐句用 Edge 神经语音（云健）生成，按章节锚点排进 128 秒时间线。

输出：
  .cache/narration.wav   旁白整轨（44.1kHz 单声道，已 gitignore）
  narration.mp3          试听用
  timeline.json          每句字幕的起止时间、每章的起止时间（画面和混音都读它）
"""
import asyncio, json, os, ssl, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")
SR = 44100
TOTAL = 128.0
GAP = 0.35          # 句间停顿（秒）
CH_GAP = 0.9        # 章节间至少停顿

os.makedirs(CACHE, exist_ok=True)
S = json.load(open(os.path.join(HERE, "script.json"), encoding="utf-8"))


async def synth_all(force=False):
    import edge_tts, edge_tts.communicate as c
    # 会话代理会替换证书，需要显式信任代理 CA
    ca = "/root/.ccr/ca-bundle.crt"
    if os.path.exists(ca):
        c._SSL_CTX = ssl.create_default_context(cafile=ca)
    k = 0
    for ch in S["chapters"]:
        for zh, en in ch["lines"]:
            out = os.path.join(CACHE, f"line_{k:02d}.mp3")
            meta = out + ".txt"
            key = f"{S['voice']}|{S['rate']}|{zh}"
            if force or not os.path.exists(out) or open(meta).read() != key:
                await edge_tts.Communicate(zh, S["voice"], rate=S["rate"]).save(out)
                open(meta, "w").write(key)
            k += 1


def load(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le",
                          "-ac", "1", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32).copy()
    # 去掉首尾静音
    env = np.convolve(np.abs(x), np.ones(441) / 441, "same")
    idx = np.where(env > 0.01)[0]
    a, b = max(idx[0] - 600, 0), min(idx[-1] + 2500, len(x))
    return x[a:b]


def main():
    asyncio.run(synth_all("--force" in sys.argv))
    track = np.zeros(int(TOTAL * SR), np.float32)
    lines, chapters = [], []
    t, k = 0.0, 0
    for ci, ch in enumerate(S["chapters"]):
        t = max(ch["start"], t + (CH_GAP if ci else 0))
        c0 = t
        for zh, en in ch["lines"]:
            x = load(os.path.join(CACHE, f"line_{k:02d}.mp3"))
            d = len(x) / SR
            i = int(t * SR)
            track[i:i + len(x)] += x[:len(track) - i]
            lines.append({"zh": zh, "en": en, "t0": round(t, 3), "t1": round(t + d, 3), "ch": ci})
            t += d + GAP
            k += 1
        chapters.append({"id": ch["id"], "era_zh": ch["era_zh"], "era_en": ch["era_en"],
                         "voice0": round(c0, 3), "voice1": round(t - GAP, 3)})
    # 画面切换点：取两章旁白之间空隙的中点
    for ci, c in enumerate(chapters):
        c["t0"] = 0.0 if ci == 0 else round((chapters[ci - 1]["voice1"] + c["voice0"]) / 2, 3)
    for ci, c in enumerate(chapters):
        c["t1"] = TOTAL if ci == len(chapters) - 1 else chapters[ci + 1]["t0"]

    end = lines[-1]["t1"]
    print(f"旁白结束于 {end:.2f}s（总长 {TOTAL}s）")
    for c in chapters:
        print(f"  {c['id']:8s} 画面 {c['t0']:6.2f}–{c['t1']:6.2f}  旁白 {c['voice0']:6.2f}–{c['voice1']:6.2f}")
    if end > TOTAL - 4:
        print("警告：旁白太长，结尾留不出片尾")

    track /= max(np.abs(track).max(), 1e-6) / 0.89
    json.dump({"total": TOTAL, "lines": lines, "chapters": chapters},
              open(os.path.join(HERE, "timeline.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    wav = os.path.join(CACHE, "narration.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                    wav], input=track.tobytes(), check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", wav, "-b:a", "128k",
                    os.path.join(HERE, "narration.mp3")], check=True)


if __name__ == "__main__":
    main()
