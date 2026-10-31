"""
《星际回响》完整 128 秒黑白动画，配第一版（小调版）音乐。
复用 render_demo.py 的光线追踪、吸积盘纹理、星空和文字图层；
画面亮度、旋转速度随音乐段落变化，并跟随音频响度轻微脉动。

运行：python3 render_full.py            -> stellar_echo_full_bw.mp4（4 进程并行）
     python3 render_full.py --still 1 20 100 120   -> 输出几张静帧检查
"""
import multiprocessing as mp
import os
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
from scipy.io import wavfile

import render_demo as rd

rd.BW = True
HERE = rd.HERE
RW, RH, OW, OH, FPS = rd.RW, rd.RH, rd.OW, rd.OH, rd.FPS
DUR = 128.0
NF = int(DUR * FPS)
AUDIO_MP3 = os.path.join(HERE, "..", "music", "stellar_echo.mp3")
OUT = os.path.join(HERE, "stellar_echo_full_bw.mp4")
WORKERS = 4


def interp(keys, t):
    ks = np.array(keys, float)
    return float(np.interp(t, ks[:, 0], ks[:, 1]))


# 音乐段落：前奏 0-16 / 主旋律 16-48 / 变奏 48-64 / 管风琴主旋律 64-96 / 高潮 96-112 / 尾声 112-128
BRIGHT = [(0, .5), (16, .7), (32, .8), (48, .85), (64, 1.05), (96, 1.1), (98, 1.3),
          (112, 1.25), (120, .85), (128, .6)]
SPEED = [(0, .6), (16, .8), (48, .9), (64, 1.2), (96, 1.4), (112, 1.5), (128, .8)]


def audio_pulse():
    """每帧的音频瞬态强度（0~1），用来让画面随低鼓等重音轻微闪动"""
    wav = os.path.join(rd.HERE, "..", "music", "stellar_echo.wav")
    if not os.path.exists(wav):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", AUDIO_MP3, wav], check=True)
    sr, x = wavfile.read(wav)
    x = x.astype(np.float32).mean(axis=1) / 32768
    hop = sr // FPS
    n = min(NF, len(x) // hop)
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-9)
    rms = np.pad(rms, (0, NF - n), mode="edge")
    long = gaussian_filter(rms, FPS * 1.0)
    p = np.clip(rms / long - 1.0, 0, 1)
    env = np.zeros_like(p)
    for i in range(NF):   # 快起慢落
        prev = env[i - 1] if i else 0
        env[i] = p[i] if p[i] > prev else prev * 0.88
    return env


def setup():
    hr, hp, esc, edir = rd.trace()
    tex = rd.disk_texture()
    bg = rd.star_background(esc, edir)
    title, logo = rd.overlay_layers()
    NR, NP = tex.shape
    layers = []
    for k in range(hr.shape[0]):
        idx = np.nonzero(~np.isnan(hr[k]))[0]
        r = hr[k][idx]
        ir = np.clip((r - rd.RIN) / (rd.ROUT - rd.RIN) * (NR - 1), 0, NR - 1).astype(np.int32)
        x = np.clip((r - rd.RIN) / (rd.ROUT - rd.RIN), 0, 1)
        prof = (rd.RIN / np.maximum(r, rd.RIN)) ** 1.6
        prof *= np.clip((r - rd.RIN * 0.9) / (rd.RIN * 0.25), 0, 1) * np.clip((rd.ROUT * 1.05 - r) / 3.5, 0, 1)
        g = (1.0 - 0.25 * x).astype(np.float32)
        omega = 0.55 * (rd.RIN / np.maximum(r, rd.RIN)) ** 1.5
        layers.append((idx, ir, hp[k][idx], prof.astype(np.float32), g, omega))
    # 旋转相位 = 速度曲线的积分（速度变化时纹理不会跳）
    ts = np.arange(NF + 1) / FPS
    sp = np.array([interp(SPEED, t) for t in ts])
    phase = np.concatenate([[0], np.cumsum((sp[1:] + sp[:-1]) / 2 / FPS)])
    return dict(tex=tex, bg=bg.reshape(-1, 3), title=title, logo=logo, layers=layers,
                phase=phase, pulse=audio_pulse())


S = None


def frame(i):
    t = i / FPS
    tex, NP = S["tex"], S["tex"].shape[1]
    bright = interp(BRIGHT, t) * (1 + 0.22 * S["pulse"][i])
    ph_t = S["phase"][i]
    trans = np.ones(RW * RH, np.float32)
    acc = np.zeros(RW * RH, np.float32)
    for idx, ir, ph, prof, g, omega in S["layers"]:
        pr = (ph + omega * ph_t) % (2 * np.pi)
        ip = (pr / (2 * np.pi) * NP).astype(np.int32) % NP
        inten = prof * tex[ir, ip]
        alpha = np.clip(inten * 1.3, 0, 0.97)
        acc[idx] += g * inten * 2.2 * bright * trans[idx] * alpha
        trans[idx] *= 1 - alpha
    bgl = S["bg"] @ np.array([0.299, 0.587, 0.114], np.float32)
    hdr = (acc + bgl * trans).reshape(RH, RW)
    small = hdr[::4, ::4]
    bloom = zoom(gaussian_filter(np.maximum(small - 0.6, 0), 10), 4, order=1)[:RH, :RW]
    hdr = hdr + (0.5 + 0.35 * max(bright - 1, 0)) * bloom
    ldr = np.clip(1 - np.exp(-hdr * 1.25), 0, 1) ** (1 / 1.15)
    z = 1.0 + 0.14 * rd.ramp(t, 0, DUR)
    cw, ch = RW / z, RH / z
    x0, y0 = (RW - cw) / 2, (RH - ch) / 2
    im = Image.fromarray((ldr * 255).astype(np.uint8)).resize((OW, OH), Image.LANCZOS,
                                                              box=(x0, y0, x0 + cw, y0 + ch))
    lum = np.asarray(im).astype(np.float32) / 255 * rd.ramp(t, 0, 4.0)
    out = np.repeat(lum[..., None], 3, axis=2)
    title_a = rd.ramp(t, 3, 5) * (1 - rd.ramp(t, 13, 15)) + rd.ramp(t, 113, 115.5)
    for layer, a in ((S["title"], title_a), (S["logo"], rd.ramp(t, 4.5, 6))):
        al = layer[..., 3:4] * a
        out = out * (1 - al) + layer[..., :3] * al
    out *= 1 - rd.ramp(t, DUR - 3, DUR)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def worker(args):
    k, i0, i1 = args
    seg = os.path.join(HERE, f".seg_{k}.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                          "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", seg],
                         stdin=subprocess.PIPE)
    for i in range(i0, i1):
        p.stdin.write(frame(i).tobytes())
        if (i - i0) % 150 == 0:
            print(f"  worker {k}: {i - i0}/{i1 - i0}", flush=True)
    p.stdin.close()
    p.wait()
    return seg


def main():
    global S
    S = setup()
    if "--still" in sys.argv:
        ts = [float(v) for v in sys.argv[sys.argv.index("--still") + 1:]]
        ims = [Image.fromarray(frame(min(int(t * FPS), NF - 1))).resize((640, 360)) for t in ts]
        sheet = Image.new("RGB", (1280, 360 * ((len(ims) + 1) // 2)))
        for n, im in enumerate(ims):
            sheet.paste(im, ((n % 2) * 640, (n // 2) * 360))
        sheet.save(os.path.join(HERE, "full_bw_preview.png"))
        print("preview saved")
        return
    bounds = np.linspace(0, NF, WORKERS + 1).astype(int)
    with mp.get_context("fork").Pool(WORKERS) as pool:
        segs = pool.map(worker, [(k, bounds[k], bounds[k + 1]) for k in range(WORKERS)])
    lst = os.path.join(HERE, ".segs.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{s}'\n" for s in segs)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-i", AUDIO_MP3, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", f"afade=t=out:st={DUR - 3}:d=3", "-c:a", "aac", "-b:a", "192k",
                    "-t", str(DUR), "-movflags", "+faststart", OUT], check=True)
    for s in segs + [lst]:
        os.remove(s)
    print("saved", OUT)


if __name__ == "__main__":
    main()
