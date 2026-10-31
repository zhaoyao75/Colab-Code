"""
《星际回响》黑白版 · 抖音竖版（1080x1920，128 秒）。
复用 render_full.py 的光线追踪、亮度/转速曲线和音频脉冲：
每帧先按横版 1920x1080 计算黑洞，取中间 4:3 放大到 1080x810 放在屏幕中部，
上下用同风格星空补满；标题全程显示，文字避开抖音顶部栏、底部文案区和右侧按钮。

运行：python3 render_vertical.py --still 8 100   -> 预览图 vertical_preview.png
     python3 render_vertical.py                 -> stellar_echo_vertical_bw.mp4（4 进程并行）
"""
import multiprocessing as mp
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, zoom

import render_full as rf

rd = rf.rd
HERE = rf.HERE
VW, VH = 1080, 1920
BAND_H = 810                 # 黑洞区域高度（4:3 裁切放大后）
BAND_Y = 470                 # 黑洞区域顶部位置
FPS, DUR, NF = rf.FPS, rf.DUR, rf.NF
OUT = os.path.join(HERE, "stellar_echo_vertical_bw.mp4")
ZH = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
EN = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def star_canvas():
    """竖版全屏星空（灰度），风格与横版一致：幂律亮度、亮星光晕、淡银河。"""
    r = np.random.default_rng(5)
    sky = np.zeros((VH, VW), np.float32)
    k = 900
    ys, xs = r.integers(0, VH, k), r.integers(0, VW, k)
    lum = np.clip(0.08 * r.random(k) ** -1.2, 0, 12.0)
    sky[ys, xs] = lum
    core = gaussian_filter(sky, 0.8) * (2 * np.pi * 0.8 ** 2)
    halo = gaussian_filter(np.where(sky > 2.0, sky, 0), 3.0) * 0.6
    yy, xx = np.mgrid[0:VH, 0:VW] / VH
    band = np.exp(-(((xx * 0.55 - yy * 0.85) + 0.35) / 0.16) ** 2)
    band *= np.clip(0.55 + 0.25 * gaussian_filter(r.standard_normal((VH, VW)), 18) * 18, 0, None)
    hdr = core + halo + 0.03 * band
    return np.clip(1 - np.exp(-hdr * 1.25), 0, 1) ** (1 / 1.15)


def text_layer():
    layer = Image.new("RGBA", (VW, VH), (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)

    def spaced(text, font, y, spacing, fill):
        widths = [dr.textlength(c, font=font) for c in text]
        x = (VW - sum(widths) - spacing * (len(text) - 1)) / 2
        for c, w in zip(text, widths):
            dr.text((x, y), c, font=font, fill=fill)
            x += w + spacing

    spaced("星际回响", ImageFont.truetype(ZH, 132), 230, 30, (245, 245, 245, 255))
    spaced("STELLAR  ECHO", ImageFont.truetype(EN, 30), 392, 13, (190, 190, 190, 255))
    spaced("致敬《星际穿越》", ImageFont.truetype(ZH, 34), 1310, 6, (165, 165, 165, 255))
    # Claude 标志（保留橙色），居中放在副标题下方
    f = ImageFont.truetype(EN, 32)
    label_w = dr.textlength("Claude", font=f)
    R = 18
    cx = (VW - (2 * R + 16 + label_w)) / 2 + R
    cy = 1392
    lr = np.random.default_rng(11)
    for i in range(12):
        ang = i * 2 * np.pi / 12 + lr.uniform(-0.12, 0.12)
        ln = R * lr.uniform(0.75, 1.0)
        x2, y2 = cx + ln * np.cos(ang), cy + ln * np.sin(ang)
        dr.line([(cx, cy), (x2, y2)], fill=(217, 119, 87, 255), width=4)
        dr.ellipse([x2 - 2, y2 - 2, x2 + 2, y2 + 2], fill=(217, 119, 87, 255))
    dr.text((cx + R + 16, cy - 20), "Claude", font=f, fill=(232, 232, 232, 255))
    return np.asarray(layer).astype(np.float32) / 255


S = None


def hole(i):
    """第 i 帧黑洞画面（灰度，BAND_H x VW）。"""
    t = i / FPS
    D = rf.S
    tex = D["tex"]
    NP = tex.shape[1]
    bright = rf.interp(rf.BRIGHT, t) * (1 + 0.22 * D["pulse"][i])
    ph_t = D["phase"][i]
    RW, RH = rd.RW, rd.RH
    trans = np.ones(RW * RH, np.float32)
    acc = np.zeros(RW * RH, np.float32)
    for idx, ir, ph, prof, g, omega in D["layers"]:
        ip = (((ph + omega * ph_t) % (2 * np.pi)) / (2 * np.pi) * NP).astype(np.int32) % NP
        inten = prof * tex[ir, ip]
        alpha = np.clip(inten * 1.3, 0, 0.97)
        acc[idx] += g * inten * 2.2 * bright * trans[idx] * alpha
        trans[idx] *= 1 - alpha
    bgl = D["bg"] @ np.array([0.299, 0.587, 0.114], np.float32)
    hdr = (acc + bgl * trans).reshape(RH, RW)
    bloom = zoom(gaussian_filter(np.maximum(hdr[::4, ::4] - 0.6, 0), 10), 4, order=1)[:RH, :RW]
    hdr = hdr + (0.5 + 0.35 * max(bright - 1, 0)) * bloom
    ldr = np.clip(1 - np.exp(-hdr * 1.25), 0, 1) ** (1 / 1.15)
    z = 1.0 + 0.10 * rd.ramp(t, 0, DUR)             # 缓慢推近
    cw, ch = 1260 / z, 945 / z                      # 中间 4:3，放大约 15%
    x0, y0 = (RW - cw) / 2, (RH - ch) / 2
    im = Image.fromarray((ldr * 255).astype(np.uint8)).resize((VW, BAND_H), Image.LANCZOS,
                                                              box=(x0, y0, x0 + cw, y0 + ch))
    return np.asarray(im).astype(np.float32) / 255


def frame(i):
    t = i / FPS
    out = S["stars"].copy()
    band = hole(i)
    out[BAND_Y:BAND_Y + BAND_H] = out[BAND_Y:BAND_Y + BAND_H] * (1 - S["feather"]) + band * S["feather"]
    out *= rd.ramp(t, 0, 2.0)
    rgb = np.repeat(out[..., None], 3, axis=2)
    al = S["text"][..., 3:4] * rd.ramp(t, 0.8, 2.6)
    rgb = rgb * (1 - al) + S["text"][..., :3] * al
    rgb *= 1 - rd.ramp(t, DUR - 3, DUR)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def setup():
    global S
    rf.S = rf.setup()
    y = np.arange(BAND_H, dtype=np.float32)
    f = np.minimum(np.clip(y / 110, 0, 1), np.clip((BAND_H - 1 - y) / 110, 0, 1))
    f = (f * f * (3 - 2 * f))[:, None]
    S = dict(stars=star_canvas(), text=text_layer(), feather=f)


def worker(args):
    k, i0, i1 = args
    seg = os.path.join(HERE, f".seg_v{k}.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{VW}x{VH}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                          "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", seg],
                         stdin=subprocess.PIPE)
    for i in range(i0, i1):
        p.stdin.write(frame(i).tobytes())
        if (i - i0) % 150 == 0:
            print(f"  worker {k}: {i - i0}/{i1 - i0}", flush=True)
    p.stdin.close()
    p.wait()
    return seg


def main():
    setup()
    if "--still" in sys.argv:
        ts = [float(v) for v in sys.argv[sys.argv.index("--still") + 1:]]
        ims = [Image.fromarray(frame(min(int(t * FPS), NF - 1))).resize((540, 960)) for t in ts]
        sheet = Image.new("RGB", (540 * len(ims), 960))
        for n, im in enumerate(ims):
            sheet.paste(im, (n * 540, 0))
        sheet.save(os.path.join(HERE, "vertical_preview.png"))
        print("preview saved")
        return
    bounds = np.linspace(0, NF, 5).astype(int)
    with mp.get_context("fork").Pool(4) as pool:
        segs = pool.map(worker, [(k, bounds[k], bounds[k + 1]) for k in range(4)])
    lst = os.path.join(HERE, ".segs_v.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{s}'\n" for s in segs)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-i", rf.AUDIO_MP3, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", f"afade=t=out:st={DUR - 3}:d=3", "-c:a", "aac", "-b:a", "192k",
                    "-t", str(DUR), "-movflags", "+faststart", OUT], check=True)
    for s in segs + [lst]:
        os.remove(s)
    print("saved", OUT)


if __name__ == "__main__":
    main()
