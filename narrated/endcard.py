"""
微信点赞彩蛋：正片结束后的 5 秒片尾动画（头像 + 点赞 / 推荐 / 转发依次点亮 + 关注）
运行：
  python3 endcard.py --still 2.8          横版 + 竖版静帧 → stills/
  python3 endcard.py                      渲染 endcard_16x9.mp4、endcard_9x16.mp4（含编钟音效）
"""
import math
import subprocess
import sys
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.io import wavfile

import render as R

FPS, DUR, SR = 30, 5.0, 44100
BG = np.array([0.15, 0.15, 0.16], np.float32)
ORANGE = np.array([0.91, 0.36, 0.17], np.float32)          # 取自头像的橙红
WHITE = np.array([0.95, 0.94, 0.92], np.float32)
TAPS = [0.9, 1.7, 2.5]                                      # 三个按钮依次点亮
FOLLOW = 3.3


@lru_cache(None)
def avatar(d):
    im = Image.open("assets/avatar_src.png").convert("RGB")
    cx, cy, r = 627, 632, 560
    im = im.crop((cx - r, cy - r, cx + r, cy + r)).resize((d, d), Image.LANCZOS)
    m = Image.new("L", (d * 4, d * 4), 0)
    ImageDraw.Draw(m).ellipse((0, 0, d * 4 - 1, d * 4 - 1), fill=255)
    rgba = im.convert("RGBA")
    rgba.putalpha(m.resize((d, d), Image.LANCZOS))
    return R.Sprite(rgba)


def icon_mask(kind, s):
    """在 100×100 的网格里画图标，4 倍超采样后缩到 s"""
    k = 4 * s / 100
    im = Image.new("L", (4 * s, 4 * s), 0)
    d = ImageDraw.Draw(im)
    P = lambda pts: [(x * k, y * k) for x, y in pts]
    if kind == "like":      # 竖大拇指
        d.rounded_rectangle(P([(8, 44), (27, 92)])[0] + P([(27, 92)])[0], radius=4 * k, fill=255)
        d.polygon(P([(32, 46), (47, 20), (52, 9), (60, 9), (64, 18), (60, 40), (86, 40), (93, 48),
                     (86, 88), (80, 92), (32, 92)]), fill=255)
    elif kind == "heart":
        d.ellipse(P([(8, 16), (52, 60)])[0] + P([(52, 60)])[0], fill=255)
        d.ellipse(P([(48, 16), (92, 60)])[0] + P([(92, 60)])[0], fill=255)
        d.polygon(P([(11, 46), (89, 46), (50, 90)]), fill=255)
    elif kind == "share":   # 转发箭头
        d.polygon(P([(56, 10), (94, 44), (56, 78), (56, 58), (40, 58), (22, 64), (8, 86), (10, 62),
                     (22, 42), (40, 32), (56, 30)]), fill=255)
    return np.asarray(im.resize((s, s), Image.LANCZOS), np.float32) / 255


@lru_cache(None)
def icons(s):
    return {k: icon_mask(k, s) for k in ("like", "heart", "share")}


@lru_cache(None)
def texts(scale):
    f = lambda t, st, sz, col: R.text_sprite(t, st, int(sz * scale), fill=col, shadow=3)
    return dict(top=f("看到这里，点个赞吧", R.SERIF_BOLD, 64, (245, 238, 228)),
                labels=[f(t, R.SANS, 32, (220, 215, 210)) for t in ("点赞", "推荐", "转发")],
                follow=f("+ 关注", R.SANS_BOLD, 40, (255, 255, 255)),
                bottom=f("关注我，下一座城市见", R.SERIF_BOLD, 44, (230, 222, 210)))


def pop(t, t0):
    """点击弹跳：先压扁再放大回弹"""
    x = t - t0
    if x < 0:
        return 1.0
    return 1 + 0.35 * math.exp(-x * 7) * math.sin(x * 22) if x < 1.2 else 1.0


def burst(frame, cx, cy, t, t0, size, seed):
    """像素方块迸散（呼应头像里的像素块）"""
    x = t - t0
    if not 0 < x < 0.9:
        return
    rng = np.random.default_rng(seed)
    for _ in range(14):
        a = rng.uniform(0, 2 * math.pi)
        v = rng.uniform(0.9, 1.6) * size
        q = int(rng.uniform(0.08, 0.16) * size)
        px = cx + math.cos(a) * v * x ** 0.6
        py = cy + math.sin(a) * v * x ** 0.6
        al = max(0.0, 1 - x / 0.9)
        col = ORANGE if rng.random() < 0.7 else WHITE
        x0, y0 = int(px - q / 2), int(py - q / 2)
        if 0 <= x0 < frame.shape[1] - q and 0 <= y0 < frame.shape[0] - q:
            reg = frame[y0:y0 + q, x0:x0 + q]
            reg[:] = reg * (1 - al) + col * al


def put_icon(frame, mask, cx, cy, sc, col):
    s = mask.shape[0]
    n = max(4, int(s * sc))
    m = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((n, n), Image.LANCZOS), np.float32) / 255
    x0, y0 = int(cx - n / 2), int(cy - n / 2)
    reg = frame[y0:y0 + n, x0:x0 + n]
    reg[:] = reg * (1 - m[..., None]) + np.asarray(col, np.float32) * m[..., None]


def frame(t, W, H):
    vert = H > W
    sc = W / 1920 if not vert else W / 1080 * 0.95
    img = np.zeros((H, W, 3), np.float32) + BG
    yy = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
    xx = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
    img *= (1.15 - 0.35 * (xx ** 2 + yy ** 2))[..., None]
    tx = texts(round(sc, 3))
    fin = R.ease(t, 0.0, 0.6) * (1 - R.ease(t, DUR - 0.6, DUR))
    # 布局
    if vert:
        ay, top_y, icon_y, fol_y = H * 0.30, H * 0.17, H * 0.52, H * 0.66
        av = int(380 * sc)
    else:
        ay, top_y, icon_y, fol_y = H * 0.34, H * 0.11, H * 0.61, H * 0.79
        av = int(300 * sc)
    sp = avatar(av)
    ring = 1 + 0.03 * math.sin(t * 3)
    R.paste(img, R.glow_dot(av * 0.18 * ring, ORANGE), W / 2, ay, alpha=0.35 * fin, center=True, add=True)
    R.paste(img, sp, W / 2, ay, alpha=fin, center=True)
    R.paste(img, tx["top"], W / 2, top_y, alpha=fin * R.ease(t, 0.2, 0.8), center=True)
    isz = int(110 * sc)
    gap = int(260 * sc)
    for i, (kind, t0) in enumerate(zip(("like", "heart", "share"), TAPS)):
        cx = W / 2 + (i - 1) * gap
        on = R.ease(t, t0, t0 + 0.12)
        col = WHITE * (1 - on) + ORANGE * on
        a_in = R.ease(t, 0.3 + i * 0.1, 0.8 + i * 0.1) * fin
        tmp = img.copy()
        put_icon(tmp, icons(isz)[kind], cx, icon_y, pop(t, t0), col)
        img = img * (1 - a_in) + tmp * a_in
        burst(img, cx, icon_y, t, t0, isz * 1.2, seed=i + 3)
        R.paste(img, tx["labels"][i], cx, icon_y + isz * 0.9, alpha=a_in, center=True)
    # 关注按钮
    fa = R.ease(t, FOLLOW - 0.3, FOLLOW + 0.2) * fin
    if fa > 0:
        bw, bh = int(260 * sc), int(76 * sc)
        p = pop(t, FOLLOW + 0.4)
        bw2, bh2 = int(bw * p), int(bh * p)
        pill = Image.new("L", (bw2 * 4, bh2 * 4), 0)
        ImageDraw.Draw(pill).rounded_rectangle((0, 0, bw2 * 4 - 1, bh2 * 4 - 1), radius=bh2 * 2, fill=255)
        pm = np.asarray(pill.resize((bw2, bh2), Image.LANCZOS), np.float32) / 255
        x0, y0 = int(W / 2 - bw2 / 2), int(fol_y - bh2 / 2)
        reg = img[y0:y0 + bh2, x0:x0 + bw2]
        reg[:] = reg * (1 - (pm * fa)[..., None]) + ORANGE * (pm * fa)[..., None]
        R.paste(img, tx["follow"], W / 2, fol_y, alpha=fa, center=True)
        R.paste(img, tx["bottom"], W / 2, fol_y + bh * 1.25, alpha=fa, center=True)
    return R.to_u8(img)


def sound():
    n = int(DUR * SR)
    out = np.zeros(n, np.float32)
    notes = [587.33, 739.99, 880.0, 1174.66]          # D5 F#5 A5 D6，越点越亮
    for t0, f in zip(TAPS + [FOLLOW + 0.4], notes):
        tt = np.arange(int(2.5 * SR)) / SR
        b = sum(a * np.sin(2 * np.pi * f * r * tt) * np.exp(-tt / tau)
                for r, a, tau in [(1, 1.0, 0.9), (2.76, 0.35, 0.35), (5.4, 0.12, 0.15)])
        b *= np.clip(tt / 0.003, 0, 1)
        i = int(t0 * SR)
        out[i:i + len(b)] += b[: n - i] * 0.25
    t = np.arange(n) / SR
    out *= np.clip((DUR - t) / 0.6, 0, 1)
    out /= np.abs(out).max() / 0.6
    return out


def render_clip(W, H, path):
    wavfile.write(".endcard.wav", SR, (np.repeat(sound()[:, None], 2, 1) * 32767).astype(np.int16))
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-i", ".endcard.wav", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(int(DUR * FPS)):
        p.stdin.write(frame(i / FPS, W, H).tobytes())
    p.stdin.close()
    p.wait()
    print("完成：", path)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--still":
        for ts in a[1:]:
            Image.fromarray(frame(float(ts), 1920, 1080)).save(f"stills/egg_h_{float(ts):.1f}.png")
            Image.fromarray(frame(float(ts), 1080, 1920)).save(f"stills/egg_v_{float(ts):.1f}.png")
            print("stills/egg", ts)
    else:
        render_clip(1920, 1080, "endcard_16x9.mp4")
        render_clip(1080, 1920, "endcard_9x16.mp4")
