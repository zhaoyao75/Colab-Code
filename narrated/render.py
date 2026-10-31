"""
《天地之中 · 郑州》画面渲染 —— 9 个场景全部程序生成（numpy + Pillow），1920×1080 @30fps

依赖 narrate.py 生成的 timeline.json（场景与字幕时间）和 mix.py 生成的 mix.wav。
运行：
  python3 render.py --still 6 20 37 ...   导出静帧到 stills/
  python3 render.py --sheet               每个场景取一帧，拼成一张预览图 preview_sheet.png
  python3 render.py --demo 28 40          渲染 28–40 秒的小样 demo.mp4（含声音）
  python3 render.py                       完整版 zhengzhou_full.mp4（4 进程并行）
"""
import json
import math
import os
import subprocess
import sys
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from script import TITLE_EN, TITLE_ZH

W, H, FPS = 1920, 1080, 30
TL = json.load(open("timeline.json"))
LENGTH = TL["length"]
SC = TL["scenes"]
SUBS = TL["subs"]
XF = 0.9                       # 场景交叉淡化时长
INK = np.array([0.11, 0.10, 0.10], np.float32)
RED = np.array([0.70, 0.17, 0.13], np.float32)
GOLD = np.array([1.00, 0.80, 0.45], np.float32)


# ================================================================ 通用工具
def font(style, size):
    f = subprocess.run(["fc-match", "-f", "%{file}:%{index}", style],
                       capture_output=True, text=True).stdout
    path, idx = f.rsplit(":", 1)
    return ImageFont.truetype(path, size, index=int(idx))


SERIF_BLACK = "Noto Serif CJK SC:style=Black"
SERIF_BOLD = "Noto Serif CJK SC:style=Bold"
SANS_BOLD = "Noto Sans CJK SC:style=Bold"
SANS = "Noto Sans CJK SC:style=Regular"


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def ease(t, a, b):
    return float(smooth((t - a) / (b - a)))


def fade(t, a, b, fi=0.5, fo=0.5):
    """a~b 之间显示，两端淡入淡出"""
    return min(ease(t, a, a + fi), 1 - ease(t, b - fo, b))


def noise2d(h, w, cell, seed, octaves=4):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        c = max(1, int(cell / 2 ** o))
        sh = (h // c + 3, w // c + 3)
        small = Image.fromarray(rng.random(sh).astype(np.float32), mode="F")
        big = np.asarray(small.resize((sh[1] * c, sh[0] * c), Image.BICUBIC))
        out += amp * big[c:c + h, c:c + w]
        tot += amp
        amp *= 0.5
    return out / tot


def ridge(n, seed, octaves=7, base=3, rough=0.55):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    y = np.zeros(n)
    amp = 1.0
    for o in range(octaves):
        k = base * 2 ** o
        y += amp * np.interp(x * k, np.arange(k + 2), rng.random(k + 2))
        amp *= rough
    return (y - y.min()) / (y.max() - y.min())


def comp(img, col, a):
    a = a[..., None]
    img *= 1 - a
    img += np.asarray(col, np.float32) * a


def vgrad(h, w, top, bottom, power=1.0):
    v = (np.linspace(0, 1, h, dtype=np.float32) ** power)[:, None, None]
    img = np.asarray(top, np.float32) * (1 - v) + np.asarray(bottom, np.float32) * v
    return np.repeat(img, w, axis=1)


def paper(h, w, seed=0, tone=(0.93, 0.89, 0.80), vig=0.35):
    n = noise2d(h, w, 160, seed, 5)
    f = noise2d(h, w, 4, seed + 1, 2)
    img = np.asarray(tone, np.float32)[None, None] * (0.93 + 0.09 * n[..., None]) \
        * (0.975 + 0.035 * f[..., None])
    yy = np.linspace(-0.5, 0.5, h, dtype=np.float32)[:, None]
    xx = np.linspace(-0.5, 0.5, w, dtype=np.float32)[None, :]
    img *= (1 - vig * (xx ** 2 + yy ** 2) * 1.6)[..., None]
    return img.astype(np.float32)


def ink_mountains(img, layers, seed, col=INK, carve=None):
    """layers: (基线, 振幅, 浓度, 雾化长度, 粗糙度, 包络函数或 None)"""
    h, w, _ = img.shape
    Y = np.arange(h, dtype=np.float32)[:, None]
    u = np.linspace(0, 1, w)
    for i, (base, amp, dark, fl, rough, env) in enumerate(layers):
        r = ridge(w, seed + i, rough=rough)
        if env is not None:
            r = r * env(u)
        top = (base - amp * r)[None, :] * h
        if carve is not None:   # 切出 V 形山谷：(谷底横坐标, 谷底高度, 谷壁斜率)
            us, yb, slope = carve
            jag = (ridge(w, seed + 70 + i, rough=0.6) - 0.5) * 0.07 * h * np.clip(np.abs(u - us) * 12, 0, 1)
            top = np.maximum(top, (yb * h - slope * np.abs(u - us) * w + jag)[None, :])
        below = Y - top
        a = np.exp(-np.maximum(below, 0) / (fl * h)) * np.clip(below / 2.5, 0, 1)
        a *= dark * (0.7 + 0.6 * noise2d(h, w, 50, seed + 20 + i, 3))
        comp(img, col, np.clip(a, 0, 1))
    return img


def to_u8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def kb(big, s, cx=0.5, cy=0.5):
    """Ken Burns：big 为 PIL 图，s ≥ 1 为放大倍数，cx/cy 为视窗中心（相对）"""
    bw, bh = big.size
    vw, vh = bw / s, bh / s
    x0 = np.clip(cx * bw - vw / 2, 0, bw - vw)
    y0 = np.clip(cy * bh - vh / 2, 0, bh - vh)
    out = big.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + vw, y0 + vh))
    return np.asarray(out, np.float32) / 255


class Sprite:
    def __init__(self, rgba):
        a = np.asarray(rgba, np.float32) / 255
        self.rgb, self.a = a[..., :3].copy(), a[..., 3].copy()
        self.h, self.w = self.a.shape


def text_sprite(txt, style, size, fill=(255, 255, 255), stroke=0,
                stroke_fill=(0, 0, 0, 200), shadow=0):
    f = font(style, size)
    box = f.getbbox(txt, stroke_width=stroke)
    pad = stroke + shadow * 3 + 4
    w, h = box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if shadow:
        sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(sh).text((pad - box[0] + 2, pad - box[1] + 3), txt, font=f,
                                fill=(0, 0, 0, 160), stroke_width=stroke,
                                stroke_fill=(0, 0, 0, 160))
        im = Image.alpha_composite(im, sh.filter(ImageFilter.GaussianBlur(shadow)))
    d = ImageDraw.Draw(im)
    d.text((pad - box[0], pad - box[1]), txt, font=f, fill=tuple(fill) + (255,)
           if len(fill) == 3 else fill, stroke_width=stroke, stroke_fill=stroke_fill)
    return Sprite(im)


def vtext_sprite(txt, style, size, fill, gap=0.08):
    f = font(style, size)
    step = int(size * (1 + gap))
    im = Image.new("RGBA", (int(size * 1.3), step * len(txt) + size // 3), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i, ch in enumerate(txt):
        d.text((im.width // 2, i * step + size // 2), ch, font=f, fill=fill, anchor="mm")
    return Sprite(im)


def seal_sprite(txt, size=90, vertical=True):
    """朱文印：红底白字，边缘做旧"""
    n = len(txt)
    if vertical:
        w, h = size, int(size * (0.15 + 0.85 * n))
    else:
        w, h = int(size * (0.15 + 0.85 * n)), size
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=size // 12,
                        fill=tuple(int(c * 255) for c in RED) + (255,))
    f = font(SERIF_BLACK, int(size * 0.62))
    for i, ch in enumerate(txt):
        if vertical:
            xy = (w / 2, size * (0.075 + 0.85 * i + 0.425))
        else:
            xy = (size * (0.075 + 0.85 * i + 0.425), h / 2)
        d.text(xy, ch, font=f, fill=(250, 240, 230, 255), anchor="mm")
    a = np.asarray(im, np.float32)
    rough = noise2d(h, w, 6, n * 7 + size, 2)
    a[..., 3] *= np.clip((rough - 0.18) * 4, 0, 1)
    return Sprite(Image.fromarray(a.astype(np.uint8)))


def paste(img, sp, x, y, alpha=1.0, center=False, add=False):
    if alpha <= 0.001:
        return
    if center:
        x, y = x - sp.w / 2, y - sp.h / 2
    x, y = int(round(x)), int(round(y))
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + sp.w, img.shape[1]), min(y + sp.h, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    a = sp.a[y0 - y:y1 - y, x0 - x:x1 - x, None] * alpha
    rgb = sp.rgb[y0 - y:y1 - y, x0 - x:x1 - x]
    if add:
        img[y0:y1, x0:x1] += rgb * a
    else:
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + rgb * a


def subs_of(k):
    return [s for s in SUBS if s["scene"] == k]


def glow_dot(r, col):
    s = int(r * 6)
    yy, xx = np.mgrid[-s:s + 1, -s:s + 1]
    d2 = (xx ** 2 + yy ** 2) / r ** 2
    a = np.clip(np.exp(-d2) + 0.25 * np.exp(-d2 / 8), 0, 1)
    sp = Sprite.__new__(Sprite)
    sp.rgb = np.ones(a.shape + (3,), np.float32) * np.asarray(col, np.float32)
    sp.a = a.astype(np.float32)
    sp.h, sp.w = a.shape
    return sp


# ================================================================ 1. 黄河出峡谷
@lru_cache(None)
def build_river():
    BW, BH = int(W * 1.08), int(H * 1.08)
    img = paper(BH, BW, seed=1)
    yy, xx = np.mgrid[0:BH, 0:BW].astype(np.float32)
    d = np.hypot(xx - BW * 0.84, yy - BH * 0.30)
    comp(img, RED * 1.1, 0.5 * smooth((66 - d) / 3))
    ink_mountains(img, [(0.47, 0.20, 0.20, 0.10, 0.55, None),
                        (0.54, 0.17, 0.30, 0.08, 0.6, None)], seed=11, carve=(0.17, 0.60, 0.75))
    # 河道：从峡谷口（画面中部）蜿蜒流向镜头，越近越宽
    y0 = BH * 0.585
    p = np.clip((yy[:, :1] - y0) / (BH - y0), 0, 1)
    q = p ** 1.5   # 源头附近先顺着谷底往下流，再开始拐弯
    xc = BW * (0.17 + 0.32 * q + 0.17 * (0.5 + 0.5 * q) * np.sin(q * 2 * np.pi * 1.1))
    wd = 3 + BW * 0.24 * p ** 1.35
    dd = (xx - xc) / wd
    lim = 1 + 0.28 * (noise2d(BH, BW, 36, 12, 3) - 0.5)          # 不规则的岸线
    soft = np.maximum(1.5, 0.12 * wd)
    m = np.clip((lim - np.abs(dd)) * wd / soft, 0, 1) * smooth((yy - y0 + 5) / 40)
    ry, rx = np.nonzero(m > 0.002)
    bank = np.exp(-(((lim - np.abs(dd)) * wd) / (2 + 0.025 * wd)) ** 2)  # 岸边一道淡墨
    # 沿河"宽度单位"累计的路程：波纹在河面上等距，透视下近大远小
    srow = np.cumsum(1.0 / wd[:, 0]) * (np.arange(BH) > y0)
    # 近景峡谷：左右两山夹住河口
    gorge = lambda u: 0.06 + 0.9 * np.exp(-((u + 0.01) / 0.10) ** 2) + 0.8 * np.exp(-((u - 0.50) / 0.22) ** 2)
    tmp = np.ones((BH, BW, 3), np.float32)
    ink_mountains(tmp, [(0.66, 0.33, 0.92, 0.07, 0.6, gorge)], seed=31, col=(0, 0, 0), carve=(0.17, 0.63, 1.0))
    fga = 1 - tmp[..., 0]
    fga_r = fga * (1 - smooth((yy - y0 - 40) / 70))   # 河口在山后，流出后在山前
    under = img[ry, rx].copy()
    comp(img, INK, fga)
    mist = 0.85 * np.exp(-((yy - y0 + 10) / 40) ** 2) * np.exp(-((xx - BW * 0.20) / (BW * 0.28)) ** 2)
    comp(img, (0.93, 0.90, 0.83), mist)
    river = dict(ry=ry, rx=rx, m=m[ry, rx], dd=dd[ry, rx], s=srow[ry], under=under,
                 bank=bank[ry, rx], fga=fga_r[ry, rx], mist=mist[ry, rx],
                 n0=noise2d(BH, BW, 60, 13, 3)[ry, rx], nm=noise2d(BH, BW, 22, 14, 3)[ry, rx])
    return img, river


def scene_river(t, tl):
    img, rv = build_river()
    frame = img.copy()
    tint = np.array([0.89, 0.79, 0.62], np.float32)
    c = rv["under"] * tint * (1 - 0.35 * rv["bank"])[:, None]
    arg = 2 * np.pi * (rv["s"] * 1.5 - tl * 0.6) + 1.6 * np.sin(rv["dd"] * 2.4 + rv["s"] * 0.5) + 2 * rv["n0"]
    crest = np.clip((np.cos(arg) - 0.90) * 9, 0, 1) * np.clip((rv["nm"] - 0.35) * 3, 0, 1)
    crest *= 1 - np.abs(rv["dd"]) ** 4
    c = c * (1 - 0.55 * crest[:, None]) + np.array([0.97, 0.94, 0.86]) * 0.55 * crest[:, None]
    m = rv["m"][:, None]
    px = rv["under"] * (1 - m) + c * m
    fa = rv["fga"][:, None]
    px = px * (1 - fa) + INK * fa
    mi = rv["mist"][:, None]
    frame[rv["ry"], rv["rx"]] = px * (1 - mi) + np.array([0.93, 0.90, 0.83]) * mi
    big = Image.fromarray(to_u8(frame))
    return kb(big, 1.0 + 0.06 * tl / 14, 0.48, 0.50)


# ================================================================ 2. 史前：彩陶与粟米
@lru_cache(None)
def build_prehistory():
    img = vgrad(H, W, (0.88, 0.74, 0.52), (0.60, 0.40, 0.24), 1.3)
    img *= (0.94 + 0.08 * noise2d(H, W, 140, 2, 5))[..., None]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = W * 0.5, H * 0.52
    ang = np.arctan2(yy - cy, xx - cx)
    rad = np.hypot(xx - cx, yy - cy)
    comp(img, (0.98, 0.86, 0.60), 0.55 * smooth((250 - rad) / 36))
    # 地面
    comp(img, (0.33, 0.20, 0.11), 0.55 * smooth((yy - H * 0.80) / 60))
    # 陶罐
    top, hgt, R = 405, 405, 192
    vv = np.linspace(0, 1, hgt)
    rr = np.interp(vv, [0, .04, .10, .20, .40, .56, .75, .93, 1],
                   [.42, .46, .37, .56, 1, .95, .70, .47, .45])
    rr = gaussian_filter1d(rr, 8, mode="nearest") * R
    drr = np.gradient(rr)
    # 地面阴影
    sh = np.exp(-(((xx - cx - 30) / 230) ** 2 + ((yy - top - hgt) / 22) ** 2) * 2)
    comp(img, (0.15, 0.08, 0.04), 0.5 * sh)
    X = np.arange(int(cx - R - 3), int(cx + R + 4))
    dx = X[None, :] - cx
    a = np.clip(rr[:, None] - np.abs(dx) + 0.5, 0, 1)
    jy, jx = np.nonzero(a > 0)
    th = np.arcsin(np.clip(dx[0, jx] / rr[jy], -1, 1))
    lam = 0.28 + 0.72 * np.clip(np.cos(th + 0.6), 0, 1) ** 0.8
    lam *= np.clip(1 + 0.5 * drr[jy], 0.55, 1.15)
    spec = 0.18 * np.exp(-((th + 0.55) / 0.12) ** 2) * np.clip(1 + drr[jy], 0, 1)
    # 纹样贴图（一圈 1600 像素）
    TW, HT = 1600, 600
    tex = Image.new("RGB", (TW, HT), (182, 98, 56))
    d = ImageDraw.Draw(tex)
    dark, white = (52, 30, 22), (236, 214, 180)
    for y in (int(HT * .19), int(HT * .235), int(HT * .455), int(HT * .49), int(HT * .64)):
        d.rectangle((0, y, TW, y + 7), fill=dark)
    for k in range(7):
        for wrap in (-TW, 0, TW):
            x = wrap + k * TW / 7 + 60
            y = HT * 0.345
            d.ellipse((x - 46, y - 46, x + 46, y + 46), fill=dark)
            d.ellipse((x - 30, y - 30, x + 30, y + 30), fill=(196, 72, 40))
            d.ellipse((x - 12, y - 12, x + 12, y + 12), fill=white)
            for j in range(16):
                aa = j * math.pi / 8
                d.line((x + 54 * math.cos(aa), y + 54 * math.sin(aa),
                        x + 72 * math.cos(aa), y + 72 * math.sin(aa)), fill=dark, width=6)
    for row, amp, col in ((0.55, 16, dark), (0.585, 12, white)):
        pts = [(x, HT * row + amp * math.sin(x / TW * 2 * math.pi * 14)) for x in range(0, TW + 1, 4)]
        d.line(pts, fill=col, width=7)
    for k in range(28):
        x = k * TW / 28
        d.polygon([(x, HT * .66), (x + TW / 56, HT * .72), (x + TW / 28, HT * .66)], fill=dark)
    TW = int(TW * R / 285)
    tex = tex.resize((TW, hgt), Image.LANCZOS)
    texa = np.asarray(tex, np.float32) / 255
    texa *= (0.9 + 0.12 * noise2d(hgt, TW, 30, 3, 3))[..., None]
    jar = dict(py=top + jy, px=X[jx], jy=jy, th=th, a=a[jy, jx], lam=lam, spec=spec,
               tex=texa, TW=TW, top=top, r0=rr[0], cx=cx)
    return img, ang, rad, jar


def millet(d, x0, y0, hgt, lean, sway, s):
    pts = []
    for i in range(13):
        f = i / 12
        bend = (lean + sway) * f * f
        pts.append((x0 + hgt * math.sin(bend) * 0.9 + 40 * lean * f, y0 - hgt * f * math.cos(bend * 0.5)))
    col = (70, 50, 22, 235)
    d.line(pts, fill=col, width=max(2, int(4 * s)))
    # 低垂的谷穗
    tx, ty = pts[-1]
    dirx = 1 if lean + sway >= 0 else -1
    for j in range(14):
        f = j / 13
        ex = tx + dirx * (40 * s * math.sin(f * 1.6))
        ey = ty + 90 * s * f * f + 10 * s * f
        r = (14 - 7 * f) * s
        d.ellipse((ex - r, ey - r * 1.3, ex + r, ey + r * 1.3), fill=(96, 68, 26, 240))


def scene_prehistory(t, tl):
    img, ang, rad, jar = build_prehistory()
    frame = img.copy()
    rays = (0.5 + 0.5 * np.cos(20 * (ang - 0.03 * tl))) ** 6
    comp(frame, (1.0, 0.93, 0.72), 0.22 * rays * smooth((900 - rad) / 600) * smooth((rad - 215) / 50))
    phase = 0.32 * tl
    col = (((jar["th"] + phase) / (2 * math.pi)) * jar["TW"]).astype(int) % jar["TW"]
    c = jar["tex"][jar["jy"], col] * jar["lam"][:, None] + jar["spec"][:, None]
    a = jar["a"][:, None]
    frame[jar["py"], jar["px"]] = frame[jar["py"], jar["px"]] * (1 - a) + c * a
    # 罐口
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    cx, top, r0 = jar["cx"], jar["top"], jar["r0"]
    d.ellipse((cx - r0, top - r0 * 0.2, cx + r0, top + r0 * 0.2), fill=(170, 92, 52, 255))
    d.ellipse((cx - r0 * 0.84, top - r0 * 0.15, cx + r0 * 0.84, top + r0 * 0.15), fill=(48, 26, 16, 255))
    # 粟米
    rng = np.random.default_rng(4)
    for k in range(34):
        side = -1 if k % 2 == 0 else 1
        x0 = W / 2 + side * (420 + rng.random() * 560)
        s = 0.7 + rng.random() * 0.6
        sway = 0.06 * math.sin(0.9 * tl + k * 1.7)
        millet(d, x0, H + 10, (260 + rng.random() * 220) * s, side * (0.15 + rng.random() * 0.25), sway, s)
    o = np.asarray(ov, np.float32) / 255
    comp(frame, o[..., :3], o[..., 3])
    return frame


# ================================================================ 3. 商都：夯土城墙与青铜方鼎
@lru_cache(None)
def build_shang():
    WW = int(W * 1.25)
    img = vgrad(H, WW, (0.10, 0.08, 0.08), (0.42, 0.28, 0.18), 1.0)
    rng = np.random.default_rng(3)
    yy = np.arange(H, dtype=np.float32)[:, None]
    top = H * 0.30 + 70 * ridge(WW, 8, rough=0.5)[None, :]
    bottom = H * 0.86
    # 一层层夯土
    wall = np.zeros((H, WW, 3), np.float32)
    y = H * 0.25
    while y < bottom:
        th = rng.uniform(18, 30)
        tone = np.array([0.60, 0.47, 0.32]) * rng.uniform(0.82, 1.08)
        band = (yy >= y) & (yy < y + th)
        wall[band[:, 0]] = tone
        line = np.abs(yy - y) < 1.5
        wall[line[:, 0]] *= 0.88
        y += th
    n = noise2d(H, WW, 30, 9, 4)
    g = noise2d(H, WW, 3, 10, 2)
    wall *= ((0.78 + 0.38 * n) * (0.9 + 0.2 * g))[..., None]
    pits = Image.new("L", (WW, H), 0)
    d = ImageDraw.Draw(pits)
    for _ in range(2600):
        x, py, r = rng.uniform(0, WW), rng.uniform(H * 0.28, bottom), rng.uniform(3, 7)
        d.ellipse((x - r, py - r * 0.7, x + r, py + r * 0.7), fill=90)
    wall *= (1 - np.asarray(pits, np.float32)[..., None] / 255 * 0.6)
    wall *= np.clip(0.55 + 0.6 * (1 - np.linspace(0, 1, WW))[None, :], 0, 1)[..., None]
    wa = np.clip((yy - top) / 2, 0, 1) * (yy < bottom)
    comp(img, wall, wa)
    comp(img, (0.20, 0.14, 0.10), smooth((yy - bottom) / 4) * np.ones((1, WW)))
    # 方鼎（2 倍绘制后缩小）
    S = 2
    bw, bh = 560 * S, 640 * S
    m = Image.new("L", (bw, bh), 0)
    deco = Image.new("L", (bw, bh), 0)
    dm, dd = ImageDraw.Draw(m), ImageDraw.Draw(deco)
    cx = bw / 2
    rim_y, body_top, body_bot = 150 * S, 175 * S, 440 * S
    for hx in (-150, 150):   # 立耳
        x = cx + hx * S
        dm.rectangle((x - 36 * S, rim_y - 135 * S, x + 36 * S, rim_y), fill=255)
        dm.rectangle((x - 17 * S, rim_y - 108 * S, x + 17 * S, rim_y - 6 * S), fill=0)
    dm.rectangle((cx - 232 * S, rim_y, cx + 232 * S, body_top), fill=255)
    dm.polygon([(cx - 222 * S, body_top), (cx + 222 * S, body_top),
                (cx + 210 * S, body_bot), (cx - 210 * S, body_bot)], fill=255)
    for lx in (-165, 165):   # 柱足
        x = cx + lx * S
        dm.polygon([(x - 30 * S, body_bot - 4), (x + 30 * S, body_bot - 4),
                    (x + 24 * S, 620 * S), (x - 24 * S, 620 * S)], fill=255)
    # 饕餮纹带 + 乳钉纹
    by0, by1 = body_top + 18 * S, body_top + 78 * S
    dd.rectangle((cx - 205 * S, by0, cx + 205 * S, by0 + 5 * S), fill=255)
    dd.rectangle((cx - 205 * S, by1, cx + 205 * S, by1 + 5 * S), fill=255)
    ey = (by0 + by1) / 2
    for ex in (-38, 38):
        dd.ellipse((cx + (ex - 18) * S, ey - 11 * S, cx + (ex + 18) * S, ey + 11 * S), fill=255)
    for k in range(-4, 5):
        if abs(k) < 1:
            continue
        x = cx + k * 44 * S
        dd.arc((x - 18 * S, ey - 18 * S, x + 18 * S, ey + 18 * S), 0, 300, fill=255, width=5 * S)
    dd.rectangle((cx - 4 * S, by0, cx + 4 * S, by1), fill=255)
    for col in (-180, -150, 150, 180):
        for k in range(5):
            y = by1 + (32 + k * 34) * S
            x = cx + col * S * (1 - (y - body_top) / (body_bot - body_top) * 0.04)
            dd.ellipse((x - 9 * S, y - 9 * S, x + 9 * S, y + 9 * S), fill=255)
    for k in range(-4, 5):
        x, y = cx + k * 34 * S, body_bot - 26 * S
        dd.ellipse((x - 9 * S, y - 9 * S, x + 9 * S, y + 9 * S), fill=255)
    Z = 1.25
    sz = (int(bw / S * Z), int(bh / S * Z))
    m = np.asarray(m.resize(sz, Image.LANCZOS), np.float32) / 255
    deco = np.asarray(deco.resize(sz, Image.LANCZOS), np.float32) / 255
    deco *= m
    hh, ww = m.shape
    n1 = noise2d(hh, ww, 26, 21, 4)
    n2 = noise2d(hh, ww, 50, 22, 3)
    base = np.array([0.22, 0.34, 0.29]) * (1 - n1[..., None]) + np.array([0.40, 0.54, 0.44]) * n1[..., None]
    rust = (n2 > 0.62)[..., None] * 0.6
    base = base * (1 - rust) + np.array([0.38, 0.28, 0.18]) * rust
    xs = np.linspace(0, 1, ww)[None, :]
    light = 0.55 + 0.65 * (1 - xs) ** 1.5
    shade = np.ones_like(m)
    b0, b1 = int(body_top / S * Z), int(body_bot / S * Z)
    shade[b0:b1] *= np.linspace(1.05, 0.75, b1 - b0)[:, None]
    dsh = np.roll(np.roll(deco, 4, 0), 3, 1)
    col = base * (light * shade)[..., None]
    col = col * (1 - 0.35 * dsh[..., None] * (1 - deco[..., None])) + 0.10 * deco[..., None]
    ding = Sprite.__new__(Sprite)
    ding.rgb, ding.a, ding.h, ding.w = col.astype(np.float32), m, hh, ww
    # 光束
    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
    dist = np.abs((X - W * 0.15) * 0.6 - (Y + 100) * 0.8)
    beam = np.exp(-(dist / 260) ** 2) * smooth(1 - Y / H)
    return img, ding, beam.astype(np.float32)


def scene_shang(t, tl):
    img, ding, beam = build_shang()
    x0 = int(np.clip(tl * 14, 0, img.shape[1] - W))
    frame = img[:, x0:x0 + W].copy()
    frame += beam[..., None] * np.array([0.30, 0.22, 0.12]) * (0.9 + 0.1 * math.sin(tl * 1.3))
    yy, xx = np.mgrid[0:H, 0:W]
    dx, dy = W * 0.60, H * 0.855
    sh = np.exp(-(((xx - dx - 40) / 450) ** 2 + ((yy - dy + 12) / 30) ** 2) * 2)
    comp(frame, (0.05, 0.04, 0.03), 0.6 * sh)
    paste(frame, ding, dx - ding.w / 2, dy - ding.h + 10)
    # 一道光扫过鼎身
    sx = int(dx - ding.w / 2)
    sy = int(dy - ding.h + 10)
    pos = (tl * 0.12 % 1.4 - 0.2) * ding.w
    band = np.exp(-((np.arange(ding.w) - pos) / 50) ** 2)[None, :] * ding.a * 0.18
    frame[sy:sy + ding.h, sx:sx + ding.w] += band[..., None] * np.array([1.0, 0.9, 0.7])
    # 浮尘
    rng = np.random.default_rng(7)
    p = rng.random((120, 5))
    for px, py, sp, ph, r in p:
        x = (px * W + tl * (8 + 20 * sp)) % W
        y = py * H + 25 * math.sin(tl * 0.5 + ph * 6)
        b = beam[int(np.clip(y, 0, H - 1)), int(x)]
        if b > 0.05:
            paste(frame, glow_dot(1 + 2 * r, (1, 0.85, 0.6)), x, y, alpha=0.5 * b, center=True, add=True)
    vig = (1 - 0.7 * (((xx / W - 0.55) * 1.2) ** 2 + ((yy / H - 0.5)) ** 2))
    return frame * np.clip(vig, 0, 1)[..., None]


# ================================================================ 4. 名字：管 · 郑韩 · 郑州
@lru_cache(None)
def build_names():
    img = paper(H, W, seed=4)
    ink_mountains(img, [(0.95, 0.18, 0.16, 0.08, 0.55, None)], seed=44)
    k = 3
    subs = subs_of(k)
    comps = []
    layouts = [
        ([("管", W * 0.46, H * 0.43, 470)], ("管城", W * 0.64, H * 0.24), ("西周 · 管国", W * 0.29, H * 0.27)),
        ([("郑", W * 0.34, H * 0.43, 380), ("韩", W * 0.62, H * 0.43, 380)], ("新郑", W * 0.80, H * 0.22), None),
        ([("郑州", W * 0.48, H * 0.40, 330)], ("郑州", W * 0.78, H * 0.20), ("公元五八三年 · 隋", W * 0.48, H * 0.70)),
    ]
    for (glyphs, seal, cap), sub in zip(layouts, subs):
        m = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(m)
        for txt, x, y, size in glyphs:
            d.text((x, y), txt, font=font(SERIF_BLACK, size), fill=255, anchor="mm")
        ma = np.asarray(m.filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255
        bleed = gaussian_filter(ma, 6) * 0.35
        ys, xs = np.nonzero(ma > 0.05)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
        diag = np.clip(((X - x0) / (x1 - x0)) * 0.55 + ((Y - y0) / (y1 - y0)) * 0.45, -1, 2)
        tex = 0.82 + 0.25 * noise2d(H, W, 8, 50 + len(comps), 3)
        a = np.clip(np.maximum(ma * tex, bleed), 0, 1)
        sl = (slice(max(y0 - 40, 0), min(y1 + 40, H)), slice(max(x0 - 40, 0), min(x1 + 40, W)))
        cs = None
        if cap:
            cs = (text_sprite(cap[0], SERIF_BOLD, 54, fill=(40, 34, 30)), cap[1], cap[2])
            if cap[0].startswith("西周"):
                cs = (vtext_sprite(cap[0].replace(" · ", "·"), SERIF_BOLD, 54, (40, 34, 30, 255)), cap[1], cap[2])
        comps.append(dict(a=a[sl], diag=diag[sl], sl=sl, start=sub["start"] - 0.2,
                          seal=(seal_sprite(seal[0], 96), seal[1], seal[2]), cap=cs))
    return img, comps


def scene_names(t, tl):
    img, comps = build_names()
    frame = img.copy()
    for i, c in enumerate(comps):
        end = comps[i + 1]["start"] if i + 1 < len(comps) else SC[3]["end"] + 1
        vis = 1 - ease(t, end - 0.5, end)
        if t < c["start"] or vis <= 0:
            continue
        r = (t - c["start"]) / 1.4
        a = c["a"] * smooth((r * 1.3 - c["diag"]) / 0.12) * vis
        comp(frame[c["sl"]], INK, a)
        sp, x, y = c["seal"]
        sa = ease(t, c["start"] + 1.1, c["start"] + 1.3) * vis
        paste(frame, sp, x, y, alpha=sa * 0.95, center=True)
        if c["cap"]:
            cp, x, y = c["cap"]
            paste(frame, cp, x, y, alpha=ease(t, c["start"] + 0.9, c["start"] + 1.6) * vis, center=True)
    return frame


# ================================================================ 5. 文脉：嵩山 · 塔林 · 钟声
def stupa(d, x, base, h, w, tiers):
    col = 255
    d.rectangle((x - w * 0.65, base - h * 0.07, x + w * 0.65, base), fill=col)
    y = base - h * 0.07
    d.rectangle((x - w / 2, y - h * 0.30, x + w / 2, y), fill=col)
    y -= h * 0.30
    bw = w
    for i in range(tiers):
        ew = bw * 1.35
        d.polygon([(x - ew / 2 - 6, y - 4), (x + ew / 2 + 6, y - 4),
                   (x + bw / 2, y - h * 0.035), (x - bw / 2, y - h * 0.035)], fill=col)
        y -= h * 0.035
        bw *= 0.86
        d.rectangle((x - bw / 2, y - h * 0.06, x + bw / 2, y), fill=col)
        y -= h * 0.06
    d.ellipse((x - bw * 0.45, y - bw * 0.6, x + bw * 0.45, y + bw * 0.2), fill=col)
    y -= bw * 0.5
    d.rectangle((x - 2, y - h * 0.12, x + 2, y), fill=col)
    d.ellipse((x - 6, y - h * 0.12 - 8, x + 6, y - h * 0.12 + 4), fill=col)


@lru_cache(None)
def build_culture():
    BW, BH = int(W * 1.06), int(H * 1.06)
    img = paper(BH, BW, seed=5)
    peak = lambda u: 0.35 + 0.65 * np.exp(-((u - 0.42) / 0.22) ** 2)
    ink_mountains(img, [(0.55, 0.40, 0.25, 0.12, 0.55, peak),
                        (0.66, 0.30, 0.45, 0.08, 0.6, lambda u: 0.4 + 0.6 * np.exp(-((u - 0.7) / 0.25) ** 2))],
                  seed=55)
    Y = np.arange(BH, dtype=np.float32)[:, None]
    comp(img, (0.95, 0.93, 0.88), 0.55 * np.exp(-((Y - BH * 0.68) / 50) ** 2) * np.ones((1, BW)))
    # 寺院屋顶
    tx, ty = BW * 0.5, BH * 0.70
    m = Image.new("L", (BW, BH), 0)
    d = ImageDraw.Draw(m)
    d.polygon([(tx - 120, ty - 20), (tx - 70, ty - 62), (tx + 70, ty - 62), (tx + 120, ty - 20),
               (tx + 95, ty - 30), (tx - 95, ty - 30)], fill=230)
    d.rectangle((tx - 80, ty - 30, tx + 80, ty + 15), fill=200)
    d.rectangle((tx - 40, ty - 82, tx + 40, ty - 60), fill=230)
    # 塔林
    rng = np.random.default_rng(15)
    for k in range(16):
        x = BW * (0.03 + 0.94 * (k + rng.random() * 0.6) / 16)
        if abs(x - BW * 0.5) < 160:
            continue
        h = rng.uniform(170, 330) * (1.2 if abs(x - BW * 0.5) > 500 else 0.85)
        stupa(d, x, BH * 0.995, h, h * rng.uniform(0.22, 0.3), int(rng.integers(1, 6)))
    a = np.asarray(m.filter(ImageFilter.GaussianBlur(1)), np.float32) / 255
    comp(img, INK * 1.2, a * (0.8 + 0.25 * noise2d(BH, BW, 30, 56, 3)))
    YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
    dist = np.hypot(XX - W * 0.5, (YY - H * 0.66) * 1.6)
    labels = [(vtext_sprite("杜甫", SERIF_BLACK, 76, (30, 26, 24, 255)), seal_sprite("巩义", 52), W * 0.86),
              (vtext_sprite("白居易", SERIF_BLACK, 76, (30, 26, 24, 255)), seal_sprite("新郑", 52), W * 0.77)]
    return Image.fromarray(to_u8(img)), dist.astype(np.float32), labels


def scene_culture(t, tl):
    big, dist, labels = build_culture()
    frame = kb(big, 1.0 + 0.04 * tl / 13, 0.5, 0.55)
    subs = subs_of(4)
    for i, (txt, seal, x) in enumerate(labels):
        a = fade(t, subs[0]["start"] + 0.3 + i * 1.6, subs[1]["start"] + 0.6, 0.8, 0.6)
        paste(frame, txt, x, H * 0.16, alpha=a)
        paste(frame, seal, x + txt.w / 2 - seal.w / 2, H * 0.16 + txt.h + 10, alpha=a)
    # 钟声涟漪
    t0 = subs[1]["start"] - 0.2
    for k in range(6):
        age = t - (t0 + k * 1.6)
        if 0 < age < 4.5:
            r = age * 230
            ring = np.exp(-((dist - r) / 7) ** 2) * 0.45 * (1 - age / 4.5)
            comp(frame, RED * 0.9, ring)
    return frame


# ================================================================ 6. 铁路交汇 + 二七塔
def rail(d, p0, p1, S, ties=True):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    if ties:
        s = 0
        while s < L:
            cx, cy = p0[0] + ux * s, p0[1] + uy * s
            d.line((cx - nx * 17 * S, cy - ny * 17 * S, cx + nx * 17 * S, cy + ny * 17 * S),
                   fill=(110, 80, 52, 255), width=5 * S)
            s += 18 * S
    for o in (-8, 8):
        d.line((p0[0] + nx * o * S, p0[1] + ny * o * S, p1[0] + nx * o * S, p1[1] + ny * o * S),
               fill=(40, 34, 30, 255), width=3 * S)


NS = ((W * 0.465, -60), (W * 0.535, H + 60))
EW = ((-60, H * 0.53), (W + 60, H * 0.47))


@lru_cache(None)
def build_rail_map():
    img = paper(H, W, seed=6, tone=(0.88, 0.80, 0.64), vig=0.6)
    S = 2
    ov = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for gx in range(0, W, 120):
        d.line((gx * S, 0, gx * S, H * S), fill=(120, 96, 70, 40), width=S)
    for gy in range(0, H, 120):
        d.line((0, gy * S, W * S, gy * S), fill=(120, 96, 70, 40), width=S)
    pts = [(x * S, (H * 0.13 + 40 * math.sin(x / 260) + 20 * math.sin(x / 90)) * S) for x in range(0, W + 1, 8)]
    d.line(pts, fill=(170, 130, 70, 200), width=16 * S)
    ov = ov.resize((W, H), Image.LANCZOS)
    o = np.asarray(ov, np.float32) / 255
    comp(img, o[..., :3], o[..., 3])
    layers = []
    for p0, p1 in (NS, EW):
        lv = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
        rail(ImageDraw.Draw(lv), (p0[0] * S, p0[1] * S), (p1[0] * S, p1[1] * S), S)
        a = np.asarray(lv.resize((W, H), Image.LANCZOS), np.float32) / 255
        layers.append(a)
    labels = [(text_sprite("京汉铁路 · 1906", SERIF_BOLD, 40, fill=(50, 40, 32)), W * 0.545, H * 0.17),
              (text_sprite("汴洛铁路 · 1909", SERIF_BOLD, 40, fill=(50, 40, 32)), W * 0.06, H * 0.44),
              (text_sprite("黄河", SERIF_BOLD, 34, fill=(120, 88, 50)), W * 0.30, H * 0.17),
              (text_sprite("郑州", SERIF_BLACK, 52, fill=(150, 40, 30)), W * 0.515, H * 0.545)]
    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
    return img, layers, labels, Y, X


def train(d, p0, p1, s, n=7):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    for i in range(n):
        a = s - i * 62
        b = a - 54
        pts = []
        for ss, o in ((a, -13), (a, 13), (b, 13), (b, -13)):
            pts.append((p0[0] + ux * ss + nx * o, p0[1] + uy * ss + ny * o))
        d.polygon(pts, fill=(36, 30, 28, 255) if i else (120, 30, 22, 255))
    return p0[0] + ux * s, p0[1] + uy * s


def scene_rail_map(t, tl):
    img, layers, labels, Y, X = build_rail_map()
    frame = img.copy()
    rv1 = ease(tl, 0.3, 2.0) * (H + 120) - 60
    rv2 = ease(tl, 1.0, 2.7) * (W + 120) - 60
    comp(frame, layers[0][..., :3], layers[0][..., 3] * (Y < rv1))
    comp(frame, layers[1][..., :3], layers[1][..., 3] * (X < rv2))
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for (p0, p1), t0, v in ((NS, 2.2, 150), (EW, 3.0, 190)):
        s = (tl - t0) * v
        if s > 0:
            for k in range(14):
                age = k * 0.22 + (tl % 0.22)
                ss = s - age * v
                if ss < 0:
                    continue
                L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
                x = p0[0] + (p1[0] - p0[0]) * ss / L + 14 * age
                y = p0[1] + (p1[1] - p0[1]) * ss / L - 10 * age
                r = 10 + 26 * age
                al = int(150 * max(0, 1 - age / 3.1))
                d.ellipse((x - r, y - r, x + r, y + r), fill=(240, 236, 228, al))
            train(d, p0, p1, s)
    o = np.asarray(ov, np.float32) / 255
    comp(frame, o[..., :3], o[..., 3])
    for i, (sp, x, y) in enumerate(labels):
        paste(frame, sp, x, y, alpha=ease(tl, 1.5 + i * 0.4, 2.3 + i * 0.4))
    cx, cy = W * 0.5, H * 0.5
    pulse = 0.5 + 0.5 * math.sin(tl * 3)
    paste(frame, glow_dot(9 + 3 * pulse, RED), cx, cy, alpha=ease(tl, 2.5, 3.2), center=True)
    return frame


def erqi(d, cx, base, S):
    """二七塔剪影：双塔并联，14 层，顶部钟楼 + 五角星"""
    col = 255
    d.rectangle((cx - 230 * S, base - 40 * S, cx + 230 * S, base), fill=col)
    d.rectangle((cx - 180 * S, base - 80 * S, cx + 180 * S, base - 40 * S), fill=col)
    wins = []
    for off in (-46, 46):
        x = cx + off * S
        y = base - 80 * S
        for i in range(11):
            hs = 46 * S
            d.rectangle((x - 44 * S, y - hs, x + 44 * S, y), fill=col)
            wins.append((x, y - hs * 0.55))
            ew = 64 * S
            d.polygon([(x - ew - 8 * S, y - hs - 2 * S), (x - 44 * S, y - hs + 9 * S),
                       (x + 44 * S, y - hs + 9 * S), (x + ew + 8 * S, y - hs - 2 * S),
                       (x + 44 * S, y - hs - 8 * S), (x - 44 * S, y - hs - 8 * S)], fill=col)
            y -= hs
    top = base - 80 * S - 11 * 46 * S
    d.rectangle((cx - 80 * S, top - 76 * S, cx + 80 * S, top), fill=col)
    d.polygon([(cx - 120 * S, top - 70 * S), (cx + 120 * S, top - 70 * S),
               (cx + 30 * S, top - 150 * S), (cx - 30 * S, top - 150 * S)], fill=col)
    d.rectangle((cx - 5 * S, top - 230 * S, cx + 5 * S, top - 150 * S), fill=col)
    return wins, (cx, top - 38 * S), (cx, top - 250 * S)


@lru_cache(None)
def build_erqi():
    img = vgrad(H, W, (0.13, 0.10, 0.22), (0.98, 0.58, 0.30), 1.6)
    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
    sun = np.hypot(X - W * 0.5, Y - H * 0.70)
    comp(img, (1.0, 0.86, 0.6), smooth((170 - sun) / 6) * 0.9)
    img += (np.exp(-sun / 420) * 0.25)[..., None] * np.array([1.0, 0.6, 0.3])
    ang = np.arctan2(Y - H * 0.70, X - W * 0.5)
    S = 2
    m = Image.new("L", (W * S, H * S), 0)
    d = ImageDraw.Draw(m)
    # 低矮的城市轮廓
    rng = np.random.default_rng(27)
    x = 0
    while x < W:
        w = rng.uniform(40, 120)
        hgt = rng.uniform(40, 170)
        d.rectangle((x * S, (H * 0.90 - hgt) * S, (x + w) * S, H * S), fill=200)
        x += w + rng.uniform(0, 10)
    wins, clock, star = erqi(d, W * 0.5 * S, H * 0.90 * S, S * 0.88)
    d.rectangle((0, int(H * 0.90 * S), W * S, H * S), fill=255)
    a = np.asarray(m.resize((W, H), Image.LANCZOS), np.float32) / 255
    comp(img, (0.07, 0.05, 0.09), a)
    wins = [(x / S, y / S) for x, y in wins]
    return img, ang, sun, wins, (clock[0] / S, clock[1] / S), (star[0] / S, star[1] / S)


def star_sprite(r):
    im = Image.new("RGBA", (int(r * 2.4), int(r * 2.4)), (0, 0, 0, 0))
    c = r * 1.2
    pts = []
    for k in range(10):
        rr = r if k % 2 == 0 else r * 0.4
        a = -math.pi / 2 + k * math.pi / 5
        pts.append((c + rr * math.cos(a), c + rr * math.sin(a)))
    ImageDraw.Draw(im).polygon(pts, fill=(230, 50, 40, 255))
    return Sprite(im)


def scene_erqi(t, tl, t_on):
    img, ang, sun, wins, clock, star = build_erqi()
    frame = img.copy()
    rays = (0.5 + 0.5 * np.cos(28 * (ang + 0.02 * tl))) ** 8
    frame += (rays * np.exp(-sun / 700) * 0.18)[..., None] * np.array([1, 0.7, 0.4]) * (sun > 170)[..., None]
    lit = ease(t, t_on + 0.5, t_on + 2.5)
    for i, (x, y) in enumerate(wins):
        fl = 0.75 + 0.25 * math.sin(tl * 2 + i)
        paste(frame, glow_dot(3.5, GOLD), x, y, alpha=lit * fl * 0.9, center=True, add=True)
    paste(frame, glow_dot(14, (1, 0.95, 0.8)), clock[0], clock[1], alpha=lit * 0.9, center=True, add=True)
    paste(frame, star_sprite(26), star[0], star[1], alpha=0.95, center=True)
    return frame


def scene_railway(t, tl):
    t_on = subs_of(5)[2]["start"] - 0.6
    if t < t_on - 0.45:
        return scene_rail_map(t, tl)
    if t > t_on + 0.45:
        return scene_erqi(t, tl, t_on)
    k = ease(t, t_on - 0.45, t_on + 0.45)
    return scene_rail_map(t, tl) * (1 - k) + scene_erqi(t, tl, t_on) * k


# ================================================================ 7/9. 天际线
@lru_cache(None)
def build_skyline():
    rng = np.random.default_rng(1954)
    base = int(H * 0.78)
    sprites = []
    for row, (col, hmin, hmax, wmin, wmax, lit) in enumerate(
            [((0.16, 0.18, 0.30), 160, 420, 50, 120, 0.25),
             ((0.06, 0.07, 0.12), 70, 300, 50, 150, 0.40)]):
        x = -40 + row * 25
        while x < W + 40:
            w = int(rng.uniform(wmin, wmax))
            center = 1 - abs((x + w / 2) / W - 0.55) * 1.2
            h = int(rng.uniform(hmin, hmax) * (0.5 + 0.7 * max(center, 0)))
            im = Image.new("RGBA", (w, h + 30), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            fill = tuple(int(c * 255) for c in col) + (255,)
            if rng.random() < 0.25:   # 斜顶 / 天线
                d.polygon([(0, 30), (w, 30), (w, 12 + 18 * rng.random()), (0, 30)], fill=fill)
                d.line((w // 2, 0, w // 2, 30), fill=fill, width=3)
            d.rectangle((0, 30, w, h + 30), fill=fill)
            for wy in range(40, h + 24, 14):
                for wx in range(6, w - 8, 12):
                    if rng.random() < lit:
                        warm = rng.random() < 0.7
                        c = (255, 205, 130) if warm else (180, 210, 255)
                        d.rectangle((wx, wy, wx + 5, wy + 7), fill=c + (int(rng.uniform(140, 255)),))
            sprites.append(dict(sp=Sprite(im), x=x, y=base - h - 30, h=h + 30,
                                order=rng.random() * 0.85))
            x += w + int(rng.uniform(-10, 18))
    # 「大玉米」（千玺广场）
    hh, ww = 470, 130
    im = Image.new("RGBA", (ww, hh), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for y in range(hh):
        v = y / hh
        r = (0.32 + 0.38 * math.sin(min(1, (1 - v) * 1.15) * math.pi * 0.85)) * ww / 2 if v < 0.97 else 0
        if v < 0.12:
            r = ww / 2 * 0.70 * math.sqrt(max(0, 1 - ((0.12 - v) / 0.12) ** 2))
        if r <= 0:
            continue
        rib = (y % 9) < 3
        c = (255, 196, 90, 255) if rib else (40, 34, 30, 255)
        d.line((ww / 2 - r, y, ww / 2 + r, y), fill=c)
    d.line((ww / 2, 0, ww / 2, 40), fill=(255, 196, 90, 255), width=3)
    sprites.append(dict(sp=Sprite(im), x=int(W * 0.60), y=base - hh, h=hh, order=1.0))
    sprites.sort(key=lambda s: s["sp"].rgb[..., 0].mean() < 0.12)  # 后排先画
    return sprites, base


def draw_skyline(frame, grow, t):
    sprites, base = build_skyline()
    for s in sprites:
        g = smooth((grow - s["order"]) / 0.15) if grow < 2 else 1.0
        if g <= 0:
            continue
        dy = (1 - g) * s["h"]
        paste(frame, s["sp"], s["x"], s["y"] + dy)
    return base


# ================================================================ 7. 省会：城市生长
@lru_cache(None)
def build_modern():
    img = vgrad(H, W, (0.03, 0.04, 0.10), (0.36, 0.24, 0.34), 1.4)
    rng = np.random.default_rng(9)
    for _ in range(260):
        x, y = rng.uniform(0, W), rng.uniform(0, H * 0.55)
        img[int(y), int(x)] += rng.uniform(0.2, 0.7)
    a = text_sprite("开封", SERIF_BOLD, 44, fill=(255, 220, 160))
    b = text_sprite("郑州", SERIF_BLACK, 56, fill=(255, 220, 160))
    y = text_sprite("1954", SERIF_BLACK, 64, fill=(255, 200, 120))
    return img, a, b, y


def scene_modern(t, tl):
    img, a, b, yr = build_modern()
    frame = img.copy()
    subs = subs_of(6)
    # 开封 → 郑州
    al = fade(t, subs[0]["start"] - 0.2, subs[1]["start"] + 0.6)
    if al > 0:
        p0, p1 = (W * 0.70, H * 0.28), (W * 0.32, H * 0.28)
        k = ease(t, subs[0]["start"] + 0.4, subs[0]["start"] + 2.4)
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        xe = p0[0] + (p1[0] - p0[0]) * k
        d.line((p0[0], p0[1], xe, p1[1]), fill=(255, 200, 120, 220), width=4)
        if k > 0.02:
            d.polygon([(xe - 4, p1[1]), (xe + 22, p1[1] - 12), (xe + 22, p1[1] + 12)], fill=(255, 200, 120, 255))
        o = np.asarray(ov, np.float32) / 255
        comp(frame, o[..., :3], o[..., 3] * al)
        paste(frame, glow_dot(7, GOLD), p0[0], p0[1], alpha=al, center=True, add=True)
        paste(frame, glow_dot(10, GOLD), p1[0], p1[1], alpha=al * k, center=True, add=True)
        paste(frame, a, p0[0], p0[1] - 60, alpha=al, center=True)
        paste(frame, b, p1[0], p1[1] - 66, alpha=al * k, center=True)
        paste(frame, yr, (p0[0] + p1[0]) / 2, p1[1] + 60, alpha=al * k, center=True)
    # 纺织：经线 + 往返的梭子
    al2 = fade(t, subs[1]["start"] - 0.3, subs[2]["start"] + 0.8, 0.6, 0.8)
    if al2 > 0:
        X = np.arange(W)
        warp = ((X % 22) < 2).astype(np.float32)[None, :] * np.ones((int(H * 0.5), 1))
        frame[:int(H * 0.5)] += warp[..., None] * 0.10 * al2 * GOLD
        lt = t - subs[1]["start"]
        rows = int(np.clip(lt * 3, 0, 40))
        for r in range(rows):
            y = int(H * 0.08 + r * 9)
            frame[y:y + 3] += 0.10 * al2 * GOLD
        sx = W * (0.5 + 0.45 * math.sin(lt * 3.0))
        paste(frame, glow_dot(8, (1, 0.9, 0.7)), sx, H * 0.08 + rows * 9, alpha=al2, center=True, add=True)
    grow = (t - (subs[0]["start"] + 1.5)) / (SC[6]["end"] - subs[0]["start"] - 3.0)
    base = draw_skyline(frame, grow, t)
    frame[base:] = frame[base:] * 0.2 + np.array([0.02, 0.025, 0.05]) * 0.8
    return frame


# ================================================================ 8. 今天：米字形高铁网
RAYS = [(-90, 400, "北京"), (90, 345, "广州"), (0, 660, "徐州"), (180, 660, "西安"),
        (-38, 540, "济南"), (-142, 540, "太原"), (38, 540, "合肥"), (142, 540, "重庆")]
CEN = (W * 0.5, H * 0.46)


@lru_cache(None)
def build_today():
    img = np.zeros((H, W, 3), np.float32) + np.array([0.02, 0.035, 0.07], np.float32)
    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
    R = np.hypot(X - CEN[0], Y - CEN[1])
    img += (np.exp(-R / 500) * 0.12)[..., None] * np.array([0.3, 0.5, 1.0])
    img[(X % 60 < 1.5) & (Y % 60 < 1.5)] += 0.08
    S = 2
    core = Image.new("L", (W * S, H * S), 0)
    d = ImageDraw.Draw(core)
    ends = []
    for ang, L, name in RAYS:
        a = math.radians(ang)
        ex, ey = CEN[0] + L * math.cos(a), CEN[1] + L * math.sin(a)
        ends.append((ex, ey, a, L, name))
        d.line((CEN[0] * S, CEN[1] * S, ex * S, ey * S), fill=255, width=4 * S)
    core = np.asarray(core.resize((W, H), Image.LANCZOS), np.float32) / 255
    glow = gaussian_filter(core, 9) * 3
    A = np.degrees(np.arctan2(Y - CEN[1], X - CEN[0]))
    sector = np.argmin([np.abs((A - ang + 180) % 360 - 180) for ang, _, _ in RAYS], axis=0)
    labels = [text_sprite(n, SANS_BOLD, 34, fill=(220, 230, 255)) for *_, n in ends]
    big = text_sprite("1300万+", SERIF_BLACK, 110, fill=(255, 205, 120))
    small = text_sprite("常住人口  Residents", SANS, 32, fill=(220, 220, 230))
    cname = text_sprite("郑州", SERIF_BLACK, 58, fill=(255, 230, 190), shadow=4)
    return img, core, glow, R, sector, ends, labels, big, small, cname


def scene_today(t, tl):
    img, core, glow, R, sector, ends, labels, big, small, cname = build_today()
    frame = img.copy()
    reach = np.array([ease(tl, 0.6 + i * 0.12, 2.0 + i * 0.12) * (L + 8) for i, (*_, L, _) in enumerate(ends)])
    mask = (R < reach[sector]).astype(np.float32)
    lc = np.array([1.0, 0.78, 0.40])
    frame += (glow * mask)[..., None] * lc * 0.35
    comp(frame, (1.0, 0.92, 0.75), core * mask)
    for i, (ex, ey, a, L, name) in enumerate(ends):
        on = ease(tl, 1.6 + i * 0.12, 2.3 + i * 0.12)
        paste(frame, glow_dot(6, lc), ex, ey, alpha=on, center=True, add=True)
        lx, ly = ex + 46 * math.cos(a), ey + 40 * math.sin(a)
        paste(frame, labels[i], lx, ly, alpha=on, center=True)
        for k in range(3):
            f = ((tl - 2.0) * 0.45 + k / 3 + i * 0.13) % 1
            if tl > 2.0:
                paste(frame, glow_dot(4.5, (1, 1, 1)), CEN[0] + f * L * math.cos(a),
                      CEN[1] + f * L * math.sin(a), alpha=0.9 * math.sin(f * math.pi), center=True, add=True)
    paste(frame, glow_dot(16 + 3 * math.sin(tl * 3), (1, 0.85, 0.6)), CEN[0], CEN[1], alpha=1, center=True, add=True)
    paste(frame, cname, CEN[0] + 80, CEN[1] - 50, alpha=ease(tl, 0.3, 1.0), center=True)
    # 航线
    s2 = subs_of(7)[1]["start"]
    if t > s2 - 0.5:
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        for j, (tx, ty, bend) in enumerate([(W * 1.05, H * 0.05, -300), (-W * 0.05, H * 0.34, -260),
                                            (W * 1.05, H * 0.55, -200)]):
            k = ease(t, s2 - 0.3 + j * 0.4, s2 + 3.0 + j * 0.4)
            pts = []
            for i in range(int(60 * k) + 1):
                u = i / 60
                mx, my = (CEN[0] + tx) / 2, (CEN[1] + ty) / 2 + bend
                x = (1 - u) ** 2 * CEN[0] + 2 * (1 - u) * u * mx + u * u * tx
                y = (1 - u) ** 2 * CEN[1] + 2 * (1 - u) * u * my + u * u * ty
                pts.append((x, y))
            if len(pts) > 1:
                d.line(pts, fill=(140, 200, 255, 200), width=3)
                hx, hy = pts[-1]
                d.ellipse((hx - 6, hy - 6, hx + 6, hy + 6), fill=(230, 245, 255, 255))
        o = np.asarray(ov, np.float32) / 255
        comp(frame, o[..., :3], o[..., 3])
        al = ease(t, s2 + 0.8, s2 + 1.6)
        paste(frame, big, W * 0.85, H * 0.63, alpha=al, center=True)
        paste(frame, small, W * 0.85, H * 0.63 + 82, alpha=al, center=True)
    return frame


# ================================================================ 9. 结尾：夜色倒影
@lru_cache(None)
def build_ending():
    img = vgrad(H, W, (0.02, 0.03, 0.08), (0.20, 0.14, 0.26), 1.2)
    ink_mountains(img, [(0.70, 0.30, 0.35, 0.10, 0.55, None)], seed=91, col=(0.30, 0.30, 0.42))
    sky = img.copy()
    base = draw_skyline(sky, 99, 0)
    refl = sky[base - (H - base) * 2:base][::-2][: H - base] * 0.45
    title = text_sprite(TITLE_ZH, SERIF_BLACK, 120, fill=(255, 225, 170), shadow=6)
    en = text_sprite(TITLE_EN, SERIF_BOLD, 44, fill=(235, 225, 210), shadow=4)
    return sky, refl.astype(np.float32), base, title, en


def scene_ending(t, tl):
    sky, refl, base, title, en = build_ending()
    frame = sky.copy()
    hgt = H - base
    rows = np.arange(hgt)
    shift = (6 * np.sin(rows * 0.35 - tl * 4) * (rows / hgt + 0.2)).astype(int)
    xs = (np.arange(W)[None, :] - shift[:, None]) % W
    frame[base:] = refl[rows[:, None], xs] + np.array([0.02, 0.03, 0.06])
    t0 = subs_of(8)[2]["start"] - 0.3
    al = ease(t, t0, t0 + 1.5)
    paste(frame, title, W / 2, H * 0.17, alpha=al, center=True)
    paste(frame, en, W / 2, H * 0.17 + 110, alpha=ease(t, t0 + 0.6, t0 + 2.0), center=True)
    paste(frame, seal_sprite("中原", 70), W / 2 + title.w / 2 + 50, H * 0.17 - 20, alpha=al, center=True)
    return frame


# ================================================================ 叠加层
LABELS = {
    "prehistory": ("陶", "约八千至五千年前 · 裴李岗 · 仰韶", "c. 6000–3000 BC · Neolithic", False),
    "shang": ("商", "约三千六百年前 · 商代都城", "c. 1600 BC · Shang Capital", True),
    "names": ("名", "周 · 春秋战国 · 隋", "Zhou · Warring States · Sui", False),
    "culture": ("文", "文脉 · 嵩山少林", "Poets & Shaolin", False),
    "railway": ("路", "1906–1923 · 铁路之城", "City of Railways", False),
    "modern": ("城", "1954 · 河南省会", "Provincial Capital", True),
    "today": ("今", "今天 · 国家中心城市", "Today · National Central City", True),
}


@lru_cache(None)
def label_sprites(sid, dark=None):
    seal, zh, en, d0 = LABELS[sid]
    dark = d0 if dark is None else dark
    c1 = (245, 235, 220) if dark else (40, 34, 30)
    c2 = (200, 195, 190) if dark else (90, 80, 70)
    return seal_sprite(seal, 64), text_sprite(zh, SERIF_BOLD, 34, fill=c1), text_sprite(en, SANS, 22, fill=c2)


@lru_cache(None)
def sub_sprites(i):
    s = SUBS[i]
    zh = text_sprite(s["zh"], SANS_BOLD, 46, stroke=3, stroke_fill=(0, 0, 0, 210), shadow=3)
    size = 28
    en = text_sprite(s["en"], SANS, size, fill=(235, 235, 235), stroke=2, stroke_fill=(0, 0, 0, 200))
    while en.w > W - 160:
        size -= 1
        en = text_sprite(s["en"], SANS, size, fill=(235, 235, 235), stroke=2, stroke_fill=(0, 0, 0, 200))
    return zh, en


@lru_cache(None)
def bottom_band():
    v = np.linspace(0, 1, 230, dtype=np.float32)
    return (smooth(v) * 0.42)[:, None]


@lru_cache(None)
def title_sprites():
    t = text_sprite(TITLE_ZH, SERIF_BLACK, 124, fill=(30, 26, 24))
    e = text_sprite(TITLE_EN, SERIF_BOLD, 44, fill=(60, 52, 46))
    return t, e, seal_sprite("中原", 74)


def overlays(frame, t):
    # 开场标题
    if t < 12:
        ts, es, seal = title_sprites()
        a = fade(t, 0.8, 10.5, 1.4, 1.5)
        paste(frame, ts, W / 2, H * 0.20, alpha=a, center=True)
        paste(frame, es, W / 2, H * 0.20 + 100, alpha=fade(t, 1.6, 10.5, 1.4, 1.5), center=True)
        paste(frame, seal, W / 2 + ts.w / 2 + 50, H * 0.20 - 20, alpha=fade(t, 2.4, 10.5, 0.3, 1.5), center=True)
    # 章节标签
    for k, s in enumerate(SC):
        if s["id"] in LABELS and s["start"] - 0.5 < t < s["end"] + 0.5:
            a = fade(t, s["start"] + 0.4, s["end"] - 0.2, 0.8, 0.6)
            dark = None
            if s["id"] == "railway":   # 后半段切到暮色中的二七塔，标签改浅色
                dark = t > subs_of(k)[2]["start"] - 0.6
            seal, zh, en = label_sprites(s["id"], dark)
            paste(frame, seal, 70, 60, alpha=a)
            paste(frame, zh, 70 + seal.w + 18, 58, alpha=a)
            paste(frame, en, 70 + seal.w + 22, 58 + zh.h + 2, alpha=a)
    # 字幕
    band = None
    for i, s in enumerate(SUBS):
        nxt = SUBS[i + 1]["start"] if i + 1 < len(SUBS) else LENGTH
        end = min(s["end"] + 0.3, nxt - 0.05)
        if s["start"] - 0.15 < t < end:
            if band is None:
                frame[H - 230:] *= 1 - bottom_band()[..., None]
                band = True
            a = fade(t, s["start"] - 0.12, end, 0.12, 0.15)
            zh, en = sub_sprites(i)
            paste(frame, zh, W / 2, H - 112, alpha=a, center=True)
            paste(frame, en, W / 2, H - 56, alpha=a, center=True)
    return frame


RENDER = {"river": scene_river, "prehistory": scene_prehistory, "shang": scene_shang,
          "names": scene_names, "culture": scene_culture, "railway": scene_railway,
          "modern": scene_modern, "today": scene_today, "ending": scene_ending}


def scene_frame(k, t):
    s = SC[k]
    return RENDER[s["id"]](t, t - s["start"])


def frame_at(t):
    k = max(i for i, s in enumerate(SC) if s["start"] <= t) if t >= 0 else 0
    img = scene_frame(k, t)
    # 与下一场景 / 上一场景交叉淡化
    if k + 1 < len(SC) and t > SC[k + 1]["start"] - XF / 2:
        w = ease(t, SC[k + 1]["start"] - XF / 2, SC[k + 1]["start"] + XF / 2)
        img = img * (1 - w) + scene_frame(k + 1, t) * w
    elif k > 0 and t < SC[k]["start"] + XF / 2:
        w = ease(t, SC[k]["start"] - XF / 2, SC[k]["start"] + XF / 2)
        img = scene_frame(k - 1, t) * (1 - w) + img * w
    img = overlays(img.astype(np.float32), t)
    img *= ease(t, 0, 1.0) * (1 - ease(t, LENGTH - 2.0, LENGTH - 0.1))
    return to_u8(img)


# ================================================================ 输出
def encode(args):
    f0, f1, path = args
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-pix_fmt", "yuv420p", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(f0, f1):
        p.stdin.write(frame_at(i / FPS).tobytes())
        if (i - f0) % 150 == 0:
            print(f"  {path}: {i - f0}/{f1 - f0}", flush=True)
    p.stdin.close()
    p.wait()
    return path


def render_range(t0, t1, out, procs=4):
    f0, f1 = int(t0 * FPS), int(t1 * FPS)
    step = math.ceil((f1 - f0) / procs)
    jobs = [(a, min(a + step, f1), f".seg_{i}.mp4") for i, a in enumerate(range(f0, f1, step))]
    with Pool(len(jobs)) as pool:
        segs = pool.map(encode, jobs)
    with open(".segs.txt", "w") as f:
        f.writelines(f"file '{s}'\n" for s in segs)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", ".segs.txt",
                    "-ss", f"{t0}", "-t", f"{t1 - t0}", "-i", "mix.wav",
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart", out], check=True)
    for s in segs:
        os.remove(s)
    os.remove(".segs.txt")
    print("完成：", out)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--still":
        os.makedirs("stills", exist_ok=True)
        for ts in a[1:]:
            Image.fromarray(frame_at(float(ts))).save(f"stills/still_{float(ts):06.2f}.png")
            print("stills/", ts)
    elif a and a[0] == "--sheet":
        times = [(s["start"] + s["end"]) / 2 for s in SC]
        times[0], times[-1] = 7.0, SC[-1]["end"] - 3.0
        times[5] = subs_of(5)[1]["start"] + 1
        tiles = [Image.fromarray(frame_at(ts)).resize((640, 360), Image.LANCZOS) for ts in times]
        sheet = Image.new("RGB", (640 * 3 + 40, 360 * 3 + 40), (20, 20, 20))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, (10 + (i % 3) * 650, 10 + (i // 3) * 370))
        sheet.save("preview_sheet.png")
        print("preview_sheet.png", [round(x, 1) for x in times])
    elif a and a[0] == "--demo":
        render_range(float(a[1]), float(a[2]), "demo.mp4")
    else:
        render_range(0, LENGTH, "zhengzhou_full.mp4")
