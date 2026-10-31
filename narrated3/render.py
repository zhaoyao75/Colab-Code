"""
《黄永玉 · 永远长不大的老头》画面渲染（128 秒，1920×1080，30fps）
8 个场景 × 16 秒，全部用代码手绘（原创卡通形象，不临摹原作），中英双语字幕烧录。

  python3 render.py --still 8 24 40 56 72 88 104 120   # 关键帧静帧 -> stills/
  python3 render.py --demo 0 16                         # 从 0 秒起 16 秒 demo -> demo.mp4
  python3 render.py                                     # 完整版 -> huang_yongyu_full.mp4（4 进程）
依赖：numpy scipy pillow，ffmpeg；先跑 tts.py / music.py / mix.py 生成 timings.json、mix.wav。
"""
import json, math, os, subprocess, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import gaussian_filter, gaussian_filter1d

W, H, FPS, LENGTH = 1920, 1080, 30, 128.0
NFRAMES = int(LENGTH * FPS)
HERE = os.path.dirname(os.path.abspath(__file__))
F_BRUSH = os.path.join(HERE, "fonts/MaShanZheng-Regular.ttf")
F_FUN = os.path.join(HERE, "fonts/ZCOOLKuaiLe-Regular.ttf")
F_EN = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
F_EN_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
PAPER = (243, 234, 214)
INK = (28, 25, 22)
RED = (200, 38, 42)
CUES = json.load(open(os.path.join(HERE, "timings.json"), encoding="utf-8"))
SCRIPT = json.load(open(os.path.join(HERE, "script.json"), encoding="utf-8"))
# 场景起点（结尾段旁白提前开始，画面也跟着提前）
STARTS = [0, 16, 32, 48, 64, 80, 96, 111.2]
XFADE = 0.8
SS = 2   # 精灵图超采样倍数


def font(path, size):
    return ImageFont.truetype(path, size)


# ================================================================ 通用工具
def smoothstep(a, b, x):
    x = np.clip((x - a) / (b - a), 0, 1) if not np.isscalar(x) else min(1, max(0, (x - a) / (b - a)))
    return x * x * (3 - 2 * x)


def ease_out_back(x, k=1.7):
    x = min(1, max(0, x)) - 1
    return 1 + (k + 1) * x ** 3 + k * x ** 2


def fbm1d(n, base, octaves=5, seed=0):
    r = np.random.default_rng(seed)
    y = np.zeros(n)
    amp = 1.0
    for o in range(octaves):
        k = max(2, int(n / base * 2 ** o))
        pts = r.standard_normal(k + 2)
        v = np.interp(np.linspace(0, k, n), np.arange(k + 2), pts)
        y += amp * gaussian_filter1d(v, n / k * 0.35)
        amp *= 0.5
    return y / np.abs(y).max()


def noise2d(h, w, scale, seed=0):
    r = np.random.default_rng(seed)
    small = r.random((max(2, h // scale), max(2, w // scale)))
    small = gaussian_filter(small, 1.2)
    img = Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    a = np.asarray(img, np.float32) / 255
    return (a - a.mean()) / (a.std() + 1e-6)


def paper(w=W, h=H, seed=1, base=PAPER, vig=True):
    a = np.ones((h, w, 3), np.float32) * np.array(base, np.float32)
    a *= (1 + 0.025 * noise2d(h, w, 60, seed))[..., None]
    a *= (1 + 0.015 * noise2d(h, w, 6, seed + 1))[..., None]
    # 纤维
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")
    r = np.random.default_rng(seed + 2)
    for _ in range(w * h // 2500):
        x, y = r.random() * w, r.random() * h
        ang, ln = r.random() * math.pi, 6 + r.random() * 18
        c = (255, 250, 240, 40) if r.random() < 0.5 else (150, 130, 100, 22)
        d.line([(x, y), (x + ln * math.cos(ang), y + ln * math.sin(ang))], fill=c, width=1)
    a = np.asarray(im, np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    vig = 1 - (0.18 if vig else 0.0) * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) ** 1.5
    return Image.fromarray(np.clip(a * vig[..., None], 0, 255).astype(np.uint8))


def rgba_from(alpha, color):
    h, w = alpha.shape
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = color
    out[..., 3] = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def ink_mountains(w, h, base_y, amp, color, alpha, seed, fall=90, mist=260):
    """水墨远山：山脊处墨色浓，往下渐淡"""
    ridge = base_y - amp * (0.5 + 0.5 * fbm1d(w, 700, 6, seed)) - amp * 0.3 * np.abs(fbm1d(w, 180, 4, seed + 9))
    yy = np.arange(h)[:, None]
    d = yy - ridge[None, :]
    a = np.where(d >= 0, (0.35 + 0.65 * np.exp(-d / fall)) * np.exp(-d / mist), 0)
    a *= smoothstep(-1.5, 1.5, d)
    a *= np.clip(1 + 0.25 * noise2d(h, w, 25, seed + 3), 0.3, 1.5)
    return rgba_from(a * alpha, color), ridge


def new_sprite(w, h):
    im = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)


def fin(im):
    return im.resize((im.width // SS, im.height // SS), Image.LANCZOS)


def place(canvas, spr, cx, cy, scale=1.0, angle=0.0, alpha=1.0):
    """把 RGBA 精灵按中心点贴到 RGB 画布上"""
    im = spr
    if scale != 1.0:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BICUBIC)
    if angle:
        im = im.rotate(angle, Image.BICUBIC, expand=True)
    if alpha < 1.0:
        if alpha <= 0:
            return
        a = im.getchannel("A").point(lambda v: int(v * alpha))
        im = im.copy()
        im.putalpha(a)
    canvas.paste(im, (int(round(cx - im.width / 2)), int(round(cy - im.height / 2))), im)


def ell(d, cx, cy, rx, ry, **kw):
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], **kw)


def seal(text, size=110, color=RED):
    """红色印章（白文）"""
    im, d = new_sprite(size, size)
    s = size * SS
    d.rounded_rectangle([4, 4, s - 4, s - 4], radius=s // 10, fill=color + (235,))
    f = font(F_BRUSH, int(s * (0.42 if len(text) > 1 else 0.7)))
    if len(text) == 2:
        for k, ch in enumerate(text):
            d.text((s / 2, s * (0.3 + 0.4 * k)), ch, font=f, fill=(250, 240, 230, 255), anchor="mm")
    else:
        for k, ch in enumerate(text):
            col, row = 1 - k // 2, k % 2
            d.text((s * (0.28 + 0.44 * col), s * (0.3 + 0.4 * row)), ch, font=f, fill=(250, 240, 230, 255), anchor="mm")
    im = fin(im)
    # 印泥斑驳
    a = np.asarray(im.getchannel("A"), np.float32)
    r = np.random.default_rng(len(text) * 7 + size)
    a *= np.clip(0.75 + 0.5 * r.random(a.shape), 0, 1)
    im.putalpha(Image.fromarray(a.astype(np.uint8)))
    return im


def perforated(w, h, fill, hole=16, gap=34):
    """邮票齿孔边框（孔透明）"""
    im, d = new_sprite(w, h)
    S = SS
    d.rectangle([0, 0, w * S, h * S], fill=fill)
    r = hole * S / 2
    for x in np.arange(gap * S / 2, w * S, gap * S):
        ell(d, x, 0, r, r, fill=(0, 0, 0, 0))
        ell(d, x, h * S, r, r, fill=(0, 0, 0, 0))
    for y in np.arange(gap * S / 2, h * S, gap * S):
        ell(d, 0, y, r, r, fill=(0, 0, 0, 0))
        ell(d, w * S, y, r, r, fill=(0, 0, 0, 0))
    return im, d


def shadowed(spr, off=(10, 14), blur=12, op=0.35):
    """给精灵加柔和投影"""
    pad = blur * 3
    out = Image.new("RGBA", (spr.width + pad * 2 + abs(off[0]), spr.height + pad * 2 + abs(off[1])), (0, 0, 0, 0))
    sh = Image.new("RGBA", spr.size, (40, 30, 20, 0))
    sh.putalpha(spr.getchannel("A").point(lambda v: int(v * op)))
    out.paste(sh, (pad + off[0], pad + off[1]), sh)
    out = out.filter(ImageFilter.GaussianBlur(blur))
    out.paste(spr, (pad, pad), spr)
    return out


# ================================================================ 角色精灵
def owl_sprite(left, right):
    """原创卡通猫头鹰；left/right = 'open' | 'closed'"""
    w, h = 520, 620
    im, d = new_sprite(w, h)
    S = SS
    brown, dark, belly, cream = (150, 92, 50, 255), (70, 40, 22, 255), (226, 190, 138, 255), (247, 233, 205, 255)
    o = 7 * S
    # 耳羽
    for sgn in (-1, 1):
        cx = w * S / 2 + sgn * 150 * S
        d.polygon([(cx - 55 * S, 175 * S), (cx + sgn * 40 * S, 30 * S), (cx + 60 * S * -sgn * -1 * 0 + sgn * 70 * S, 190 * S)],
                  fill=brown, outline=dark, width=o)
    # 身体
    ell(d, w * S / 2, 400 * S, 205 * S, 210 * S, fill=brown, outline=dark, width=o)
    ell(d, w * S / 2, 430 * S, 135 * S, 160 * S, fill=belly)
    for row in range(5):
        for k in range(-2, 3):
            x = w * S / 2 + k * 48 * S + (row % 2) * 24 * S
            y = (330 + row * 46) * S
            if abs(x - w * S / 2) < (125 - row * 8) * S:
                d.arc([x - 16 * S, y - 12 * S, x + 16 * S, y + 12 * S], 20, 160, fill=(150, 100, 60, 255), width=4 * S)
    # 翅膀
    for sgn in (-1, 1):
        cx = w * S / 2 + sgn * 185 * S
        d.chord([cx - 70 * S, 290 * S, cx + 70 * S, 560 * S], 90 - sgn * 90 + (0 if sgn > 0 else 0), 270 - sgn * 90,
                fill=(118, 70, 38, 255), outline=dark, width=o)
    # 头
    ell(d, w * S / 2, 215 * S, 215 * S, 165 * S, fill=brown, outline=dark, width=o)
    # 脸盘 + 眼睛
    for sgn, state in ((-1, left), (1, right)):
        cx, cy = w * S / 2 + sgn * 92 * S, 215 * S
        ell(d, cx, cy, 92 * S, 92 * S, fill=cream, outline=dark, width=5 * S)
        if state == "open":
            ell(d, cx, cy, 62 * S, 62 * S, fill=(250, 196, 40, 255), outline=dark, width=6 * S)
            ell(d, cx + 4 * S, cy + 4 * S, 34 * S, 34 * S, fill=(20, 15, 12, 255))
            ell(d, cx + 16 * S, cy - 10 * S, 11 * S, 11 * S, fill=(255, 255, 255, 255))
        else:
            d.arc([cx - 50 * S, cy - 40 * S, cx + 50 * S, cy + 30 * S], 15, 165, fill=(20, 15, 12, 255), width=11 * S)
    # 喙
    cx = w * S / 2
    d.polygon([(cx - 24 * S, 262 * S), (cx + 24 * S, 262 * S), (cx, 312 * S)], fill=(240, 140, 40, 255), outline=dark, width=4 * S)
    # 爪
    for sgn in (-1, 1):
        cx = w * S / 2 + sgn * 70 * S
        for k in (-1, 0, 1):
            ell(d, cx + k * 20 * S, 600 * S, 11 * S, 15 * S, fill=(240, 140, 40, 255), outline=dark, width=3 * S)
    return fin(im)


def monkey_stamp():
    """原创红底小金猴邮票；返回 (邮票底, 猴子层, 尾巴层, 尾根坐标)，三层同尺寸"""
    w, h = 560, 700
    im, d = perforated(w, h, (252, 248, 240, 255))
    S = SS
    m = 34 * S
    d.rectangle([m, m, w * S - m, h * S - m], fill=(196, 30, 36, 255))
    f = font(F_BRUSH, 64 * S)
    for k, ch in enumerate("庚申年"):
        d.text((92 * S, (110 + k * 70) * S), ch, font=f, fill=(255, 225, 120, 255), anchor="mm")
    d.text((w * S - 70 * S, h * S - 70 * S), "1980", font=font(F_EN_B, 30 * S), fill=(255, 240, 220, 255), anchor="rm")
    bg = fin(im)
    gold, dark, face = (228, 168, 52, 255), (95, 50, 18, 255), (250, 218, 170, 255)
    o = 6 * S
    # 尾巴：从身体右后方伸出、向上卷成螺旋
    tim, td = new_sprite(w, h)
    pts = []
    for k in range(80):
        u = k / 79
        ang = math.radians(114 - 420 * u)
        r = 110 - 88 * u
        pts.append(((430 + r * math.cos(ang)) * S, (390 + r * math.sin(ang)) * S))
    td.line(pts, fill=dark, width=30 * S, joint="curve")
    td.line(pts, fill=gold, width=18 * S, joint="curve")
    tail = fin(tim)
    root = (385, 490)
    mim, d = new_sprite(w, h)
    cx, cy = 290 * S, 430 * S
    ell(d, cx - 70 * S, cy + 120 * S, 70 * S, 45 * S, fill=gold, outline=dark, width=o)
    ell(d, cx + 60 * S, cy + 125 * S, 70 * S, 45 * S, fill=gold, outline=dark, width=o)
    ell(d, cx, cy + 30 * S, 120 * S, 130 * S, fill=gold, outline=dark, width=o)
    ell(d, cx + 10 * S, cy + 50 * S, 70 * S, 85 * S, fill=face)
    hx, hy = cx + 10 * S, cy - 150 * S
    for sgn in (-1, 1):
        ell(d, hx + sgn * 112 * S, hy + 5 * S, 40 * S, 40 * S, fill=gold, outline=dark, width=o)
        ell(d, hx + sgn * 112 * S, hy + 5 * S, 22 * S, 22 * S, fill=face)
    ell(d, hx, hy, 105 * S, 98 * S, fill=gold, outline=dark, width=o)
    ell(d, hx - 32 * S, hy - 8 * S, 42 * S, 44 * S, fill=face)
    ell(d, hx + 32 * S, hy - 8 * S, 42 * S, 44 * S, fill=face)
    ell(d, hx, hy + 38 * S, 66 * S, 44 * S, fill=face)
    for sgn in (-1, 1):
        ell(d, hx + sgn * 30 * S, hy - 10 * S, 11 * S, 14 * S, fill=(25, 15, 10, 255))
        ell(d, hx + sgn * 30 * S + 4 * S, hy - 15 * S, 4 * S, 4 * S, fill=(255, 255, 255, 255))
    d.arc([hx - 26 * S, hy + 20 * S, hx + 26 * S, hy + 58 * S], 20, 160, fill=dark, width=5 * S)
    ell(d, hx - 9 * S, hy + 22 * S, 4 * S, 3 * S, fill=dark)
    ell(d, hx + 9 * S, hy + 22 * S, 4 * S, 3 * S, fill=dark)
    px, py = cx + 15 * S, cy - 5 * S
    ell(d, px, py, 52 * S, 48 * S, fill=(250, 140, 150, 255), outline=(160, 50, 60, 255), width=4 * S)
    d.line([(px, py - 45 * S), (px + 8 * S, py + 40 * S)], fill=(230, 100, 115, 255), width=3 * S)
    d.polygon([(px, py - 45 * S), (px + 45 * S, py - 75 * S), (px + 20 * S, py - 35 * S)], fill=(80, 160, 70, 255))
    for sgn in (-1, 1):
        ell(d, px + sgn * 48 * S, py + 12 * S, 26 * S, 22 * S, fill=gold, outline=dark, width=o)
    return bg, fin(mim), tail, root


def compose_monkey(bg, monkey, tail, root, tail_ang=0.0):
    t = tail.rotate(tail_ang, Image.BICUBIC, center=root) if tail_ang else tail
    a = np.asarray(t.getchannel("A")).copy()
    a[:34, :] = 0; a[-34:, :] = 0; a[:, :34] = 0; a[:, -34:] = 0      # 只留在红底内
    t = t.copy(); t.putalpha(Image.fromarray(a))
    out = Image.alpha_composite(bg, t)
    return Image.alpha_composite(out, monkey)


def rabbit_stamp():
    """原创蓝兔子邮票"""
    w, h = 520, 660
    im, d = perforated(w, h, (252, 250, 245, 255))
    S = SS
    m = 32 * S
    d.rectangle([m, m, w * S - m, h * S - m], fill=(255, 246, 232, 255), outline=(220, 200, 180, 255), width=3 * S)
    blue, dark, pink = (102, 150, 228, 255), (35, 60, 130, 255), (250, 170, 185, 255)
    o = 6 * S
    cx, cy = 270 * S, 420 * S
    # 耳朵
    for sgn, tilt in ((-1, -12), (1, 14)):
        ex = cx + sgn * 55 * S
        ear, ed = Image.new("RGBA", (120 * S, 300 * S), (0, 0, 0, 0)), None
        ed = ImageDraw.Draw(ear)
        ell(ed, 60 * S, 150 * S, 50 * S, 140 * S, fill=blue, outline=dark, width=o)
        ell(ed, 60 * S, 165 * S, 24 * S, 100 * S, fill=pink)
        ear = ear.rotate(-tilt, Image.BICUBIC, expand=True)
        im.paste(ear, (int(ex - ear.width / 2), int(cy - 380 * S - ear.height / 2 + 150 * S)), ear)
    d = ImageDraw.Draw(im)
    # 身体
    ell(d, cx, cy + 120 * S, 115 * S, 120 * S, fill=blue, outline=dark, width=o)
    ell(d, cx, cy + 140 * S, 65 * S, 80 * S, fill=(220, 232, 252, 255))
    # 头
    ell(d, cx, cy - 40 * S, 120 * S, 108 * S, fill=blue, outline=dark, width=o)
    for sgn in (-1, 1):
        ell(d, cx + sgn * 44 * S, cy - 50 * S, 14 * S, 18 * S, fill=(25, 20, 40, 255))
        ell(d, cx + sgn * 44 * S + 5 * S, cy - 57 * S, 5 * S, 5 * S, fill=(255, 255, 255, 255))
        ell(d, cx + sgn * 75 * S, cy - 5 * S, 20 * S, 12 * S, fill=pink)
    ell(d, cx, cy - 15 * S, 10 * S, 8 * S, fill=(230, 100, 130, 255))
    d.arc([cx - 22 * S, cy - 18 * S, cx, cy + 10 * S], 0, 150, fill=dark, width=4 * S)
    d.arc([cx, cy - 18 * S, cx + 22 * S, cy + 10 * S], 30, 180, fill=dark, width=4 * S)
    # 胡萝卜
    d.polygon([(cx + 70 * S, cy + 90 * S), (cx + 150 * S, cy + 70 * S), (cx + 85 * S, cy + 130 * S)],
              fill=(245, 140, 50, 255), outline=(170, 80, 20, 255))
    d.polygon([(cx + 150 * S, cy + 70 * S), (cx + 185 * S, cy + 40 * S), (cx + 170 * S, cy + 80 * S)], fill=(80, 160, 70, 255))
    for sgn in (-1, 1):
        ell(d, cx + sgn * 95 * S, cy + 95 * S, 30 * S, 24 * S, fill=blue, outline=dark, width=o)
    f = font(F_BRUSH, 56 * S)
    for k, ch in enumerate("癸卯年"):
        d.text((w * S - 85 * S, (105 + k * 62) * S), ch, font=f, fill=(196, 30, 36, 255), anchor="mm")
    d.text((70 * S, h * S - 70 * S), "2023", font=font(F_EN_B, 28 * S), fill=(35, 60, 130, 255), anchor="lm")
    return fin(im)


def lotus_flower(size=320, open_=1.0, seed=0):
    w = h = size
    im, d = new_sprite(w, h)
    S = SS
    cx, cy = w * S / 2, h * S * 0.62
    r = np.random.default_rng(seed)
    petals = [(-62, 0.95), (62, 0.95), (-35, 1.0), (35, 1.0), (-10, 1.05), (12, 1.05), (0, 0.9)]
    for ang, ln in petals:
        ang = ang * open_ + r.normal(0, 3)
        L = size * 0.42 * ln * S
        a = math.radians(ang - 90)
        tip = (cx + L * math.cos(a), cy + L * math.sin(a))
        pts = []
        for k in range(30):
            u = k / 29
            wdt = math.sin(math.pi * u) ** 0.8 * L * 0.26
            x = cx + (tip[0] - cx) * u
            y = cy + (tip[1] - cy) * u
            pts.append((x + wdt * -math.sin(a), y + wdt * math.cos(a)))
        for k in range(29, -1, -1):
            u = k / 29
            wdt = math.sin(math.pi * u) ** 0.8 * L * 0.26
            x = cx + (tip[0] - cx) * u
            y = cy + (tip[1] - cy) * u
            pts.append((x - wdt * -math.sin(a), y - wdt * math.cos(a)))
        d.polygon(pts, fill=(232, 80, 110, 235), outline=(170, 30, 60, 255), width=3 * S)
        # 花瓣尖端提亮
        inner = [(cx + (px - cx) * 0.7 + (tip[0] - cx) * 0.3, cy + (py - cy) * 0.7 + (tip[1] - cy) * 0.3) for px, py in pts]
        d.polygon(inner, fill=(250, 170, 185, 200))
    ell(d, cx, cy - size * 0.05 * S, size * 0.09 * S, size * 0.05 * S, fill=(245, 200, 60, 255), outline=(170, 120, 20, 255))
    return fin(im)


def lotus_leaf(size, seed, color=(38, 82, 56)):
    r = np.random.default_rng(seed)
    n = size
    yy, xx = np.mgrid[0:n, 0:n] - n / 2
    ang = np.arctan2(yy, xx)
    rad = np.hypot(xx, yy)
    edge = n * 0.45 * (1 + 0.08 * np.sin(ang * 5 + r.random() * 6) + 0.05 * np.sin(ang * 11 + r.random() * 6))
    a = smoothstep(edge, edge - 6, rad)
    tex = gaussian_filter(r.random((n, n)), 2)
    a = a * (0.7 + 0.3 * (tex - tex.min()) / (tex.max() - tex.min()))
    a *= 0.75 + 0.25 * np.clip(rad / (n * 0.45), 0, 1)       # 中间淡、边缘浓（泼墨感）
    # 叶脉
    vein = np.zeros_like(a)
    for k in range(9):
        th = k * 2 * math.pi / 9 + r.random() * 0.2
        dd = np.abs(-np.sin(th) * xx + np.cos(th) * yy)
        along = np.cos(th) * xx + np.sin(th) * yy
        vein = np.maximum(vein, np.exp(-dd / 2.0) * (along > 0) * (rad < edge * 0.92))
    rgb = np.ones((n, n, 3)) * np.array(color, np.float32)
    rgb = rgb * (1 + 0.6 * vein[..., None])
    out = np.zeros((n, n, 4), np.uint8)
    out[..., :3] = np.clip(rgb, 0, 255)
    out[..., 3] = np.clip(a * 235, 0, 255)
    return Image.fromarray(out, "RGBA")


def pipe_sprite():
    w, h = 420, 220
    im, d = new_sprite(w, h)
    S = SS
    dark = (60, 32, 15, 255)
    # 烟杆
    d.line([(40 * S, 70 * S), (180 * S, 120 * S), (300 * S, 150 * S)], fill=dark, width=26 * S, joint="curve")
    d.line([(40 * S, 70 * S), (180 * S, 120 * S), (300 * S, 150 * S)], fill=(40, 30, 25, 255), width=16 * S, joint="curve")
    # 烟斗锅
    d.rounded_rectangle([270 * S, 50 * S, 380 * S, 200 * S], radius=40 * S, fill=(140, 80, 38, 255), outline=dark, width=6 * S)
    ell(d, 325 * S, 55 * S, 55 * S, 16 * S, fill=(50, 30, 20, 255), outline=dark, width=5 * S)
    d.arc([285 * S, 80 * S, 340 * S, 180 * S], 120, 240, fill=(190, 120, 60, 255), width=6 * S)
    return fin(im)


def car_body():
    w, h = 520, 200
    im, d = new_sprite(w, h)
    S = SS
    red, dark = (215, 35, 40, 255), (110, 15, 20, 255)
    body = [(20, 140), (30, 105), (120, 92), (190, 50), (330, 45), (400, 90), (495, 105), (505, 140), (490, 160), (30, 160)]
    d.polygon([(x * S, y * S) for x, y in body], fill=red, outline=dark, width=6 * S)
    d.polygon([(x * S, y * S) for x, y in [(205, 60), (320, 56), (375, 92), (175, 95)]], fill=(170, 215, 240, 255), outline=dark, width=4 * S)
    d.line([(272 * S, 58 * S), (272 * S, 94 * S)], fill=dark, width=5 * S)
    ell(d, 480 * S, 118 * S, 12 * S, 8 * S, fill=(255, 230, 120, 255))
    return fin(im)


def wheel(angle, r=42):
    im, d = new_sprite(2 * r + 4, 2 * r + 4)
    S = SS
    c = (r + 2) * S
    ell(d, c, c, r * S, r * S, fill=(25, 25, 28, 255))
    ell(d, c, c, r * 0.55 * S, r * 0.55 * S, fill=(190, 190, 195, 255))
    for k in range(5):
        a = angle + k * 2 * math.pi / 5
        d.line([(c, c), (c + r * 0.5 * S * math.cos(a), c + r * 0.5 * S * math.sin(a))], fill=(90, 90, 95, 255), width=5 * S)
    return fin(im)


def dog(phase):
    """原创小狗，phase 控制跑步腿和尾巴"""
    w, h = 300, 220
    im, d = new_sprite(w, h)
    S = SS
    cream, brown, dark = (245, 228, 196, 255), (150, 95, 55, 255), (70, 45, 25, 255)
    o = 5 * S
    for k, base in enumerate((95, 125, 185, 215)):
        sw = math.sin(phase + k * math.pi / 2 + (math.pi if k % 2 else 0)) * 22
        d.line([(base * S, 140 * S), ((base + sw) * S, 195 * S)], fill=dark, width=16 * S)
        d.line([(base * S, 140 * S), ((base + sw) * S, 195 * S)], fill=cream, width=9 * S)
    ta = math.sin(phase * 2) * 0.5
    d.line([(70 * S, 120 * S), ((70 - 45 * math.cos(0.9 + ta)) * S, (120 - 45 * math.sin(0.9 + ta)) * S)], fill=dark, width=12 * S)
    ell(d, 155 * S, 125 * S, 95 * S, 45 * S, fill=cream, outline=dark, width=o)
    ell(d, 130 * S, 115 * S, 30 * S, 20 * S, fill=brown)
    ell(d, 235 * S, 85 * S, 48 * S, 44 * S, fill=cream, outline=dark, width=o)
    ell(d, 212 * S, 95 * S, 18 * S, 34 * S, fill=brown, outline=dark, width=4 * S)
    ell(d, 252 * S, 78 * S, 7 * S, 8 * S, fill=(20, 15, 10, 255))
    ell(d, 280 * S, 92 * S, 9 * S, 7 * S, fill=(20, 15, 10, 255))
    d.arc([245 * S, 90 * S, 280 * S, 115 * S], 20, 120, fill=dark, width=3 * S)
    return fin(im)


def boy(phase):
    """木刻风小人：背着包袱走路（黑色剪影）"""
    w, h = 160, 260
    im, d = new_sprite(w, h)
    S = SS
    k = INK + (255,)
    for sgn in (-1, 1):
        sw = math.sin(phase) * 20 * sgn
        d.line([(80 * S, 160 * S), ((80 + sw) * S, 245 * S)], fill=k, width=16 * S)
    d.polygon([(55 * S, 85 * S), (105 * S, 85 * S), (115 * S, 170 * S), (45 * S, 170 * S)], fill=k)
    ell(d, 80 * S, 55 * S, 28 * S, 28 * S, fill=k)
    d.line([(60 * S, 100 * S), (140 * S, 20 * S)], fill=k, width=7 * S)
    ell(d, 140 * S, 30 * S, 22 * S, 20 * S, fill=k)
    return fin(im)


# ================================================================ 场景
class Scene:
    def frame(self, tl):
        raise NotImplementedError


class Fenghuang(Scene):
    """凤凰：水墨远山 + 吊脚楼 + 沱江 + 小船"""
    def __init__(self):
        LW = 2400
        self.LW = LW
        self.bg = paper(LW, H, 3)
        sky = np.asarray(self.bg, np.float32)
        yy = np.arange(H)[:, None, None]
        sky = sky * (1 - 0.04 * np.exp(-yy / 400)) + np.array([0, 4, 10]) * np.exp(-yy / 500)
        self.bg = Image.fromarray(np.clip(sky, 0, 255).astype(np.uint8))
        self.far, _ = ink_mountains(LW, H, 520, 260, (95, 105, 115), 0.45, 11, fall=60, mist=180)
        self.mid, _ = ink_mountains(LW, H, 600, 200, (55, 62, 68), 0.7, 23, fall=50, mist=140)
        # 吊脚楼一排
        self.houses = self._houses(LW)
        # 倒影
        self.refl = self.houses.transpose(Image.FLIP_TOP_BOTTOM).filter(ImageFilter.GaussianBlur(3))
        self.boat = self._boat()
        self.sun = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        ImageDraw.Draw(self.sun).ellipse([10, 10, 190, 190], fill=(214, 70, 52, 200))
        self.sun = self.sun.filter(ImageFilter.GaussianBlur(1.5))
        self.seal = seal("凤凰", 96)
        self.title_layer = self._title()

    def _houses(self, LW):
        im = Image.new("RGBA", (LW * SS, 520 * SS), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        r = np.random.default_rng(5)
        S = SS
        x = 40
        water = 470          # 图层内水面位置
        while x < LW - 120:
            w = r.uniform(110, 200)
            hgt = r.uniform(110, 190)
            base = water - r.uniform(50, 85)
            wood = tuple(int(v) for v in (r.uniform(60, 85), r.uniform(42, 55), r.uniform(32, 40))) + (255,)
            for px in np.linspace(x + 8, x + w - 8, int(w // 32) + 2):
                d.line([(px * S, base * S), (px * S, (water + 8) * S)], fill=(45, 35, 30, 255), width=5 * S)
            d.line([(x * S, (base + 25) * S), ((x + w) * S, (base + 25) * S)], fill=(45, 35, 30, 255), width=3 * S)
            d.rectangle([x * S, (base - hgt) * S, (x + w) * S, base * S], fill=wood)
            d.line([(x * S, (base - hgt * 0.32) * S), ((x + w) * S, (base - hgt * 0.32) * S)], fill=(40, 28, 22, 255), width=4 * S)
            for wx in np.arange(x + 18, x + w - 30, 42):
                lit = r.random() < 0.6
                d.rectangle([wx * S, (base - hgt * 0.8) * S, (wx + 24) * S, (base - hgt * 0.5) * S],
                            fill=(236, 196, 110, 255) if lit else (35, 28, 25, 255))
            top = base - hgt
            up = r.uniform(14, 26)
            roof = [(x - w * 0.16, top - up), (x - w * 0.06, top + 6), (x + w * 1.06, top + 6), (x + w * 1.16, top - up),
                    (x + w * 0.95, top - hgt * 0.22), (x + w * 0.05, top - hgt * 0.22)]
            d.polygon([(px * S, py * S) for px, py in roof], fill=(30, 28, 30, 255))
            x += w + r.uniform(-10, 60)
        return im.resize((LW, 520), Image.LANCZOS)

    def _boat(self):
        im, d = new_sprite(260, 110)
        S = SS
        d.chord([0, -60 * S, 260 * S, 100 * S], 20, 160, fill=(40, 32, 28, 255))
        d.pieslice([60 * S, 10 * S, 190 * S, 110 * S], 180, 360, fill=(25, 22, 22, 255))
        d.line([(200 * S, 70 * S), (250 * S, 0)], fill=(60, 45, 35, 255), width=4 * S)
        return fin(im)

    def _title(self):
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.text((W / 2, 250), "黄永玉", font=font(F_BRUSH, 190), fill=INK + (255,), anchor="mm")
        d.text((W / 2, 400), "永远长不大的老头", font=font(F_BRUSH, 72), fill=(70, 60, 50, 255), anchor="mm")
        d.text((W / 2, 470), "The Old Man Who Never Grew Up", font=font(F_EN, 34), fill=(90, 80, 70, 255), anchor="mm")
        return im

    def frame(self, tl):
        tl = max(0.0, tl)
        pan = 360 * smoothstep(0, 16, tl)
        c = self.bg.crop((int(pan * 0.2), 0, int(pan * 0.2) + W, H)).copy()
        place(c, self.sun, 1500 - pan * 0.1, 210)
        c.paste(self.far, (-int(pan * 0.35), 0), self.far)
        c.paste(self.mid, (-int(pan * 0.6), 0), self.mid)
        # 江面
        wy = 760
        river = Image.new("RGBA", (W, H - wy + 40), (150, 170, 180, 0))
        ra = np.zeros((H - wy + 40, W, 4), np.uint8)
        ra[..., :3] = (140, 160, 168)
        ra[..., 3] = (np.linspace(90, 150, H - wy + 40)[:, None] * np.ones((1, W))).astype(np.uint8)
        river = Image.fromarray(ra, "RGBA")
        c.paste(river, (0, wy - 10), river)
        # 倒影（逐行水波位移）
        ox = -int(pan)
        refl = self.refl.crop((-ox, 0, -ox + W, 300))
        arr = np.asarray(refl).copy()
        for y in range(arr.shape[0]):
            s = int(4 * math.sin(y * 0.15 + tl * 2.5) * (y / 300 + 0.2))
            arr[y] = np.roll(arr[y], s, axis=0)
        arr[..., 3] = (arr[..., 3] * np.linspace(0.45, 0.0, arr.shape[0])[:, None]).astype(np.uint8)
        rim = Image.fromarray(arr, "RGBA")
        c.paste(rim, (0, wy), rim)
        c.paste(self.houses, (ox, wy - 470), self.houses)
        # 水波纹
        d = ImageDraw.Draw(c, "RGBA")
        r = np.random.default_rng(3)
        for k in range(40):
            y = wy + 20 + r.random() * 280
            x = (r.random() * 2600 - pan * 1.0 + tl * 12 * (1 + r.random())) % 2400 - 200
            ln = 30 + r.random() * 60
            d.line([(x, y), (x + ln, y)], fill=(250, 250, 245, 90), width=2)
        # 小船
        bx = 200 + tl * 55 - pan * 0.9
        place(c, self.boat, bx, wy + 120 + 4 * math.sin(tl * 1.7), angle=2 * math.sin(tl * 1.3))
        # 飞鸟
        for k in range(5):
            bx2 = (1250 + k * 90 + tl * 70) % 2200 - 100
            by = 120 + k * 18 + 10 * math.sin(tl + k)
            flap = 10 * math.sin(tl * 8 + k)
            d.line([(bx2 - 14, by - flap), (bx2, by), (bx2 + 14, by - flap)], fill=(40, 40, 40, 200), width=3)
        place(c, self.seal, 1780, 160, alpha=smoothstep(1, 2.5, tl))
        # 片头标题
        ta = smoothstep(0.2, 1.4, tl) * (1 - smoothstep(6.0, 7.5, tl))
        if ta > 0:
            place(c, self.title_layer, W / 2, H / 2, alpha=ta)
        return c


class Woodcut(Scene):
    """黑白木刻：少年背着包袱沿路远行；开头有「刻刀」刻出画面的效果"""
    def __init__(self):
        self.bg = paper(W, H, 7)
        self.carved = self._carve()
        self.knife = self._knife()
        self.seal = seal("木刻", 96)
        # 路径：从近到远
        self.path = [(820, 1010, 1.0), (930, 860, 0.7), (1010, 760, 0.5), (1060, 690, 0.35)]

    def _carve(self):
        """白色 = 刻掉的部分（露出纸色）"""
        S = SS
        im = Image.new("L", (W * S, H * S), 0)
        d = ImageDraw.Draw(im)
        r = np.random.default_rng(4)
        sx, sy = 1380 * S, 300 * S
        # 天空：围绕太阳的同心刻痕
        for rad in np.arange(170, 1900, 24):
            a0 = r.random() * 360
            while a0 < 360 + 360:
                seg = r.uniform(8, 40) * 300 / rad + 6
                wdt = int(max(2, 7 - rad / 400) * S)
                d.arc([sx - rad * S, sy - rad * S, sx + rad * S, sy + rad * S], a0, a0 + seg, fill=255, width=wdt)
                a0 += seg + r.uniform(4, 18) * 300 / rad + 3
        # 太阳
        ell(d, sx, sy, 125 * S, 125 * S, fill=255)
        for rr in (95, 65, 35):
            d.arc([sx - rr * S, sy - rr * S, sx + rr * S, sy + rr * S], 0, 360, fill=0, width=5 * S)
        # 远山与近山：山脊白线 + 平行刻线；山体以下遮住天空刻痕
        for base, amp, seed, spacing in ((640, 180, 31, 20), (760, 140, 47, 16)):
            ridge = base - amp * (0.5 + 0.5 * fbm1d(W, 600, 5, seed))
            poly = [(0, H * S)] + [(x * S, ridge[x] * S) for x in range(0, W, 4)] + [(W * S, H * S)]
            d.polygon(poly, fill=0)
            d.line([(x * S, ridge[x] * S) for x in range(0, W, 4)], fill=255, width=6 * S)
            for k in range(1, 14):
                pts = [(x * S, (ridge[x] + k * spacing + 6 * math.sin(x / 90 + k)) * S) for x in range(0, W, 6)]
                d.line(pts, fill=255, width=max(2, int((4 - k * 0.2) * S)))
        # 道路（透视）
        road_l = [(560, 1080), (760, 900), (960, 760), (1050, 680)]
        road_r = [(1250, 1080), (1120, 900), (1080, 760), (1072, 680)]
        d.polygon([(x * S, y * S) for x, y in road_l + road_r[::-1]], fill=255)
        for k in range(26):
            u = k / 26
            y = 1080 - (1080 - 690) * (1 - (1 - u) ** 2)
            xl = np.interp(-y, [-p[1] for p in road_l], [p[0] for p in road_l])
            xr = np.interp(-y, [-p[1] for p in road_r], [p[0] for p in road_r])
            wdt = max(1, int((1 - u) * 5 * S))
            for j in range(3):
                xa = xl + (xr - xl) * (0.15 + 0.3 * j + r.random() * 0.1)
                d.line([(xa * S, y * S), ((xa + (xr - xl) * 0.12) * S, y * S)], fill=0, width=wdt)
        # 近处草丛
        for _ in range(160):
            x, y = r.random() * W, 900 + r.random() * 180
            if 520 < x < 1280 and y > 850:
                continue
            hgt = 20 + r.random() * 30
            d.line([(x * S, y * S), ((x - 6) * S, (y - hgt) * S)], fill=255, width=3 * S)
            d.line([(x * S, y * S), ((x + 7) * S, (y - hgt * 0.8) * S)], fill=255, width=3 * S)
        im = im.resize((W, H), Image.LANCZOS)
        return np.asarray(im, np.float32) / 255

    def _knife(self):
        im, d = new_sprite(120, 300)
        S = SS
        d.rounded_rectangle([40 * S, 90 * S, 80 * S, 290 * S], radius=12 * S, fill=(175, 120, 70, 255), outline=(90, 55, 30, 255), width=4 * S)
        d.polygon([(45 * S, 92 * S), (75 * S, 92 * S), (70 * S, 10 * S), (50 * S, 25 * S)], fill=(200, 205, 210, 255), outline=(90, 95, 100, 255))
        return fin(im)

    def frame(self, tl):
        tl = max(0.0, tl)
        # 刻刀从左到右「刻」出画面
        prog = smoothstep(0.0, 3.2, tl)
        xx = np.arange(W)[None, :] + 0.25 * np.arange(H)[:, None]
        front = -300 + prog * (W + 600)
        n = 30 * np.sin(np.arange(H) / 37)[:, None]
        mask = np.clip((front + n - xx) / 40, 0, 1)
        white = self.carved * mask
        ink = 1 - white
        tex = (np.asarray(self.bg, np.float32) / 255)
        speck = 0.92 + 0.08 * np.random.default_rng(int(tl * 3)).random((H, W))   # 印刷斑点（每 1/3 秒变一次）
        out = tex * (1 - ink * 0.9 * speck)[..., None] + np.array(INK, np.float32)[None, None] / 255 * (ink * 0.9 * speck)[..., None] * 0
        c = Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8))
        if prog < 1:
            fy = 540 + 200 * math.sin(tl * 6)
            place(c, self.knife, front - 0.25 * fy + 30, fy - 120, angle=-30)
        # 少年
        u = smoothstep(2.0, 16.0, tl)
        ys = [p[1] for p in self.path]
        k = min(len(self.path) - 2, int(u * (len(self.path) - 1)))
        f = u * (len(self.path) - 1) - k
        x = self.path[k][0] + (self.path[k + 1][0] - self.path[k][0]) * f
        y = self.path[k][1] + (self.path[k + 1][1] - self.path[k][1]) * f
        s = self.path[k][2] + (self.path[k + 1][2] - self.path[k][2]) * f
        b = boy(tl * 6)
        place(c, b, x, y - 130 * s, scale=s, alpha=smoothstep(1.5, 2.5, tl))
        place(c, self.seal, 130, 120, alpha=smoothstep(3, 4, tl))
        return c


class Beijing(Scene):
    """《边城》书页 + 墙上挂着木刻 + 1953 北京"""
    def __init__(self, woodcut):
        self.bg = paper(W, H, 9, base=(240, 230, 208))
        self.book = self._book()
        self.print_ = self._framed(woodcut)
        self.gate = self._gate()
        self.text = Image.new("RGBA", (600, 160), (0, 0, 0, 0))
        d = ImageDraw.Draw(self.text)
        d.text((300, 60), "1953年  北京", font=font(F_BRUSH, 96), fill=INK + (255,), anchor="mm")
        d.text((300, 135), "Beijing", font=font(F_EN, 34), fill=(90, 80, 70, 255), anchor="mm")

    def _book(self):
        w, h = 980, 640
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([10, 30, w - 10, h - 10], radius=18, fill=(130, 70, 45, 255))
        pg = paper(w // 2 - 30, h - 70, 13, base=(250, 244, 228), vig=False)
        im.paste(pg, (25, 40))
        im.paste(pg.transpose(Image.FLIP_LEFT_RIGHT), (w // 2 + 5, 40))
        gut = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        gd = ImageDraw.Draw(gut)
        for k in range(60):
            a = int(70 * (1 - k / 60) ** 2)
            gd.line([(w // 2 - k, 40), (w // 2 - k, h - 30)], fill=(80, 60, 40, a))
            gd.line([(w // 2 + 5 + k, 40), (w // 2 + 5 + k, h - 30)], fill=(80, 60, 40, a))
        im = Image.alpha_composite(im, gut)
        d = ImageDraw.Draw(im)
        # 左页：竖排书名
        d.text((330, 200), "边", font=font(F_BRUSH, 150), fill=INK + (255,), anchor="mm")
        d.text((330, 360), "城", font=font(F_BRUSH, 150), fill=INK + (255,), anchor="mm")
        for k, ch in enumerate("沈从文"):
            d.text((170, 230 + k * 56), ch, font=font(F_BRUSH, 48), fill=(80, 70, 60, 255), anchor="mm")
        d.rectangle([150, 410, 192, 452], fill=RED + (220,))
        # 右页：小幅水墨山水
        pw, ph = 380, 300
        mini = paper(pw, ph, 17, base=(250, 244, 228), vig=False)
        m1, _ = ink_mountains(pw, ph, 170, 90, (80, 90, 95), 0.6, 61, fall=25, mist=60)
        m2, _ = ink_mountains(pw, ph, 210, 60, (40, 45, 50), 0.8, 73, fall=20, mist=40)
        mini.paste(m1, (0, 0), m1)
        mini.paste(m2, (0, 0), m2)
        md = ImageDraw.Draw(mini, "RGBA")
        md.rectangle([0, 230, pw, ph], fill=(150, 170, 175, 110))
        md.chord([200, 228, 260, 262], 0, 180, fill=(40, 32, 28, 255))
        md.rectangle([60, 190, 100, 230], fill=(70, 50, 40, 255))
        md.polygon([(52, 192), (80, 172), (108, 192)], fill=(30, 28, 30, 255))
        im.paste(mini, (w // 2 + 55, 130))
        return shadowed(im, off=(14, 18), blur=14, op=0.3)

    def _framed(self, woodcut):
        pic = woodcut.frame(16).resize((460, 259), Image.LANCZOS)
        fw, fh = pic.width + 60, pic.height + 60
        im = Image.new("RGBA", (fw, fh + 120), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.line([(fw / 2, 10), (20, 120)], fill=(90, 70, 50, 255), width=3)
        d.line([(fw / 2, 10), (fw - 20, 120)], fill=(90, 70, 50, 255), width=3)
        ell(d, fw / 2, 10, 8, 8, fill=(60, 50, 40, 255))
        d.rectangle([0, 120, fw, 120 + fh], fill=(110, 70, 40, 255))
        d.rectangle([14, 134, fw - 14, 106 + fh], fill=(245, 238, 222, 255))
        im.paste(pic, (30, 150))
        return shadowed(im, off=(10, 14), blur=10, op=0.3)

    def _gate(self):
        im, d = new_sprite(520, 300)
        S = SS
        red, roof, dark = (190, 45, 40, 255), (232, 175, 50, 255), (90, 30, 25, 255)
        d.rectangle([0, 190 * S, 520 * S, 300 * S], fill=red)
        for x in (140, 260, 380):
            d.chord([(x - 34) * S, 215 * S, (x + 34) * S, 330 * S], 180, 360, fill=dark)
        d.rectangle([90 * S, 120 * S, 430 * S, 190 * S], fill=red)
        for x in range(110, 430, 40):
            d.line([(x * S, 125 * S), (x * S, 190 * S)], fill=dark, width=4 * S)
        d.polygon([(x * S, y * S) for x, y in [(40, 125), (70, 95), (450, 95), (480, 125)]], fill=roof)
        d.rectangle([140 * S, 60 * S, 380 * S, 95 * S], fill=red)
        d.polygon([(x * S, y * S) for x, y in [(100, 65), (140, 30), (380, 30), (420, 65)]], fill=roof)
        return fin(im)

    def frame(self, tl):
        tl = max(0.0, tl)
        c = self.bg.copy()
        a = smoothstep(0, 1.0, tl)
        place(c, self.book, 640, 520 + 30 * (1 - ease_out_back(tl / 1.2)), alpha=a)
        ang = 2.5 * math.sin(tl * 1.6) * math.exp(-tl * 0.08)
        place(c, self.print_, 1500, 290, angle=ang, alpha=smoothstep(1.0, 2.0, tl))
        ga = smoothstep(3.8, 5.0, tl)
        place(c, self.gate, 1500, 770 + 40 * (1 - ga), scale=0.55, alpha=ga)
        place(c, self.text, 1500, 610, alpha=smoothstep(4.0, 5.0, tl))
        # 镜头缓推
        z = 1 + 0.05 * smoothstep(0, 16, tl)
        if z > 1.001:
            cw, ch = W / z, H / z
            c = c.crop((int((W - cw) / 2), int((H - ch) / 2), int((W + cw) / 2), int((H + ch) / 2))).resize((W, H), Image.BICUBIC)
        return c


class Owl(Scene):
    """夜里的猫头鹰：睁一只眼闭一只眼，还会换眼眨"""
    def __init__(self):
        p = np.asarray(paper(W, H, 21), np.float32) / 255
        yy = np.linspace(0, 1, H)[:, None, None]
        sky = np.array([28, 36, 78]) * (1 - yy) + np.array([60, 70, 120]) * yy
        self.bg = Image.fromarray(np.clip(sky * (0.85 + 0.15 * p) * 1.0, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(self.bg, "RGBA")
        for rr, al in ((260, 20), (200, 30), (150, 60)):
            ell(d, 1450, 280, rr, rr, fill=(255, 240, 190, al))
        ell(d, 1450, 280, 120, 120, fill=(252, 240, 200, 255))
        ell(d, 1410, 250, 22, 22, fill=(235, 220, 175, 255))
        ell(d, 1490, 320, 14, 14, fill=(235, 220, 175, 255))
        r = np.random.default_rng(8)
        self.stars = [(r.random() * W, r.random() * 650, r.random() * 6.28, 1 + r.random() * 2.5) for _ in range(120)]
        self.branch = self._branch()
        self.owls = {k: owl_sprite(*k) for k in [("open", "closed"), ("closed", "closed"), ("closed", "open")]}
        self.leaf = self._leaf()

    def _branch(self):
        im, d = new_sprite(W, 400)
        S = SS
        pts = [(x * S, (200 + 40 * math.sin(x / 300) + (x - 900) ** 2 / 20000) * S) for x in range(0, W, 10)]
        for wdt, col in ((54, (45, 30, 22, 255)), (40, (85, 58, 38, 255))):
            d.line(pts, fill=col, width=wdt * S, joint="curve")
        d.line([(1300 * S, 215 * S), (1600 * S, 80 * S), (1750 * S, 60 * S)], fill=(45, 30, 22, 255), width=22 * S, joint="curve")
        d.line([(400 * S, 215 * S), (250 * S, 90 * S)], fill=(45, 30, 22, 255), width=18 * S)
        return fin(im)

    def _leaf(self):
        im, d = new_sprite(80, 50)
        S = SS
        ell(d, 40 * S, 25 * S, 36 * S, 16 * S, fill=(70, 130, 80, 255), outline=(30, 60, 35, 255), width=3 * S)
        d.line([(6 * S, 25 * S), (74 * S, 25 * S)], fill=(30, 60, 35, 255), width=2 * S)
        return fin(im)

    def frame(self, tl):
        tl = max(0.0, tl)
        c = self.bg.copy()
        d = ImageDraw.Draw(c, "RGBA")
        for x, y, ph, s in self.stars:
            a = int(140 + 110 * math.sin(tl * 2 + ph))
            ell(d, x, y, s, s, fill=(255, 250, 220, a))
        c.paste(self.branch, (0, 700), self.branch)
        for k, (lx, ly) in enumerate([(1560, 760), (1700, 740), (300, 790), (220, 780)]):
            place(c, self.leaf, lx, ly, angle=20 * math.sin(tl * 2 + k) + k * 40)
        # 眨眼时间表：左眼睁、右眼闭；偶尔眨一下；10 秒后换眼
        if 9.6 < tl < 10.4:
            key = ("closed", "closed")
        elif tl >= 10.4 and tl < 14.0:
            key = ("closed", "open")
        else:
            key = ("open", "closed")
        for b0 in (3.5, 6.8, 13.0):
            if b0 < tl < b0 + 0.18:
                key = ("closed", "closed")
        ang = 4 * math.sin(tl * 1.2)
        bob = 6 * math.sin(tl * 2.4)
        place(c, self.owls[key], 960, 640 + bob, angle=ang, scale=0.95)
        z = 1 + 0.06 * smoothstep(0, 16, tl)
        cw, ch = W / z, H / z
        return c.crop((int((W - cw) / 2), int((H - ch) / 2), int((W + cw) / 2), int((H + ch) / 2))).resize((W, H), Image.BICUBIC)


class Monkey(Scene):
    """红底金猴邮票飞入、尾巴摇摆、闪光"""
    def __init__(self):
        self.bg = paper(W, H, 25)
        d = ImageDraw.Draw(self.bg, "RGBA")
        r = np.random.default_rng(12)
        for _ in range(70):                       # 彩纸屑
            x, y = r.random() * W, r.random() * H
            if 600 < x < 1320:
                continue
            col = [(220, 60, 60, 120), (240, 180, 50, 120), (80, 150, 220, 100)][r.integers(3)]
            a = r.random() * 6.28
            d.polygon([(x + 12 * math.cos(a + k * 2.1), y + 12 * math.sin(a + k * 2.1)) for k in range(3)], fill=col)
        self.parts = monkey_stamp()
        self.stamp = compose_monkey(*self.parts)
        self.pad = 48
        self.shadow = shadowed(self.parts[0], off=(16, 22), blur=16, op=0.35)
        self.seal = seal("猴年", 100)

    def frame(self, tl):
        tl = max(0.0, tl)
        c = self.bg.copy()
        u = ease_out_back(tl / 1.4, 1.2)
        y = -500 + (500 + 500) * u
        ang = 25 * (1 - u)
        sc = 1.0
        st = self.shadow.copy()
        top = compose_monkey(*self.parts, tail_ang=10 * math.sin(tl * 2.2))
        st.paste(top, (self.pad, self.pad), top)
        merged = st
        place(c, merged, 960, y, scale=sc, angle=ang)
        # 闪光
        d = ImageDraw.Draw(c, "RGBA")
        r = np.random.default_rng(2)
        for k in range(10):
            ph = r.random() * 6.28
            a = max(0, math.sin(tl * 3 + ph)) * smoothstep(1.4, 2.0, tl)
            x = 960 + (r.random() - 0.5) * 900
            y2 = 560 + (r.random() - 0.5) * 900
            if abs(x - 960) < 360 and abs(y2 - 560) < 440:
                continue
            s = 14 + 12 * a
            col = (240, 180, 40, int(220 * a))
            d.polygon([(x, y2 - s), (x + s * 0.25, y2), (x, y2 + s), (x - s * 0.25, y2)], fill=col)
            d.polygon([(x - s, y2), (x, y2 + s * 0.25), (x + s, y2), (x, y2 - s * 0.25)], fill=col)
        place(c, self.seal, 1500, 260, alpha=smoothstep(4, 5, tl), angle=-8)
        return c


class Lotus(Scene):
    """泼墨荷塘 + 稿纸上写出书名"""
    def __init__(self):
        self.bg = paper(W, H, 31)
        r = np.random.default_rng(3)
        self.leaves = []
        for k, (x, y, s) in enumerate([(250, 780, 620), (650, 900, 520), (1050, 760, 560), (420, 470, 420),
                                        (1400, 930, 460), (1700, 700, 500), (900, 420, 300)]):
            self.leaves.append((x, y, lotus_leaf(s, 40 + k, color=(30 + k * 4, 75 + k * 3, 50))))
        self.flowers = [(520, 560, lotus_flower(340, 1.0, 1)), (1150, 470, lotus_flower(300, 0.8, 2)),
                        (1560, 520, lotus_flower(260, 0.55, 3))]
        self.dragonfly = self._dragonfly()
        self.sheet = self._sheet()
        # 墨点飞溅
        sp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        sd = ImageDraw.Draw(sp)
        for _ in range(60):
            x, y = r.normal(700, 400), r.normal(650, 220)
            rr = abs(r.normal(0, 6)) + 1
            ell(sd, x, y, rr, rr, fill=(25, 40, 30, 200))
        self.splash = sp

    def _dragonfly(self):
        im, d = new_sprite(160, 120)
        S = SS
        for sgn in (-1, 1):
            for dy in (-12, 12):
                ell(d, (80 + sgn * 40) * S, (50 + dy * 0.5) * S, 40 * S, 10 * S, fill=(200, 230, 245, 170), outline=(90, 120, 140, 200), width=2 * S)
        d.line([(80 * S, 30 * S), (80 * S, 115 * S)], fill=(200, 60, 40, 255), width=8 * S)
        ell(d, 80 * S, 28 * S, 11 * S, 11 * S, fill=(60, 120, 60, 255))
        return fin(im)

    def _sheet(self):
        w, h = 640, 820
        pg = paper(w, h, 41, base=(250, 245, 232), vig=False)
        im = pg.convert("RGBA")
        d = ImageDraw.Draw(im)
        self.cells = []
        cols, rows, cs = 9, 12, 60
        ox, oy = (w - cols * cs) / 2, 60
        for rr in range(rows):
            for cc in range(cols):
                x, y = ox + cc * cs, oy + rr * cs
                d.rectangle([x + 4, y + 4, x + cs - 4, y + cs - 4], outline=(200, 80, 70, 200), width=2)
                self.cells.append((x + cs / 2, y + cs / 2))
        d.rectangle([5, 5, w - 5, h - 5], outline=(200, 80, 70, 230), width=3)
        return im

    def frame(self, tl):
        tl = max(0.0, tl)
        c = self.bg.copy()
        sa = smoothstep(0, 1.2, tl)
        if sa > 0:
            place(c, self.splash, W / 2, H / 2, alpha=sa * 0.8)
        for k, (x, y, lf) in enumerate(self.leaves):
            g = ease_out_back((tl - k * 0.12) / 1.0, 1.0)
            if g > 0.02:
                place(c, lf, x, y, scale=max(0.02, g), angle=3 * math.sin(tl * 0.9 + k))
        d = ImageDraw.Draw(c, "RGBA")
        for k, (x, y, fl) in enumerate(self.flowers):
            sw = 4 * math.sin(tl * 1.1 + k * 1.7)
            d.line([(x, y + 80), (x - 10 + sw * 2, H + 20)], fill=(40, 70, 45, 230), width=7)
            g = ease_out_back((tl - 0.8 - k * 0.25) / 0.9, 1.4)
            if g > 0.02:
                place(c, fl, x + sw * 3, y, scale=max(0.02, g), angle=sw)
        # 蜻蜓
        dx = 300 + tl * 90
        dy = 300 + 60 * math.sin(tl * 1.4) + 25 * math.sin(tl * 4.1)
        place(c, self.dragonfly, dx, dy, angle=-70 + 10 * math.sin(tl * 3), alpha=smoothstep(1.5, 2.5, tl))
        # 稿纸：后半段从右侧滑入，逐字写出书名
        su = smoothstep(7.0, 8.6, tl)
        if su > 0:
            sheet = self.sheet.copy()
            sd = ImageDraw.Draw(sheet)
            f = font(F_BRUSH, 50)
            title = "无愁河的浪荡汉子"
            n = int(np.clip((tl - 8.8) / 0.32, 0, len(title)))
            for i in range(n):
                x, y = self.cells[i + 9 * 1 + 0]
                sd.text((x, y), title[i], font=f, fill=INK + (255,), anchor="mm")
            if tl > 8.8 + 0.32 * len(title) + 0.3:
                for i, ch in enumerate("黄永玉　著"):
                    x, y = self.cells[9 * 3 + 4 + i]
                    sd.text((x, y), ch, font=font(F_BRUSH, 40), fill=(80, 70, 60, 255), anchor="mm")
            m = int(max(0, (tl - 12.0) / 0.12))
            r = np.random.default_rng(9)
            for i in range(min(m, 9 * 6)):
                x, y = self.cells[9 * 5 + i]
                ln = 14 + r.random() * 14
                sd.line([(x - ln / 2, y + r.normal(0, 3)), (x + ln / 2, y + r.normal(0, 3))], fill=(120, 110, 100, 160), width=4)
            sh = shadowed(sheet, off=(12, 16), blur=12, op=0.3)
            place(c, sh, W + 400 - (W + 400 - 1500) * su, 560, angle=3 * (1 - su) + 2)
        return c


class Rascal(Scene):
    """老顽童：烟斗冒烟圈、小狗跑过、红色跑车呼啸而过；后半段蓝兔子邮票跳出来"""
    def __init__(self):
        self.bg = paper(W, H, 35, base=(246, 236, 206))
        d = ImageDraw.Draw(self.bg, "RGBA")
        d.line([(0, 720), (W, 720)], fill=(120, 100, 80, 120), width=4)
        self.pipe = pipe_sprite()
        self.car = car_body()
        self.rabbit = shadowed(rabbit_stamp(), off=(14, 20), blur=14, op=0.35)
        self.seal = seal("童心", 100)
        self.label = Image.new("RGBA", (700, 140), (0, 0, 0, 0))
        ld = ImageDraw.Draw(self.label)
        ld.text((350, 55), "老顽童", font=font(F_BRUSH, 110), fill=RED + (255,), anchor="mm")
        ld.text((350, 125), "The Old Rascal", font=font(F_EN, 32), fill=(110, 80, 60, 255), anchor="mm")

    def frame(self, tl):
        tl = max(0.0, tl)
        c = self.bg.copy()
        d = ImageDraw.Draw(c, "RGBA")
        out = smoothstep(6.0, 7.0, tl)          # 前半段元素退场
        A = 1 - out
        # 烟斗 + 烟圈
        px, py = 420, 420
        if A > 0:
            for k in range(14):
                t0 = k * 0.45
                age = (tl - t0) % 6.3
                if tl < t0:
                    continue
                x = px + 110 + age * 25 + 30 * math.sin(age * 1.5 + k)
                y = py - 80 - age * 70
                rr = 14 + age * 12
                a = int(150 * max(0, 1 - age / 6.3) * A)
                d.ellipse([x - rr, y - rr * 0.7, x + rr, y + rr * 0.7], outline=(120, 120, 125, a), width=5)
            place(c, self.pipe, px, py, angle=4 * math.sin(tl * 0.8), alpha=A)
            place(c, self.label, 1350, 260, alpha=smoothstep(0.5, 1.5, tl) * A)
        # 小狗从左跑到右
        dxp = -200 + (tl - 0.5) * 260
        if A > 0 and -300 < dxp < W + 300:
            place(c, dog(tl * 14), dxp, 640 - abs(math.sin(tl * 7)) * 16, alpha=A)
        # 跑车：3.0–5.5 秒从右呼啸到左
        u = (tl - 3.0) / 2.5
        if 0 <= u <= 1:
            x = W + 400 - (W + 800) * u
            for k in range(5):
                d.line([(x + 270 + k * 10, 620 + k * 18), (x + 400 + k * 40, 620 + k * 18)], fill=(150, 120, 100, 150), width=4)
            place(c, self.car, x, 640)
            for wx in (-150, 160):
                place(c, wheel(-tl * 25), x + wx, 690)
        # 蓝兔子邮票
        g = ease_out_back((tl - 6.6) / 1.0, 1.6)
        if g > 0.01:
            place(c, self.rabbit, 960, 470 + 6 * math.sin(tl * 2.5), scale=max(0.02, g * 0.95), angle=-3 + 2 * math.sin(tl * 1.7))
            place(c, self.seal, 1380, 230, alpha=smoothstep(8.5, 9.5, tl), angle=6)
        return c


class Ending(Scene):
    """烟斗静静放着 → 猴子、猫头鹰、荷花依次出现并眨眼 → 片尾"""
    def __init__(self, monkey, owl, lotus, rascal):
        self.bg = paper(W, H, 51)
        self.pipe = rascal.pipe
        self.monkey = shadowed(monkey.stamp, off=(10, 14), blur=10, op=0.3)
        self.owls = owl.owls
        self.flower = lotus.flowers[0][2]
        self.leaf = lotus.leaves[1][2]
        self.rabbit = rascal.rabbit
        # 结尾里各元素出现的时刻（相对本场景，按旁白「那只猴子、那只猫头鹰、那池荷花」对齐）
        cue = [c for c in CUES if c["scene"] == 7]
        base = STARTS[7]
        t1 = cue[1]["start"] - base
        self.t_m, self.t_o, self.t_l = t1 + 0.1, t1 + 1.3, t1 + 2.6
        self.t_wink = cue[1]["end"] - base - 0.9
        self.t_card = cue[2]["start"] - base - 0.6
        self.card = self._card()

    def _card(self):
        im = paper(W, H, 53).convert("RGBA")
        d = ImageDraw.Draw(im)
        d.text((W / 2, 360), "黄永玉", font=font(F_BRUSH, 200), fill=INK + (255,), anchor="mm")
        d.text((W / 2, 520), "1924 — 2023", font=font(F_EN, 52), fill=(80, 70, 60, 255), anchor="mm")
        d.text((W / 2, 630), "永远长不大的老头", font=font(F_BRUSH, 76), fill=RED + (255,), anchor="mm")
        d.text((W / 2, 700), "The Old Man Who Never Grew Up", font=font(F_EN, 32), fill=(100, 90, 80, 255), anchor="mm")
        d.text((W - 40, H - 30), "Visuals & music generated by code · AI voice",
               font=font(F_EN, 20), fill=(130, 120, 110, 255), anchor="rs")
        d.text((W - 40, H - 62), "画面与配乐由代码生成，AI 配音", font=font(F_FUN, 26), fill=(130, 120, 110, 255), anchor="rs")
        return im

    def frame(self, tl):
        tl = max(0.0, tl)
        c = self.bg.copy()
        d = ImageDraw.Draw(c, "RGBA")
        # 开头：烟斗 + 最后一缕烟
        pa = 1 - smoothstep(self.t_m - 1.2, self.t_m, tl)
        if pa > 0:
            place(c, self.pipe, 960, 600, alpha=pa)
            for k in range(6):
                age = (tl - k * 0.8) % 4.8
                x = 960 + 110 + 20 * math.sin(age * 1.3 + k)
                y = 520 - age * 55
                a = int(110 * max(0, 1 - age / 4.8) * pa)
                d.ellipse([x - 10 - age * 6, y - 8 - age * 4, x + 10 + age * 6, y + 8 + age * 4], outline=(120, 120, 125, a), width=4)
        wink = self.t_wink < tl < self.t_wink + 0.5
        gm = ease_out_back((tl - self.t_m) / 0.7, 1.6)
        if gm > 0.01:
            place(c, self.monkey, 470, 520, scale=gm * 0.62, angle=-6)
        go = ease_out_back((tl - self.t_o) / 0.7, 1.6)
        if go > 0.01:
            key = ("closed", "open") if wink else ("open", "closed")
            place(c, self.owls[key], 960, 500, scale=go * 0.75, angle=3 * math.sin(tl * 1.5))
        gl = ease_out_back((tl - self.t_l) / 0.7, 1.4)
        if gl > 0.01:
            place(c, self.leaf, 1450, 640, scale=gl * 0.9)
            place(c, self.flower, 1450, 470, scale=gl * 1.05, angle=4 * math.sin(tl * 1.2))
        if tl > self.t_wink:                     # 一起「眨眼」：轻轻一跳
            j = max(0, math.sin((tl - self.t_wink) * 6)) * math.exp(-(tl - self.t_wink) * 2)
            if j > 0.01:
                for x in (470, 1450):
                    d.text((x + 120, 300 - 30 * j), "✦", font=font(F_EN, 40), fill=(240, 180, 40, int(255 * j)), anchor="mm")
        ca = smoothstep(self.t_card, self.t_card + 1.2, tl)
        if ca > 0:
            place(c, self.card, W / 2, H / 2, alpha=ca)
        fo = smoothstep(LENGTH - STARTS[7] - 0.8, LENGTH - STARTS[7], tl)
        if fo > 0:
            c = Image.blend(c, Image.new("RGB", (W, H), PAPER), fo * 0.6)
        return c


# ================================================================ 字幕
def wrap_en(text, f, maxw):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if f.getlength(t) > maxw and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = t
    lines.append(cur)
    return lines


def build_subs():
    fz, fe = font(F_FUN, 52), font(F_EN, 32)
    subs = []
    for c in CUES:
        en_lines = wrap_en(c["en"], fe, 1720)
        hgt = 76 + 42 * len(en_lines) + 24
        im = Image.new("RGBA", (W, hgt + 20), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        zw = fz.getlength(c["zh"])
        ew = max(fe.getlength(l) for l in en_lines)
        bw = max(zw, ew) + 70
        d.rounded_rectangle([(W - bw) / 2, 4, (W + bw) / 2, hgt + 4], radius=22, fill=(20, 16, 12, 120))
        d.text((W / 2, 46), c["zh"], font=fz, fill=(255, 252, 240, 255), anchor="mm", stroke_width=3, stroke_fill=(30, 22, 15, 255))
        for k, l in enumerate(en_lines):
            d.text((W / 2, 100 + 42 * k), l, font=fe, fill=(235, 228, 210, 255), anchor="mm", stroke_width=2, stroke_fill=(30, 22, 15, 255))
        subs.append((c["start"] - 0.05, c["end"] + 0.35, im))
    return subs


# ================================================================ 合成
SCENES = None
SUBS = None


def setup():
    global SCENES, SUBS
    fh = Fenghuang()
    wc = Woodcut()
    bj = Beijing(wc)
    ow = Owl()
    mk = Monkey()
    lo = Lotus()
    ra = Rascal()
    en = Ending(mk, ow, lo, ra)
    SCENES = [fh, wc, bj, ow, mk, lo, ra, en]
    SUBS = build_subs()


def render(t):
    i = max(k for k in range(len(STARTS)) if STARTS[k] <= t)
    img = SCENES[i].frame(t - STARTS[i])
    # 场景交叉淡化
    if i + 1 < len(STARTS) and t > STARTS[i + 1] - XFADE / 2:
        nxt = SCENES[i + 1].frame(t - STARTS[i + 1])
        img = Image.blend(img, nxt, smoothstep(STARTS[i + 1] - XFADE / 2, STARTS[i + 1] + XFADE / 2, t))
    elif i > 0 and t < STARTS[i] + XFADE / 2:
        prv = SCENES[i - 1].frame(t - STARTS[i - 1])
        img = Image.blend(prv, img, smoothstep(STARTS[i] - XFADE / 2, STARTS[i] + XFADE / 2, t))
    if t < 0.6:
        img = Image.blend(Image.new("RGB", (W, H), PAPER), img, t / 0.6)
    for s0, s1, im in SUBS:
        if s0 <= t <= s1:
            a = min(1, (t - s0) / 0.15, (s1 - t) / 0.15)
            place(img, im, W / 2, H - 40 - im.height / 2, alpha=a)
    return img


def encode(args):
    f0, f1, path = args
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                          "-crf", "18", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for f in range(f0, f1):
        p.stdin.write(render(f / FPS).tobytes())
        if (f - f0) % 150 == 0:
            print(f"  {path}: {f - f0}/{f1 - f0}", flush=True)
    p.stdin.close()
    p.wait()
    return path


def make_video(t0, t1, out, procs=4):
    f0, f1 = int(t0 * FPS), int(t1 * FPS)
    step = math.ceil((f1 - f0) / procs)
    jobs = [(a, min(f1, a + step), f".seg_{k}.mp4") for k, a in enumerate(range(f0, f1, step))]
    with Pool(len(jobs)) as pool:
        segs = pool.map(encode, jobs)
    with open(".segs.txt", "w") as fh:
        fh.writelines(f"file '{s}'\n" for s in segs)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", ".segs.txt",
                    "-ss", str(t0), "-t", str(t1 - t0), "-i", "mix.wav", "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", out], check=True)
    for s in segs:
        os.remove(s)
    os.remove(".segs.txt")
    print("->", out)


if __name__ == "__main__":
    os.chdir(HERE)
    setup()
    if len(sys.argv) > 1 and sys.argv[1] == "--still":
        os.makedirs("stills", exist_ok=True)
        for a in sys.argv[2:]:
            render(float(a)).save(f"stills/t{float(a):05.1f}.jpg", quality=90)
            print("stills/", a)
    elif len(sys.argv) > 1 and sys.argv[1] == "--demo":
        t0, dur = float(sys.argv[2]), float(sys.argv[3])
        make_video(t0, t0 + dur, "demo.mp4")
    else:
        make_video(0, LENGTH, "huang_yongyu_full.mp4")
