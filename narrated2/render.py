"""《青城 · 呼和浩特》画面渲染：程序化分层风景 + 剪影 + 双语字幕。

画面全部用 numpy / Pillow 现场绘制，没有外部素材。每个场景由若干图层组成
（天空、云、远山、近山、建筑、草地……），镜头横移时各层速度不同，形成视差。
场景切换点、字幕时间都读 timeline.json（由 tts.py 生成）。

用法：
  python3 render.py --still 5 22 38 52 70 86 102 122   各时间点出静帧（stills/）
  python3 render.py --demo 60 10                         从 60 秒起渲 10 秒 demo（含声音）
  python3 render.py                                      完整 128 秒，4 进程
"""
import json, os, subprocess, sys
import multiprocessing as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import zoom, gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")
W, H, FPS = 1920, 1080, 30
WW = W + 720                     # 图层比画面宽，留出横移余量
XF = 1.4                         # 场景交叉淡化时长（秒）
WORKERS = 4
TL = json.load(open(os.path.join(HERE, "timeline.json"), encoding="utf-8"))
SCRIPT = json.load(open(os.path.join(HERE, "script.json"), encoding="utf-8"))
TOTAL = TL["total"]

FONT_DIR = "/usr/share/fonts/opentype/noto/"
def font(name, size):
    return ImageFont.truetype(FONT_DIR + name + ".ttc", size, index=2)   # index 2 = 简体中文


def C(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


def mix(a, b, t):
    return a * (1 - t) + b * t


# ====================================================================== 噪声
def noise1d(n, seed, octaves=6, f0=0.0025, rough=0.55, ridged=False):
    rng = np.random.default_rng(seed)
    x = np.arange(n, dtype=np.float64)
    out = np.zeros(n)
    amp, f = 1.0, f0
    for _ in range(octaves):
        pts = rng.standard_normal(int(n * f) + 3)
        xi = x * f
        i = xi.astype(int)
        s = (1 - np.cos(np.pi * (xi - i))) / 2
        v = pts[i] * (1 - s) + pts[i + 1] * s
        out += amp * ((1 - np.abs(v)) * 2 - 1 if ridged else v)
        amp *= rough
        f *= 2
    return out / np.abs(out).max()


def noise2d(h, w, seed, cell=128, octaves=4, rough=0.5):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, c = 1.0, cell
    for _ in range(octaves):
        g = rng.standard_normal((h // c + 3, w // c + 3))
        out += amp * zoom(g, c, order=3)[:h, :w]
        amp *= rough
        c = max(c // 2, 2)
    return out / np.abs(out).max()


# ====================================================================== 图层
class Layer:
    """y0 以下的一条横带：rgb (h, WW, 3)、alpha (h, WW)，横移速度 speed（像素/秒）"""
    def __init__(self, y0, rgb, a, speed, wave=None):
        self.y0, self.rgb, self.a, self.speed, self.wave = y0, rgb.astype(np.float32), a.astype(np.float32), speed, wave

    def comp(self, frame, tl):
        ox = 30 + self.speed * tl
        h = self.rgb.shape[0]
        if self.wave is None:
            i = int(ox); f = ox - i
            rgb = self.rgb[:, i:i + W] * (1 - f) + self.rgb[:, i + 1:i + 1 + W] * f
            a = self.a[:, i:i + W] * (1 - f) + self.a[:, i + 1:i + 1 + W] * f
        else:
            # 草被风吹：越靠近草尖横向位移越大
            amp, top, hgt, gust = self.wave
            yy = np.arange(h)[:, None]
            k = np.clip(1 - (yy - top) / hgt, 0, 1) ** 1.6
            x = np.arange(W)[None, :]
            disp = amp * k * (np.sin(x * 0.0062 - tl * 1.9 + yy * 0.01) * 0.6
                              + np.sin(x * 0.0021 - tl * 0.8) * (0.4 + gust * np.sin(tl * 0.5)))
            xs = np.clip(x + ox + disp, 0, WW - 2)
            i = xs.astype(int); f = (xs - i)[..., None]
            rows = np.broadcast_to(yy, i.shape)
            rgb = self.rgb[rows, i] * (1 - f) + self.rgb[rows, i + 1] * f
            a = self.a[rows, i] * (1 - f[..., 0]) + self.a[rows, i + 1] * f[..., 0]
        sub = frame[self.y0:self.y0 + h]
        sub *= 1 - a[..., None]
        sub += rgb * a[..., None]


def crop_layer(rgb, a, speed, wave=None):
    rows = np.where(a.max(axis=1) > 0.002)[0]
    y0 = int(rows[0]) if len(rows) else H - 1
    if wave:
        wave = (wave[0], wave[1] - y0, wave[2], wave[3])
    return Layer(y0, rgb[y0:], a[y0:], speed, wave)


def ridge(seed, base, amp, top, bot, speed, fade=160, octaves=6, f0=0.0025, rough=0.55, ridged=False):
    h = base - amp * noise1d(WW, seed, octaves, f0, rough, ridged)
    Y = np.arange(H, dtype=np.float32)[:, None]
    a = np.clip(Y - h[None, :] + 0.5, 0, 1)
    d = np.clip((Y - h[None, :]) / fade, 0, 1)[..., None]
    rgb = mix(top, bot, d)
    return crop_layer(rgb, a, speed), h


def haze(y, spread, col, alpha, speed=0):
    Y = np.arange(H, dtype=np.float32)[:, None]
    a = alpha * np.exp(-((Y - y) / spread) ** 2) * np.ones((1, WW), np.float32)
    rgb = np.broadcast_to(col, (H, WW, 3)).copy()
    return crop_layer(rgb, a, speed)


def clouds(seed, y_top, y_bot, col_hi, col_lo, alpha, speed, thr=0.15, stretch=3.0, cell=160):
    hh = y_bot - y_top
    n = noise2d(hh, int(WW / stretch) + 4, seed, cell=max(int(cell / stretch), 24), octaves=5)
    n = zoom(n, (1, stretch), order=1)[:, :WW]   # 横向拉长：层云
    yy = np.linspace(0, 1, hh)[:, None]
    band = np.sin(np.pi * yy) ** 0.8
    a = np.clip((n - thr) * 2.5, 0, 1) * band * alpha
    rgb = mix(col_lo, col_hi, np.clip(0.5 + n[..., None], 0, 1))
    A = np.zeros((H, WW), np.float32); R = np.zeros((H, WW, 3), np.float32)
    A[y_top:y_bot] = a; R[y_top:y_bot] = rgb
    return crop_layer(R, A, speed)


def grass(seed, base, tip, root, ground, speed, blade=(40, 150), density=1.1, amp=10, gust=0.5):
    """前景草地：上千根草叶（2 倍超采样绘制），随风摆动"""
    rng = np.random.default_rng(seed)
    S = 2
    img = Image.new("L", (WW * S, H * S), 0)
    d = ImageDraw.Draw(img)
    gh = base + 40 * noise1d(WW, seed + 1, 4, 0.002)
    d.polygon([(0, H * S)] + [(x * S, (gh[x] + 18) * S) for x in range(0, WW, 8)] + [(WW * S, H * S)], fill=255)
    for _ in range(int(WW * density)):
        x = rng.uniform(0, WW)
        y = gh[min(int(x), WW - 1)] + rng.uniform(0, 70)
        L = rng.uniform(*blade) * (0.6 + 0.4 * (y - base + 40) / 110)
        lean = rng.normal(0.25, 0.35) * L
        w = rng.uniform(2.0, 4.5)
        tipx, tipy = x + lean, y - L
        midx, midy = x + lean * 0.35, y - L * 0.55
        d.polygon([((x - w) * S, y * S), ((midx - w * 0.6) * S, midy * S), (tipx * S, tipy * S),
                   ((midx + w * 0.6) * S, midy * S), ((x + w) * S, y * S)], fill=255)
    a = np.asarray(img.resize((WW, H), Image.LANCZOS), np.float32) / 255
    Y = np.arange(H, dtype=np.float32)[:, None, None]
    t = np.clip((Y - (base - 150)) / 220, 0, 1)
    rgb = np.where(t < 0.5, mix(tip, root, t * 2), mix(root, ground, (t - 0.5) * 2)) * np.ones((1, WW, 1), np.float32)
    return crop_layer(rgb, a, speed, wave=(amp, base - blade[1], blade[1] + 40, gust))


def paint(draw_fn, speed, scale=2):
    """在 2 倍超采样的 RGBA 画布上画建筑、人物等，缩小后做成图层"""
    img = Image.new("RGBA", (WW * scale, H * scale), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(img), scale)
    arr = np.asarray(img.resize((WW, H), Image.LANCZOS), np.float32) / 255
    return crop_layer(arr[..., :3], arr[..., 3], speed)


def rgb255(c, a=255):
    return tuple(int(v * 255) for v in c) + (a,)


# ====================================================================== 天空、光、粒子
def sky(stops):
    ys = np.linspace(0, 1, H)
    col = np.stack([np.interp(ys, [s for s, _ in stops], [C(c)[k] for _, c in stops]) for k in range(3)], -1)
    return np.broadcast_to(col[:, None, :].astype(np.float32), (H, W, 3)).copy()


YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def glow(frame, x, y, col, r_disc, r_glow, strength=1.0, disc=True):
    r = np.sqrt((XX - x) ** 2 + (YY - y) ** 2)
    g = strength * (0.55 * np.exp(-r / r_glow) + 0.35 * np.exp(-r / (r_glow * 3.5)))
    if disc:
        g = g + np.clip((r_disc - r) / 2.0 + 0.5, 0, 1)
    frame[:] = 1 - (1 - frame) * (1 - np.clip(g, 0, 1)[..., None] * col)


def starfield(seed, n, ymax):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(2):
        img = np.zeros((H, W), np.float32)
        xs = rng.integers(0, W, n); ys = (rng.random(n) ** 1.3 * ymax).astype(int)
        mag = rng.pareto(2.2, n) * 0.25 + 0.15
        np.add.at(img, (ys, xs), np.clip(mag, 0, 1.5))
        out.append(gaussian_filter(img, 0.7) * 2.2)
    return out


def kernel(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return np.exp(-(x * x + y * y) / (2 * (r / 2.2) ** 2)).astype(np.float32)


def stamp(buf, x, y, k, s=1.0):
    r = k.shape[0] // 2
    x, y = int(x), int(y)
    x0, x1, y0, y1 = max(x - r, 0), min(x + r + 1, buf.shape[1]), max(y - r, 0), min(y + r + 1, buf.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    buf[y0:y1, x0:x1] += k[y0 - y + r:y1 - y + r, x0 - x + r:x1 - x + r] * s


def add_light(frame, buf, col):
    frame[:] = 1 - (1 - frame) * (1 - np.clip(buf, 0, 1)[..., None] * col)


def over(frame, a, col):
    a = np.clip(a, 0, 1)[..., None]
    frame[:] = frame * (1 - a) + col * a


class Motes:
    """漂浮的尘埃 / 火星 / 光点"""
    def __init__(self, seed, n, y_range, vx, vy, size=(2, 4), sway=8):
        rng = np.random.default_rng(seed)
        self.x0 = rng.uniform(0, W, n); self.y0 = rng.uniform(*y_range, n)
        self.vx = vx * rng.uniform(0.5, 1.5, n); self.vy = vy * rng.uniform(0.5, 1.5, n)
        self.ph = rng.uniform(0, 6.28, n); self.b = rng.uniform(0.3, 1, n)
        self.ks = [kernel(r) for r in range(size[0], size[1] + 1)]
        self.kid = rng.integers(0, len(self.ks), n)
        self.yr = y_range; self.sway = sway

    def draw(self, buf, tl, flicker=0.0):
        h = self.yr[1] - self.yr[0]
        x = (self.x0 + self.vx * tl + self.sway * np.sin(tl * 0.7 + self.ph)) % W
        y = self.yr[0] + (self.y0 - self.yr[0] + self.vy * tl) % h
        b = self.b * (1 - flicker + flicker * (0.5 + 0.5 * np.sin(tl * 3.1 + self.ph * 5)))
        for i in range(len(x)):
            stamp(buf, x[i], y[i], self.ks[self.kid[i]], b[i])


# ====================================================================== 精灵（人物、动物）
def sprite(draw_fn, w, h, col, S=4):
    img = Image.new("L", (w * S, h * S), 0)
    draw_fn(ImageDraw.Draw(img), S)
    a = np.asarray(img.resize((w, h), Image.LANCZOS), np.float32) / 255
    return a, col


def put(frame, spr, x, y, alpha=1.0, flip=False):
    """把精灵贴到画面上；(x, y) 为精灵底边中点"""
    a, col = spr
    if flip:
        a = a[:, ::-1]
    h, w = a.shape
    x0, y0 = int(round(x - w / 2)), int(round(y - h))
    xa, xb, ya, yb = max(x0, 0), min(x0 + w, W), max(y0, 0), min(y0 + h, H)
    if xa >= xb or ya >= yb:
        return
    aa = a[ya - y0:yb - y0, xa - x0:xb - x0, None] * alpha
    frame[ya:yb, xa:xb] = frame[ya:yb, xa:xb] * (1 - aa) + col * aa


def P(k, pts, ox=0, oy=0):
    return [((x + ox) * k, (y + oy) * k) for x, y in pts]


def draw_horse(d, k, oy=40, rider=True, banner=False):
    d.ellipse([22 * k, (40 + oy) * k, 78 * k, (62 + oy) * k], fill=255)
    d.polygon(P(k, [(66, 48), (74, 30), (80, 18), (90, 16), (97, 24), (93, 29), (84, 29), (78, 54)], 0, oy), fill=255)
    d.polygon(P(k, [(81, 19), (83, 11), (86, 18)], 0, oy), fill=255)
    for a, b in [((70, 56), (73, 100)), ((64, 56), (60, 99)), ((30, 56), (25, 99)), ((37, 56), (39, 100))]:
        d.line(P(k, [a, b], 0, oy), fill=255, width=int(5 * k))
    d.polygon(P(k, [(24, 44), (12, 62), (15, 72), (24, 55)], 0, oy), fill=255)
    if rider:
        d.polygon(P(k, [(43, 42), (46, 16), (58, 15), (61, 42)], 0, oy), fill=255)
        d.ellipse(P(k, [(47, 2), (58, 14)], 0, oy), fill=255)
        d.line(P(k, [(53, 40), (58, 60)], 0, oy), fill=255, width=int(6 * k))
        d.line(P(k, [(56, 22), (70, 34)], 0, oy), fill=255, width=int(4 * k))
        if banner:
            d.line(P(k, [(46, 30), (42, -38)], 0, oy), fill=255, width=int(2.5 * k))
            d.polygon(P(k, [(42, -38), (20, -32), (24, -24), (21, -16), (43, -20)], 0, oy), fill=255)


def draw_camel(d, k, oy=40, rider=True):
    d.ellipse(P(k, [(22, 48), (82, 72)], 0, oy), fill=255)
    d.ellipse(P(k, [(30, 30), (50, 58)], 0, oy), fill=255)
    d.ellipse(P(k, [(54, 32), (73, 58)], 0, oy), fill=255)
    d.polygon(P(k, [(76, 52), (90, 60), (97, 40), (103, 30), (113, 30), (116, 37), (107, 39), (101, 58), (92, 72), (78, 68)], 0, oy), fill=255)
    for a, b in [((30, 66), (28, 100)), ((38, 66), (41, 100)), ((70, 66), (67, 100)), ((76, 66), (81, 100))]:
        d.line(P(k, [a, b], 0, oy), fill=255, width=int(5 * k))
    d.line(P(k, [(23, 56), (16, 72)], 0, oy), fill=255, width=int(3 * k))
    if rider:
        d.polygon(P(k, [(46, 38), (49, 12), (59, 11), (61, 38)], 0, oy), fill=255)
        d.ellipse(P(k, [(49, 0), (59, 11)], 0, oy), fill=255)
    else:
        d.rectangle(P(k, [(40, 26), (64, 46)], 0, oy), fill=255)   # 货物


def draw_sitter(d, k, cx, lean=0):
    d.ellipse(P(k, [(cx - 6 + lean, 4), (cx + 6 + lean, 16)]), fill=255)
    d.polygon(P(k, [(cx - 14, 46), (cx - 8 + lean, 16), (cx + 8 + lean, 16), (cx + 14, 46)]), fill=255)


def bird_sprites(col):
    up = sprite(lambda d, k: d.line(P(k, [(1, 3), (8, 7), (12, 9), (16, 7), (23, 3)]), fill=255, width=int(2 * k)), 24, 12, col)
    dn = sprite(lambda d, k: d.line(P(k, [(1, 10), (8, 8), (12, 9), (16, 8), (23, 10)]), fill=255, width=int(2 * k)), 24, 12, col)
    return up, dn


class Flock:
    def __init__(self, seed, n, x0, y0, vx, vy, col, spread=120):
        rng = np.random.default_rng(seed)
        self.dx = rng.normal(0, spread, n); self.dy = rng.normal(0, spread * 0.35, n)
        self.ph = rng.uniform(0, 6.28, n); self.fr = rng.uniform(5, 7, n)
        self.x0, self.y0, self.vx, self.vy = x0, y0, vx, vy
        self.spr = bird_sprites(col)

    def draw(self, frame, tl):
        for i in range(len(self.dx)):
            x = self.x0 + self.vx * tl + self.dx[i] + 6 * np.sin(tl * 0.9 + self.ph[i])
            y = self.y0 + self.vy * tl + self.dy[i] + 4 * np.sin(tl * 1.3 + self.ph[i])
            put(frame, self.spr[int(tl * self.fr[i] + self.ph[i]) % 2], x, y, 0.85)


# ====================================================================== 中式建筑
def roof(d, k, cx, ey, w, h, col, rim=None):
    """歇山式屋顶：两端起翘"""
    pts = [(cx - w * 1.08, ey - h * 0.30), (cx - w * 0.92, ey), (cx + w * 0.92, ey), (cx + w * 1.08, ey - h * 0.30),
           (cx + w * 0.86, ey - h * 0.22), (cx + w * 0.62, ey - h * 0.42), (cx + w * 0.42, ey - h),
           (cx - w * 0.42, ey - h), (cx - w * 0.62, ey - h * 0.42), (cx - w * 0.86, ey - h * 0.22)]
    d.polygon([(x * k, y * k) for x, y in pts], fill=col)
    for s in (-1, 1):   # 正脊两端的鸱吻
        d.polygon([((cx + s * w * 0.42) * k, (ey - h) * k), ((cx + s * w * 0.48) * k, (ey - h * 1.28) * k),
                   ((cx + s * w * 0.36) * k, (ey - h) * k)], fill=col)
    if rim:
        d.line([((cx - w * 1.08) * k, (ey - h * 0.30) * k), ((cx - w * 0.92) * k, ey * k),
                ((cx + w * 0.92) * k, ey * k), ((cx + w * 1.08) * k, (ey - h * 0.30) * k)], fill=rim, width=int(2 * k))


def pagoda(d, k, cx, base, light, mid, shade, eave, scale=1.0):
    """辽代八角七层楼阁式塔（万部华严经塔的简化造型）"""
    s = scale
    d.rectangle([(cx - 150 * s) * k, (base - 46 * s) * k, (cx + 150 * s) * k, base * k], fill=mid)
    d.rectangle([(cx - 150 * s) * k, (base - 46 * s) * k, (cx - 40 * s) * k, base * k], fill=light)
    y = base - 46 * s
    for i in range(7):
        bw = (82 - 5.5 * i) * s; bh = (66 - 3.5 * i) * s
        if i == 0:
            bh = 92 * s
        top = y - bh
        # 三个立面：左侧亮、正面中、右侧暗
        d.rectangle([(cx - bw) * k, top * k, (cx - bw * 0.42) * k, y * k], fill=light)
        d.rectangle([(cx - bw * 0.42) * k, top * k, (cx + bw * 0.42) * k, y * k], fill=mid)
        d.rectangle([(cx + bw * 0.42) * k, top * k, (cx + bw) * k, y * k], fill=shade)
        aw, ah = bw * 0.18, bh * 0.42      # 券门 / 窗
        d.rounded_rectangle([(cx - aw) * k, (y - bh * 0.15 - ah) * k, (cx + aw) * k, (y - bh * 0.15) * k],
                            radius=aw * k, fill=eave)
        ew = bw * 1.32
        d.polygon([((cx - ew * 1.06) * k, (top - 10 * s) * k), ((cx - ew) * k, top * k), ((cx + ew) * k, top * k),
                   ((cx + ew * 1.06) * k, (top - 10 * s) * k), ((cx + bw * 0.9) * k, (top - 14 * s) * k),
                   ((cx - bw * 0.9) * k, (top - 14 * s) * k)], fill=eave)
        d.rectangle([(cx - bw * 0.95) * k, (top - 20 * s) * k, (cx + bw * 0.95) * k, (top - 13 * s) * k], fill=mid)
        y = top - 20 * s
    # 塔刹
    d.polygon([((cx - 40 * s) * k, y * k), ((cx + 40 * s) * k, y * k), ((cx + 10 * s) * k, (y - 34 * s) * k),
               ((cx - 10 * s) * k, (y - 34 * s) * k)], fill=eave)
    d.ellipse([(cx - 16 * s) * k, (y - 60 * s) * k, (cx + 16 * s) * k, (y - 28 * s) * k], fill=mid)
    d.line([(cx * k, (y - 60 * s) * k), (cx * k, (y - 110 * s) * k)], fill=eave, width=int(5 * s * k))
    for j in range(4):
        r = (9 - 1.5 * j) * s
        yy = y - 70 * s - j * 10 * s
        d.ellipse([(cx - r) * k, (yy - 3 * s) * k, (cx + r) * k, (yy + 3 * s) * k], fill=eave)


def city_wall(d, k, y_top, y_bot, col, x0=0, x1=WW, merlon=24):
    d.rectangle([x0 * k, y_top * k, x1 * k, y_bot * k], fill=col)
    for x in range(int(x0), int(x1), merlon):
        d.rectangle([x * k, (y_top - 12) * k, (x + merlon * 0.55) * k, y_top * k], fill=col)


def gate_tower(d, k, cx, y_wall, col, rim, dark):
    d.rounded_rectangle([(cx - 48) * k, (y_wall + 30) * k, (cx + 48) * k, (y_wall + 140) * k], radius=46 * k, fill=dark)
    d.rectangle([(cx - 120) * k, (y_wall - 18) * k, (cx + 120) * k, y_wall * k], fill=col)
    d.rectangle([(cx - 100) * k, (y_wall - 90) * k, (cx + 100) * k, (y_wall - 18) * k], fill=col)
    for i in range(6):
        x = cx - 85 + i * 34
        d.rectangle([x * k, (y_wall - 78) * k, (x + 16) * k, (y_wall - 30) * k], fill=dark)
    roof(d, k, cx, y_wall - 86, 150, 46, col, rim)
    d.rectangle([(cx - 82) * k, (y_wall - 176) * k, (cx + 82) * k, (y_wall - 128) * k], fill=col)
    roof(d, k, cx, y_wall - 172, 128, 62, col, rim)


def temple_hall(d, k, cx, base, col, rim, dark, s=1.0):
    d.rectangle([(cx - 260 * s) * k, (base - 40 * s) * k, (cx + 260 * s) * k, base * k], fill=col)
    d.rectangle([(cx - 210 * s) * k, (base - 150 * s) * k, (cx + 210 * s) * k, (base - 40 * s) * k], fill=col)
    for i in range(7):
        x = cx - 190 * s + i * 60 * s
        d.rectangle([x * k, (base - 132 * s) * k, (x + 30 * s) * k, (base - 50 * s) * k], fill=dark)
    roof(d, k, cx, base - 146 * s, 290 * s, 70 * s, col, rim)
    d.rectangle([(cx - 170 * s) * k, (base - 250 * s) * k, (cx + 170 * s) * k, (base - 210 * s) * k], fill=col)
    roof(d, k, cx, base - 246 * s, 220 * s, 92 * s, col, rim)
    d.ellipse([(cx - 14 * s) * k, (base - 372 * s) * k, (cx + 14 * s) * k, (base - 344 * s) * k], fill=col)  # 宝顶


# ====================================================================== 场景
class Scene:
    def __init__(self):
        self.layers = []     # [(z, Layer)]，z 小的先画
        self.sky = None

    def frame(self, tl):
        f = self.sky.copy()
        self.pre(f, tl)
        for L in self.layers:
            if isinstance(L, Layer):
                L.comp(f, tl)
            else:
                L(f, tl)
        self.post(f, tl)
        return f

    def pre(self, f, tl): pass
    def post(self, f, tl): pass


class Grassland(Scene):
    """开篇 / 结尾：清晨的阴山与草原（主色调：青）"""
    def __init__(self, end=False):
        super().__init__()
        self.end = end
        self.sky = sky([(0, "#0d1d31"), (0.42, "#2c5671"), (0.6, "#93b0b6"), (0.69, "#ecc896"), (1, "#ecc896")] if not end else
                       [(0, "#1b3a5a"), (0.42, "#4f84a3"), (0.6, "#b4ccd0"), (0.69, "#f3dbae"), (1, "#f3dbae")])
        hz = C("#ecc896") if not end else C("#f3dbae")
        self.cl = clouds(3, 150, 520, C("#f1c9a5"), C("#7d8ea0"), 0.45, 4)
        far, _ = ridge(10, 640, 95, mix(C("#5f7f90"), hz, 0.35), mix(C("#4e6b7b"), hz, 0.2), 3, ridged=True)
        mid, _ = ridge(11, 705, 70, C("#3e5c6c"), C("#2f4856"), 6, f0=0.003, ridged=True)
        river = paint(lambda d, k: self._river(d, k), 12)
        near, _ = ridge(12, 790, 34, C("#33493d"), C("#26382f"), 12, f0=0.0018, octaves=4)
        self.layers = [self.cl, far, haze(680, 40, hz, 0.55), mid, haze(745, 30, hz, 0.4), near, river,
                       grass(13, 940, C("#8a9a63") if end else C("#6e7f55"), C("#2a3a26"), C("#141d14"), 26,
                             amp=16 if end else 9, gust=0.9 if end else 0.4)]
        self.motes = Motes(5, 70, (300, 1000), 6, -4, (2, 4), 14)

    def _river(self, d, k):
        c1 = rgb255(C("#e9d0a1"), 210)
        pts_t = [(x, 812 + 10 * np.sin(x / 260) + 6 * np.sin(x / 90)) for x in range(-50, WW + 60, 20)]
        pts_b = [(x, y + 8 + 4 * np.sin(x / 140)) for x, y in pts_t][::-1]
        d.polygon([(x * k, y * k) for x, y in pts_t + pts_b], fill=c1)

    def pre(self, f, tl):
        y = 640 - (tl * 2.2 if not self.end else 90 + tl * 1.5)
        glow(f, 1290, y, C("#ffd7a0"), 34, 120, 0.9)

    def post(self, f, tl):
        b = np.zeros((H, W), np.float32)
        self.motes.draw(b, tl, 0.4)
        add_light(f, b * 0.45, C("#ffe2b0"))


class Prehistoric(Scene):
    """远古：星空下的篝火"""
    def __init__(self):
        super().__init__()
        self.sky = sky([(0, "#04070f"), (0.5, "#121c33"), (0.72, "#3a3a52"), (1, "#3a3a52")])
        self.stars = starfield(21, 2600, 760)
        far, _ = ridge(22, 690, 90, C("#141a28"), C("#0f1420"), 2, ridged=True)
        near, _ = ridge(23, 820, 40, C("#0d1118"), C("#080b10"), 6, octaves=4)
        rocks = paint(self._rocks, 14)
        self.people = sprite(lambda d, k: [draw_sitter(d, k, 20, 3), draw_sitter(d, k, 60, -2),
                                           draw_sitter(d, k, 150, -4)], 175, 48, C("#05060a"))
        self.layers = [far, haze(705, 40, C("#2a2b3a"), 0.4), near, rocks,
                       grass(24, 960, C("#2a2a24"), C("#0d0f0c"), C("#050605"), 20, amp=6)]
        self.sparks = Motes(25, 40, (560, 870), 3, -45, (1, 3), 10)
        self.k_flame = kernel(14)

    def _rocks(self, d, k):
        rng = np.random.default_rng(26)
        for _ in range(40):
            x = rng.uniform(0, WW); y = rng.uniform(870, 1000); r = rng.uniform(10, 40)
            d.ellipse([(x - r * 1.4) * k, (y - r) * k, (x + r * 1.4) * k, (y + r * 0.3) * k], fill=(10, 11, 14, 255))

    def pre(self, f, tl):
        tw = 0.5 + 0.5 * np.sin(tl * 0.9)
        st = self.stars[0] * (0.7 + 0.3 * tw) + self.stars[1] * (1 - 0.3 * tw)
        add_light(f, st, C("#dfe6ff"))

    def post(self, f, tl):
        fx, fy = 960 - 18 * tl, 885
        fl = 0.85 + 0.1 * np.sin(tl * 17) + 0.06 * np.sin(tl * 29 + 1)
        glow(f, fx, fy, C("#ff9a4a"), 0, 95 * fl, 1.0, disc=False)
        put(f, self.people, fx - 12, fy + 10)
        b = np.zeros((H, W), np.float32)
        for j in range(6):
            stamp(b, fx + 9 * np.sin(tl * 7 + j * 2), fy - 6 - j * 6 - 4 * np.sin(tl * 11 + j), self.k_flame, 0.45)
        add_light(f, b, C("#ffc069"))
        s = np.zeros((H, W), np.float32)
        sp = self.sparks
        x = fx + (sp.x0 % 60 - 30) + 20 * np.sin(tl + sp.ph)
        y = fy - ((sp.y0 + 60 * tl * sp.b) % 320)
        for i in range(len(x)):
            stamp(s, x[i], y[i], sp.ks[sp.kid[i]], (1 - (fy - y[i]) / 320) * 0.9)
        add_light(f, s, C("#ffb35c"))


class ZhaoWall(Scene):
    """战国：山脊上的赵长城、烽火台、骑兵"""
    def __init__(self):
        super().__init__()
        self.sky = sky([(0, "#3a3a42"), (0.45, "#7d7262"), (0.64, "#cfae7c"), (1, "#cfae7c")])
        hz = C("#cfae7c")
        self.cl = clouds(31, 80, 520, C("#c9b08d"), C("#57525a"), 0.65, 5, thr=0.0, stretch=2.2)
        far, _ = ridge(32, 610, 80, mix(C("#6c6458"), hz, 0.3), C("#5a5348"), 3, ridged=True)
        mid, self.h_mid = ridge(33, 700, 85, C("#4a4136"), C("#3a3229"), 7, f0=0.0022, ridged=True)
        wall = paint(self._wall, 7)
        near, _ = ridge(34, 860, 25, C("#5b4b35"), C("#3e3325"), 14, octaves=4, f0=0.0015)
        self.riders = [sprite(lambda d, k, b=(i == 0): draw_horse(d, k, banner=b), 100, 140, C("#1f1912"), S=4)
                       for i in range(5)]
        self.layers = [self.cl, far, haze(650, 40, hz, 0.4), mid, wall, haze(760, 50, hz, 0.35), near,
                       self._riders,
                       grass(35, 960, C("#9a8455"), C("#3d3221"), C("#1e1910"), 24, amp=10, gust=0.6)]
        self.k_smoke = kernel(26)
        self.beacon_x = int(np.argmin(self.h_mid[900:1500])) + 900

    def _wall(self, d, k):
        h = self.h_mid
        col = (58, 49, 38, 255)
        pts_t = [(x, h[x] - 16) for x in range(0, WW, 6)]
        pts_b = [(x, h[x] + 4) for x in range(0, WW, 6)][::-1]
        d.polygon([(x * k, y * k) for x, y in pts_t + pts_b], fill=col)
        for x in range(150, WW - 40, 330):
            y = h[x]
            d.polygon([((x - 22) * k, (y + 2) * k), ((x - 16) * k, (y - 44) * k), ((x + 16) * k, (y - 44) * k),
                       ((x + 22) * k, (y + 2) * k)], fill=col)
        bx = int(np.argmin(h[900:1500])) + 900
        y = h[bx]
        d.polygon([((bx - 34) * k, (y + 4) * k), ((bx - 24) * k, (y - 70) * k), ((bx + 24) * k, (y - 70) * k),
                   ((bx + 34) * k, (y + 4) * k)], fill=col)

    def _riders(self, f, tl):
        for i, spr in enumerate(self.riders):
            x = 1500 + i * 115 - 42 * tl + (i % 2) * 20
            put(f, spr, x, 905 + (i % 2) * 14 + 2 * np.sin(tl * 7 + i), flip=True)

    def post(self, f, tl):
        ox = 30 + 7 * tl
        bx, by = self.beacon_x - ox, self.h_mid[self.beacon_x] - 72
        a = np.zeros((H, W), np.float32)
        for j in range(26):
            age = (tl * 0.35 + j / 26) % 1.0
            stamp(a, bx + 160 * age ** 1.4 + 10 * np.sin(j + tl), by - 380 * age, self.k_smoke, 0.3 * (1 - age) * min(age * 8, 1))
        over(f, a * 1.6, C("#4a433d"))
        glow(f, bx, by + 6, C("#ff8a3a"), 0, 14, 0.8, disc=False)


class HanTomb(Scene):
    """西汉：夕阳下的青冢和出塞的队伍"""
    def __init__(self):
        super().__init__()
        self.sky = sky([(0, "#24172b"), (0.35, "#6e2f3a"), (0.55, "#d36f45"), (0.66, "#f4c06a"), (1, "#f4c06a")])
        hz = C("#f4c06a")
        self.cl = clouds(41, 120, 560, C("#ffb07a"), C("#5a2a3a"), 0.6, 4, thr=0.08, stretch=4)
        far, _ = ridge(42, 680, 70, mix(C("#8a4a4a"), hz, 0.3), C("#6e3a3a"), 3, ridged=True)
        mound = paint(self._mound, 8)
        near, self.h_near = ridge(43, 880, 22, C("#3e2424"), C("#2a1818"), 14, octaves=3, f0=0.0015)
        self.parade = [sprite(lambda d, k: draw_horse(d, k, banner=True), 100, 140, C("#1d1012")),
                       sprite(lambda d, k: draw_camel(d, k, rider=False), 120, 140, C("#1d1012")),
                       sprite(lambda d, k: draw_horse(d, k), 100, 140, C("#1d1012")),
                       sprite(lambda d, k: draw_camel(d, k, rider=True), 120, 140, C("#1d1012")),
                       sprite(lambda d, k: draw_horse(d, k), 100, 140, C("#1d1012"))]
        self.layers = [self.cl, far, haze(700, 45, hz, 0.5), mound, haze(800, 30, hz, 0.25), near, self._parade,
                       grass(44, 975, C("#7a4a30"), C("#2a1712"), C("#140b09"), 22, amp=9)]
        self.birds = Flock(45, 9, 1500, 330, -38, -4, C("#2b1520"))

    def _mound(self, d, k):
        cx, base = 1620, 878
        col = (58, 34, 34, 255)
        pts = [(cx + x, base - 250 * np.exp(-(x / 330) ** 2 * 1.4) * (1 + 0.03 * np.sin(x / 23))) for x in range(-760, 761, 8)]
        d.polygon([(x * k, y * k) for x, y in pts] + [((cx + 760) * k, (base + 40) * k), ((cx - 760) * k, (base + 40) * k)], fill=col)
        d.rectangle([(cx - 230) * k, (base - 72) * k, (cx - 214) * k, (base - 10) * k], fill=col)   # 石碑
        for tx, s in [(cx - 420, 1.0), (cx + 380, 0.8), (cx + 470, 0.65)]:
            d.line([(tx * k, base * k), (tx * k, (base - 70 * s) * k)], fill=col, width=int(5 * k))
            d.ellipse([(tx - 36 * s) * k, (base - 130 * s) * k, (tx + 36 * s) * k, (base - 52 * s) * k], fill=col)

    def _parade(self, f, tl):
        for i, spr in enumerate(self.parade):
            x = 1180 + i * 125 - 26 * tl
            put(f, spr, x, 912 + 2 * np.sin(tl * 5 + i * 1.3), flip=True)

    def pre(self, f, tl):
        glow(f, 560, 590 + tl * 2.5, C("#ffa25c"), 58, 150, 1.0)

    def post(self, f, tl):
        self.birds.draw(f, tl)


class LiaoPagoda(Scene):
    """北魏 · 辽：白塔"""
    def __init__(self):
        super().__init__()
        self.sky = sky([(0, "#5e7f9c"), (0.45, "#a5bccb"), (0.72, "#e8e1cd"), (1, "#e8e1cd")])
        hz = C("#e8e1cd")
        self.cl = clouds(51, 60, 600, C("#ffffff"), C("#9fb0bf"), 0.8, 5, thr=0.05, stretch=1.8, cell=220)
        far, _ = ridge(52, 700, 60, mix(C("#7f93a0"), hz, 0.4), C("#6f8390"), 3, ridged=True)
        town = paint(self._ruins, 7)
        tower = paint(lambda d, k: pagoda(d, k, 1000, 905, rgb255(C("#f4efe4")), rgb255(C("#d9d2c4")),
                                          rgb255(C("#9f998e")), rgb255(C("#4b4740")), 1.12), 9)
        near, _ = ridge(53, 900, 18, C("#5f6c4c"), C("#3c4632"), 9, octaves=3, f0=0.0012)
        self.layers = [self.cl, far, haze(740, 50, hz, 0.5), town, near, tower,
                       grass(54, 985, C("#a3a472"), C("#3b452b"), C("#1b2214"), 22, amp=10)]
        self.birds = Flock(55, 14, 300, 300, 44, -6, C("#3a3a3a"), 150)

    def _ruins(self, d, k):
        # 丰州故城残垣：低矮、顶部不平的夯土墙
        col = rgb255(C("#8d927f"))
        rng = np.random.default_rng(56)
        x = -40
        while x < WW:
            w = rng.uniform(160, 420); h = rng.uniform(14, 34)
            xs = np.linspace(x, x + w, 24)
            top = 885 - h + rng.normal(0, 4, 24)
            top[[0, -1]] = 890
            d.polygon([(a * k, b * k) for a, b in zip(xs, top)] + [((x + w) * k, 905 * k), (x * k, 905 * k)], fill=col)
            x += w + rng.uniform(120, 400)

    def pre(self, f, tl):
        glow(f, 420, 210, C("#fff3d8"), 0, 260, 0.55, disc=False)

    def post(self, f, tl):
        self.birds.draw(f, tl)


class MingCity(Scene):
    """明：归化城城楼与大召寺"""
    def __init__(self):
        super().__init__()
        self.sky = sky([(0, "#3f3a52"), (0.4, "#b07a4e"), (0.68, "#f3cf88"), (1, "#f3cf88")])
        hz = C("#f3cf88")
        self.cl = clouds(61, 90, 520, C("#ffd59a"), C("#8a5a50"), 0.55, 4, thr=0.1, stretch=3.5)
        far, _ = ridge(62, 690, 75, mix(C("#8a6a58"), hz, 0.35), C("#7a5a48"), 3, ridged=True)
        col, rim, dark = rgb255(C("#3a271b")), rgb255(C("#f0b45c")), rgb255(C("#1e140e"))
        temple = paint(lambda d, k: [temple_hall(d, k, 1520, 770, col, rim, dark, 0.95),
                                     temple_hall(d, k, 2220, 760, col, rim, dark, 0.7)], 8)
        wall = paint(lambda d, k: [city_wall(d, k, 800, 930, col), gate_tower(d, k, 760, 800, col, rim, dark)], 10)
        self.layers = [self.cl, far, haze(720, 50, hz, 0.5), temple, self._smoke, haze(800, 35, hz, 0.25), wall,
                       grass(64, 1000, C("#8c6a3a"), C("#2e1f12"), C("#140d08"), 20, blade=(30, 110), amp=8)]
        self.k_smoke = kernel(18)
        self.birds = Flock(65, 7, 1900, 280, -30, 3, C("#3a2420"))

    def _smoke(self, f, tl):
        a = np.zeros((H, W), np.float32)
        ox = 30 + 8 * tl
        for sx in (1400, 1640):
            for j in range(18):
                age = (tl * 0.18 + j / 18) % 1.0
                stamp(a, sx - ox + 60 * age + 14 * np.sin(age * 6 + j), 690 - 300 * age, self.k_smoke, 0.12 * (1 - age))
        over(f, a, C("#e8d2b0"))

    def pre(self, f, tl):
        glow(f, 260, 640 + tl * 1.5, C("#ffd28a"), 44, 140, 0.9)

    def post(self, f, tl):
        self.birds.draw(f, tl)


class QingCaravan(Scene):
    """清：暮色沙丘上的驼队"""
    def __init__(self):
        super().__init__()
        self.sky = sky([(0, "#15183a"), (0.3, "#4a3366"), (0.48, "#b5566a"), (0.6, "#f19a5c"), (1, "#f19a5c")])
        self.stars = starfield(71, 900, 300)
        hz = C("#f19a5c")
        towns = paint(self._towns, 2)
        d1, _ = ridge(72, 660, 40, mix(C("#a65a4a"), hz, 0.3), C("#7a4040"), 4, octaves=3, f0=0.0016, fade=90)
        d2, self.h2 = ridge(73, 790, 55, C("#c0704e"), C("#6a3438"), 9, octaves=3, f0=0.0013, fade=120)
        d3, _ = ridge(74, 940, 45, C("#8a4636"), C("#3a1d22"), 18, octaves=3, f0=0.0011, fade=140)
        self.camels = [sprite(lambda d, k, r=(i % 3 == 0): draw_camel(d, k, rider=r), 120, 140, C("#24121a"))
                       for i in range(8)]
        self.layers = [towns, d1, haze(690, 30, hz, 0.35), d2, self._caravan, d3]
        self.moon = sprite(lambda d, k: [d.ellipse([0, 0, 60 * k, 60 * k], fill=255),
                                         d.ellipse([14 * k, -6 * k, 74 * k, 54 * k], fill=0)], 64, 64, C("#fff1d0"))

    def _towns(self, d, k):
        col = rgb255(mix(C("#7a4a50"), C("#f19a5c"), 0.25))
        for cx in (260, 520):
            city_wall(d, k, 640, 660, col, cx - 110, cx + 110, merlon=12)
            roof(d, k, cx, 640, 34, 16, col)

    def _caravan(self, f, tl):
        ox = 30 + 9 * tl
        for i, spr in enumerate(self.camels):
            x = 520 + i * 96 + 24 * tl
            sx = int(np.clip(x + ox, 0, WW - 1))
            put(f, spr, x, self.h2[sx] + 6 + 2 * np.sin(tl * 4 + i))

    def pre(self, f, tl):
        tw = 0.5 + 0.5 * np.sin(tl * 0.7)
        add_light(f, (self.stars[0] * tw + self.stars[1] * (1 - tw)) * np.clip(tl / 8, 0.3, 1), C("#fff4e0"))
        put(f, self.moon, 1580, 230)


class ModernCity(Scene):
    """今天：夜色中的城市天际线、车流、数据网络"""
    def __init__(self):
        super().__init__()
        self.sky = sky([(0, "#040817"), (0.42, "#0c1c3c"), (0.68, "#24507e"), (0.8, "#3b6b95"), (1, "#3b6b95")])
        rng = np.random.default_rng(81)
        n = 70
        self.nodes = np.stack([rng.uniform(0, W, n), rng.uniform(40, 460, n)], 1)
        dist = np.linalg.norm(self.nodes[:, None] - self.nodes[None], axis=-1)
        self.edges = [(i, j) for i in range(n) for j in range(i + 1, n) if dist[i, j] < 210]
        img = Image.new("L", (W * 2, H * 2), 0)
        dd = ImageDraw.Draw(img)
        for i, j in self.edges:
            dd.line([tuple(self.nodes[i] * 2), tuple(self.nodes[j] * 2)], fill=70, width=2)
        for x, y in self.nodes:
            dd.ellipse([x * 2 - 5, y * 2 - 5, x * 2 + 5, y * 2 + 5], fill=255)
        self.net = np.asarray(img.resize((W, H), Image.LANCZOS), np.float32) / 255
        self.pulse_e = rng.integers(0, len(self.edges), 40); self.pulse_ph = rng.random(40)
        self.k_p = kernel(5)
        back = self._skyline(82, 730, 140, 360, C("#14243c"), C("#5a7da8"), 4, tower=True)
        front = self._skyline(83, 860, 50, 190, C("#0a1222"), C("#ffd79a"), 10)
        self.trail_tex = self._trails(84)
        ground, _ = ridge(85, 900, 8, C("#070b14"), C("#05080e"), 14, octaves=2)
        self.layers = [back[0], self._win(back), haze(780, 40, C("#3b6b95"), 0.35), front[0], self._win(front),
                       self._road, ground,
                       grass(86, 1010, C("#1b2a30"), C("#0a1014"), C("#05080a"), 22, blade=(25, 80), amp=6)]

    def _skyline(self, seed, base, hmin, hmax, col, wcol, speed, tower=False):
        rng = np.random.default_rng(seed)
        S = 2
        img = Image.new("L", (WW * S, H * S), 0); win = [Image.new("L", (WW * S, H * S), 0) for _ in range(2)]
        d = ImageDraw.Draw(img); dw = [ImageDraw.Draw(w) for w in win]
        x = -20
        while x < WW:
            w = rng.uniform(50, 140); h = rng.uniform(hmin, hmax) * (1.5 if rng.random() < 0.12 else 1)
            top = base - h
            d.rectangle([x * S, top * S, (x + w) * S, H * S], fill=255)
            if rng.random() < 0.3:
                d.polygon([(x * S, top * S), ((x + w) * S, top * S), ((x + w / 2) * S, (top - 30) * S)], fill=255)
            for wy in np.arange(top + 12, base + 140, 16):
                for wx in np.arange(x + 8, x + w - 10, 13):
                    for m in range(2):
                        if rng.random() < 0.28:
                            dw[m].rectangle([wx * S, wy * S, (wx + 6) * S, (wy + 8) * S], fill=255)
            x += w + rng.uniform(-10, 60)
        if tower:   # 电视塔
            tx = 980
            d.polygon([((tx - 24) * S, base * S), ((tx - 5) * S, (base - 420) * S), ((tx + 5) * S, (base - 420) * S),
                       ((tx + 24) * S, base * S)], fill=255)
            d.ellipse([(tx - 34) * S, (base - 400) * S, (tx + 34) * S, (base - 352) * S], fill=255)
            d.line([(tx * S, (base - 420) * S), (tx * S, (base - 520) * S)], fill=255, width=6)
        a = np.asarray(img.resize((WW, H), Image.LANCZOS), np.float32) / 255
        wins = [np.asarray(w.resize((WW, H), Image.LANCZOS), np.float32) / 255 * a for w in win]
        rgb = np.broadcast_to(col, (H, WW, 3)).copy()
        return crop_layer(rgb, a, speed), wins, wcol, speed

    def _win(self, sk):
        _, wins, wcol, speed = sk
        def f(frame, tl):
            ox = int(30 + speed * tl)
            k = 0.5 + 0.5 * np.sin(tl * 0.6)
            w = wins[0][:, ox:ox + W] * (0.55 + 0.45 * k) + wins[1][:, ox:ox + W] * (1 - 0.45 * k)
            add_light(frame, w * 0.9, wcol)
        return f

    def _trails(self, seed):
        rng = np.random.default_rng(seed)
        tex = np.zeros((2, 30, W * 2), np.float32)
        for lane in range(2):
            for _ in range(70):
                x = rng.integers(0, W * 2 - 200); y = rng.integers(4, 26); L = rng.integers(60, 200)
                tex[lane, y, x:x + L] += np.linspace(0, 1, L) ** 2 if lane == 0 else np.linspace(1, 0, L) ** 2
        return [gaussian_filter(t, (0.8, 1.6)) * 3 for t in tex]

    def _road(self, f, tl):
        for lane, (y, v, col) in enumerate([(842, 240, C("#ff5a4a")), (858, -300, C("#fff2d0"))]):
            o = int(v * tl) % W
            strip = self.trail_tex[lane][:, o:o + W] if v < 0 else self.trail_tex[lane][:, W - o:2 * W - o]
            buf = np.zeros((H, W), np.float32); buf[y:y + 30] = strip
            add_light(f, buf * 0.8, col)

    def pre(self, f, tl):
        glow(f, 960, 900, C("#4d7fb8"), 0, 420, 0.5, disc=False)
        b = self.net * 0.28 * (0.8 + 0.2 * np.sin(tl * 1.2))
        for e, ph in zip(self.pulse_e, self.pulse_ph):
            i, j = self.edges[e]
            s = (tl * 0.6 + ph) % 1.0
            p = self.nodes[i] * (1 - s) + self.nodes[j] * s
            stamp(b, p[0], p[1], self.k_p, 0.9)
        add_light(f, b, C("#7fd6ff"))


# ====================================================================== 时间轴
def build_timeline():
    ch = {c["id"]: c for c in TL["chapters"]}
    ln = TL["lines"]
    anc = [l for l in ln if l["ch"] == 1]
    split = (anc[1]["t1"] + anc[2]["t0"]) / 2           # 远古篝火 → 战国长城
    seq = [("open", 0.0, Grassland), ("prehistoric", ch["ancient"]["t0"], Prehistoric),
           ("zhao", split, ZhaoWall), ("han", ch["han"]["t0"], HanTomb), ("liao", ch["liao"]["t0"], LiaoPagoda),
           ("ming", ch["ming"]["t0"], MingCity), ("qing", ch["qing"]["t0"], QingCaravan),
           ("modern", ch["modern"]["t0"], ModernCity), ("end", ch["end"]["t0"], lambda: Grassland(end=True))]
    return seq


SEQ = build_timeline()
_scenes = {}


def scene(i):
    if i not in _scenes:
        _scenes.clear() if len(_scenes) > 2 else None
        _scenes[i] = SEQ[i][2]()
    return _scenes[i]


def visual(t):
    starts = [s[1] for s in SEQ]
    i = max(k for k in range(len(SEQ)) if starts[k] <= t)
    tl = t - starts[i] + XF
    f = scene(i).frame(tl)
    if i + 1 < len(SEQ) and t > starts[i + 1] - XF / 2:
        a = (t - (starts[i + 1] - XF / 2)) / XF
        a = a * a * (3 - 2 * a)
        g = scene(i + 1).frame(t - starts[i + 1] + XF)
        f = f * (1 - a) + g * a
    elif i > 0 and t < starts[i] + XF / 2:
        a = (t - (starts[i] - XF / 2)) / XF
        a = a * a * (3 - 2 * a)
        g = scene(i - 1).frame(t - starts[i - 1] + XF)
        f = g * (1 - a) + f * a
    return f


# ====================================================================== 文字
def text_img(lines, shadow=6, pad=40):
    """lines = [(文字, 字体, 颜色, 行距)]，返回居中排版的 RGBA（带柔和阴影）"""
    sizes = [ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), s, font=f) for s, f, _, _ in lines]
    w = max(b[2] - b[0] for b in sizes) + pad * 2
    h = sum(g for *_, g in lines) + pad * 2
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sh = Image.new("L", (w, h), 0)
    d, ds = ImageDraw.Draw(img), ImageDraw.Draw(sh)
    y = pad
    for (s, f, col, g), b in zip(lines, sizes):
        x = (w - (b[2] - b[0])) / 2 - b[0]
        ds.text((x, y), s, font=f, fill=255)
        d.text((x, y), s, font=f, fill=col)
        y += g
    sh = sh.filter(ImageFilter.GaussianBlur(shadow))
    base = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    base.putalpha(sh.point(lambda v: int(v * 0.85)))
    return np.asarray(Image.alpha_composite(base, img), np.float32) / 255


def blit(frame, rgba, cx, cy, alpha):
    if alpha <= 0:
        return
    h, w = rgba.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    xa, xb, ya, yb = max(x0, 0), min(x0 + w, W), max(y0, 0), min(y0 + h, H)
    src = rgba[ya - y0:yb - y0, xa - x0:xb - x0]
    a = src[..., 3:] * alpha
    frame[ya:yb, xa:xb] = frame[ya:yb, xa:xb] * (1 - a) + src[..., :3] * a


class Texts:
    def __init__(self):
        zh, en = font("NotoSansCJK-Regular", 46), font("NotoSansCJK-Regular", 27)
        self.subs = [(l["t0"], l["t1"], text_img([(l["zh"], zh, (255, 255, 255), 62), (l["en"], en, (215, 222, 230), 36)]))
                     for l in TL["lines"]]
        se, sz = font("NotoSerifCJK-Bold", 64), font("NotoSansCJK-Regular", 24)
        self.eras = [(c["t0"], c["t1"], text_img([(c["era_zh"], se, (255, 244, 225), 84),
                                                    (c["era_en"].upper(), sz, (225, 215, 195), 30)], shadow=8, pad=30))
                     for c in TL["chapters"] if c["era_zh"]]
        big, mid, small = font("NotoSerifCJK-Bold", 150), font("NotoSerifCJK-Regular", 50), font("NotoSansCJK-Regular", 28)
        self.title = text_img([("青 城", big, (255, 250, 238), 190), ("呼 和 浩 特", mid, (245, 236, 220), 70),
                               ("HOHHOT · THE BLUE CITY", small, (230, 225, 215), 40)], shadow=10)
        tiny = font("NotoSansCJK-Regular", 22)
        self.credit = text_img([("旁白：AI 语音合成（微软云健）  ·  配乐与画面：代码生成", tiny, (225, 225, 225), 32),
                                ("Narration: AI voice  ·  Music & visuals generated with code", tiny, (205, 205, 205), 30)], shadow=5)
        self.end_line = TL["lines"][-1]["t1"]
        y = np.arange(H, dtype=np.float32)[:, None, None]
        self.shade = 1 - 0.42 * np.clip((y - 800) / 280, 0, 1) ** 1.5             # 字幕区压暗
        r = np.sqrt(((XX - W / 2) / (W * 0.62)) ** 2 + ((YY - H / 2) / (H * 0.62)) ** 2)
        self.vig = (1 - 0.38 * np.clip(r - 0.55, 0, 1) ** 1.3)[..., None]

    @staticmethod
    def fade(t, t0, t1, fi=0.35, fo=0.35):
        return float(np.clip(min((t - t0) / fi, (t1 - t) / fo), 0, 1))

    def draw(self, f, t):
        f *= self.shade
        f *= self.vig
        for t0, t1, im in self.subs:
            a = self.fade(t, t0 - 0.15, t1 + 0.35, 0.25, 0.3)
            if a > 0:
                blit(f, im, W / 2, 955, a)
        for t0, t1, im in self.eras:
            a = self.fade(t, t0 + 0.6, t1 - 0.3, 0.8, 0.7)
            if a > 0:
                blit(f, im, 90 + im.shape[1] / 2 + 12 * (1 - a), 70 + im.shape[0] / 2, a)
        a = self.fade(t, 0.6, 11.0, 1.6, 1.5)
        a2 = self.fade(t, self.end_line + 1.0, TOTAL + 5, 2.0, 1)
        if a > 0:
            blit(f, self.title, W / 2, 360, a)
        if a2 > 0:
            blit(f, self.title, W / 2, 430, a2)
            blit(f, self.credit, W / 2, 760, a2 * 0.9)
        k = np.clip(min(t / 1.2, (TOTAL - t) / 2.0), 0, 1)       # 片头片尾黑场
        f *= k


_texts = None
_grain = None


def render_frame(t):
    global _texts, _grain
    if _texts is None:
        _texts = Texts()
        rng = np.random.default_rng(99)
        _grain = rng.normal(0, 0.008, (H, W, 1)).astype(np.float32)   # 固定颗粒：不影响压缩
    f = visual(t)
    _texts.draw(f, t)
    f += _grain
    return (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)


# ====================================================================== 输出
def encode(frames_t, path, crf=18):
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                          "-crf", str(crf), "-tune", "film", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for t in frames_t:
        p.stdin.write(render_frame(t).tobytes())
    p.stdin.close(); p.wait()


def seg_job(args):
    k, a, b = args
    path = os.path.join(HERE, f".seg_{k}.mp4")
    encode([i / FPS for i in range(a, b)], path)
    return path


def mux(video, out, t0=0.0, dur=None):
    audio = os.path.join(CACHE, "final_audio.wav")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-ss", str(t0)]
    if dur:
        cmd += ["-t", str(dur)]
    cmd += ["-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)


def main():
    args = sys.argv[1:]
    if args and args[0] == "--still":
        os.makedirs(os.path.join(HERE, "stills"), exist_ok=True)
        ims = []
        for s in args[1:]:
            im = Image.fromarray(render_frame(float(s)))
            im.save(os.path.join(HERE, "stills", f"still_{float(s):06.1f}.jpg"), quality=92)
            ims.append(im)
            print("still", s)
        if len(ims) > 1:   # 拼一张缩略总览
            cols = 3; rows = (len(ims) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * 640, rows * 360))
            for i, im in enumerate(ims):
                sheet.paste(im.resize((640, 360), Image.LANCZOS), ((i % cols) * 640, (i // cols) * 360))
            sheet.save(os.path.join(HERE, "stills", "contact_sheet.jpg"), quality=90)
        return
    if args and args[0] == "--demo":
        t0, dur = float(args[1]), float(args[2])
        n0, n = int(t0 * FPS), int(dur * FPS)
        tmp = os.path.join(HERE, ".seg_demo.mp4")
        encode([(n0 + i) / FPS for i in range(n)], tmp)
        mux(tmp, os.path.join(HERE, "hohhot_demo.mp4"), t0, dur)
        os.remove(tmp)
        return
    n = int(TOTAL * FPS)
    cuts = np.linspace(0, n, WORKERS + 1).astype(int)
    with mp.get_context("fork").Pool(WORKERS) as pool:
        segs = pool.map(seg_job, [(k, cuts[k], cuts[k + 1]) for k in range(WORKERS)])
    lst = os.path.join(HERE, ".segs.txt")
    open(lst, "w").write("".join(f"file '{s}'\n" for s in segs))
    tmp = os.path.join(HERE, ".seg_all.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", tmp], check=True)
    mux(tmp, os.path.join(HERE, "hohhot_full.mp4"))
    for s in segs + [tmp, lst]:
        os.remove(s)


if __name__ == "__main__":
    main()
