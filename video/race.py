"""
《红线》(Redline) 赛车动画 —— 配 ../music/redline.mp3
夜间赛道追车镜头：镜头跟在 F1 赛车正后方，赛道、广告墙、灯柱逐像素光线投射生成，
多次采样做运动模糊；赛车用矢量绘制（2 倍超采样抗锯齿），车底擦出火花。

运行：python3 race.py --still 4 60      先出静帧（race_still_4.png ...）
     python3 race.py --demo 10          前 N 秒 demo
     python3 race.py                    完整 128 秒（4 进程）
依赖：numpy、scipy、Pillow、ffmpeg
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import gaussian_filter, zoom

HERE = os.path.dirname(os.path.abspath(__file__))
OW, OH = 1280, 720            # 输出分辨率
SS = 1.0                      # 环境超采样倍数
RW, RH = int(OW * SS), int(OH * SS)
FPS = 30
AUDIO = os.path.join(HERE, "..", "music", "redline.mp3")
LETTER = 88                   # 2.39:1 宽银幕黑边（像素，输出分辨率）

# 相机 / 场景（单位：米）
HC = 1.75                     # 相机高度
HOR = 0.40                    # 地平线位置（画面高度比例）
FOV = 66                      # 水平视场
ROAD = 7.0                    # 赛道半宽
KERB = 1.1                    # 路肩宽
WALL = 13.0                   # 护墙到中心距离
WALL_H, FENCE_H = 1.0, 4.6    # 护墙、防护网高度
POLE = 42.0                   # 灯柱间距
LAMP_H = 10.0
CAR_Z = 6.4                   # 赛车离相机距离

rng = np.random.default_rng(5)
TEX = gaussian_filter(rng.standard_normal((512, 512)).astype(np.float32), 0.8)
TEX /= TEX.std()
SPECK = (rng.random((512, 512)) > 0.985).astype(np.float32)


def smooth(t, a, b):
    x = np.clip((t - a) / (b - a), 0, 1)
    return x * x * (3 - 2 * x)


# ------------------------------------------------------------------ 时间线（和音乐段落对齐）
SEC = 12.8   # 8 小节


def speed_at(t):      # 米/秒
    keys = [(0, 42), (6, 62), (12.8, 85), (38.4, 100), (50.9, 106), (52.4, 128), (76.8, 122),
            (80, 78), (89.6, 84), (101.5, 100), (102.4, 150), (115.2, 148), (124, 60), (128, 42)]
    xs, ys = zip(*keys)
    return float(np.interp(t, xs, ys))


_TT = np.linspace(0, 130, 13001)
_PH = np.concatenate([[0], np.cumsum([speed_at(x) * 0.01 for x in _TT[:-1]])])


def phase_at(t):
    return float(np.interp(t, _TT, _PH)) + 300.0


def curve_at(t):      # 赛道曲率：缓缓左右弯
    return 0.0011 * np.sin(0.21 * t + 0.6) + 0.0006 * np.sin(0.53 * t + 2.0)


# ------------------------------------------------------------------ 环境（光线投射）
F = RW / 2 / np.tan(np.radians(FOV / 2))
CX, CY = RW / 2, RH * HOR
_X, _Y = np.meshgrid(np.arange(RW, dtype=np.float32) + 0.5, np.arange(RH, dtype=np.float32) + 0.5)


def sky(dx, dy, heading):
    h = np.clip(dy * 2.2, 0, 1)
    top = np.array([0.004, 0.006, 0.016])
    mid = np.array([0.012, 0.016, 0.04])
    hor = np.array([0.07, 0.06, 0.09])     # 城市光晕
    c = np.where(h[..., None] < 0.18, hor + (mid - hor) * (h[..., None] / 0.18),
                 mid + (top - mid) * np.clip((h[..., None] - 0.18) / 0.6, 0, 1) ** 0.7)
    c = c * (0.85 + 0.15 * np.exp(-((dx + 0.15) / 0.5) ** 2))[..., None]
    # 星星（固定在天空，随赛道方向缓慢平移）
    sx = np.floor((np.arctan(dx) * 180 + heading) * 6).astype(np.int64)
    sy = np.floor(dy * 1100).astype(np.int64)
    hsh = (np.sin(sx * 12.9898 + sy * 78.233) * 43758.5453) % 1
    star = (hsh > 0.9965) & (dy > 0.06)
    c = c + (star * ((hsh - 0.9965) / 0.0035) ** 2)[..., None] * np.array([0.8, 0.85, 1.0]) * 1.2
    # 城市天际线
    ang = np.arctan(dx) * 180 + heading
    cell = np.floor(ang / 2.3)
    hgt = (0.006 + 0.034 * (np.sin(cell * 12.9898) * 43758.5453 % 1) ** 2.2)
    tower = (np.sin(cell * 78.233) * 9631.1 % 1) > 0.88
    hgt = np.where(tower, hgt + 0.028, hgt)
    bld = dy < hgt
    win = ((np.floor(ang * 9) % 2 == 0) & (np.floor(dy * 420) % 3 == 0)
           & ((np.sin(np.floor(ang * 9) * 3.1 + np.floor(dy * 420) * 7.7) * 999 % 1) > 0.35))
    bcol = np.array([0.012, 0.012, 0.02]) + win[..., None] * np.array([0.95, 0.7, 0.4]) * 0.6
    c = np.where(bld[..., None], bcol, c)
    return c


def lamp_light(u, s):
    """路面受泛光灯照亮的程度"""
    L = np.zeros_like(u)
    base = np.floor(s / POLE) * POLE
    for side, off in ((-1, 0.0), (1, POLE / 2)):
        for k in (-1, 0, 1, 2):
            zl = base + k * POLE + off
            d2 = (u - side * WALL) ** 2 + (s - zl) ** 2 + LAMP_H ** 2
            L += 1640 / (d2 * np.sqrt(d2))
    return L


def C(*v):
    return np.array(v, np.float32)


def env_geom(curv, heading):
    """与相位无关的几何：每帧只算一次（运动模糊各次采样共用）"""
    dx = ((_X - CX) / F).ravel()
    dy = (-(_Y - CY) / F).ravel()
    G = {"sky": sky(dx, dy, heading).astype(np.float32)}
    g = dy < -1e-4
    tg = np.where(g, np.minimum(HC / np.maximum(-dy, 1e-4), 5000), 5000.0).astype(np.float32)

    a = curv if abs(curv) > 1e-9 else 1e-9
    best_t = np.full(dx.shape, 9000.0, dtype=np.float32)
    best_side = np.zeros(dx.shape, dtype=np.float32)
    for side in (-1, 1):
        b, c = -dx, side * WALL
        disc = b * b - 4 * a * c
        ok = disc >= 0
        sq = np.sqrt(np.maximum(disc, 0))
        q = -0.5 * (b + np.where(b >= 0, 1, -1) * sq)
        q = np.where(np.abs(q) < 1e-9, 1e-9, q)
        t1, t2 = q / a, c / q
        t1 = np.where((t1 > 0.5) & ok, t1, 9000.0)
        t2 = np.where((t2 > 0.5) & ok, t2, 9000.0)
        tw = np.minimum(t1, t2)
        yw = HC + dy * tw
        hit = (tw < best_t) & (yw > 0) & (yw < LAMP_H + 1.0)
        best_t = np.where(hit, tw, best_t)
        best_side = np.where(hit, side, best_side)
    vis = (best_t < 8000) & (best_t < tg)
    gr = g & (tg < 4999) & ~vis

    gi = np.nonzero(gr)[0]
    z = tg[gi]
    u = dx[gi] * z - curv * z * z
    G.update(gi=gi, z=z, u=u, au=np.abs(u),
             fres=(np.clip(1 - (-dy[gi]) * 6, 0, 1) ** 3).astype(np.float32),
             fog=np.exp(-z / 260)[:, None].astype(np.float32))
    wi = np.nonzero(vis)[0]
    tw = best_t[wi]
    G.update(wi=wi, tw=tw, yw=(HC + dy[wi] * tw).astype(np.float32), side=best_side[wi],
             fogw=np.exp(-tw / 260)[:, None].astype(np.float32))
    return G


def env_shade(G, phase):
    img = G["sky"].copy()

    # ---- 地面
    z, u, au = G["z"], G["u"], G["au"]
    s = z + phase
    tex = TEX[(np.floor(u * 30).astype(np.int64) % 512), (np.floor(s * 6).astype(np.int64) % 512)]
    spk = SPECK[(np.floor(u * 60).astype(np.int64) % 512), (np.floor(s * 12).astype(np.int64) % 512)]
    detail = np.exp(-z / 30)
    asphalt = 0.085 + 0.012 * tex * detail + 0.05 * spk * detail
    col = asphalt[:, None] * C(1.0, 1.0, 1.05)
    groove = np.exp(-((au - 0.85) / 0.35) ** 2) * 0.35          # 磨胎痕
    col *= (1 - groove * 0.5)[:, None]
    on_kerb = (au > ROAD) & (au < ROAD + KERB)                  # 红白路肩
    stripe = (np.floor(s / 1.6) % 2)[:, None]
    kerb_c = stripe * C(0.75, 0.75, 0.73) + (1 - stripe) * C(0.7, 0.06, 0.05)
    kerb_c = kerb_c + (C(0.72, 0.4, 0.39) - kerb_c) * np.clip((z[:, None] - 25) / 60, 0, 1)
    col = np.where(on_kerb[:, None], kerb_c, col)
    line = (au > ROAD - 0.28) & (au < ROAD - 0.1)
    col = np.where(line[:, None], C(0.8, 0.8, 0.78), col)
    runoff = au >= ROAD + KERB
    col = np.where(runoff[:, None], col * 0.8 + C(0.0, 0.015, 0.03), col)
    L = lamp_light(u, s)                                        # 灯光 + 掠射反光
    col = col * (0.22 + 2.6 * L)[:, None] + G["fres"][:, None] * C(0.1, 0.12, 0.18) * 0.45
    fog = G["fog"]
    col = col * fog + C(0.05, 0.05, 0.07) * (1 - fog)
    img[G["gi"]] = col

    # ---- 护墙 / 防护网 / 灯柱
    tw, yw, side, fogw = G["tw"], G["yw"], G["side"], G["fogw"]
    w = img[G["wi"]]
    sw = tw + phase
    sp = sw - np.where(side > 0, POLE / 2, 0)                   # 两侧灯柱错开半个间距
    m = sp - np.floor(sp / POLE) * POLE
    panel = (np.floor(sw / 14) % 4).astype(int)                 # 广告牌
    pal = np.array([[0.8, 0.36, 0.12], [0.7, 0.7, 0.68], [0.05, 0.05, 0.06], [0.08, 0.3, 0.36]], np.float32)
    wall_c = pal[panel]
    band = (yw > 0.42) & (yw < 0.6) & (panel != 1)
    wall_c = np.where(band[:, None], wall_c * 0.4 + 0.55, wall_c)
    Lw = lamp_light(np.full_like(tw, WALL - 1.5) * side, sw)
    wall_c = wall_c * (0.12 + 0.55 * Lw)[:, None]
    wall_c = wall_c * fogw + C(0.05, 0.05, 0.07) * (1 - fogw)
    is_wall = yw <= WALL_H
    w = np.where(is_wall[:, None], wall_c, w)
    led = (yw > WALL_H - 0.06) & (yw <= WALL_H) & ((sw % 3) < 1.2)
    w = np.where(led[:, None], C(1.6, 0.5, 0.2) * fogw + 0.4 * (1 - fogw), w)
    is_fence = (yw > WALL_H) & (yw < FENCE_H)                  # 防护网：半透明网格
    mesh = ((np.abs(((sw * 1.3) % 1) - 0.5) < 0.06) | (np.abs(((yw * 1.3) % 1) - 0.5) < 0.06))
    post = (sw % 4) < 0.12
    fence_a = (np.where(post, 0.9, np.where(mesh, 0.25, 0.06)) * np.exp(-tw / 120))[:, None]
    fence_c = C(0.3, 0.3, 0.32) * (0.12 + 0.7 * Lw)[:, None]
    w = np.where(is_fence[:, None], w * (1 - fence_a) + fence_c * fence_a, w)
    is_pole = (m < 0.35) & (yw > WALL_H) & (yw < LAMP_H)        # 灯柱 + 灯头
    w = np.where(is_pole[:, None], C(0.08, 0.08, 0.09) * fogw + 0.1 * (1 - fogw), w)
    is_lamp = (m > POLE - 1.8) & (yw > LAMP_H - 0.25) & (yw < LAMP_H + 0.45)
    w = np.where(is_lamp[:, None], C(3.2, 3.0, 2.6), w)
    img[G["wi"]] = w
    return img


def env_blur(t, n=5):
    """运动模糊：快门时间内多次采样平均（几何只算一次）"""
    v = speed_at(t)
    ph0 = phase_at(t)
    curv = curve_at(t)
    G = env_geom(curv, ph0 * curv * 40)
    acc = 0
    for k in np.linspace(-0.5, 0.5, n):
        acc = acc + env_shade(G, ph0 + v * k / FPS * 0.9)
    img = (acc / n).reshape(RH, RW, 3)
    small = np.stack([zoom(img[..., c], 1 / SS, order=1) for c in range(3)], axis=2)
    return small[:OH, :OW]


# ------------------------------------------------------------------ 赛车（矢量绘制，后视）
ORANGE = (236, 104, 34)   # 赛车涂装橙
LOGO = (217, 119, 87)


def draw_car(t, sway=0.0, bounce=0.0, flash=0.0):
    """三维透视绘制 F1 赛车（从后上方看），远到近依次画（画家算法）"""
    S = 2   # 超采样
    W, H = OW * S, OH * S
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = OW / 2 / np.tan(np.radians(FOV / 2))
    cx, cy = OW / 2 + sway, OH * HOR + bounce

    def P(x, y, dz=0.0):
        z = CAR_Z + dz
        return ((cx + x * f / z) * S, (cy + (HC - y) * f / z) * S)

    def poly(pts, fill):
        d.polygon([P(*p) for p in pts], fill=fill)

    def box(x1, x2, y1, y2, z1, z2, face, top, side=None, r=0.0):
        """长方体：画顶面、朝相机的面（z1 面）和朝内的侧面"""
        poly([(x1, y2, z1), (x2, y2, z1), (x2, y2, z2), (x1, y2, z2)], top)
        if side is not None:
            xs = x1 if x1 > 0 else x2      # 朝向车中心的那一侧
            poly([(xs, y1, z1), (xs, y2, z1), (xs, y2, z2), (xs, y1, z2)], side)
        a, b = P(x1, y2, z1), P(x2, y1, z1)
        d.rounded_rectangle([a, b], radius=r * f / CAR_Z * S, fill=face)

    OR = ORANGE + (255,)
    OR_T = tuple(min(255, int(c * 1.18)) for c in ORANGE) + (255,)   # 顶面更亮
    OR_D = tuple(int(c * 0.62) for c in ORANGE) + (255,)              # 背光面
    CB, CB_T = (24, 24, 28, 255), (48, 48, 54, 255)
    TY, TY_T, TY_S = (30, 30, 33, 255), (78, 78, 84, 255), (44, 44, 48, 255)

    # 车底阴影
    sh = Image.new("L", (W, H), 0)
    ImageDraw.Draw(sh).polygon([P(-1.2, 0, -0.5), P(1.2, 0, -0.5), P(1.1, 0, 5.2), P(-1.1, 0, 5.2)], fill=220)
    sh = sh.filter(ImageFilter.GaussianBlur(9 * S))

    # 前翼
    poly([(-0.95, 0.16, 4.75), (0.95, 0.16, 4.75), (0.95, 0.16, 5.15), (-0.95, 0.16, 5.15)], CB_T)
    for sg in (-1, 1):
        poly([(sg * 0.95, 0.05, 4.7), (sg * 0.95, 0.3, 4.7), (sg * 0.95, 0.3, 5.2), (sg * 0.95, 0.05, 5.2)], OR_D)
    # 前轮
    for sg in (-1, 1):
        x1, x2 = sorted((sg * 0.62, sg * 0.95))
        box(x1, x2, 0.0, 0.66, 3.9, 4.55, TY, TY_T, TY_S, 0.1)
    # 鼻锥
    poly([(-0.13, 0.42, 3.0), (0.13, 0.42, 3.0), (0.07, 0.3, 4.85), (-0.07, 0.3, 4.85)], OR_T)
    # 前悬挂
    for sg in (-1, 1):
        d.line([P(sg * 0.12, 0.38, 3.9), P(sg * 0.66, 0.36, 4.2)], fill=CB, width=int(0.03 * f / 9 * S) + 1)
    # 座舱 / 单体壳
    poly([(-0.42, 0.62, 1.6), (0.42, 0.62, 1.6), (0.2, 0.5, 3.05), (-0.2, 0.5, 3.05)], OR_T)
    poly([(-0.2, 0.63, 2.0), (0.2, 0.63, 2.0), (0.17, 0.6, 2.75), (-0.17, 0.6, 2.75)], (10, 10, 12, 255))
    # 头盔 + halo
    a, b = P(-0.13, 0.92, 2.35), P(0.13, 0.62, 2.35)
    d.ellipse([a[0], a[1], b[0], b[1]], fill=(245, 210, 55, 255))
    a, b = P(-0.1, 0.84, 2.36), P(0.1, 0.76, 2.36)
    d.rectangle([a, b], fill=(25, 25, 30, 255))                           # 面罩
    for sg in (-1, 1):
        d.line([P(sg * 0.2, 0.64, 2.0), P(sg * 0.17, 0.93, 2.3), P(0, 0.95, 2.75)], fill=CB,
               width=int(0.04 * f / 8 * S) + 1)
    # 侧箱（可乐瓶形，前高后低）
    for sg in (-1, 1):
        xi, xo = sg * 0.3, sg * 0.78
        poly([(xo, 0.56, 2.7), (xi, 0.62, 2.7), (xi * 0.8, 0.5, 0.4), (sg * 0.5, 0.32, 0.4)], OR_T)
        poly([(xo, 0.56, 2.7), (sg * 0.5, 0.32, 0.4), (sg * 0.5, 0.1, 0.4), (xo, 0.1, 2.7)], OR_D)
    # 发动机盖 + 进气口 + 鲨鱼鳍
    for sg in (-1, 1):
        poly([(0, 1.02, 1.85), (sg * 0.24, 0.66, 1.85), (sg * 0.16, 0.42, 0.05), (0, 0.66, 0.05)],
             OR_T if sg < 0 else OR)
    poly([(-0.11, 0.98, 1.86), (0.11, 0.98, 1.86), (0, 1.16, 1.9)], CB)      # 防滚架进气口
    poly([(0, 1.02, 1.85), (0, 0.66, 0.05), (0.008, 0.66, 0.05)], (250, 250, 250, 255))  # 中线
    # 后轮
    for sg in (-1, 1):
        x1, x2 = sorted((sg * 0.6, sg * 1.0))
        box(x1, x2, 0.0, 0.72, 0.0, 0.7, TY, TY_T, TY_S, 0.12)
        xo = x1 if sg < 0 else x2
        d.line([P(xo, 0.1, 0), P(xo, 0.62, 0)], fill=(95, 95, 100, 255), width=max(2, int(0.015 * f / CAR_Z * S)))
        xs = x2 - 0.035 if sg < 0 else x1 + 0.035
        d.line([P(xs, 0.18, 0), P(xs, 0.54, 0)], fill=(215, 180, 40, 255), width=max(2, int(0.02 * f / CAR_Z * S)))
    # 后悬挂
    for sg in (-1, 1):
        for y0, y1 in ((0.4, 0.44), (0.27, 0.22)):
            d.line([P(sg * 0.15, y0, -0.05), P(sg * 0.64, y1, 0.2)], fill=(8, 8, 10, 255), width=int(0.03 * f / CAR_Z * S))
    # 扩散器 + 变速箱
    poly([(-0.55, 0.04, -0.15), (0.55, 0.04, -0.15), (0.42, 0.3, -0.05), (-0.42, 0.3, -0.05)], (10, 10, 12, 255))
    for k in np.linspace(-0.38, 0.38, 6):
        poly([(k - 0.012, 0.05, -0.15), (k + 0.012, 0.05, -0.15), (k * 0.8 + 0.01, 0.28, -0.05), (k * 0.8 - 0.01, 0.28, -0.05)],
             (44, 44, 50, 255))
    a, b = P(-0.15, 0.42, -0.08), P(0.15, 0.25, -0.08)
    d.rounded_rectangle([a, b], radius=6, fill=CB)
    # 梁翼 + 尾翼支架
    poly([(-0.46, 0.5, -0.12), (0.46, 0.5, -0.12), (0.46, 0.44, -0.12), (-0.46, 0.44, -0.12)], CB)
    poly([(-0.03, 0.42, -0.12), (0.03, 0.42, -0.12), (0.03, 0.82, -0.25), (-0.03, 0.82, -0.25)], CB)
    # 尾翼：主翼面（顶面可见）+ 襟翼 + 端板
    poly([(-0.5, 0.82, -0.3), (0.5, 0.82, -0.3), (0.5, 0.84, 0.15), (-0.5, 0.84, 0.15)], CB_T)
    poly([(-0.5, 0.78, -0.3), (0.5, 0.78, -0.3), (0.5, 0.82, -0.3), (-0.5, 0.82, -0.3)], CB)
    poly([(-0.5, 0.9, -0.35), (0.5, 0.9, -0.35), (0.5, 0.95, -0.15), (-0.5, 0.95, -0.15)], CB_T)
    poly([(-0.5, 0.86, -0.35), (0.5, 0.86, -0.35), (0.5, 0.9, -0.35), (-0.5, 0.9, -0.35)], CB)
    d.line([P(-0.5, 0.95, -0.15), P(0.5, 0.95, -0.15)], fill=OR_T, width=max(2, int(0.012 * f / CAR_Z * S)))
    for sg in (-1, 1):
        x = sg * 0.52
        poly([(x, 0.55, -0.4), (x, 0.98, -0.4), (x, 0.98, 0.3), (x, 0.6, 0.3)], CB_T)
        a, b = P(min(x, x + sg * 0.03), 0.98, -0.4), P(max(x, x + sg * 0.03), 0.55, -0.4)
        d.rectangle([a, b], fill=OR)

    arr = np.asarray(layer).astype(np.float32) / 255
    # 顶光：越往上越亮
    yy = np.arange(H)[:, None] / S
    yr = (yy - (cy + (HC - 1.1) * f / CAR_Z)) / ((1.1) * f / CAR_Z)
    arr[..., :3] *= np.clip(1.2 - 0.45 * yr, 0.6, 1.25)[..., None]
    al = arr[..., 3]
    edge = np.clip(al - np.roll(al, 2 * S, axis=0), 0, 1)
    arr[..., :3] += edge[..., None] * np.array([0.35, 0.3, 0.25])
    # 尾灯
    gl = Image.new("L", (W, H), 0)
    ImageDraw.Draw(gl).rectangle([P(-0.07, 0.38, -0.1), P(0.07, 0.32, -0.1)], fill=255)
    gl = np.asarray(gl).astype(np.float32) / 255
    lamp_on = 0.45 + 0.55 * flash
    arr[..., :3] = np.where(gl[..., None] > 0, np.array([1.0, 0.1, 0.08]) * (1.5 + 2.5 * lamp_on), arr[..., :3])

    def down(x):
        return x.reshape(OH, S, OW, S, *x.shape[2:]).mean(axis=(1, 3))

    return down(arr), down(np.asarray(sh).astype(np.float32) / 255), down(gl) * lamp_on, (cx, cy, f)


def sparks(t, geo):
    """车底滑块擦地的火花：从车底中部向后飞向镜头"""
    cx, cy, f = geo
    img = Image.new("RGB", (OW, OH), 0)
    d = ImageDraw.Draw(img)
    r = np.random.default_rng(int(t * FPS) + 99)
    burst = max(0.0, np.sin(t * 2.7) * np.sin(t * 1.3 + 1))
    n_sp = int(60 * burst)
    if 102.4 <= t < 115.2:                       # 高潮：火花持续喷射
        n_sp = int(110 + 50 * burst)
    for _ in range(n_sp):
        age = r.random() * 0.12
        vx, vz, vy = r.normal(0, 2.5), -r.uniform(20, 45), r.uniform(0.3, 2.5)
        x = r.uniform(-0.25, 0.25) + vx * age
        z = CAR_Z + 1.0 + vz * age
        y = max(0.0, 0.02 + vy * age - 4.9 * age * age)
        if z < 0.8:
            continue
        x2, z2, y2 = x - vx * 0.012, z - vz * 0.012, y
        p1 = (cx + x * f / z, cy + (HC - y) * f / z)
        p2 = (cx + x2 * f / z2, cy + (HC - y2) * f / z2)
        c = 1 - age / 0.12
        d.line([p1, p2], fill=(255, int(200 * c + 40), int(90 * c)), width=2 if z < 4 else 1)
    return np.asarray(img).astype(np.float32) / 255 * 1.5


# ------------------------------------------------------------------ 叠加：标题、标志
def overlay_layers():
    title = Image.new("RGBA", (OW, OH), (0, 0, 0, 0))
    dr = ImageDraw.Draw(title)
    zh = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 64)
    en = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)

    def spaced(text, font, y, spacing, fill):
        widths = [dr.textlength(c, font=font) for c in text]
        total = sum(widths) + spacing * (len(text) - 1)
        x = (OW - total) / 2
        for c, w in zip(text, widths):
            dr.text((x, y), c, font=font, fill=fill)
            x += w + spacing

    spaced("红线", zh, 104, 40, (245, 240, 232, 255))
    spaced("R E D L I N E", en, 184, 6, (230, 90, 60, 255))
    title = title.filter(ImageFilter.GaussianBlur(0.3))

    logo = Image.new("RGBA", (OW, OH), (0, 0, 0, 0))
    dl = ImageDraw.Draw(logo)
    cx, cy, R = OW - 132, OH - LETTER - 34, 13
    lr = np.random.default_rng(11)
    for i in range(12):   # Claude 星芒标志（简化绘制）
        ang = i * 2 * np.pi / 12 + lr.uniform(-0.12, 0.12)
        ln = R * lr.uniform(0.75, 1.0)
        x2, y2 = cx + ln * np.cos(ang), cy + ln * np.sin(ang)
        dl.line([(cx, cy), (x2, y2)], fill=LOGO + (255,), width=3)
        dl.ellipse([x2 - 1.5, y2 - 1.5, x2 + 1.5, y2 + 1.5], fill=LOGO + (255,))
    lf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 21)
    dl.text((cx + 22, cy - 13), "Claude", font=lf, fill=(235, 230, 222, 255))
    to_np = lambda im: np.asarray(im).astype(np.float32) / 255
    return to_np(title), to_np(logo)


def title_alpha(t, total):
    return float(smooth(t, 1.0, 2.5) * (1 - smooth(t, 9.0, 11.0)) + smooth(t, total - 9, total - 6.5))


# ------------------------------------------------------------------ 合成一帧
_VIG = None


def frame(t, layers, total=128.0, pulse=0.0):
    global _VIG
    title, logo = layers
    r = np.random.default_rng(int(t * 1000))
    v = speed_at(t)
    shake = (v / 90) ** 2 * 1.2
    sway = 6 * np.sin(t * 0.9) + r.normal(0, shake)
    bounce = r.normal(0, shake * 0.7)
    img = env_blur(t)
    car, shadow, tail, geo = draw_car(t, sway, bounce, flash=pulse)
    img *= (1 - 0.75 * shadow)[..., None]
    # 尾灯在路面上的红色反光
    refl = gaussian_filter(tail, 18) * 40
    img += refl[..., None] * np.array([0.9, 0.05, 0.03])
    a = car[..., 3:4]
    img = img * (1 - a) + car[..., :3] * a
    img += sparks(t, geo) * min(1, v / 70)
    # 泛光
    bright = np.clip(img - 0.85, 0, None)
    bloom = np.stack([gaussian_filter(bright[..., c], 6) + gaussian_filter(bright[..., c], 22) for c in range(3)], 2)
    img = img + bloom * (0.55 + 0.4 * pulse)
    # 调色：暗部偏青、亮部偏橙
    lum = img.mean(axis=2, keepdims=True)
    img = img + (np.array([-0.01, 0.01, 0.025]) * (1 - np.clip(lum * 2, 0, 1))
                 + np.array([0.03, 0.0, -0.03]) * np.clip(lum, 0, 1))
    img = img / (1 + img * 0.35)                                # 柔和高光压缩
    if _VIG is None:
        yy, xx = np.mgrid[0:OH, 0:OW]
        _VIG = (1 - 0.45 * (((xx - OW / 2) / (OW / 2)) ** 2 + ((yy - OH / 2) / (OH / 1.4)) ** 2))[..., None]
    img *= _VIG
    img += r.normal(0, 0.012, (OH, OW, 1))
    img = np.clip(img, 0, 1)
    # 叠加文字
    ta = title[..., 3:4] * title_alpha(t, total)
    img = img * (1 - ta) + title[..., :3] * ta
    img = img * (1 - logo[..., 3:4]) + logo[..., :3] * logo[..., 3:4]
    img[:LETTER] = 0
    img[OH - LETTER:] = 0
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def audio_pulse(nf):
    """每帧的音频瞬态强度（0~1）：底鼓/重音一响，尾灯和泛光就闪一下"""
    from scipy.io import wavfile
    wav = os.path.join(HERE, "..", "music", "redline.wav")
    if not os.path.exists(wav):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", AUDIO, wav], check=True)
    sr, x = wavfile.read(wav)
    x = x.astype(np.float32).mean(axis=1) / 32768
    hop = sr // FPS
    n = min(nf, len(x) // hop)
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-9)
    rms = np.pad(rms, (0, nf - n), mode="edge")
    p = np.clip(rms / gaussian_filter(rms, FPS * 0.5) - 1.0, 0, 1) * 2.5
    env = np.zeros_like(p)
    for i in range(nf):   # 快起慢落
        prev = env[i - 1] if i else 0
        env[i] = min(1.0, p[i]) if p[i] > prev else prev * 0.8
    return env


TOTAL = 128.0
LAYERS = PULSE = None
T0, DUR = 0.0, TOTAL


def render_frame(i):
    t = T0 + i / FPS
    img = frame(t, LAYERS, TOTAL, float(PULSE[min(int(t * FPS), len(PULSE) - 1)])).astype(np.float32)
    fade = min(1.0, (i + 1) / (FPS * 0.5), (DUR * FPS - i) / (FPS * (3.0 if DUR == TOTAL else 0.6)))
    return (img * max(0.0, fade)).astype(np.uint8)


def worker(args):
    k, i0, i1 = args
    seg = os.path.join(HERE, f".seg_{k}.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                          "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", seg], stdin=subprocess.PIPE)
    for i in range(i0, i1):
        p.stdin.write(render_frame(i).tobytes())
        if (i - i0) % 100 == 0:
            print(f"  worker {k}: {i - i0}/{i1 - i0}", flush=True)
    p.stdin.close()
    p.wait()
    return seg


def main():
    global LAYERS, PULSE, T0, DUR
    import multiprocessing as mp
    import warnings
    warnings.filterwarnings("ignore")
    LAYERS = overlay_layers()
    if "--still" in sys.argv:
        ts = [float(x) for x in sys.argv[sys.argv.index("--still") + 1:]]
        for t in ts:
            Image.fromarray(frame(t, LAYERS)).save(os.path.join(HERE, "race_still_%g.png" % t))
            print("saved race_still_%g.png" % t)
        return
    out = os.path.join(HERE, "redline.mp4")
    if "--demo" in sys.argv:            # --demo 起始秒 时长
        k = sys.argv.index("--demo")
        T0, DUR = float(sys.argv[k + 1]), float(sys.argv[k + 2])
        out = os.path.join(HERE, "redline_demo.mp4")
    PULSE = audio_pulse(int(TOTAL * FPS) + 1)
    nf = int(round(DUR * FPS))
    workers = 4
    bounds = np.linspace(0, nf, workers + 1).astype(int)
    with mp.get_context("fork").Pool(workers) as pool:
        segs = pool.map(worker, [(k, bounds[k], bounds[k + 1]) for k in range(workers)])
    lst = os.path.join(HERE, ".segs.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{x}'\n" for x in segs)
    afade = f"afade=t=in:d=0.3,afade=t=out:st={DUR - 0.6}:d=0.6" if DUR < TOTAL else ""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-ss", str(T0), "-i", AUDIO, "-map", "0:v", "-map", "1:a", "-c:v", "copy"]
                   + (["-af", afade] if afade else [])
                   + ["-c:a", "aac", "-b:a", "192k", "-t", str(DUR), "-movflags", "+faststart", out], check=True)
    for x in segs + [lst]:
        os.remove(x)
    print("saved", out)


if __name__ == "__main__":
    main()
