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


def speed_at(t):      # 米/秒：起跑 0 → 加速 → 比赛 → 终点前收一收让白旗看得清 → 冲线后减速
    keys = [(0, 0), (9.6, 0), (10.8, 38), (12.8, 80), (25, 92), (38.4, 100), (50.9, 106), (52.4, 128),
            (76.8, 122), (80, 78), (89.6, 84), (101.5, 100), (102.4, 150), (115.2, 140), (117, 92),
            (119, 84), (121.6, 105), (122.8, 70), (128, 38)]
    xs, ys = zip(*keys)
    return float(np.interp(t, xs, ys))


_TT = np.linspace(0, 130, 13001)
_PH = np.concatenate([[0], np.cumsum([speed_at(x) * 0.01 for x in _TT[:-1]])])


def phase_at(t):
    return float(np.interp(t, _TT, _PH)) + 300.0


def curve_at(t):      # 赛道曲率：缓缓左右弯；发车区和终点直道保持笔直
    base = 0.0011 * np.sin(0.21 * t + 0.6) + 0.0006 * np.sin(0.53 * t + 2.0)
    return float(base * smooth(t, 11, 16) * (1 - smooth(t, 114, 118)))


# ------------------------------------------------------------------ 剧本：发车、超车、冲线
from scipy.interpolate import PchipInterpolator

S_CAR0 = 300.0 + CAR_Z            # 起点处本车车尾的赛道位置
T_LIGHTS = [5.6, 6.4, 7.2, 8.0, 8.8]   # 5 盏起跑灯依次亮起（每 2 拍一盏）
T_OUT = 9.6                       # 灯灭 = 起步（对上引擎呼啸 + 上升音效）
T_FIN = 121.6                     # 冲线（对上结尾的重击）
S_GANTRY = S_CAR0 + 58.0
S_FIN = float(np.interp(T_FIN, _TT, _PH)) + 300.0 + CAR_Z + 5.2   # 本车车头越线时的赛道位置
# 发车格：(领先本车的距离 d, 横向位置 u)；最后两项是本车和身后的 R6
GRID = [(8, 2.0), (16, -2.0), (24, 2.0), (32, -2.0), (40, 2.0), (0, -2.0), (-8, 2.0)]


def _cruise(ug, uc):
    return [(0, ug), (10.4, ug), (15, uc)]


# d：对手车尾领先本车车尾的距离（米，负数 = 已被超过）；u：对手所在车道
RIVALS = [
    dict(col=(205, 205, 212), d=[(0, 8), (9.6, 8), (13, 22), (17, 30), (21, 22), (23.6, 13), (25.6, 0.5),
                                 (26.8, -6.5), (28, -13)], u=_cruise(2.0, 0.0)),                 # 25.6 超车
    dict(col=(0, 150, 150), d=[(0, 16), (9.6, 16), (13, 32), (25, 40), (40, 36), (47, 22), (49.4, 12),
                               (51.2, 0.5), (52.4, -6.5), (53.6, -13)], u=_cruise(-2.0, 1.5)),   # 51.2 超车
    dict(col=(40, 80, 210), d=[(0, 24), (9.6, 24), (13, 40), (30, 52), (52, 48), (59, 26), (61.8, 13),
                               (64, 0.5), (65.2, -6.5), (66.4, -13)], u=_cruise(2.0, -1.5)),      # 64 超车
    dict(col=(240, 205, 20), d=[(0, 32), (9.6, 32), (13, 50), (40, 70), (80, 62), (94, 44), (99.2, 20),
                                (100.8, 13), (102.4, 0.5), (103.6, -6.5), (104.8, -13)], u=_cruise(-2.0, 0.5)),  # 102.4 超车
    dict(col=(60, 185, 80), d=[(0, 40), (9.6, 40), (13, 60), (40, 80), (80, 74), (100, 56), (104, 28),
                               (105.6, 14), (107.2, 0.5), (108.4, -6.5), (109.6, -13)], u=_cruise(2.0, -1.0)),  # 107.2 超车
    dict(col=(215, 30, 35), d=[(0, -8), (9.6, -8), (11, -20), (13, -40), (60, -60), (100, -70), (110, -50),
                               (114, -22), (116.2, -8), (117.6, 0), (118.8, 3.0), (120, 2.0), (121.0, 0.5),
                               (121.6, -0.9), (123, -4), (124.5, -8), (128, -14)],
         u=[(0, 2.0), (10.4, 2.0), (15, 0.0), (110, 0.0), (114, 2.5), (117, 3.6), (128, 3.6)]),       # 终点前并排，冲线领先
]
for _r in RIVALS:
    _r["fd"] = PchipInterpolator(*zip(*_r["d"]))
    _r["fu"] = PchipInterpolator(*zip(*_r["u"]))
    _r["t0"], _r["t1"] = _r["d"][0][0], _r["d"][-1][0]

# 本车横向位置（镜头跟着本车）：发车格在左列 → 回到中线；每次超车先拐到对手旁边的车道
_UC = [(0, -2.0), (10.4, -2.0), (14, 0.0),
       (20.6, 0), (23.2, -3.3), (27.4, -3.3), (29.6, 0),            # 25.6 左侧超车
       (46.2, 0), (48.8, 4.8), (53.0, 4.8), (55.2, 0),              # 51.2 右侧
       (59.0, 0), (61.6, -4.8), (65.8, -4.8), (68.0, 0),            # 64 左侧
       (97.4, 0), (100.0, -2.8), (103.2, -2.8), (105.6, 2.3), (109.0, 2.3), (111.2, 0)]  # 102.4 / 107.2
_FUC = PchipInterpolator(*zip(*_UC))


def ucam_at(t):
    return float(_FUC(np.clip(t, _UC[0][0], _UC[-1][0])))


def rival_state(r, t):
    """(d, u)：t 超出关键帧范围时取端点值"""
    tt = float(np.clip(t, r["t0"], r["t1"]))
    return float(r["fd"](tt)), float(r["fu"](np.clip(t, r["u"][0][0], r["u"][-1][0])))


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


def env_geom(curv, heading, ucam=0.0):
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
        b, c = -dx, side * WALL - ucam
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
    u = dx[gi] * z - curv * z * z + ucam
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
    if phase < S_CAR0 + 70:                                     # 发车格白线
        for dd, uc in GRID:
            sl = S_CAR0 + dd + 6.3
            m1 = (np.abs(s - sl) < 0.09) & (np.abs(u - uc) < 1.35)
            m2 = (np.abs(np.abs(u - uc) - 1.35) < 0.07) & (s > sl - 1.3) & (s < sl)
            col = np.where((m1 | m2)[:, None], C(0.75, 0.75, 0.72), col)
    if S_FIN - phase < 450:                                     # 终点线
        fin = (s > S_FIN) & (s < S_FIN + 1.6) & (au < ROAD)
        chk = ((np.floor(u / 0.8) + np.floor((s - S_FIN) / 0.8)) % 2 == 0)
        col = np.where(fin[:, None], np.where(chk[:, None], C(0.8, 0.8, 0.8), C(0.03, 0.03, 0.03)), col)
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
    G = env_geom(curv, ph0 * curv * 40, ucam_at(t))
    acc = 0
    for k in np.linspace(-0.5, 0.5, n):
        acc = acc + env_shade(G, ph0 + v * k / FPS * 0.9)
    img = (acc / n).reshape(RH, RW, 3)
    small = np.stack([zoom(img[..., c], 1 / SS, order=1) for c in range(3)], axis=2)
    return small[:OH, :OW]


# ------------------------------------------------------------------ 赛车（矢量绘制，后视）
ORANGE = (236, 104, 34)   # 赛车涂装橙
LOGO = (217, 119, 87)


def car_patch(zc, xc, base, flash=0.0, sway=0.0, bounce=0.0):
    """三维透视绘制一辆 F1 赛车（从后上方看）：zc = 车尾离相机的距离，xc = 横向偏移（米），base = 车身颜色。
    只在车周围的小画布上绘制，返回 (预乘颜色, 透明度, 阴影, 尾灯, x0, y0)；太近/不在画面内返回 None"""
    S = 2   # 超采样
    f = OW / 2 / np.tan(np.radians(FOV / 2))
    cx, cy = OW / 2 + sway, OH * HOR + bounce
    if zc < 1.8:
        return None

    def pr(x, y, dz):
        z = zc + dz
        return cx + (xc + x) * f / z, cy + (HC - y) * f / z

    pts = [pr(x, y, dz) for x in (-1.3, 1.3) for y in (-0.05, 1.25) for dz in (-0.5, 5.4)]
    x0 = int(max(0, min(q[0] for q in pts) - 40)); bx1 = int(min(OW, max(q[0] for q in pts) + 40))
    y0 = int(max(0, min(q[1] for q in pts) - 40)); by1 = int(min(OH, max(q[1] for q in pts) + 40))
    if bx1 - x0 < 4 or by1 - y0 < 4:
        return None
    W, H = (bx1 - x0) * S, (by1 - y0) * S
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    kpx = f / zc * S * 0.8          # 每米约多少像素（用于线宽）

    def lw(m, lo=1):
        return max(lo, int(m * kpx))

    def P(x, y, dz=0.0):
        px, py = pr(x, y, dz)
        return ((px - x0) * S, (py - y0) * S)

    def poly(pts, fill):
        d.polygon([P(*p) for p in pts], fill=fill)

    def box(x1, x2, y1, y2, z1, z2, face, top, side=None, r=0.0):
        """长方体：画顶面、朝相机的面（z1 面）和朝内的侧面"""
        poly([(x1, y2, z1), (x2, y2, z1), (x2, y2, z2), (x1, y2, z2)], top)
        if side is not None:
            xs = x1 if x1 > 0 else x2      # 朝向车中心的那一侧
            poly([(xs, y1, z1), (xs, y2, z1), (xs, y2, z2), (xs, y1, z2)], side)
        a, b = P(x1, y2, z1), P(x2, y1, z1)
        d.rounded_rectangle([a, b], radius=r * f / zc * S, fill=face)

    OR = tuple(base) + (255,)
    OR_T = tuple(min(255, int(c * 1.18)) for c in base) + (255,)   # 顶面更亮
    OR_D = tuple(int(c * 0.62) for c in base) + (255,)              # 背光面
    CB, CB_T = (24, 24, 28, 255), (48, 48, 54, 255)
    TY, TY_T, TY_S = (30, 30, 33, 255), (78, 78, 84, 255), (44, 44, 48, 255)

    # 车底阴影
    sh = Image.new("L", (W, H), 0)
    ImageDraw.Draw(sh).polygon([P(-1.2, 0, -0.5), P(1.2, 0, -0.5), P(1.1, 0, 5.2), P(-1.1, 0, 5.2)], fill=220)
    sh = sh.filter(ImageFilter.GaussianBlur(max(1, 9 * S * 6.4 / zc)))

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
        d.line([P(sg * 0.12, 0.38, 3.9), P(sg * 0.66, 0.36, 4.2)], fill=CB, width=lw(0.03 * 0.7))
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
               width=lw(0.04 * 0.8))
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
        d.line([P(xo, 0.1, 0), P(xo, 0.62, 0)], fill=(95, 95, 100, 255), width=lw(0.015, 1))
        xs = x2 - 0.035 if sg < 0 else x1 + 0.035
        d.line([P(xs, 0.18, 0), P(xs, 0.54, 0)], fill=(215, 180, 40, 255), width=lw(0.02, 1))
    # 后悬挂
    for sg in (-1, 1):
        for ya, yb in ((0.4, 0.44), (0.27, 0.22)):
            d.line([P(sg * 0.15, ya, -0.05), P(sg * 0.64, yb, 0.2)], fill=(8, 8, 10, 255), width=lw(0.03))
    # 扩散器 + 变速箱
    poly([(-0.55, 0.04, -0.15), (0.55, 0.04, -0.15), (0.42, 0.3, -0.05), (-0.42, 0.3, -0.05)], (10, 10, 12, 255))
    for k in np.linspace(-0.38, 0.38, 6):
        poly([(k - 0.012, 0.05, -0.15), (k + 0.012, 0.05, -0.15), (k * 0.8 + 0.01, 0.28, -0.05), (k * 0.8 - 0.01, 0.28, -0.05)],
             (44, 44, 50, 255))
    a, b = P(-0.15, 0.42, -0.08), P(0.15, 0.25, -0.08)
    d.rounded_rectangle([a, b], radius=max(1, 6 * 6.4 / zc), fill=CB)
    # 梁翼 + 尾翼支架
    poly([(-0.46, 0.5, -0.12), (0.46, 0.5, -0.12), (0.46, 0.44, -0.12), (-0.46, 0.44, -0.12)], CB)
    poly([(-0.03, 0.42, -0.12), (0.03, 0.42, -0.12), (0.03, 0.82, -0.25), (-0.03, 0.82, -0.25)], CB)
    # 尾翼：主翼面（顶面可见）+ 襟翼 + 端板
    poly([(-0.5, 0.82, -0.3), (0.5, 0.82, -0.3), (0.5, 0.84, 0.15), (-0.5, 0.84, 0.15)], CB_T)
    poly([(-0.5, 0.78, -0.3), (0.5, 0.78, -0.3), (0.5, 0.82, -0.3), (-0.5, 0.82, -0.3)], CB)
    poly([(-0.5, 0.9, -0.35), (0.5, 0.9, -0.35), (0.5, 0.95, -0.15), (-0.5, 0.95, -0.15)], CB_T)
    poly([(-0.5, 0.86, -0.35), (0.5, 0.86, -0.35), (0.5, 0.9, -0.35), (-0.5, 0.9, -0.35)], CB)
    d.line([P(-0.5, 0.95, -0.15), P(0.5, 0.95, -0.15)], fill=OR_T, width=lw(0.012, 1))
    for sg in (-1, 1):
        x = sg * 0.52
        poly([(x, 0.55, -0.4), (x, 0.98, -0.4), (x, 0.98, 0.3), (x, 0.6, 0.3)], CB_T)
        a, b = P(min(x, x + sg * 0.03), 0.98, -0.4), P(max(x, x + sg * 0.03), 0.55, -0.4)
        d.rectangle([a, b], fill=OR)

    arr = np.asarray(layer).astype(np.float32) / 255
    # 顶光：越往上越亮
    yy = np.arange(H)[:, None] / S + y0
    yr = (yy - (cy + (HC - 1.1) * f / (zc + 1.0))) / ((1.1) * f / (zc + 1.0))
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

    arr[..., :3] *= arr[..., 3:4]                                   # 预乘透明度，边缘不发黑

    def down(x):
        return x.reshape(by1 - y0, S, bx1 - x0, S, *x.shape[2:]).mean(axis=(1, 3))

    rgba = down(arr)
    return (rgba[..., :3], rgba[..., 3:4], down(np.asarray(sh).astype(np.float32) / 255),
            down(gl) * lamp_on, x0, y0)


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


# ------------------------------------------------------------------ 场景物体
def world_layer(t, phase, curv, ucam):
    """起跑灯架、终点门架、挥白旗的工作人员（按赛道坐标投影到画面）。返回 (预乘颜色, 透明度, 发光) 或 None"""
    show_start = t < 14 and S_GANTRY - phase > 1.8
    show_fin = S_FIN - phase < 230
    if not (show_start or show_fin):
        return None
    S = 2
    f = OW / 2 / np.tan(np.radians(FOV / 2))
    body = Image.new("RGBA", (OW * S, OH * S), (0, 0, 0, 0))
    db = ImageDraw.Draw(body)
    glow = Image.new("RGB", (OW * S, OH * S), (0, 0, 0))
    dg = ImageDraw.Draw(glow)

    def pj(u, y, sw):
        z = sw - phase
        if z < 1.8:
            return None
        x = u - ucam + curv * z * z
        return ((OW / 2 + x * f / z) * S, (OH * HOR + (HC - y) * f / z) * S, f / z * S)

    def quad(pts, fill, dr=None):
        q = [pj(*pt) for pt in pts]
        if any(v is None for v in q):
            return
        (dr or db).polygon([(v[0], v[1]) for v in q], fill=fill)

    def gantry(sw, start):
        for sg in (-1, 1):
            quad([(sg * 10 - 0.3, 0, sw), (sg * 10 + 0.3, 0, sw), (sg * 10 + 0.3, 7.8, sw), (sg * 10 - 0.3, 7.8, sw)],
                 (30, 30, 34, 255))
        quad([(-10.3, 6.0, sw), (10.3, 6.0, sw), (10.3, 7.8, sw), (-10.3, 7.8, sw)], (22, 22, 26, 255))
        if start:
            for i, u in enumerate((-4, -2, 0, 2, 4)):
                quad([(u - 0.55, 4.5, sw), (u + 0.55, 4.5, sw), (u + 0.55, 6.0, sw), (u - 0.55, 6.0, sw)], (10, 10, 12, 255))
                lit = T_LIGHTS[i] <= t < T_OUT
                for y in (5.6, 4.95):
                    c = pj(u, y, sw)
                    if c is None:
                        continue
                    rad = 0.25 * c[2]
                    box = [c[0] - rad, c[1] - rad, c[0] + rad, c[1] + rad]
                    db.ellipse(box, fill=(255, 70, 50, 255) if lit else (70, 10, 10, 255))
                    if lit:
                        dg.ellipse(box, fill=(255, 60, 40))
        else:                                   # 黑白格横幅
            for i in range(28):
                for j in range(2):
                    if (i + j) % 2 == 0:
                        x0_ = -10.3 + i * 0.736
                        quad([(x0_, 6.0 + j * 0.9, sw), (x0_ + 0.736, 6.0 + j * 0.9, sw),
                              (x0_ + 0.736, 6.9 + j * 0.9, sw), (x0_, 6.9 + j * 0.9, sw)], (235, 235, 235, 255))

    def marshal(u, sw, side, seed, scale):
        p0, p1, p2 = pj(u, 0, sw), pj(u, 0.9, sw), pj(u, 1.55, sw)
        if p0 is None:
            return
        k = p0[2]
        hw = 0.2 * k
        db.rectangle([p0[0] - hw, p1[1], p0[0] + hw, p0[1]], fill=(35, 35, 40, 255))               # 腿
        db.rectangle([p0[0] - hw * 1.1, p2[1], p0[0] + hw * 1.1, p1[1]], fill=(235, 110, 20, 255))  # 橙色马甲
        db.rectangle([p0[0] - hw * 1.1, (p1[1] + p2[1]) / 2 - 0.03 * k, p0[0] + hw * 1.1, (p1[1] + p2[1]) / 2 + 0.03 * k],
                     (230, 230, 230, 255))
        hd = pj(u, 1.7, sw)
        db.ellipse([hd[0] - 0.12 * k, hd[1] - 0.12 * k, hd[0] + 0.12 * k, hd[1] + 0.12 * k], fill=(225, 225, 230, 255))
        # 旗杆：绕手摆动
        th = 0.95 * np.sin(t * 7.5 + seed)
        L = 1.5 * scale
        hu, hy = u - side * 0.25, 1.5
        tu, ty = hu - side * L * np.sin(th) * 0.9, hy + L * np.cos(th)
        h0, h1 = pj(hu, hy, sw), pj(tu, ty, sw)
        db.line([(h0[0], h0[1]), (h1[0], h1[1])], fill=(60, 60, 64, 255), width=max(1, int(0.04 * k)))
        # 白旗：随时间起伏的布面，沿杆顶向外飘
        top, bot = [], []
        for i in range(6):
            wv = np.sin(t * 11 + seed + i * 0.9) * 0.1 * scale * (i / 5)
            uu = tu - side * (0.95 * scale) * i / 5 * (-1)
            top.append(pj(uu, ty + wv, sw))
            bot.append(pj(uu, ty - 0.7 * scale + wv * 1.3, sw))
        if all(v is not None for v in top + bot):
            poly_ = [(v[0], v[1]) for v in top] + [(v[0], v[1]) for v in bot[::-1]]
            db.polygon(poly_, fill=(240, 240, 244, 255))
            dg.polygon(poly_, fill=(110, 110, 118))

    if show_start:
        gantry(S_GANTRY, True)
    if show_fin:
        gantry(S_FIN + 0.4, False)
        ms = [(S_FIN - 6 - 13 * j, -1 if j % 2 == 0 else 1, j) for j in range(12)] + [(S_FIN + 9, -1, 12), (S_FIN + 9, 1, 13)]
        for sw, side, j in sorted(ms, key=lambda m_: -m_[0]):
            marshal(side * (WALL - 1.4), sw, side, j * 1.7, 1.5 if j < 4 else 1.15)

    arr = np.asarray(body).astype(np.float32) / 255
    arr[..., :3] *= arr[..., 3:4]
    dn = lambda x: x.reshape(OH, S, OW, S, *x.shape[2:]).mean(axis=(1, 3))
    return dn(arr[..., :3]), dn(arr[..., 3:4]), dn(np.asarray(glow).astype(np.float32) / 255) * 2.6


# ------------------------------------------------------------------ 特写：方向盘转速灯 → REDLINE
DASH_T0, DASH_T1 = 99.2, 102.4        # 对上高潮前的上升音效 + 军鼓滚奏，102.4 的重击硬切回赛道
_FONTS = {}


def _font(size):
    if size not in _FONTS:
        _FONTS[size] = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    return _FONTS[size]


def _gauge(d, g, S, cx, cy, R, vmin, vmax, val, major, minors, red=None, title="", fmt="{:.0f}", heat=0.0,
           hero=False, needle=(255, 112, 28)):
    """通用机械仪表：金属表圈 + 黑色表盘 + 刻度/数字 + 红区 + 锥形指针。cx, cy, R 为放大 S 倍后的像素。"""
    span = vmax - vmin
    th = lambda v_: np.radians(225 - 270 * (v_ - vmin) / span)
    pt = lambda v_, rad: (cx + rad * np.cos(th(v_)), cy - rad * np.sin(th(v_)))
    fnt = lambda size: _font(max(8, int(size)))
    T = int(0.14 * R)
    for k in range(T):                                              # 金属表圈（同心环模拟拉丝）
        rad = R + T - k
        c = int(46 + 64 * abs(np.sin(k * 0.33 + 0.6)))
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(c, c, int(c * 1.04)))
    d.arc([cx - R - T * 0.55, cy - R - T * 0.55, cx + R + T * 0.55, cy + R + T * 0.55], 195, 255,
          fill=(235, 235, 240), width=max(2, int(T * 0.2)))
    for k in range(30):                                             # 表盘
        rad = R * (1 - k / 30)
        c = int(5 + 15 * (k / 30) ** 2)
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(c, c, c + 2))
    if red:
        vs = np.linspace(red[0], red[1], 30)
        band = [pt(x, R * 0.97) for x in vs] + [pt(x, R * 0.88) for x in vs[::-1]]
        d.polygon(band, fill=(190, 20, 20))
        g.polygon(band, fill=(int(255 * (0.3 + 0.7 * heat)), int(30 * heat), int(25 * heat)))
    n = int(round(span / major * minors))
    for i in range(n + 1):
        v_ = vmin + i * major / minors
        is_major = i % minors == 0
        in_red = red is not None and red[0] <= v_ <= red[1] + 1e-6
        ln = 0.13 if is_major else 0.06
        wd = max(1, int(R * (0.026 if is_major else 0.013)))
        col = (255, 70, 55) if in_red else (228, 228, 232)
        d.line([pt(v_, R * 0.88), pt(v_, R * (0.88 - ln))], fill=col, width=wd)
        if in_red and is_major:
            g.line([pt(v_, R * 0.88), pt(v_, R * (0.88 - ln))], fill=(255, 40, 30), width=wd)
        if is_major:
            x_, y_ = pt(v_, R * 0.66)
            d.text((x_, y_), fmt.format(v_), font=fnt(R * (0.125 if hero else 0.15)), fill=(255, 80, 65) if in_red else (232, 232, 236), anchor="mm")
    if title:
        d.text((cx, cy - R * 0.36), title, font=fnt(R * 0.085), fill=(150, 150, 158), anchor="mm")
    a_ = th(val)
    ca, sa = np.cos(a_), np.sin(a_)
    w = R * 0.04
    loc = [(R * 0.90, 0), (R * 0.10, w), (-R * 0.20, w * 0.8), (-R * 0.20, -w * 0.8), (R * 0.10, -w)]
    nd = lambda off=(0, 0): [(cx + off[0] + x * ca + y * sa, cy + off[1] - x * sa + y * ca) for x, y in loc]
    d.polygon(nd((R * 0.035, R * 0.045)), fill=(2, 2, 3))
    d.polygon(nd(), fill=needle)
    d.line([(cx + R * 0.10 * ca, cy - R * 0.10 * sa), (cx + R * 0.88 * ca, cy - R * 0.88 * sa)],
           fill=(255, 205, 140), width=max(1, int(R * 0.008)))
    if heat > 0:
        g.polygon(nd(), fill=(int(200 * heat), int(55 * heat), int(10 * heat)))
    for rad, col in ((R * 0.13, (30, 30, 34)), (R * 0.115, (150, 150, 156)), (R * 0.085, (70, 70, 76)), (R * 0.05, (200, 200, 205))):
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=col)


def dash_frame(t):
    """仪表台特写：大转速表（主角）+ 时速表 + 水温 + 机油 + 涡轮增压 + 挡位 + 警示灯。
    转速表指针冲进红区，其余各表也一起进入危险区，整个画面泛红闪烁；3 秒内镜头缓慢推向转速表。"""
    S = 2
    p = float(np.clip((t - DASH_T0) / (DASH_T1 - DASH_T0), 0, 1))
    W, H = OW * S, OH * S
    base = Image.new("RGB", (W, H), (7, 7, 10))
    d = ImageDraw.Draw(base)
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    g = ImageDraw.Draw(glow)
    rr = np.random.default_rng(int(round(t * FPS)) + 7)
    hx, hy = 520.0, 365.0
    z = 1.0 + 0.10 * p                                              # 缓慢推近
    Z = lambda x, y, r: ((hx + (x - hx) * z) * S, (hy + (y - hy) * z) * S, r * z * S)

    # 仪表台底板：碳纤维斜纹 + 橙色饰线
    for k_ in range(-700, 1400, 16):
        d.line([(k_ * S, 0), ((k_ - 360) * S, H)], fill=(13, 13, 17), width=S)
    d.rounded_rectangle([20 * S, 96 * S, 1260 * S, 624 * S], radius=50 * S, outline=(30, 30, 36), width=3 * S)
    d.line([(60 * S, 616 * S), (1220 * S, 616 * S)], fill=(236, 104, 34), width=3 * S)

    # 读数
    v = 5.0 + 10.2 * (1 - (1 - p) ** 2.2)
    v += 0.16 * np.sin(t * 95) * smooth(v, 13.8, 14.6) + rr.normal(0, 0.05) * p
    heat = float(np.clip((v - 13.5) / 1.2, 0, 1))
    spd = 250 + 100 * p ** 1.2 + rr.normal(0, 1.5) * heat
    tmp = 88 + 36 * smooth(p, 0.1, 1.0) + rr.normal(0, 0.4) * heat
    oil = 5.0 - 1.4 * p + 0.15 * np.sin(t * 60) * p
    bst = 0.8 + 1.9 * smooth(p, 0.0, 0.9) + 0.05 * np.sin(t * 70) * heat

    # 副表（先画，主表压在上面）：时速表（中）、水温 / 机油（小）、涡轮增压（小）
    cx_, cy_, r_ = Z(965, 232, 128)
    _gauge(d, g, S, cx_, cy_, r_, 0, 400, spd, 50, 5, red=(360, 400), title="KM/H", heat=heat)
    cx_, cy_, r_ = Z(893, 478, 82)
    _gauge(d, g, S, cx_, cy_, r_, 60, 130, tmp, 10, 2, red=(110, 130), title="WATER", heat=smooth(tmp, 108, 118))
    cx_, cy_, r_ = Z(1087, 478, 82)
    _gauge(d, g, S, cx_, cy_, r_, 0, 8, oil, 1, 2, red=(0, 1.5), title="OIL", heat=0.0)
    cx_, cy_, r_ = Z(158, 262, 92)
    _gauge(d, g, S, cx_, cy_, r_, 0, 3, bst, 0.5, 5, red=(2.5, 3.0), title="BOOST", fmt="{:.1f}", heat=smooth(bst, 2.3, 2.8))

    # 左下：挡位小屏 + 三个警示灯
    gx, gy, gw = Z(158, 440, 60)
    d.rounded_rectangle([gx - gw, gy - gw * 0.85, gx + gw, gy + gw * 0.85], radius=gw * 0.2, fill=(3, 3, 5),
                        outline=(70, 70, 78), width=2 * S)
    d.text((gx, gy - gw * 0.1), str(7 if p < 0.55 else 8), font=_font(int(gw * 1.15)), fill=(236, 120, 40), anchor="mm")
    d.text((gx, gy + gw * 0.62), "GEAR", font=_font(int(gw * 0.2)), fill=(130, 130, 140), anchor="mm")
    flash = int(t * 8) % 2 == 0
    for i_, (name, lit, col) in enumerate((("ENG", heat > 0.2 and flash, (255, 160, 20)),
                                           ("TEMP", tmp > 110 and flash, (255, 50, 40)),
                                           ("OIL", False, (255, 50, 40)))):
        lx, ly, lw = Z(98 + i_ * 60, 550, 24)
        d.rounded_rectangle([lx - lw, ly - lw * 0.6, lx + lw, ly + lw * 0.6], radius=lw * 0.2,
                            fill=col if lit else tuple(int(c * 0.14) for c in col), outline=(60, 60, 66), width=S)
        d.text((lx, ly), name, font=_font(int(lw * 0.5)), fill=(20, 8, 8) if lit else (90, 90, 96), anchor="mm")
        if lit:
            g.rounded_rectangle([lx - lw, ly - lw * 0.6, lx + lw, ly + lw * 0.6], radius=lw * 0.2, fill=col)

    # 主角：转速表（最大），REDLINE 粗刻线 + 警示字
    cx_, cy_, r_ = Z(hx, hy, 235)
    _gauge(d, g, S, cx_, cy_, r_, 0, 16, v, 1, 4, red=(13.5, 16.0), title="RPM x1000", heat=heat, hero=True)
    ang = np.radians(225 - 270 * 14 / 16)
    p1 = (cx_ + r_ * 1.0 * np.cos(ang), cy_ - r_ * 1.0 * np.sin(ang))
    p2 = (cx_ + r_ * 0.62 * np.cos(ang), cy_ - r_ * 0.62 * np.sin(ang))
    d.line([p1, p2], fill=(255, 40, 30), width=int(9 * z * S))
    g.line([p1, p2], fill=(255, 40, 30), width=int(9 * z * S))
    on = v > 14 and int(t * 8) % 2 == 0
    for dr_ in ((d, g) if on else (d,)):
        dr_.text((cx_, cy_ + r_ * 0.58), "REDLINE", font=_font(int(r_ * 0.15)), fill=(255, 45, 35) if on else (60, 14, 14), anchor="mm")
    # 主表玻璃反光
    refl = Image.new("L", (W, H), 0)
    ImageDraw.Draw(refl).polygon([(cx_ - r_ * 0.95, cy_ - r_ * 0.15), (cx_ - r_ * 0.35, cy_ - r_ * 0.92),
                                  (cx_ + r_ * 0.30, cy_ - r_ * 0.92), (cx_ - r_ * 0.62, cy_ + r_ * 0.12)], fill=13)
    msk = Image.new("L", (W, H), 0)
    ImageDraw.Draw(msk).ellipse([cx_ - r_, cy_ - r_, cx_ + r_, cy_ + r_], fill=255)
    rf = np.asarray(refl).astype(np.float32) / 255 * np.asarray(msk).astype(np.float32) / 255

    arr = np.asarray(base).astype(np.float32) / 255 + rf[..., None]
    gl = np.asarray(glow).astype(np.float32) / 255 * (1.3 + 1.4 * heat)
    dn = lambda x: x.reshape(OH, S, OW, S, 3).mean(axis=(1, 3))
    img = dn(arr) + dn(gl)
    yy, xx = np.mgrid[0:OH, 0:OW]
    edge = np.clip(np.hypot((xx - OW / 2) / (OW / 2), (yy - OH / 2) / (OH / 2)), 0, 1.3)
    pulse = 0.6 + 0.4 * np.sin(t * 2 * np.pi * 4)
    img += (heat * pulse * (0.12 + 0.45 * edge ** 2))[..., None] * np.array([1.0, 0.04, 0.03], np.float32)
    sh = 1.0 + 7 * heat
    return np.roll(img, (int(rr.normal(0, sh)), int(rr.normal(0, sh))), axis=(0, 1)).astype(np.float32)


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
    """片头标题在起跑灯亮起前淡出；结尾冲线后出现"""
    return float(smooth(t, 0.8, 2.2) * (1 - smooth(t, 4.0, 5.2)) + smooth(t, 123.0, 124.6))


# ------------------------------------------------------------------ 合成一帧
_VIG = None


def _exterior(t, pulse):
    """赛道外景：环境 + 场景物体 + 所有赛车 + 火花"""
    r = np.random.default_rng(int(t * 1000))
    v = speed_at(t)
    phase, curv, ucam = phase_at(t), curve_at(t), ucam_at(t)
    shake = (v / 90) ** 2 * 1.2 + (0.3 if v < 8 else 0.0)         # 怠速时车身也在微微抖动
    sway = 6 * np.sin(t * 0.9) * min(1, v / 30) + r.normal(0, shake)
    bounce = r.normal(0, shake * 0.7)
    img = env_blur(t)

    # 场景物体（灯架、门架、工作人员）
    wl = world_layer(t, phase, curv, ucam)
    glow_add = 0
    if wl is not None:
        img = img * (1 - wl[1]) + wl[0]
        glow_add = wl[2]

    # 所有赛车：本车 + 对手；先画阴影，再从远到近画车身
    f = OW / 2 / np.tan(np.radians(FOV / 2))
    cars = [dict(zc=CAR_Z, xc=0.0, base=ORANGE, flash=pulse, sway=sway, bounce=bounce)]
    for k, rv in enumerate(RIVALS):
        d, u = rival_state(rv, t)
        zc = CAR_Z + d
        if zc < 2.2 or zc > 260:
            continue
        xc = u - ucam + curv * (zc + 2.5) ** 2
        rr = np.random.default_rng(int(t * 1000) + k)
        cars.append(dict(zc=zc, xc=xc, base=rv["col"], flash=0.35 + 0.4 * pulse, sway=0.0,
                         bounce=rr.normal(0, 0.4) * min(1, v / 40)))
    patches = []
    for c in sorted(cars, key=lambda c_: -c_["zc"]):
        pt = car_patch(c["zc"], c["xc"], c["base"], c["flash"], c["sway"], c["bounce"])
        if pt is not None:
            patches.append(pt)
    tail = np.zeros((OH, OW), np.float32)
    for rgb, a_, sh, tl, x0, y0 in patches:
        h, w = sh.shape
        img[y0:y0 + h, x0:x0 + w] *= (1 - 0.75 * sh)[..., None]
    for rgb, a_, sh, tl, x0, y0 in patches:
        h, w = sh.shape
        sl = (slice(y0, y0 + h), slice(x0, x0 + w))
        img[sl] = img[sl] * (1 - a_) + rgb
        tail[sl] = np.maximum(tail[sl], tl)
    refl = gaussian_filter(tail, 18) * 40                           # 尾灯在路面上的红色反光
    img += refl[..., None] * np.array([0.9, 0.05, 0.03])
    img += sparks(t, (OW / 2 + sway, OH * HOR + bounce, f)) * min(1, v / 70)
    img = img + glow_add
    return img


def frame(t, layers, total=128.0, pulse=0.0):
    global _VIG
    title, logo = layers
    r = np.random.default_rng(int(t * 1000))
    v = speed_at(t)
    img = dash_frame(t) if DASH_T0 <= t < DASH_T1 else _exterior(t, pulse)
    if DASH_T1 <= t < DASH_T1 + 0.3:                               # 切回赛道的那一拍：白闪
        img = img + 0.6 * (1 - (t - DASH_T1) / 0.3)
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
    """渲染一小块（默认 4 秒）；已存在的块直接跳过，所以中断后可以接着渲"""
    k, i0, i1, tag = args
    seg = os.path.join(HERE, f".seg_{tag}_{k:03d}.mp4")
    if os.path.exists(seg):
        return seg
    tmp = seg + ".part.mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                          "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    for i in range(i0, i1):
        p.stdin.write(render_frame(i).tobytes())
    p.stdin.close()
    p.wait()
    os.replace(tmp, seg)
    print(f"  chunk {k} done ({i1 - i0} frames)", flush=True)
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
    workers = os.cpu_count() or 4
    tag = "demo" if "--demo" in sys.argv else "full"
    CH = 120
    jobs = [(k, i0, min(nf, i0 + CH), tag) for k, i0 in enumerate(range(0, nf, CH))]
    done = sum(os.path.exists(os.path.join(HERE, f".seg_{tag}_{k:03d}.mp4")) for k, *_ in jobs)
    print(f"{len(jobs)} chunks, {done} already done, {workers} workers", flush=True)
    with mp.get_context("fork").Pool(workers) as pool:
        segs = pool.map(worker, jobs, chunksize=1)
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
