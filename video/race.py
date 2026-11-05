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
DUSK = "--dusk" in sys.argv             # 黄昏版：落日在赛道尽头、逆光剪影、暖色雾和路面反光
TEASER = "--teaser" in sys.argv         # 16 秒抖音预告片：按拍点剪，画面取自完整版的几个时刻，音乐取 89.6–105.6 秒连续一段
DOUYIN = "--douyin" in sys.argv or TEASER          # 抖音版：竖版 + 从起跑灯亮起处（场景 5.6 秒）开始，总长仍 128 秒（尾声延长）
START_OFF = 5.6 if DOUYIN and not TEASER else 0.0
# 预告片分镜（场景时间, 时长）：起跑灯 → 灯灭起步 → REDLINE 座舱特写 → 高潮超车；视频 9.6 秒处硬切，对上音乐的上升音效
TEASER_SEGS = [(100.8, 3.2),      # 1–2 座舱 REDLINE（钩子）→ 1.6 秒处高潮重击 + 超车 #1
               (2000.15, 0.8),    # 3 赛道边呼啸（橙车 + 黄车），插在 104.0–104.8 秒之间
               (104.8, 3.2),      # 4–5 追拍冲刺 → 超车 #2（连续）
               (101.6, 0.8),      # 6 座舱回闪
               (10.5, 3.2),       # 7 加速冲刺闪回：起步后车速已约 30 米/秒，一路加速、穿过灯架（不放静止的发车格，否则前后高速镜头之间会「突然慢下来」）
               (118.4, 3.2),      # 8–9 终点直道：红车并排、挥白旗、门架逼近，121.6 冲线那一刻正好是 14.4 秒的重击
               (1000.0, 1.6)]     # 10 赛道边呼啸（橙车 + 红车）→ WIN
TEASER_SEGS_DUSK = [(108.8, 1.6),   # 1 冲向落日的逆光剪影追拍（钩子）+ F1 RACING
                    (102.4, 1.6),   # 2 1.6 秒高潮重击：超车 #1（黄车）
                    (2000.15, 0.8), # 3 赛道边呼啸 #1（落日在防护网后面）
                    (104.8, 3.2),   # 4–5 追拍逼近 → 超车 #2（绿车，6.4 秒）
                    (100.8, 1.6),   # 6 座舱 REDLINE（halo 外是晚霞）
                    (10.6, 2.4),    # 7 加速穿过灯架，灯架是落日前的剪影
                    (118.4, 3.2),   # 8 终点直道：红车并排、挥白旗、门架逼近，14.4 秒重击
                    (1000.0, 1.6)]  # 9 赛道边呼啸 #2 → WIN
if DUSK:
    TEASER_SEGS = TEASER_SEGS_DUSK
TEASER_AUDIO0 = 100.8                 # 音乐窗口起点：开头是上升音效，1.6 秒处高潮重击，14.4 秒处尾声的大镲 + 重击
HELMET = "--helmet" in sys.argv           # 车手头盔视角（第一人称，从面罩后面往外看）4 秒片段
COVER = "--cover" in sys.argv            # 封面静帧：不画名次、片名、Claude 标志
SIZE = tuple(int(v_) for v_ in sys.argv[sys.argv.index("--size") + 1].split("x")) if "--size" in sys.argv else None   # 例如 --size 1440x1080
PORTRAIT = "--portrait" in sys.argv or DOUYIN or HELMET or bool(SIZE and SIZE[1] > SIZE[0])      # 抖音竖版：720x1280（最后用 ffmpeg 升到 1080x1920）
OW, OH = SIZE or ((720, 1280) if PORTRAIT else (1280, 720))   # 输出分辨率
SS = 1.0                      # 环境超采样倍数
RW, RH = int(OW * SS), int(OH * SS)
FPS = 30
AUDIO = os.path.join(HERE, "..", "music", "redline_douyin.mp3" if DOUYIN else "redline.mp3")
LETTER = 0 if (PORTRAIT or SIZE) else 88   # 横版：2.39:1 宽银幕黑边（像素）；竖版不加

# 相机 / 场景（单位：米）
HC = 0.95 if HELMET else (2.3 if PORTRAIT else 1.75)   # 相机高度（竖版抬高，让车落到画面中下部）
HOR = 0.42 if HELMET else (0.45 if PORTRAIT else 0.40)    # 地平线位置（画面高度比例）
FOV = 72 if HELMET else (60 if PORTRAIT else 66)      # 水平视场
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
            (119, 84), (121.6, 105), (122.8, 70), (128, 38), (133.6, 12)]
    if TEASER:                        # 预告片只有 16 秒，终点直道不必为看清白旗放慢：保持高速，和前面的高潮接得上
        keys = sorted([k for k in keys if not 115.0 <= k[0] <= 121.7] + [(115.2, 140), (117, 132), (119, 128), (121.6, 138)])
    xs, ys = zip(*keys)
    return float(np.interp(t, xs, ys))


_TT = np.linspace(0, 140, 14001)
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
    if DUSK:                               # 黄昏：地平线橙 → 粉紫 → 深蓝紫
        top, mid, hor = np.array([0.07, 0.07, 0.19]), np.array([0.45, 0.18, 0.26]), np.array([1.0, 0.46, 0.17])
        h = np.clip(dy * 1.1, 0, 1)                                 # 晚霞的渐变拉得更高
    c = np.where(h[..., None] < 0.18, hor + (mid - hor) * (h[..., None] / 0.18),
                 mid + (top - mid) * np.clip((h[..., None] - 0.18) / 0.6, 0, 1) ** 0.7)
    c = c * (0.85 + 0.15 * np.exp(-((dx + 0.15) / 0.5) ** 2))[..., None]
    # 星星（固定在天空，随赛道方向缓慢平移）
    sx = np.floor((np.arctan(dx) * 180 + heading) * 6).astype(np.int64)
    sy = np.floor(dy * 1100).astype(np.int64)
    hsh = (np.sin(sx * 12.9898 + sy * 78.233) * 43758.5453) % 1
    star = (hsh > 0.9965) & (dy > 0.06)
    if not DUSK:
        c = c + (star * ((hsh - 0.9965) / 0.0035) ** 2)[..., None] * np.array([0.8, 0.85, 1.0]) * 1.2
    else:                                  # 落日：固定在视野正前方偏右一点，正好落在赛道尽头
        sa = np.arctan(dx) * 180
        rs = np.hypot((sa - 8) / 180, dy - 0.05)
        c = c + (np.exp(-rs / 0.05) * 0.9)[..., None] * np.array([1.0, 0.5, 0.2]) + (np.exp(-rs / 0.25) * 0.6)[..., None] * np.array([0.5, 0.22, 0.1])
        c = c + (np.exp(-((dy - 0.05) / 0.004) ** 2) * np.exp(-np.abs(sa - 8) / 180 / 0.5) * 0.7)[..., None] * np.array([1.0, 0.6, 0.45])  # 横向光晕
        c = np.where((rs < 0.03)[..., None], np.array([2.4, 1.6, 0.85]), c)
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
    if DUSK:                               # 天际线变成逆光剪影，只有零星亮灯
        bcol = np.array([0.06, 0.03, 0.045]) + win[..., None] * np.array([0.95, 0.7, 0.4]) * 0.18
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
    asphalt = (0.085 + 0.012 * tex * detail + 0.05 * spk * detail) * (1.5 if DUSK else 1.0)
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
    if DUSK:                                                    # 天光更亮、灯光作用小；路面掠射反光映出夕阳
        col = col * (0.6 + 1.1 * L)[:, None] + G["fres"][:, None] * C(1.0, 0.5, 0.22) * 0.8
        fog = G["fog"]
        col = col * fog + C(0.62, 0.3, 0.17) * (1 - fog)
    else:
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
    if DUSK:
        wall_c = wall_c * (0.35 + 0.35 * Lw)[:, None] * C(1.05, 0.85, 0.75)
        wall_c = wall_c * fogw + C(0.62, 0.3, 0.17) * (1 - fogw)
    else:
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
    if DUSK:
        arr[..., :3] *= 0.6
        arr[..., :3] += edge[..., None] * np.array([1.1, 0.6, 0.25])
    else:
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
    if TEASER:                                                     # 预告片里火花不断，别出现长时间的空档
        burst = max(burst, 0.6)
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


def dash_frame(t):
    """真实 F1 风格的座舱特写（车手视角）：halo 防滚架 + 碳纤维方向盘 + 顶部 15 颗转速灯 + 中央液晶屏（环形转速弧 + 挡位）
    + 两侧彩色旋钮 + 戴手套的双手。转速弧一段段点亮，冲进红区后屏幕、转速灯、整个画面一起报警。"""
    S = 2
    p = float(np.clip((t - DASH_T0) / (DASH_T1 - DASH_T0), 0, 1))
    W, H = OW * S, OH * S
    fg = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(fg)
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    g = ImageDraw.Draw(glow)
    rr = np.random.default_rng(int(round(t * FPS)) + 7)
    kd = 0.88 if PORTRAIT else 1.0                                  # 竖版：整体缩小到画面宽度内
    Y0 = 830.0 if PORTRAIT else 500.0
    hx, hy = (OW / 2, 900.0) if PORTRAIT else (640.0, 500.0)
    z = 1.0 + (0.05 if PORTRAIT else 0.12) * p                      # 缓慢推近（竖版幅度小，免得旋钮被切）
    M = lambda x, y: (OW / 2 + (x - 640) * kd, Y0 + (y - 500) * kd)
    T = lambda x, y: (lambda X, Y: ((hx + (X - hx) * z) * S, (hy + (Y - hy) * z) * S))(*M(x, y))
    K = z * S * kd

    def poly(pts, fill, dr=None):
        (dr or d).polygon([T(*q) for q in pts], fill=fill)

    def rrect(x0, y0, x1, y1, rad, fill, outline=None, width=0, dr=None):
        (dr or d).rounded_rectangle([*T(x0, y0), *T(x1, y1)], radius=rad * K, fill=fill, outline=outline, width=int(width * K))

    def ell(x, y, r, fill, outline=None, width=0, dr=None, ry=None):
        cx_, cy_ = T(x, y)
        (dr or d).ellipse([cx_ - r * K, cy_ - (ry or r) * K, cx_ + r * K, cy_ + (ry or r) * K], fill=fill, outline=outline, width=int(width * K))

    def line(pts, fill, width, dr=None):
        (dr or d).line([T(*q) for q in pts], fill=fill, width=max(1, int(width * K)))

    def text(x, y, s_, size, fill, dr=None, anchor="mm"):
        (dr or d).text(T(x, y), s_, font=_font(max(8, int(size * K))), fill=fill, anchor=anchor)

    # ---- 读数
    v = 5.0 + 10.2 * (1 - (1 - p) ** 2.2)
    v += 0.16 * np.sin(t * 95) * smooth(v, 13.8, 14.6) + rr.normal(0, 0.05) * p
    heat = float(np.clip((v - 13.5) / 1.2, 0, 1))
    alert = v > 14.0
    flash = int(t * 8) % 2 == 0
    spd = int(250 + 100 * p ** 1.2)
    orange = (236, 104, 34, 255)

    # ---- halo 防滚架（在方向盘前方，中间立柱 + 两侧弧臂）
    poly([(610, 170), (670, 170), (682, 340), (598, 340)], (66, 68, 74, 255))
    line([(640, 170), (640, 340)], (118, 120, 128, 255), 2.5)
    for sg in (-1, 1):
        line([(640 + sg * 8, 175), (640 + sg * 400, 118), (640 + sg * 600, 150)], (58, 60, 66, 255), 24)
        line([(640 + sg * 8, 168), (640 + sg * 400, 111), (640 + sg * 600, 143)], (112, 114, 122, 255), 3)

    # ---- 座舱仪表台（碳纤维 + 橙色涂装饰线），两侧低中间高
    xs = np.linspace(-40, 1320, 40)
    top = [(x, 318 + 0.00028 * (x - 640) ** 2) for x in xs]
    poly(top + [(1320, 2200), (-40, 2200)], (12, 12, 15, 255))
    line(top, orange, 5)

    # ---- 方向盘主体：中央身 + 左右握把（碳纤维黑 + 橙色边线）
    carbon, edge = (17, 17, 21, 255), (44, 44, 52, 255)
    for sg in (-1, 1):
        rrect(640 + sg * 540 - 105, 350, 640 + sg * 540 + 105, 2200, 80, carbon, edge, 3)
    rrect(246, 316, 1034, 2200, 52, carbon, edge, 3)
    line([(270, 322), (1010, 322)], orange, 4)
    line([(270, 342), (1010, 342)], (30, 30, 36, 255), 2)

    # ---- 15 颗转速灯（绿 5 / 红 5 / 蓝 5）+ 两侧报警灯
    n_on = int(np.clip((v - 7.0) / 7.0, 0, 1) * 15 + 0.5)
    for i in range(15):
        x = 640 + (i - 7) * 26
        col = (40, 255, 90) if i < 5 else (255, 40, 30) if i < 10 else (80, 130, 255)
        on = i < n_on
        if on and alert:
            on = flash or i % 2 == 0
        ell(x, 336, 9, col + (255,) if on else tuple(int(c * 0.14) for c in col) + (255,), (36, 36, 42, 255), 1.5)
        if on:
            ell(x, 336, 10, col, dr=g)
    for sg in (-1, 1):
        lit = alert and flash
        ell(640 + sg * 215, 338, 8, (255, 40, 30, 255) if lit else (50, 10, 10, 255), (36, 36, 42, 255), 1.5)
        if lit:
            ell(640 + sg * 215, 338, 9, (255, 40, 30), dr=g)

    # ---- 中央液晶屏
    border = (255, 50, 40, 255) if (alert and flash) else (70, 70, 80, 255)
    rrect(438, 362, 842, 590, 22, (3, 4, 7, 255), border, 3)
    N = 40
    ccx, ccy, r_in, r_out = 640, 482, 66, 88
    for i in range(N):
        a0 = np.radians(220 - 260 * i / N)
        a1 = np.radians(220 - 260 * (i + 0.8) / N)
        pts = [(ccx + r_out * np.cos(a0), ccy - r_out * np.sin(a0)), (ccx + r_out * np.cos(a1), ccy - r_out * np.sin(a1)),
               (ccx + r_in * np.cos(a1), ccy - r_in * np.sin(a1)), (ccx + r_in * np.cos(a0), ccy - r_in * np.sin(a0))]
        fi = (i + 0.5) / N
        col = (60, 230, 110) if fi < 0.5 else (250, 210, 50) if fi < 0.72 else (255, 130, 30) if fi < 0.84 else (255, 40, 30)
        lit = (i + 0.5) / N <= v / 16.0
        poly(pts, col + (255,) if lit else (28, 30, 36, 255))
        if lit:
            poly(pts, col, g)
    ang = np.radians(220 - 260 * 0.84)                              # 红线刻度
    line([(ccx + (r_in - 8) * np.cos(ang), ccy - (r_in - 8) * np.sin(ang)), (ccx + (r_out + 8) * np.cos(ang), ccy - (r_out + 8) * np.sin(ang))],
         (255, 255, 255, 255), 2.5)
    text(ccx, ccy - 6, str(7 if p < 0.55 else 8), 78, (255, 70, 55, 255) if alert else (236, 236, 240, 255))
    text(ccx, ccy + 42, "RPM %.1f" % v, 15, (150, 150, 160, 255))
    for (lx, rows) in ((488, (("LAP", "3/3", (236, 236, 240)), ("DELTA", "-0.412", (60, 230, 110)), ("ERS", "78%", (250, 210, 50)))),
                       (792, (("KM/H", str(spd), (236, 236, 240)), ("BIAS", "57.5", (236, 236, 240)), ("TEMP", "112C" if p > 0.7 else "104C", (255, 70, 55) if p > 0.7 else (236, 236, 240))))):
        for k_, (lab, val, col) in enumerate(rows):
            y0_ = 412 + k_ * 52
            text(lx, y0_, lab, 13, (130, 130, 140, 255))
            text(lx, y0_ + 20, val, 24, col + (255,))
    # REDLINE 横幅：进入红区后红底白字闪烁
    if alert:
        rrect(442, 366, 838, 394, 8, (230, 25, 25, 255) if flash else (70, 10, 10, 255))
        text(640, 380, "REDLINE", 20, (255, 255, 255, 255) if flash else (140, 60, 60, 255))
        if flash:
            rrect(442, 366, 838, 394, 8, (255, 40, 30), dr=g)

    # ---- 两侧彩色旋钮 / 按键
    for (bx, by, col) in ((300, 430, (230, 40, 40)), (372, 430, (240, 200, 40)), (300, 502, (60, 120, 235)), (372, 502, (60, 200, 90)),
                          (908, 430, (235, 235, 238)), (980, 430, (230, 40, 40)), (908, 502, (60, 200, 90)), (980, 502, (240, 200, 40))):
        ell(bx, by, 29, (8, 8, 10, 255), (84, 84, 92, 255), 2)
        for k_ in range(12):
            a_ = k_ * np.pi / 6
            line([(bx + 24 * np.cos(a_), by + 24 * np.sin(a_)), (bx + 29 * np.cos(a_), by + 29 * np.sin(a_))], (150, 150, 158, 255), 2)
        ell(bx, by, 18, col + (255,), (20, 20, 24, 255), 1.5)
        line([(bx, by), (bx + 12 * np.cos(bx * 0.05), by - 12 * np.sin(bx * 0.05 + 1.0))], (20, 20, 24, 255), 3)
    for (bx, by, lab, col) in ((330, 372, "OT", (225, 35, 35)), (950, 372, "DRS", (60, 120, 235))):
        rrect(bx - 26, by - 15, bx + 26, by + 15, 8, col + (255,), (20, 20, 24, 255), 2)
        text(bx, by, lab, 14, (255, 255, 255, 255))

    # ---- 戴手套的双手（握住握把，大拇指搭在内侧）
    glove, glove_e, glove_l = (40, 42, 58, 255), (96, 100, 124, 255), (66, 70, 94, 255)
    for sg in (-1, 1):
        gx = 640 + sg * 540
        ell(gx, 610, 100, glove, glove_e, 3, ry=130)
        for k_ in range(4):                                         # 四指
            yk = 520 + k_ * 38
            ell(gx - sg * 18, yk, 82, glove, glove_e, 2.5, ry=21)
            line([(gx - 90, yk - 6), (gx + 54, yk - 8)], glove_l, 3)
        ell(gx - sg * 78, 420, 34, glove, glove_e, 3, ry=52)        # 大拇指
        line([(gx - sg * 78 - 20, 405), (gx - sg * 78 + 20, 395)], glove_l, 3)
        line([(gx - 96, 650), (gx + 96, 650)], orange, 8)           # 手套橙色饰条

    # ---- 背景：赛道前方景象（虚化 + 压暗），上方透出
    bg = gaussian_filter(env_blur(t), (5, 5, 0)) * 0.55
    arr = np.asarray(fg).astype(np.float32) / 255
    # 碳纤维编织纹
    yy_, xx_ = np.mgrid[0:H, 0:W]
    weave = (((xx_ // (7 * S) + yy_ // (7 * S)) % 2).astype(np.float32) - 0.5) * 0.028
    dn3 = lambda x: x.reshape(OH, S, OW, S, 3).mean(axis=(1, 3))
    a1 = arr[..., 3:4]
    rgb = (arr[..., :3] + weave[..., None] * (arr[..., :3].mean(axis=2, keepdims=True) < 0.2)) * a1
    rgb, al = dn3(rgb), dn3(np.repeat(a1, 3, axis=2))[..., :1]
    gl = dn3(np.asarray(glow).astype(np.float32) / 255) * (1.3 + 1.4 * heat)
    img = bg * (1 - al) + rgb + gl
    # 进入红区：整个画面泛红、随节拍闪烁
    yy, xx = np.mgrid[0:OH, 0:OW]
    edge_ = np.clip(np.hypot((xx - OW / 2) / (OW / 2), (yy - OH / 2) / (OH / 2)), 0, 1.3)
    pulse = 0.6 + 0.4 * np.sin(t * 2 * np.pi * 4)
    img += (heat * pulse * (0.10 + 0.40 * edge_ ** 2))[..., None] * np.array([1.0, 0.04, 0.03], np.float32)
    sh = 1.0 + 6 * heat
    return np.roll(img, (int(rr.normal(0, sh)), int(rr.normal(0, sh))), axis=(0, 1)).astype(np.float32)


# ------------------------------------------------------------------ 叠加：标题、标志
def overlay_layers():
    title = Image.new("RGBA", (OW, OH), (0, 0, 0, 0))
    dr = ImageDraw.Draw(title)
    zh = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 150 if PORTRAIT else 64)
    en = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28 if PORTRAIT else 18)

    def spaced(text, font, y, spacing, fill):
        widths = [dr.textlength(c, font=font) for c in text]
        total = sum(widths) + spacing * (len(text) - 1)
        x = (OW - total) / 2
        for c, w in zip(text, widths):
            dr.text((x, y), c, font=font, fill=fill)
            x += w + spacing

    spaced("红线", zh, 250 if PORTRAIT else 104, 70 if PORTRAIT else 40, (245, 240, 232, 255))
    spaced("R E D L I N E", en, 450 if PORTRAIT else 184, 12 if PORTRAIT else 6, (230, 90, 60, 255))
    title = title.filter(ImageFilter.GaussianBlur(0.3))

    logo = Image.new("RGBA", (OW, OH), (0, 0, 0, 0))
    dl = ImageDraw.Draw(logo)
    cx, cy, R = (44, 150, 13) if PORTRAIT else (OW - 132, OH - LETTER - 34, 13)   # 竖版放左上，避开抖音右侧按钮和底部文案
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
    if COVER or HELMET:
        return to_np(title) * 0, to_np(logo) * 0
    return to_np(title), to_np(logo)


POS_EVENTS = [(25.6, 5), (51.2, 4), (64.0, 3), (102.4, 2), (107.2, 1)]   # 每次超车后的名次（发车 P6）


def hud_layer(t):
    if COVER or HELMET:
        return None
    """竖版名次显示：P6 → P1，每次超车数字弹一下并变橙；冲线后显示 WINNER"""
    pos, t_ev = 6, -99.0
    for te, ps in POS_EVENTS:
        if t >= te:
            pos, t_ev = ps, te
    if TEASER:
        return None
    if t < (START_OFF + 4.4 if DOUYIN else 5.4):          # 抖音版：标题淡出后再出名次
        return None
    im = Image.new("RGBA", (OW, OH), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    age = t - t_ev
    pop = 1.0 + 0.45 * float(np.exp(-age / 0.18)) if age >= 0 else 1.0
    col = (255, 150, 50, 255) if age < 0.5 else (245, 245, 245, 255)
    big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(120 * pop))
    lab = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    d.text((44, 206), "WINNER" if t >= T_FIN else "POSITION", font=lab, fill=(205, 205, 212, 255), anchor="lm")
    d.text((40, 290), "P%d" % pos, font=big, fill=col, anchor="lm", stroke_width=4, stroke_fill=(20, 10, 8, 255))
    arr_ = np.asarray(im).astype(np.float32) / 255
    arr_[..., 3] *= 1 - float(smooth(t, 122.4, 123.2))            # 片名再现前淡出，避免重叠
    return arr_


def title_alpha(t, total):
    """片头标题在起跑灯亮起前淡出；结尾冲线后出现"""
    if TEASER or HELMET:                                           # 预告片的字幕由后期叠加（video/teaser_text.py）
        return 0.0
    if DOUYIN:                                                     # 视频一开始就是起跑灯：标题 0.1–4.2 秒
        tv = t - START_OFF
        return float(smooth(tv, 0.1, 0.6) * (1 - smooth(tv, 3.4, 4.2)) + smooth(t, 123.0, 124.6))
    return float(smooth(t, 0.8, 2.2) * (1 - smooth(t, 4.0, 5.2)) + smooth(t, 123.0, 124.6))


# ------------------------------------------------------------------ 合成一帧
_VIG = None


# ------------------------------------------------------------------ 赛道边「呼啸而过」分镜（侧面机位，从左闯入、从右冲出）
DRIVEBY_T = 1000.0                    # 场景时间 >= 1000 表示进入这个分镜（t - 1000 = 分镜内的秒数）
DB_KS = 208.0                         # 车所在车道处的 像素/米（机位离车道约 3 米，车很大，一次只能看到一部分车身）
DB_V = 12000.0                        # 车速（像素/秒，约 58 米/秒；一辆车约 0.16 秒闯过画面）
DB_T0 = 0.30                          # 第一辆车的车心从画面外 -450px 起算的时刻
DB_GAP = 0.18                         # 追车晚多久


def side_car_sprite(base, helmet=(245, 210, 55)):
    """赛车侧面剪影（车头朝右），超采样 2 倍。返回预乘 RGBA float32（高 1.4 米、宽 5.8 米的画布）"""
    ppm = DB_KS * 2
    Wc, Hc = int(5.8 * ppm), int(1.4 * ppm)
    im = Image.new("RGBA", (Wc, Hc), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    X = lambda x_: (x_ + 0.1) * ppm
    Y = lambda y_: Hc - (y_ + 0.1) * ppm
    poly = lambda pts, fill: d.polygon([(X(a), Y(b)) for a, b in pts], fill=fill)
    OR = tuple(base) + (255,)
    OR_T = tuple(min(255, int(c * 1.25)) for c in base) + (255,)
    OR_D = tuple(int(c * 0.6) for c in base) + (255,)
    DK, DK2 = (14, 14, 18, 255), (30, 30, 36, 255)

    def wheel(cx_, cy_, r_):
        d.ellipse([X(cx_ - r_), Y(cy_ + r_), X(cx_ + r_), Y(cy_ - r_)], fill=(14, 14, 16, 255))
        d.ellipse([X(cx_ - r_ * 0.84), Y(cy_ + r_ * 0.84), X(cx_ + r_ * 0.84), Y(cy_ - r_ * 0.84)], outline=(46, 46, 52, 255), width=int(0.012 * ppm))
        d.arc([X(cx_ - r_ * 0.9), Y(cy_ + r_ * 0.9), X(cx_ + r_ * 0.9), Y(cy_ - r_ * 0.9)], 200, 330, fill=(215, 180, 40, 255), width=int(0.02 * ppm))
        d.ellipse([X(cx_ - r_ * 0.58), Y(cy_ + r_ * 0.58), X(cx_ + r_ * 0.58), Y(cy_ - r_ * 0.58)], fill=(58, 58, 66, 255), outline=(90, 90, 100, 255), width=int(0.01 * ppm))
        for k_ in range(5):                                         # 轮辐
            a_ = k_ * 2 * np.pi / 5 + 0.3
            d.line([(X(cx_), Y(cy_)), (X(cx_ + 0.55 * r_ * np.cos(a_)), Y(cy_ + 0.55 * r_ * np.sin(a_)))], fill=(30, 30, 34, 255), width=int(0.025 * ppm))
        d.ellipse([X(cx_ - 0.1 * r_), Y(cy_ + 0.1 * r_), X(cx_ + 0.1 * r_), Y(cy_ - 0.1 * r_)], fill=(170, 170, 176, 255))

    # 悬挂（在轮子后面）
    d.line([(X(1.1), Y(0.36)), (X(1.7), Y(0.30))], fill=DK2, width=int(0.035 * ppm))
    d.line([(X(4.1), Y(0.34)), (X(3.5), Y(0.34))], fill=DK2, width=int(0.035 * ppm))
    # 地板 / 扩散器
    poly([(0.45, 0.12), (0.6, 0.04), (4.55, 0.04), (4.65, 0.15), (0.5, 0.2)], DK)
    # 车身主体
    poly([(0.5, 0.18), (0.45, 0.55), (0.9, 0.64), (1.5, 0.82), (2.1, 1.04), (2.55, 0.9), (3.0, 0.66), (3.5, 0.6),
          (4.3, 0.44), (5.0, 0.28), (5.35, 0.2), (5.3, 0.14), (4.6, 0.14), (4.5, 0.18)], OR)
    poly([(1.4, 0.18), (1.5, 0.62), (3.2, 0.55), (3.35, 0.42), (3.3, 0.18)], OR_D)               # 侧箱（暗面）
    poly([(3.05, 0.28), (3.38, 0.34), (3.38, 0.52), (3.05, 0.52)], DK)                         # 侧箱进气口
    poly([(0.9, 0.64), (1.5, 0.82), (2.1, 1.04), (2.22, 1.0), (1.56, 0.78), (0.96, 0.6)], OR_T)  # 发动机盖高光
    poly([(0.55, 0.40), (3.0, 0.48), (3.0, 0.53), (0.55, 0.45)], (240, 240, 244, 255))         # 涂装白线
    poly([(2.55, 0.9), (3.0, 0.66), (3.45, 0.62), (3.2, 0.8), (2.8, 0.92)], (4, 4, 6, 255))    # 座舱
    d.ellipse([X(2.71), Y(1.07), X(2.99), Y(0.79)], fill=tuple(helmet) + (255,))                  # 头盔
    poly([(2.86, 0.99), (3.0, 0.97), (3.0, 0.88), (2.88, 0.88)], (18, 18, 24, 255))            # 面罩
    d.line([(X(3.5), Y(0.62)), (X(3.25), Y(0.9)), (X(2.9), Y(1.08)), (X(2.45), Y(0.98)), (X(2.3), Y(0.88))],
           fill=(78, 80, 88, 255), width=int(0.05 * ppm))                                         # halo
    # 前翼
    poly([(4.85, 0.11), (5.55, 0.11), (5.55, 0.2), (4.9, 0.2)], (20, 20, 24, 255))
    poly([(5.3, 0.08), (5.6, 0.08), (5.6, 0.36), (5.3, 0.3)], OR_D)
    # 尾翼
    poly([(0.55, 0.5), (0.63, 0.5), (0.5, 0.9), (0.42, 0.9)], DK)
    poly([(0.0, 0.92), (0.78, 0.92), (0.78, 1.02), (0.0, 1.02)], (24, 24, 28, 255))
    poly([(0.0, 0.62), (0.8, 0.7), (0.8, 1.1), (0.0, 1.1)], OR)
    poly([(0.0, 1.04), (0.8, 1.04), (0.8, 1.1), (0.0, 1.1)], OR_T)
    # 车轮（压在车身上）
    wheel(1.05, 0.36, 0.36)
    wheel(4.15, 0.33, 0.33)
    d.line([(X(0.5), Y(0.58)), (X(4.3), Y(0.46))], fill=OR_T, width=int(0.012 * ppm))              # 上缘高光
    sp = np.asarray(im).astype(np.float32) / 255
    sp[..., :3] *= sp[..., 3:4]
    return sp


_DB_CACHE = {}


def _driveby_bg():
    """赛道边的固定背景：夜空 + 城市灯光、防护网、广告墙、灯柱、对岸路面、前景红白路肩（虚化）"""
    if "bg" in _DB_CACHE:
        return _DB_CACHE["bg"]
    f = OW / 2 / np.tan(np.radians(FOV / 2))
    y0 = OH * 0.45
    yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
    dx = (xx + 0.5 - OW / 2) / f
    dy = (y0 - (yy + 0.5)) / f
    img = sky(dx.ravel(), np.maximum(dy, 0).ravel(), 0.0).reshape(OH, OW, 3).astype(np.float32) * 0.9
    r_ = np.random.default_rng(21)
    HCb = 1.0                                                                     # 机位高度（米）
    ground = yy > y0 + 1
    dd = np.where(ground, HCb * f / np.maximum(yy + 0.5 - y0, 1), 1e4)             # 地面点离机位的横向距离（米）
    wx = dx * dd                                                                   # 沿赛道方向的位置（米）
    tex = (r_.standard_normal((OH, OW)).astype(np.float32))
    asph = 0.05 + 0.012 * gaussian_filter(tex, 1.2) * 6 * np.clip(1 / (dd * 0.2 + 0.4), 0, 1)
    col = np.repeat(asph[..., None], 3, axis=2) * np.array([1.0, 1.0, 1.06], np.float32)
    kerb = (dd > 1.2) & (dd < 2.2)
    stripe = (np.floor((wx + 40) / 1.6) % 2)[..., None]
    kc = stripe * np.array([0.30, 0.30, 0.30], np.float32) + (1 - stripe) * np.array([0.32, 0.03, 0.025], np.float32)
    col = np.where(kerb[..., None], kc, col)
    edge = (dd > 4.3) & (dd < 4.5)
    col = np.where(edge[..., None], np.array([0.5, 0.5, 0.48], np.float32), col)
    lamp_x = 130.0
    glow = np.exp(-(((xx - lamp_x) / 260) ** 2) - (((yy - 700) / 260) ** 2)).astype(np.float32)   # 灯柱在地面的光斑
    col = col * (1 + 3.0 * glow[..., None]) + glow[..., None] * np.array([0.10, 0.07, 0.04], np.float32)
    if DUSK:
        col = col * 1.6 + np.array([0.07, 0.035, 0.018], np.float32) + (np.clip(1 - (dd - 1) / 25, 0, 1) ** 2 * 0)[..., None]
    img = np.where(ground[..., None], col, img)
    # 对岸：防护网、广告墙、LED 灯带（离机位 17 米）
    Dw = 17.0
    y_wall_b, y_wall_t = y0 + HCb * f / Dw, y0 - (1.0 - HCb) * f / Dw
    y_fence_t = y0 - (4.6 - HCb) * f / Dw
    wall = (yy >= y_wall_t) & (yy <= y_wall_b)
    pidx = (np.floor((dx * Dw + 200) / 14) % 4).astype(int)
    pal = np.array([[0.55, 0.22, 0.07], [0.45, 0.45, 0.43], [0.04, 0.04, 0.05], [0.05, 0.2, 0.24]], np.float32)
    wc = pal[pidx] * (0.5 + 1.8 * glow[..., None])
    img = np.where(wall[..., None], wc, img)
    led = (yy >= y_wall_t - 3) & (yy < y_wall_t)
    img = np.where(led[..., None], np.array([1.4, 0.45, 0.18], np.float32), img)
    fence = (yy >= y_fence_t) & (yy < y_wall_t - 3)
    mesh = (np.abs(((dx * Dw * 1.3) % 1) - 0.5) < 0.05) | (np.abs((((y0 - yy) / f * Dw * 1.3) % 1) - 0.5) < 0.05)
    img = np.where(fence[..., None], img + mesh[..., None] * np.array([0.10, 0.10, 0.11], np.float32) * (0.4 + glow[..., None] * 3), img)
    # 灯柱 + 灯头 + 光晕
    lx0 = int(lamp_x)
    img[int(y0 - 9.5 * f / Dw + 20):int(y_wall_b), lx0 - 3:lx0 + 3] = np.array([0.05, 0.05, 0.06], np.float32)
    ytop = int(y0 - (10 - HCb) * f / Dw)
    img[ytop:ytop + 14, lx0 - 34:lx0 + 34] = np.array([3.4, 3.2, 2.8], np.float32)
    lam = np.exp(-(((xx - lamp_x) / 120) ** 2) - (((yy - ytop) / 90) ** 2)).astype(np.float32)
    img += lam[..., None] * np.array([0.5, 0.42, 0.3], np.float32)
    for lx_, ly_ in ((lx0, ytop + 7), (570, ytop + 40)):                      # 变形镜头的横向光晕
        fl = np.exp(-(((yy - ly_) / 4.0) ** 2) - (np.abs(xx - lx_) / 330.0) ** 1.3).astype(np.float32)
        img += fl[..., None] * np.array([0.55, 0.7, 1.2], np.float32)
    img[int(ytop + 40):int(y_wall_b), 567:573] = np.array([0.05, 0.05, 0.06], np.float32)
    img[int(ytop + 34):int(ytop + 46), 548:592] = np.array([2.6, 2.5, 2.3], np.float32)
    # 城市远景的光点（虚化的圆斑）
    for _ in range(26):
        bx, by, br = r_.uniform(0, OW), r_.uniform(y0 - 150, y0 - 20), r_.uniform(4, 11)
        c_ = np.array([1.0, 0.7, 0.35], np.float32) * r_.uniform(0.25, 0.7)
        m_ = np.exp(-(((xx - bx) ** 2 + (yy - by) ** 2) / (2 * br * br))).astype(np.float32)
        img += m_[..., None] * c_
    # 景深：前景（路肩和近处路面）强虚化，对岸轻微虚化
    near = np.clip((yy - 690) / 90, 0, 1)[..., None]
    far = np.clip((y0 - 40 - yy) / 120, 0, 1)[..., None]
    img = img * (1 - near) + gaussian_filter(img, (11, 11, 0)) * near
    img = img * (1 - 0.6 * far) + gaussian_filter(img, (2.2, 2.2, 0)) * 0.6 * far
    out = np.clip(img, 0, 6).astype(np.float32)
    _DB_CACHE["bg"] = (out, y0, f, HCb)
    return _DB_CACHE["bg"]


def driveby_frame(lt, pair=0):
    """lt：分镜内的秒数。固定机位，橙车（lt≈0.30–0.47）和红色追车（晚 0.18 秒）从左闯入、从右冲出，强烈运动模糊"""
    bg, y0, f, HCb = _driveby_bg()
    img = bg.copy()
    gy = y0 + HCb * DB_KS                                                      # 车轮接地线
    rr = np.random.default_rng(int(lt * 1000) + 5)
    shake = 0.0
    second = ((215, 30, 35), (240, 240, 245)) if pair == 0 else ((240, 205, 20), (245, 245, 250))     # 追车颜色：0 红车，1 黄车
    for dl_, col, helm in ((0.0, ORANGE, (245, 210, 55)), (DB_GAP, *second)):
        if ("car", col) not in _DB_CACHE:
            sp2 = side_car_sprite(col, helm)
            small = np.stack([np.asarray(Image.fromarray(sp2[..., c]).resize((sp2.shape[1] // 2, sp2.shape[0] // 2), Image.LANCZOS)) for c in range(4)], axis=2)
            _DB_CACHE[("car", col)] = small
        sp = _DB_CACHE[("car", col)]
        h_, w_ = sp.shape[:2]
        xc = -w_ / 2 + DB_V * (lt - DB_T0 - dl_)                        # lt=DB_T0 时车头刚好从左边露出
        if xc < -w_ or xc > OW + w_:
            continue
        shake = max(shake, 1.0)
        # 水平运动模糊：快门时间内的位移，多次采样平均
        blur_px = DB_V * 0.5 / FPS
        acc = np.zeros((h_, w_ + int(blur_px) + 4, 4), np.float32)
        n_s = 28
        for k_ in range(n_s):
            off = int(round((k_ / (n_s - 1) - 0.5) * blur_px)) + int(blur_px / 2) + 2
            acc[:, off:off + w_] += sp / n_s
        acc = gaussian_filter(acc, (1.2, 5, 0))                              # 采样之间的缝隙抹平
        x0_ = int(xc - w_ / 2 - blur_px / 2 - 2)
        y0_ = int(gy - h_ + 0.1 * DB_KS)
        # 阴影
        sh = np.zeros((OH, OW), np.float32)
        ya, yb = int(gy - 6), int(gy + 14)
        xa, xb = max(0, x0_ + 40), min(OW, x0_ + acc.shape[1] - 40)
        if xb > xa:
            sh[ya:yb, xa:xb] = 1
            sh = gaussian_filter(sh, 7)
            img *= (1 - 0.7 * sh)[..., None]
        xa_, xb_ = max(0, x0_), min(OW, x0_ + acc.shape[1])
        ya_, yb_ = max(0, y0_), min(OH, y0_ + h_)
        if xb_ > xa_ and yb_ > ya_:
            c_ = acc[ya_ - y0_:yb_ - y0_, xa_ - x0_:xb_ - x0_]
            # 灯柱方向的侧光：车身越靠上越亮
            lit = np.linspace(1.25, 0.8, c_.shape[0])[:, None, None]
            reg = img[ya_:yb_, xa_:xb_]
            img[ya_:yb_, xa_:xb_] = reg * (1 - c_[..., 3:4]) + c_[..., :3] * lit
        # 尾灯拖出的红色光带 + 地板擦出的火花（只有橙车）
        tail_x, tail_y = xc - w_ / 2 + 0.55 * DB_KS, gy - 0.38 * DB_KS
        trail = np.zeros((OH, OW), np.float32)
        L_ = 260
        xs0 = int(tail_x - L_)
        if xs0 < OW and tail_x > 0:
            xs = np.arange(max(0, xs0), min(OW, int(tail_x)))
            trail[int(tail_y) - 3:int(tail_y) + 3, xs] = ((xs - xs0) / L_) ** 1.5
            trail = gaussian_filter(trail, (2.5, 6))
            img += trail[..., None] * np.array([2.2, 0.15, 0.1], np.float32)
        if dl_ == 0.0:
            ln_ = Image.new("RGB", (OW, OH), 0)
            dr_ = ImageDraw.Draw(ln_)
            for _ in range(46):
                sx = tail_x + rr.uniform(-20, 30) - rr.uniform(0, 240)
                sy = gy + rr.uniform(-14, 6)
                dr_.line([(sx, sy), (sx - rr.uniform(30, 150), sy + rr.uniform(-8, 12))], fill=(255, int(rr.uniform(120, 210)), 70), width=2)
            img += np.asarray(ln_).astype(np.float32) / 255 * 1.5
    if shake:
        img = np.roll(img, int(rr.normal(0, 4)), axis=0)
    return img.astype(np.float32)


# ------------------------------------------------------------------ 车手头盔视角（第一人称）
HELMET_T0 = 102.4                     # 4 秒片段对应的场景时间起点：绿车在前方 42 米 → 7 米，一路逼近
HEL_CAM = (0.0, 0.95, 2.35)           # 眼睛在车里的位置（x 横向, y 高度, z 距车尾）


def helmet_cockpit(vt, pulse):
    """面罩后看到的座舱：前翼、前轮、鼻锥、halo 立柱与弧臂、方向盘上沿 + 转速灯、手套。按真实 F1 比例做 3D 透视"""
    S = 2
    W, H = OW * S, OH * S
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = OW / 2 / np.tan(np.radians(FOV / 2))
    cx0, cy0 = OW / 2, OH * HOR
    cam = HEL_CAM

    def P(x, y, z):
        zz = max(z - cam[2], 0.12)
        return ((cx0 + (x - cam[0]) * f / zz) * S, (cy0 + (cam[1] - y) * f / zz) * S)

    def poly(pts, fill, outline=None):
        d.polygon([P(*p_) for p_ in pts], fill=fill, outline=outline)

    OR_T = (232, 112, 40, 255)
    OR = (190, 86, 30, 255)
    OR_D = (110, 48, 20, 255)
    DK, DK2 = (14, 14, 18, 255), (34, 34, 40, 255)
    # 前翼（最远）：主翼片 + 两侧端板
    poly([(-0.95, 0.16, 4.75), (0.95, 0.16, 4.75), (0.95, 0.16, 5.15), (-0.95, 0.16, 5.15)], (28, 28, 34, 255))
    for sg in (-1, 1):
        poly([(sg * 0.95, 0.05, 4.7), (sg * 0.95, 0.32, 4.7), (sg * 0.95, 0.32, 5.2), (sg * 0.95, 0.05, 5.2)], OR_D)
    # 前轮：顶面、朝车内的侧面、朝前的正面（轮胎黑、黄色胎壁标记）
    for sg in (-1, 1):
        x1, x2 = sorted((sg * 0.62, sg * 0.95))
        poly([(x1, 0.66, 3.9), (x2, 0.66, 3.9), (x2, 0.66, 4.55), (x1, 0.66, 4.55)], (52, 52, 58, 255))
        xi = sg * 0.62
        poly([(xi, 0.0, 3.9), (xi, 0.66, 3.9), (xi, 0.66, 4.55), (xi, 0.0, 4.55)], (36, 36, 42, 255))
        poly([(x1, 0.0, 3.9), (x2, 0.0, 3.9), (x2, 0.66, 3.9), (x1, 0.66, 3.9)], (22, 22, 26, 255), outline=(70, 70, 78, 255))
        a_, b_ = P(sg * 0.9, 0.18, 3.9), P(sg * 0.9, 0.5, 3.9)
        d.line([a_, b_], fill=(215, 180, 40, 255), width=max(2, int(0.02 * f / 1.5 * S)))
        d.line([P(sg * 0.1, 0.38, 3.7), P(sg * 0.62, 0.34, 4.1)], fill=DK2, width=max(2, int(0.03 * f / 1.4 * S)))
    # 鼻锥
    poly([(-0.13, 0.42, 3.0), (0.13, 0.42, 3.0), (0.07, 0.30, 4.85), (-0.07, 0.30, 4.85)], OR_T)
    poly([(-0.13, 0.42, 3.0), (-0.07, 0.30, 4.85), (-0.2, 0.2, 4.85), (-0.2, 0.3, 3.0)], OR_D)
    poly([(0.13, 0.42, 3.0), (0.07, 0.30, 4.85), (0.2, 0.2, 4.85), (0.2, 0.3, 3.0)], OR)
    d.line([P(0, 0.42, 3.0), P(0, 0.30, 4.85)], fill=(250, 250, 250, 255), width=max(2, S))
    # halo 立柱 + 两侧弧臂：立柱管径约 4 厘米、在眼前约 1 米处；弧臂细而带弧度，向两侧上方延伸出视野
    pz = 3.35
    top = P(0, 1.03, pz)
    poly([(-0.022, 0.60, pz), (0.022, 0.60, pz), (0.022, 1.03, pz), (-0.022, 1.03, pz)], (66, 68, 76, 255))
    d.line([P(-0.006, 0.60, pz), P(-0.006, 1.03, pz)], fill=(130, 132, 142, 255), width=max(1, S))        # 立柱高光
    for sg in (-1, 1):
        arm = [top, (top[0] + sg * 130 * S / 2, top[1] - 6 * S / 2), (top[0] + sg * 270 * S / 2, top[1] - 26 * S / 2),
               (top[0] + sg * 420 * S / 2, top[1] - 62 * S / 2), (top[0] + sg * 520 * S / 2, top[1] - 100 * S / 2)]
        d.line(arm, fill=(58, 60, 68, 255), width=int(14 * S / 2), joint="curve")
        d.line([(x_, y_ - 3) for x_, y_ in arm], fill=(120, 122, 132, 255), width=max(1, int(2 * S / 2)), joint="curve")
    # 方向盘上沿 + 15 颗转速灯
    wy = P(0, 0.5, 2.95)[1]
    wx = 0.2 * f / 0.6 * S
    d.rounded_rectangle([cx0 * S - wx, wy, cx0 * S + wx, H + 200], radius=int(40 * S), fill=(17, 17, 21, 255), outline=(44, 44, 52, 255), width=3 * S)
    d.line([(cx0 * S - wx + 40 * S, wy + 6 * S), (cx0 * S + wx - 40 * S, wy + 6 * S)], fill=(236, 104, 34, 255), width=3 * S)
    n_on = int(np.clip(10 + 5 * vt / 2.0, 0, 15))
    flash = int(vt * 12) % 2 == 0
    for i in range(15):
        x = cx0 + (i - 7) * 20
        col = (40, 255, 90) if i < 5 else (255, 40, 30) if i < 10 else (80, 130, 255)
        on = i < n_on and (flash or i < 10 or vt < 2.0)
        d.ellipse([(x - 7) * S, wy + 16 * S, (x + 7) * S, wy + 30 * S], fill=col + (255,) if on else tuple(int(c * 0.15) for c in col) + (255,))
    # 戴手套的双手（握在方向盘两侧）
    glove, glove_e = (40, 42, 58, 255), (96, 100, 124, 255)
    for sg in (-1, 1):
        gx = cx0 + sg * 0.27 * f / 0.6
        d.ellipse([(gx - 95) * S, (OH * 0.80) * S, (gx + 95) * S, (OH * 0.80 + 260) * S], fill=glove, outline=glove_e, width=3 * S)
        for k_ in range(3):
            yk = OH * 0.80 + 60 + k_ * 50
            d.line([((gx - 80) * S, yk * S), ((gx + 60) * S, (yk + 6) * S)], fill=(66, 70, 94, 255), width=3 * S)
        d.ellipse([(gx - sg * 55 - 38) * S, (OH * 0.80 - 30) * S, (gx - sg * 55 + 38) * S, (OH * 0.80 + 70) * S], fill=glove, outline=glove_e, width=3 * S)
    arr = np.asarray(im).astype(np.float32) / 255
    arr[..., :3] *= arr[..., 3:4]
    dn = lambda x: x.reshape(OH, S, OW, S, *x.shape[2:]).mean(axis=(1, 3))
    a = dn(arr)
    return a[..., :3], a[..., 3:4]


_HEL_FX = {}


def helmet_frame(t, pulse):
    """世界（含对手）+ 座舱，整体随头部倾斜/抖动，再罩上面罩：暗角、遮阳条、脏点、撕膜（tear-off）前后的雾感"""
    from scipy.ndimage import rotate
    vt = t - HELMET_T0
    rng = np.random.default_rng(int(round(vt * FPS)) + 3)
    world = _exterior(t, pulse)
    cock, ca = helmet_cockpit(vt, pulse)
    scene = world * (1 - ca) + cock
    # 头部：横滚 ±5°、车身高频振动、1.8 秒一次路面颠簸
    roll = 5.0 * np.sin(vt * 3.2) + 1.2 * rng.normal()
    bump = 11.0 * np.exp(-((vt - 1.8) / 0.05) ** 2)
    dx, dy = rng.normal(0, 1.4), rng.normal(0, 1.8) + bump
    scene = rotate(scene, roll, axes=(1, 0), reshape=False, order=1, mode="nearest")
    scene = np.roll(scene, (int(dy), int(dx)), axis=(0, 1))
    yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
    # 撕膜：2.3–2.65 秒从右向左撕掉，之前面罩上有一层雾膜（轻微模糊 + 发白 + 脏点），之后画面清晰、反差更大
    p = float(np.clip((vt - 2.3) / 0.35, 0, 1))
    xb = OW * (1 - (p * p * (3 - 2 * p))) + 14 * np.sin(yy / 55.0)
    hazy = (xx < xb).astype(np.float32)
    if p >= 1:
        hazy[:] = 0
    if "dirt" not in _HEL_FX:
        r_ = np.random.default_rng(9)
        dirt = np.zeros((OH, OW), np.float32)
        for _ in range(70):
            cx_, cy_, rr_ = r_.uniform(0, OW), r_.uniform(0, OH), r_.uniform(1.5, 5)
            dirt = np.maximum(dirt, np.exp(-(((xx - cx_) ** 2 + (yy - cy_) ** 2) / (2 * rr_ * rr_))) * r_.uniform(0.15, 0.5))
        for _ in range(6):                                          # 几道污迹
            x0_, y0_, ang = r_.uniform(0, OW), r_.uniform(0, OH), r_.uniform(0, np.pi)
            for k_ in range(60):
                dirt = np.maximum(dirt, np.exp(-(((xx - x0_ - k_ * 3 * np.cos(ang)) ** 2 + (yy - y0_ - k_ * 3 * np.sin(ang)) ** 2) / 8.0)) * 0.12)
        _HEL_FX["dirt"] = dirt
    hz = gaussian_filter(scene, (3, 3, 0))
    veil = np.array([0.30, 0.20, 0.14], np.float32) * 0.5
    hazed = hz * 0.85 + veil + _HEL_FX["dirt"][..., None] * np.array([0.5, 0.4, 0.3], np.float32) * 0.5
    img = scene * (1 - hazy[..., None]) + hazed * hazy[..., None]
    # 撕膜卷边：边界处一条发亮的半透明窄带 + 阴影线；撕下后的碎片飞走
    if 0 < p < 1:
        band = np.exp(-(((xx - xb - 10) / 14) ** 2)).astype(np.float32)
        img += band[..., None] * np.array([0.7, 0.62, 0.5], np.float32) * 0.8
        img *= 1 - 0.35 * np.exp(-(((xx - xb + 6) / 5) ** 2))[..., None]
    if 0.0 < vt - 2.65 < 0.25:
        q = (vt - 2.65) / 0.25
        cxs, cys = OW * (0.2 - 0.5 * q), OH * (0.45 - 0.35 * q)
        piece = np.exp(-((np.abs(xx - cxs) / 150) ** 4 + (np.abs(yy - cys) / 40) ** 4)).astype(np.float32)
        img = img * (1 - 0.45 * piece[..., None]) + 0.45 * piece[..., None] * np.array([0.8, 0.75, 0.7], np.float32)
    # 太阳晃眼：围绕落日位置的眩光，面罩有雾时更强，撕膜后收窄变锐利
    sx, sy_ = OW / 2 + np.tan(np.radians(8)) * (OW / 2 / np.tan(np.radians(FOV / 2))), OH * HOR - 0.05 * (OW / 2 / np.tan(np.radians(FOV / 2)))
    r2 = ((xx - sx) ** 2 + (yy - sy_) ** 2)
    glare = np.exp(-r2 / (2 * 130.0 ** 2)) * (0.55 * hazy + 0.22 * (1 - hazy)) + np.exp(-r2 / (2 * 40.0 ** 2)) * 0.5
    img += glare[..., None] * np.array([1.0, 0.62, 0.32], np.float32)
    # 面罩与头盔：四周暗角、顶部遮阳条（琥珀色渐变）、底部下巴线
    rx, ry = (xx - OW / 2) / (OW * 0.62), (yy - OH / 2) / (OH * 0.58)
    vig = np.clip(1 - np.clip(np.hypot(rx, ry) - 0.72, 0, 1) * 2.6, 0, 1)
    img *= vig[..., None]
    shade = np.clip(1 - yy / 110.0, 0, 1) ** 1.5
    img = img * (1 - 0.6 * shade[..., None]) + shade[..., None] * np.array([0.10, 0.05, 0.01], np.float32) * 0.6
    img *= 1 - 0.7 * np.clip((yy - (OH - 50)) / 50.0, 0, 1)[..., None]
    return img.astype(np.float32)


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
    cars = [] if HELMET else [dict(zc=CAR_Z, xc=0.0, base=ORANGE, flash=pulse, sway=sway, bounce=bounce)]
    for k, rv in enumerate(RIVALS):
        d, u = rival_state(rv, t)
        zc = (d - 2.35) if HELMET else CAR_Z + d
        if zc < (2.6 if HELMET else 2.2) or zc > 260:
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
    if not HELMET:
        img += sparks(t, (OW / 2 + sway, OH * HOR + bounce, f)) * min(1, v / 70)
    img = img + glow_add
    return img


def frame(t, layers, total=128.0, pulse=0.0):
    global _VIG
    title, logo = layers
    r = np.random.default_rng(int(t * 1000))
    v = speed_at(t)
    img = helmet_frame(t, pulse) if HELMET else (driveby_frame(t - 2000.0, 1) if t >= 2000.0 else driveby_frame(t - DRIVEBY_T, 0)) if t >= DRIVEBY_T else (dash_frame(t) if DASH_T0 <= t < DASH_T1 else _exterior(t, pulse))
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
    if PORTRAIT:
        hud = hud_layer(t)
        if hud is not None:
            img = img * (1 - hud[..., 3:4]) + hud[..., :3] * hud[..., 3:4]
    img[:LETTER] = 0
    img[OH - LETTER:] = 0
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def audio_pulse(nf):
    """每帧的音频瞬态强度（0~1）：底鼓/重音一响，尾灯和泛光就闪一下"""
    from scipy.io import wavfile
    wav = os.path.splitext(AUDIO)[0] + ".wav"
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
    t = T0 + i / FPS + START_OFF                         # 场景时间（抖音版比成片时间晚 5.6 秒）
    pt = t
    if TEASER:                                           # 预告片：成片时间 → 分镜片段 → 场景时间；闪光脉冲取自实际播放的音乐窗口
        vt, base = i / FPS, 0.0
        for s0, dur in TEASER_SEGS:
            if vt < base + dur - 1e-9:
                t = s0 + (vt - base)
                break
            base += dur
        pt = TEASER_AUDIO0 + i / FPS
    img = frame(t, LAYERS, TOTAL, float(PULSE[min(int(pt * FPS), len(PULSE) - 1)])).astype(np.float32)
    fade_in = FPS * (0.1 if (DOUYIN or HELMET) else 0.5)
    fade = min(1.0, (i + 1) / fade_in, (DUR * FPS - i) / (FPS * (3.0 if DUR == TOTAL else (0.25 if HELMET else 0.6))))
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
        ts = [float(x) for x in sys.argv[sys.argv.index("--still") + 1:] if not x.startswith("--")]
        pre = "helmet_" if HELMET else ("cover%dx%d_" % (OW, OH)) if COVER else (("dusk_" if DUSK else "") + ("p" if PORTRAIT else ""))
        for t in ts:
            Image.fromarray(frame(t, LAYERS)).save(os.path.join(HERE, "race_still_%s%g.png" % (pre, t)))
            print("saved race_still_%s%g.png" % (pre, t))
        return
    out = os.path.join(HERE, "redline_douyin_720.mp4" if DOUYIN else ("redline_portrait.mp4" if PORTRAIT else "redline.mp4"))
    if HELMET:                          # 头盔视角 4 秒：场景时间从 HELMET_T0 起，不带音乐（音效另外合成再混）
        T0, DUR = HELMET_T0, 4.0
        out = os.path.join(HERE, "redline_helmet_720_silent.mp4")
    if TEASER:
        DUR = float(sum(d_ for _, d_ in TEASER_SEGS))
        out = os.path.join(HERE, "redline_dusk_720.mp4" if DUSK else "redline_teaser_720.mp4")
    if "--demo" in sys.argv:            # --demo 起始秒 时长
        k = sys.argv.index("--demo")
        T0, DUR = float(sys.argv[k + 1]), float(sys.argv[k + 2])
        out = os.path.join(HERE, "redline_demo.mp4")
    PULSE = np.zeros(int((TOTAL + START_OFF + 1) * FPS) + 1) if HELMET else audio_pulse(int((TOTAL + START_OFF + 1) * FPS) + 1)
    nf = int(round(DUR * FPS))
    workers = os.cpu_count() or 4
    tag = ("helmet" if HELMET else "") + ("dusk" if DUSK else "") + ("teaser" if TEASER else ("demo" if "--demo" in sys.argv else "full")) + ("_pd" if DOUYIN else ("_p" if PORTRAIT else ""))
    CH = 120
    jobs = [(k, i0, min(nf, i0 + CH), tag) for k, i0 in enumerate(range(0, nf, CH))]
    done = sum(os.path.exists(os.path.join(HERE, f".seg_{tag}_{k:03d}.mp4")) for k, *_ in jobs)
    print(f"{len(jobs)} chunks, {done} already done, {workers} workers", flush=True)
    with mp.get_context("fork").Pool(workers) as pool:
        segs = pool.map(worker, jobs, chunksize=1)
    lst = os.path.join(HERE, ".segs.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{x}'\n" for x in segs)
    if HELMET:                          # 无声视频，音效由 helmet_sfx.py 合成后再混入
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c:v", "copy",
                        "-t", str(DUR), "-movflags", "+faststart", out], check=True)
        for x in segs + [lst]:
            os.remove(x)
        print("saved", out)
        return
    afade = f"afade=t=in:d=0.3,afade=t=out:st={DUR - 0.6}:d=0.6" if DUR < TOTAL else ("afade=t=in:d=0.15" if DOUYIN else "")
    if TEASER:
        afade = f"afade=t=in:d=0.03,afade=t=out:st={DUR - 0.5}:d=0.5"           # 开头那一下镲要保住
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-ss", str(TEASER_AUDIO0 if TEASER else T0 + START_OFF), "-i", AUDIO, "-map", "0:v", "-map", "1:a", "-c:v", "copy"]
                   + (["-af", afade] if afade else [])
                   + ["-c:a", "aac", "-b:a", "192k", "-t", str(DUR), "-movflags", "+faststart", out], check=True)
    for x in segs + [lst]:
        os.remove(x)
    print("saved", out)


if __name__ == "__main__":
    main()
