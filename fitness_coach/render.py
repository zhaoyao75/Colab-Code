"""
《小燃 16 秒居家挑战》竖屏 1080x1920 · 30fps · 16 秒。
代码绘制的卡通 AI 教练「小燃」：关节坐标直接驱动，粗线条胶囊画肢体，2 倍超采样抗锯齿。
时间轴（120BPM，一拍 0.5s）：0-3 钩子 | 3-7 开合跳×4 | 7-11 深蹲×2 | 11-14 高抬腿×6 | 14-16 收尾打卡

运行：python3 render.py --still 1 3.25 3.75 8 9 11.25 11.75 15   -> preview/stills/*.png + 联系图 preview/sheet.png
     python3 render.py                                           -> .tmp/video_silent.mp4 → xiaoran_16s.mp4（带音频）
"""
import json, math, os, subprocess, sys
import multiprocessing as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__)); TMP = os.path.join(HERE, ".tmp")
W, H, FPS, DUR = 1080, 1920, 30, 16.0
NF = int(FPS * DUR)
ZH = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
GROUND, CX = 1190, 540            # 脚底线 / 画面中线
SS = 2                            # 角色层超采样倍数
LAY = (0, 430, W, 900)            # 角色层区域 x,y,w,h

# ---------- 配色 ----------
SKIN, SKIN_D = (255, 214, 186), (240, 180, 150)
HAIR, HAIR_D = (255, 138, 61), (222, 100, 30)
TOP, TOP_D = (28, 214, 190), (14, 160, 150)
PANTS, PANTS_D = (52, 56, 112), (36, 40, 86)
SHOE, SOLE = (255, 235, 90), (255, 255, 255)
INK = (46, 34, 60)

# ---------- 身体尺寸（像素，1 倍）----------
HEAD_R, TORSO, SH_HALF, HIP_HALF = 64, 168, 50, 30
UARM, FARM, THIGH, SHIN, LIMB_W = 98, 94, 150, 150, 34

def lerp(a, b, u): return a + (b - a) * u
def ease(u): u = min(max(u, 0), 1); return u * u * (3 - 2 * u)
def vadd(p, q): return (p[0] + q[0], p[1] + q[1])
def polar(deg, L):                 # 0°=向下，正角=朝画面右，90°=水平向右，180°=向上
    a = math.radians(deg); return (math.sin(a) * L, math.cos(a) * L)

# ---------- 分镜 ----------
SEGS = [(0, 3, "hook"), (3, 7, "jack"), (7, 11, "squat"), (11, 14, "knees"), (14, 16.01, "outro")]
def seg_of(t):
    for a, b, n in SEGS:
        if a <= t < b: return n, t - a, a, b
    return "outro", t - 14, 14, 16

# ---------- 姿势：返回关节字典 ----------
def pose(t):
    name, lt, a0, b0 = seg_of(t)
    P = dict(name=name, mood="smile", lean=0.0, lift=0.0, thumb=False)
    hip_y = GROUND - (THIGH + SHIN)          # 站直时髋高
    arms = None; legs = None
    st = lambda s: (s * (HIP_HALF), 0)
    if name == "hook":
        # 左手叉腰，右手挥手，轻微点头弹跳
        bob = 4 * math.sin(t * 2 * math.pi * 2)
        hip = (CX, hip_y + bob); sh_y = hip[1] - TORSO
        sh = {-1: (CX - SH_HALF, sh_y), 1: (CX + SH_HALF, sh_y)}
        wave = math.sin(t * 2 * math.pi * 3.2)
        el1 = vadd(sh[1], polar(62, UARM)); el1 = (el1[0], sh[1][1] - 20 + 0 * wave) if False else vadd(sh[1], polar(100, UARM))
        h1 = vadd(el1, polar(165 + 22 * wave, FARM))
        el_1 = vadd(sh[-1], polar(-52, UARM)); h_1 = vadd(el_1, polar(40, FARM))   # 叉腰
        arms = {-1: (el_1, h_1), 1: (el1, h1)}
        legs = stand(hip, 0)
        P["mood"] = "grin"
    elif name == "jack":
        p = lt % 1.0                                   # 周期 1s：0 并拢 / 0.5 张开
        u = ease(0.5 - 0.5 * math.cos(2 * math.pi * p))
        lift = 24 * abs(math.sin(2 * math.pi * p))
        la = lerp(2, 20, u)                            # 腿外展角
        arm = lerp(12, 168, u)
        hip_y2 = GROUND - lift - (THIGH + SHIN) * math.cos(math.radians(la))
        hip = (CX, hip_y2); sh_y = hip[1] - TORSO
        sh = {-1: (CX - SH_HALF, sh_y), 1: (CX + SH_HALF, sh_y)}
        arms = {}
        for s in (-1, 1):
            el = vadd(sh[s], (s * polar(arm, UARM)[0], polar(arm, UARM)[1]))
            hd = vadd(el, (s * polar(arm - 4, FARM)[0], polar(arm - 4, FARM)[1]))
            arms[s] = (el, hd)
        legs = {}
        for s in (-1, 1):
            hp = (CX + s * HIP_HALF, hip[1])
            kn = vadd(hp, (s * polar(la, THIGH)[0], polar(la, THIGH)[1]))
            an = vadd(kn, (s * polar(la, SHIN)[0], polar(la, SHIN)[1]))
            legs[s] = (hp, kn, an, 0)
        P.update(lift=lift, mood="grin" if u > .5 else "smile")
    elif name == "squat":
        p = (lt % 2.0) / 2.0                           # 周期 2s：0 站 / 0.5 蹲到底
        d = ease(0.5 - 0.5 * math.cos(2 * math.pi * p))
        hh = lerp(THIGH + SHIN, 196, d)                # 髋高
        hip = (CX, GROUND - hh)
        sh_y = hip[1] - lerp(TORSO, TORSO * 0.88, d)   # 前倾→躯干透视缩短
        sh = {-1: (CX - SH_HALF, sh_y), 1: (CX + SH_HALF, sh_y)}
        legs = {}
        for s in (-1, 1):
            hp = (CX + s * HIP_HALF, hip[1])
            an = (CX + s * lerp(44, 86, d), GROUND)
            kn = (CX + s * lerp(40, 104, d), lerp(hp[1] + 150, hp[1] + 0.40 * (GROUND - hp[1]), d))
            legs[s] = (hp, kn, an, s * 18 * d)         # 脚尖略外八
        # 手臂：站起时前平举(正面看=双手在胸前合十)，下蹲时保持
        arms = {}
        for s in (-1, 1):
            el = (CX + s * lerp(62, 56, d), sh_y + 62)
            hd = (CX + s * 10, sh_y + 52 + 12 * d)
            arms[s] = (el, hd)
        P.update(mood="effort" if d > .55 else "smile", lean=d)
    elif name == "knees":
        q = lt * 2.0                                   # 一拍 0.5s 换一条腿
        side = -1 if int(q) % 2 == 0 else 1            # 抬起的那条腿
        lf = math.sin(math.pi * (q % 1.0))
        lf = ease(lf) if True else lf
        bob = 10 * (1 - lf) - 4
        hip = (CX, GROUND - (THIGH + SHIN) - 8 + bob * 0.0 - 10 * lf)
        hip = (CX, hip[1] + 0)
        sh_y = hip[1] - TORSO; sh = {-1: (CX - SH_HALF, sh_y), 1: (CX + SH_HALF, sh_y)}
        legs = {}
        for s in (-1, 1):
            hp = (CX + s * HIP_HALF, hip[1]); l = lf if s == side else 0
            kn = (hp[0] + s * lerp(8, 34, l), hp[1] + lerp(150, 52, l))        # 大腿透视缩短、抬到近髋高
            an = (kn[0] - s * lerp(0, 10, l), kn[1] + lerp(150, 100, l))        # 小腿垂落
            if l == 0: an = (hp[0] + s * 12, GROUND)
            legs[s] = (hp, kn, an, 0)
            if l == 0: legs[s] = (hp, (hp[0] + s * 8, (hp[1] + GROUND) / 2 + 4), (hp[0] + s * 12, GROUND), 0)
        # 手臂：对侧摆臂，肘 90°，拳在胸前
        arms = {}
        for s in (-1, 1):
            up = lf if s == -side else 0
            up = up if s == -side else lf * 0 + (1 - lf if s == side else 0) * 0
            a = lf if s == -side else (1 - lf) * 0.0
            a = lf if s == -side else 0.0
            el = (CX + s * lerp(70, 74, a), sh_y + lerp(80, 62, a))
            hd = (CX + s * lerp(62, 44, a), sh_y + lerp(150, 6, a))
            arms[s] = (el, hd)
        P.update(mood="effort", lift=0)
    else:  # outro：点赞 + 微笑
        bob = 5 * abs(math.sin(lt * 2 * math.pi * 2))
        hip = (CX, hip_y + bob * 0.6); sh_y = hip[1] - TORSO
        sh = {-1: (CX - SH_HALF, sh_y), 1: (CX + SH_HALF, sh_y)}
        el1 = (CX + SH_HALF + 52, sh_y + 52); h1 = (CX + SH_HALF + 62, sh_y - 38)
        el0 = (CX - SH_HALF - 20, sh_y + 84); h0 = (CX - SH_HALF - 6, sh_y + 168)
        arms = {-1: (el0, h0), 1: (el1, h1)}
        legs = stand(hip, 0); P.update(mood="grin", thumb=True)
    P.update(hip=hip, sh=sh, arms=arms, legs=legs)
    P["head"] = (CX, sh[1][1] - 44 - HEAD_R + (10 * P["lean"]))
    return P

def stand(hip, spread):
    legs = {}
    for s in (-1, 1):
        hp = (CX + s * HIP_HALF, hip[1])
        legs[s] = (hp, (hp[0] + s * 4, (hp[1] + GROUND) / 2), (hp[0] + s * 8, GROUND), 0)
    return legs

# ---------- 角色绘制（SS 倍层）----------
def cap(d, p, q, w, fill):
    d.line([p, q], fill=fill, width=int(w))
    r = w / 2
    for c in (p, q): d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=fill)

def draw_coach(P, t, voice):
    ox, oy, lw, lh = LAY
    img = Image.new("RGBA", (lw * SS, lh * SS), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    X = lambda p: ((p[0] - ox) * SS, (p[1] - oy) * SS)
    cp = lambda a, b, w, f: cap(d, X(a), X(b), w * SS, f)
    hip, sh, head = P["hip"], P["sh"], P["head"]

    # 地面阴影
    sh_w = 190 - 1.0 * P["lift"]
    d.ellipse([*X((CX - sh_w, GROUND - 12)), *X((CX + sh_w, GROUND + 26))], fill=(10, 8, 30, 90))

    # 马尾（在身体后面）：随头部运动甩动
    prev = pose(max(t - 1 / FPS, 0))["head"]; vy = (head[1] - prev[1]) * FPS
    vx = (head[0] - prev[0]) * FPS
    sway = max(-1, min(1, -vy / 500)) * 40 + math.sin(t * 9) * 5
    base = (head[0] - 6, head[1] - HEAD_R * 0.55)
    pts = [base, (base[0] + 52, base[1] - 24), (base[0] + 78 + sway * 0.4, base[1] + 40), (base[0] + 80 + sway, base[1] + 118)]
    for i in range(3): cp(pts[i], pts[i + 1], 40 - i * 8, HAIR_D)
    d.ellipse([*X((pts[3][0] - 20, pts[3][1] - 14)), *X((pts[3][0] + 22, pts[3][1] + 32))], fill=HAIR_D)

    def leg(s, front):
        hp, kn, an, toe = P["legs"][s]
        cp(hp, kn, LIMB_W + 6, PANTS); cp(kn, an, LIMB_W, PANTS)
        # 鞋
        sx, sy = an
        d.ellipse([*X((sx - 30 + s * 6, sy - 16)), *X((sx + 30 + s * 6 + toe * 0.6, sy + 12))], fill=SHOE)
        d.ellipse([*X((sx - 30 + s * 6, sy - 2)), *X((sx + 30 + s * 6 + toe * 0.6, sy + 12))], fill=SOLE)
    for s in (-1, 1): leg(s, False)

    # 躯干：梯形 + 圆角
    sy = sh[1][1]; hy = hip[1]
    poly = [(CX - SH_HALF - 12, sy - 6), (CX + SH_HALF + 12, sy - 6), (CX + HIP_HALF + 16, hy + 10), (CX - HIP_HALF - 16, hy + 10)]
    d.polygon([X(p) for p in poly], fill=TOP)
    for p in poly[:2]: d.ellipse([*X((p[0] - 14, p[1] - 14)), *X((p[0] + 14, p[1] + 14))], fill=TOP)
    d.rectangle([*X((CX - HIP_HALF - 14, hy - 8)), *X((CX + HIP_HALF + 14, hy + 24))], fill=PANTS)   # 腰带
    d.ellipse([*X((CX - HIP_HALF - 16, hy - 2)), *X((CX + HIP_HALF + 16, hy + 40))], fill=PANTS)
    # 运动背心的斜纹 + AI 标志
    for k in (0.35, 0.6):
        yy = lerp(sy, hy, k); d.line([X((CX - 40, yy + 10)), X((CX + 40, yy - 10))], fill=TOP_D, width=int(7 * SS))
    f = ImageFont.truetype(ZH, 30 * SS); tx, ty = X((CX, lerp(sy, hy, 0.5)))
    d.text((tx, ty), "AI", font=f, fill=(255, 255, 255, 255), anchor="mm", stroke_width=2 * SS, stroke_fill=TOP_D)

    # 手臂
    for s in (-1, 1):
        el, hd = P["arms"][s]; shp = (sh[s][0] + s * 4, sh[s][1] + 2)
        cp(shp, el, LIMB_W - 2, SKIN); cp(el, hd, LIMB_W - 6, SKIN)
        mid = (lerp(shp[0], el[0], 0.5), lerp(shp[1], el[1], 0.5))
        cp(shp, mid, LIMB_W + 4, TOP)                      # 袖子
        d.ellipse([*X((hd[0] - 21, hd[1] - 21)), *X((hd[0] + 21, hd[1] + 21))], fill=SKIN)   # 拳/手
        if P["thumb"] and s == 1:
            d.rounded_rectangle([*X((hd[0] - 10, hd[1] - 46)), *X((hd[0] + 12, hd[1] - 6))], radius=10 * SS, fill=SKIN)
        # 腕带
        w0 = (lerp(el[0], hd[0], 0.82), lerp(el[1], hd[1], 0.82)); cp(w0, (lerp(el[0], hd[0], 0.9), lerp(el[1], hd[1], 0.9)), LIMB_W - 2, (255, 120, 80))
    # 脖子
    cp((head[0], head[1] + HEAD_R * 0.6), (head[0], sy - 6), 30, SKIN_D)

    # 头
    hx, hy2 = head
    d.ellipse([*X((hx - HEAD_R, hy2 - HEAD_R)), *X((hx + HEAD_R, hy2 + HEAD_R))], fill=SKIN)
    d.ellipse([*X((hx - HEAD_R - 3, hy2 - HEAD_R - 4)), *X((hx + HEAD_R + 3, hy2 + 6))], fill=HAIR)  # 头发盖
    d.pieslice([*X((hx - HEAD_R - 3, hy2 - HEAD_R - 4)), *X((hx + HEAD_R + 3, hy2 + 28))], 0, 180, fill=HAIR)
    d.ellipse([*X((hx - 50, hy2 - 10)), *X((hx + 50, hy2 + HEAD_R))], fill=SKIN)                      # 脸下部盖住发盖下半
    d.polygon([X((hx - HEAD_R, hy2 - 6)), X((hx - 8, hy2 - 30)), X((hx + 40, hy2 - 24)), X((hx + HEAD_R, hy2 - 4)), X((hx + 60, hy2 - 40)), X((hx - 60, hy2 - 40))], fill=HAIR)  # 刘海
    d.rectangle([*X((hx - HEAD_R - 2, hy2 - 38)), *X((hx + HEAD_R + 2, hy2 - 26))], fill=(255, 255, 255))                                  # 发带
    d.ellipse([*X((hx - HEAD_R - 2, hy2 - 38)), *X((hx - HEAD_R + 10, hy2 - 26))], fill=(255, 255, 255))
    # 耳机（AI 感）+ 发光天线点
    d.arc([*X((hx - HEAD_R - 4, hy2 - HEAD_R - 6)), *X((hx + HEAD_R + 4, hy2 + HEAD_R * 0.8))], 200, 340, fill=(60, 60, 90), width=int(7 * SS))
    for s in (-1, 1):
        d.ellipse([*X((hx + s * HEAD_R - 14, hy2 - 4)), *X((hx + s * HEAD_R + 14, hy2 + 26))], fill=(60, 60, 90))
        d.ellipse([*X((hx + s * HEAD_R - 6, hy2 + 4)), *X((hx + s * HEAD_R + 6, hy2 + 16))], fill=(90, 255, 230))
    # 脸
    blink = (t % 3.2) < 0.12
    ey = hy2 + 8
    for s in (-1, 1):
        ex = hx + s * 24
        if P["mood"] == "effort" or blink:
            d.arc([*X((ex - 11, ey - 4)), *X((ex + 11, ey + 14))], 200, 340, fill=INK, width=int(5 * SS))
        else:
            d.ellipse([*X((ex - 10, ey - 12)), *X((ex + 10, ey + 14))], fill=INK)
            d.ellipse([*X((ex - 4, ey - 8)), *X((ex + 3, ey - 1))], fill=(255, 255, 255))
        d.ellipse([*X((ex - 20, ey + 12)), *X((ex + 4 if s < 0 else ex + 20, ey + 26))], fill=(255, 150, 150, 120)) if False else None
        d.ellipse([*X((ex + s * 6 - 14, ey + 16)), *X((ex + s * 6 + 14, ey + 30))], fill=(255, 160, 160, 140))
        d.line([X((ex - 12, ey - 24 + (4 if s == 1 else 0))), X((ex + 12, ey - 26 - (4 if s == 1 else 0) + (4 if s == -1 else 0)))], fill=HAIR_D, width=int(5 * SS))
    my = hy2 + 34
    mo = 1 if voice > 0.05 else 0
    if mo:
        o = int(8 + min(voice * 25, 9))
        d.ellipse([*X((hx - 11, my - 3)), *X((hx + 11, my - 3 + o))], fill=(180, 60, 80)); 
    elif P["mood"] == "effort":
        d.rounded_rectangle([*X((hx - 14, my)), *X((hx + 14, my + 12))], radius=6 * SS, fill=(180, 60, 80))
    else:
        d.arc([*X((hx - 16, my - 10)), *X((hx + 16, my + 8))], 20, 160, fill=(180, 60, 80), width=int(5 * SS))
    img = img.resize((lw, lh), Image.LANCZOS)
    return img

# ---------- 背景 ----------
_bg = None
def background():
    global _bg
    if _bg is not None: return _bg.copy()
    y = np.linspace(0, 1, H)[:, None, None]
    top, mid, bot = np.array([34, 24, 84]), np.array([92, 48, 140]), np.array([255, 120, 90])
    c = np.where(y < 0.55, top + (mid - top) * (y / 0.55), mid + (bot - mid) * np.clip((y - 0.55) / 0.45, 0, 1) ** 1.6)
    arr = np.repeat(c, W, axis=1).astype(np.uint8)
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im, "RGBA")
    rng = np.random.default_rng(7)
    for _ in range(26):                       # 柔和圆点装饰
        x, yy, r = rng.integers(0, W), rng.integers(0, 1500), rng.integers(8, 34)
        d.ellipse([x - r, yy - r, x + r, yy + r], fill=(255, 255, 255, int(rng.integers(14, 40))))
    # 地板 + 瑜伽垫
    d.rectangle([0, GROUND - 20, W, H], fill=(48, 28, 84, 255))
    d.polygon([(0, GROUND - 20), (W, GROUND - 20), (W, GROUND - 8), (0, GROUND - 8)], fill=(255, 255, 255, 40))
    d.rounded_rectangle([CX - 330, GROUND - 22, CX + 330, GROUND + 62], radius=22, fill=(24, 190, 170, 255))
    d.rounded_rectangle([CX - 330, GROUND + 40, CX + 330, GROUND + 62], radius=12, fill=(14, 140, 130, 255))
    _bg = im.convert("RGB"); return _bg.copy()

# ---------- 文字 ----------
def font(sz): return ImageFont.truetype(ZH, sz)
def stext(dr, xy, s, sz, fill, stroke=6, sc=INK, anchor="mm"):
    dr.text(xy, s, font=font(sz), fill=fill, anchor=anchor, stroke_width=stroke, stroke_fill=sc)

def pop(t0, t, dur=0.28):      # 弹出缩放 0.6→1.12→1
    u = (t - t0) / dur
    if u < 0: return 0.01
    if u >= 1: return 1.0
    return 0.6 + 0.52 * math.sin(u * math.pi * 0.62) ** 1 if u < 0.7 else lerp(1.12, 1.0, (u - 0.7) / 0.3)

def overlay(t, name, lt, timings):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
    beat = (t % 0.5) / 0.5

    # 标题区（安全区 y≈230–430）
    titles = {"hook": ("16秒挑战", "你敢接吗？"), "jack": ("① 开合跳", ""), "squat": ("② 深蹲", ""),
              "knees": ("③ 高抬腿", ""), "outro": ("量力而行", "不舒服就停一停")}
    big, small = titles[name]
    a0 = {n: a for a, b, n in SEGS}[name]
    sc = pop(a0, t)
    if name == "hook":
        # 前 3 秒强钩子：超大字 + 闪烁箭头
        sz = int(190 * sc)
        stext(dr, (CX, 285), big, sz, (255, 235, 90), stroke=12)
        stext(dr, (CX, 408), small, int(70 * sc), (255, 255, 255), stroke=8)
    else:
        sz = int(124 * sc)
        stext(dr, (CX, 300), big, sz, (255, 235, 90) if name != "outro" else (120, 255, 225), stroke=10)
        if small: stext(dr, (CX, 400), small, int(58 * sc), (255, 255, 255), stroke=7)

    # 进度条 y≈ 470
    x0, x1, yb = 130, 830, 480
    dr.rounded_rectangle([x0, yb - 11, x1, yb + 11], radius=11, fill=(255, 255, 255, 70))
    dr.rounded_rectangle([x0, yb - 11, x0 + (x1 - x0) * t / DUR, yb + 11], radius=11, fill=(255, 235, 90, 255))
    for a, b, n in SEGS[:-1]:
        xx = x0 + (x1 - x0) * b / DUR; dr.ellipse([xx - 7, yb - 7, xx + 7, yb + 7], fill=(255, 255, 255, 230))

    # 倒计时圈（左侧，右侧留给抖音按钮）
    rem = max(1, math.ceil(DUR - t - 1e-6))
    cxm, cym, R = 150, 640, 74 + int(6 * (1 - beat))
    dr.ellipse([cxm - R, cym - R, cxm + R, cym + R], fill=(30, 20, 70, 190), outline=(255, 235, 90, 255), width=7)
    stext(dr, (cxm, cym - 6), str(rem), 86, (255, 255, 255), stroke=5, sc=(30, 20, 70))
    stext(dr, (cxm, cym + 50), "秒", 28, (255, 235, 90), stroke=3, sc=(30, 20, 70))

    # 次数计数
    cnt = None
    if name == "jack": cnt = (int(lt) + 1, 4)
    if name == "squat": cnt = (int(lt // 2) + 1, 2)
    if name == "knees": cnt = (min(int(lt * 2) + 1, 6), 6)
    if cnt:
        dr.rounded_rectangle([CX + 150, 590, CX + 390, 690], radius=50, fill=(30, 20, 70, 190), outline=(255, 235, 90, 255), width=5) if False else None
        stext(dr, (880, 640), f"{cnt[0]}/{cnt[1]}", 76, (255, 255, 255), stroke=6, sc=(30, 20, 70))

    # 角色标签
    dr.rounded_rectangle([60, 1236, 360, 1300], radius=32, fill=(30, 20, 70, 210), outline=(120, 255, 225, 255), width=3)
    stext(dr, (210, 1268), "AI虚拟教练·小燃", 30, (255, 255, 255), stroke=0)

    # 字幕 y≈1340（避开底部 25% 文案区）
    cur = [x for x in timings if x["start"] - 0.05 <= t <= x["end"] + 0.35]
    if cur:
        stext(dr, (CX, 1372), cur[0]["text"], 56, (255, 255, 255), stroke=8)
    # 收尾互动提示
    if name == "outro":
        bob = 8 * math.sin(t * 8)
        dr.rounded_rectangle([200, 1020 + 0, 880, 1100], radius=40, fill=(0, 0, 0, 0)) if False else None
        stext(dr, (CX, 1450 + bob), "评论区打卡  你做了几个？", 46, (255, 235, 90), stroke=7) if False else None
    return ov

def beat_ring(t, im):
    """脚下随节拍扩散的光环"""
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
    b = (t % 0.5) / 0.5; r = 230 + 160 * b; al = int(110 * (1 - b))
    cy = 880
    dr.ellipse([CX - r, cy - r, CX + r, cy + r], outline=(255, 255, 255, al), width=10)
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0)); gd = ImageDraw.Draw(glow)
    gd.ellipse([CX - 300, cy - 300, CX + 300, cy + 300], fill=(255, 190, 120, int(46 + 30 * (1 - b))))
    glow = glow.filter(ImageFilter.GaussianBlur(70))
    im.alpha_composite(glow); im.alpha_composite(ov)

_voice_env = None
def frame(t, timings):
    global _voice_env
    if _voice_env is None:
        p = os.path.join(TMP, "voice_env.npy"); _voice_env = np.load(p) if os.path.exists(p) else np.zeros(1)
    v = float(_voice_env[min(int(t * 44100), len(_voice_env) - 1)])
    P = pose(t); name, lt, a0, b0 = seg_of(t)
    im = background().convert("RGBA")
    beat_ring(t, im)
    layer = draw_coach(P, t, v); im.alpha_composite(layer, (LAY[0], LAY[1]))
    im.alpha_composite(overlay(t, name, lt, timings))
    return im.convert("RGB")

# ---------- 入口 ----------
def load_timings():
    p = os.path.join(TMP, "timings.json"); return json.load(open(p)) if os.path.exists(p) else []

def do_stills(ts):
    out = os.path.join(HERE, "preview", "stills"); os.makedirs(out, exist_ok=True)
    tim = load_timings(); ims = []
    for t in ts:
        im = frame(t, tim); im.save(os.path.join(out, f"t{t:05.2f}.png")); ims.append(im)
    cols = min(4, len(ims)); rows = math.ceil(len(ims) / cols)
    sw, sh = 405, 720; sheet = Image.new("RGB", (cols * sw, rows * sh), (0, 0, 0))
    for i, im in enumerate(ims): sheet.paste(im.resize((sw, sh), Image.LANCZOS), ((i % cols) * sw, (i // cols) * sh))
    sheet.save(os.path.join(HERE, "preview", "sheet.png")); print("saved", len(ims))

def render_chunk(args):
    k, a, b = args; tim = load_timings()
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", os.path.join(TMP, f"seg{k}.mp4")], stdin=subprocess.PIPE)
    for i in range(a, b): p.stdin.write(frame(i / FPS, tim).tobytes())
    p.stdin.close(); p.wait(); return k

def do_full():
    n = 4; cuts = [NF * i // n for i in range(n + 1)]
    with mp.Pool(n) as pool: pool.map(render_chunk, [(k, cuts[k], cuts[k + 1]) for k in range(n)])
    open(os.path.join(TMP, "segs.txt"), "w").write("".join(f"file 'seg{k}.mp4'\n" for k in range(n)))
    out = os.path.join(HERE, "xiaoran_16s.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", os.path.join(TMP, "segs.txt"), "-i", os.path.join(TMP, "audio.wav"),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-t", str(DUR), "-movflags", "+faststart", out], check=True)
    print("done", out)

if __name__ == "__main__":
    if "--still" in sys.argv:
        do_stills([float(x) for x in sys.argv[sys.argv.index("--still") + 1:]])
    else:
        do_full()
