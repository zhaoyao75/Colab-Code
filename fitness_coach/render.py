"""
《小燃 16 秒居家挑战》竖屏 1080x1920 · 30fps · 16 秒 —— 斜侧面机位版。
角色是 3D 骨骼（关节三维坐标 + 两节骨 IK），按斜侧面（YAW）正交投影到屏幕，再按深度排序画粗线胶囊/多边形，
这样深蹲、高抬腿的透视缩短自动正确。风格：卡通插画，黑色齐刘海波波头、奶油色运动装、金色麦田蓝天（参考图的配色与造型，非真人）。
时间轴（120BPM，一拍 0.5s）：0-3 钩子 | 3-7 开合跳×4 | 7-11 深蹲×2 | 11-14 高抬腿×6 | 14-16 收尾打卡

运行：python3 render.py --still 1 3.5 8 11.25 15   -> preview/stills/*.png + 联系图 preview/sheet.png
     python3 render.py                              -> xiaoran_16s.mp4（需先 python3 make_audio.py）
"""
import json, math, os, subprocess, sys
import multiprocessing as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__)); TMP = os.path.join(HERE, ".tmp")
W, H, FPS, DUR = 1080, 1920, 30, 16.0
NF = int(FPS * DUR)
ZH = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
GROUND, CX = 1250, 520
K = 1.12                          # 角色整体放大（以脚底为轴）
SS = 2
LAY = (0, 430, W, 900)
YAW = math.radians(38)             # 机位：角色朝向画面右前方 38°
CY_, SY_ = math.cos(YAW), math.sin(YAW)

# ---------- 配色 ----------
SKIN, SKIN_D = (255, 218, 192), (238, 184, 154)
HAIR, HAIR_L = (38, 30, 48), (88, 76, 110)
TOP, TOP_D = (252, 240, 212), (226, 205, 165)
PANTS, PANTS_D = (104, 160, 212), (70, 118, 176)
SHOE, SHOE_A = (255, 255, 255), (255, 140, 110)
INK = (58, 34, 40)
AICOL = (40, 200, 190)

# ---------- 身体尺寸 ----------
HEAD_R, TORSO, SH, HIPW = 62, 160, 46, 28
UARM, FARM, THIGH, SHIN = 96, 92, 150, 148
LEG = THIGH + SHIN; ANK = 16

def V(*a): return np.array(a, float)
def lerp(a, b, u): return a + (b - a) * u
def ease(u): u = min(max(u, 0), 1); return u * u * (3 - 2 * u)
def pr(p): return (CX + p[0] * CY_ + p[2] * SY_, GROUND - p[1])      # 投影（x 角色左为正，z 向前，y 向上）
def dp(p): return -p[0] * SY_ + p[2] * CY_                           # 朝向观众的深度

def limb(s, a, f):                  # 向下为 0；a 外展、f 前屈（度）
    a, f = math.radians(a), math.radians(f)
    return V(s * math.sin(a), -math.cos(a) * math.cos(f), math.cos(a) * math.sin(f))

def ik(a, t, L1, L2, pole):
    d = t - a; dist = float(np.linalg.norm(d)); dc = min(max(dist, 1e-3), (L1 + L2) * 0.998)
    n = d / max(dist, 1e-9); x = (L1 * L1 - L2 * L2 + dc * dc) / (2 * dc); r = math.sqrt(max(L1 * L1 - x * x, 0))
    pv = pole - n * float(np.dot(pole, n)); m = float(np.linalg.norm(pv)); pv = pv / m if m > 1e-6 else V(0, 0, 1)
    return a + n * x + pv * r, a + n * dc

# ---------- 分镜 ----------
SEGS = [(0, 3, "hook"), (3, 7, "jack"), (7, 11, "squat"), (11, 14, "knees"), (14, 16.01, "outro")]
def seg_of(t):
    for a, b, n in SEGS:
        if a <= t < b: return n, t - a
    return "outro", t - 14

def body(py, pz=0.0, lean=0.0):
    P = V(0, py, pz); l = math.radians(lean); u = V(0, math.cos(l), math.sin(l))
    C = P + u * TORSO; hd = V(0, math.cos(l * 0.4), math.sin(l * 0.4))
    return dict(P=P, C=C, sh={s: C + V(s * SH, 0, 0) for s in (-1, 1)}, head=C + hd * (HEAD_R + 24), u=u)

def pose(t):
    name, lt = seg_of(t)
    mood, thumb, arms, legs = "smile", False, {}, {}
    if name == "hook":
        b = body(ANK + LEG - 6 + 4 * math.sin(t * 4 * math.pi), 0, 2)
        for s in (-1, 1):
            legs[s] = ik(b["P"] + V(s * HIPW, 0, 0), V(s * 40, ANK, 0), THIGH, SHIN, V(s * .3, 0, 1)) + (V(s * 40, ANK, 0),)
        w = math.sin(t * 2 * math.pi * 3.2)
        sh_n, sh_f = b["sh"][-1], b["sh"][1]
        e, h = ik(sh_n, sh_n + V(-62 + 40 * w, 150, 30), UARM, FARM, V(-1, -.5, -.2)); arms[-1] = (sh_n, e, h)
        e, h = ik(sh_f, b["P"] + V(HIPW + 34, 22, 14), UARM, FARM, V(1, -.1, -.9)); arms[1] = (sh_f, e, h)
        mood = "grin"
    elif name == "jack":
        p = lt % 1.0; u = ease(0.5 - 0.5 * math.cos(2 * math.pi * p)); lift = 24 * abs(math.sin(2 * math.pi * p))
        la = math.radians(lerp(2, 21, u)); b = body(ANK + lift + LEG * math.cos(la) * .995, 0, 0)
        aa = lerp(12, 160, u)
        for s in (-1, 1):
            sh = b["sh"][s]; tgt = sh + limb(s, aa, 0) * (UARM + FARM) * .985
            e, h = ik(sh, tgt, UARM, FARM, V(0, 0, -1)); arms[s] = (sh, e, h)
            ank = V(s * (HIPW + LEG * math.sin(la)), ANK + lift, 0)
            legs[s] = ik(b["P"] + V(s * HIPW, 0, 0), ank, THIGH, SHIN, V(s * .3, 0, 1)) + (ank,)
        mood = "grin" if u > .5 else "smile"
    elif name == "squat":
        p = (lt % 2.0) / 2.0; d = ease(0.5 - 0.5 * math.cos(2 * math.pi * p))
        b = body(lerp(ANK + LEG - 4, 214, d), lerp(0, -92, d), lerp(2, 36, d))
        for s in (-1, 1):
            ank = V(s * lerp(46, 58, d), ANK, 0)
            legs[s] = ik(b["P"] + V(s * HIPW, 0, 0), ank, THIGH, SHIN, V(s * .25 * d, 0, 1)) + (ank,)
            sh = b["sh"][s]; tgt = sh + V(-s * 12, -10 + 8 * d, 172 - 6 * d)
            e, h = ik(sh, tgt, UARM, FARM, V(s * .5, -1, 0)); arms[s] = (sh, e, h)
        mood = "effort" if d > .5 else "smile"
    elif name == "knees":
        q = lt * 2.0; side = -1 if int(q) % 2 == 0 else 1; lf = ease(math.sin(math.pi * (q % 1.0)))
        b = body(ANK + LEG + 6 * lf, 0, 6)
        for s in (-1, 1):
            hp = b["P"] + V(s * HIPW, 0, 0)
            if s == side:
                kn = hp + V(s * 3, 0, 0) + limb(0, 0, lerp(0, 82, lf)) * THIGH
                an = kn + limb(0, 0, lerp(0, -8, lf)) * SHIN
            else:
                kn = hp + V(0, -THIGH, 0) + V(s * 2, 0, 5 * lf); an = hp + V(0, -LEG, 0)
            legs[s] = (kn, an, an)
        for s in (-1, 1):
            sh = b["sh"][s]; a_ = lf if s == -side else 1 - lf if s == side else 0
            fu = lerp(-28, 48, a_); uvec = limb(s, 10, fu); el = sh + uvec * UARM
            hand = el + limb(s, 8, fu + 88) * FARM; arms[s] = (sh, el, hand)
        mood = "effort"
    else:
        b = body(ANK + LEG - 5 + 3 * abs(math.sin(lt * 4 * math.pi)), 0, 2)
        for s in (-1, 1):
            legs[s] = ik(b["P"] + V(s * HIPW, 0, 0), V(s * 40, ANK, 0), THIGH, SHIN, V(s * .3, 0, 1)) + (V(s * 40, ANK, 0),)
        sh = b["sh"][-1]; el = sh + limb(-1, 34, 20) * UARM; arms[-1] = (sh, el, el + limb(-1, 6, 172) * FARM)
        sh = b["sh"][1]; el = sh + limb(1, 12, 8) * UARM; arms[1] = (sh, el, el + limb(1, 8, 28) * FARM)
        mood, thumb = "grin", True
    return dict(name=name, b=b, arms=arms, legs=legs, mood=mood, thumb=thumb)

# ---------- 绘制 ----------
def shade(c, f): return tuple(int(v * f) for v in c)

def hull(pts):
    pts = sorted(set(map(tuple, pts)))
    cr = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]

def draw_coach(P, t, voice):
    ox, oy, lw, lh = LAY; S2 = SS * K
    img = Image.new("RGBA", (lw * SS, lh * SS), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    X = lambda p: ((CX + (p[0] - CX) * K - ox) * SS, (GROUND + (p[1] - GROUND) * K - oy) * SS)
    def cap(p3, q3, w, col):
        p, q = X(pr(p3)), X(pr(q3)); d.line([p, q], fill=col, width=int(w * S2)); r = w * S2 / 2
        for c in (p, q): d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=col)
    def disc(p3, r, col, off=(0, 0)):
        c = X(pr(p3)); c = (c[0] + off[0] * S2, c[1] + off[1] * S2); r *= S2; d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=col)
    def poly(pts3, col, round_=12):
        pts = [X(pr(p)) for p in pts3]; pts = hull(pts)
        d.polygon(pts, fill=col); d.line(pts + [pts[0]], fill=col, width=int(round_ * S2), joint="curve")
        for p in pts: d.ellipse([p[0] - round_ * S2 / 2, p[1] - round_ * S2 / 2, p[0] + round_ * S2 / 2, p[1] + round_ * S2 / 2], fill=col)

    b = P["b"]; prims = []
    far = lambda dep: 0.9 if dep < -6 else 1.0
    add = lambda dep, fn: prims.append((dep, len(prims), fn))

    for s in (-1, 1):                                         # 腿 + 鞋
        hp = b["P"] + V(s * HIPW, 0, 0); kn_pt, an_pt = P["legs"][s][0], P["legs"][s][1]
        f = far(dp(hp))
        add(dp(hp + kn_pt) / 2 - 1, lambda hp=hp, kn_pt=kn_pt, f=f: cap(hp, kn_pt, 46, shade(PANTS, f)))
        add(dp(kn_pt + an_pt) / 2 - 1, lambda kn_pt=kn_pt, an_pt=an_pt, f=f: (cap(kn_pt, an_pt, 37, shade(PANTS, f)), disc(kn_pt, 21, shade(PANTS, f))))
        heel = an_pt + V(0, -8, -14); toe = an_pt + V(s * 6, -8, 36)
        add(dp(an_pt) + 1, lambda heel=heel, toe=toe, f=f: (cap(heel, toe, 28, shade(SHOE, f)), disc(toe, 10, shade(SHOE_A, f), (0, 2))))

    def torso():
        C, Pp = b["C"], b["P"]; mid = Pp + b["u"] * TORSO * 0.5
        pts = [C + V(sx * (SH + 2), 10, sz * 21) for sx in (-1, 1) for sz in (-1, 1)]
        pts += [mid + V(sx * SH * .66, 0, sz * 18) for sx in (-1, 1) for sz in (-1, 1)]
        pts += [Pp + V(sx * (HIPW + 10), 6, sz * 22) for sx in (-1, 1) for sz in (-1, 1)]
        poly(pts, TOP, 14)
        for k in (0.38, 0.62):                                  # 运动背心纹
            q = Pp + b["u"] * TORSO * k; cap(q + V(-SH * .6, -5, 20), q + V(SH * .6, 9, 20), 6, TOP_D)
        poly([Pp + V(sx * (HIPW + 13), dy, sz * 23) for sx in (-1, 1) for sz in (-1, 1) for dy in (-24, 8)], PANTS, 26)
        c = X(pr(b["C"] * .45 + Pp * .55 + V(0, 0, 22)))
        d.text(c, "AI", font=ImageFont.truetype(ZH, 28 * S2), fill=(255, 255, 255, 255), anchor="mm", stroke_width=2 * S2, stroke_fill=AICOL)
    add(0, torso)

    def neck(): cap(b["C"] + V(0, 6, 4), b["head"] + V(0, -HEAD_R + 10, 0), 28, SKIN_D)
    add(1, neck)

    for s in (-1, 1):                                          # 手臂
        sh, el, hd = P["arms"][s]; f = far(dp(sh)); dep = dp(sh + el) / 2
        add(dep, lambda sh=sh, el=el, f=f: cap(sh, el, 34, shade(SKIN, f)))
        add(dp(el + hd) / 2, lambda el=el, hd=hd, f=f, s=s: (cap(el, hd, 30, shade(SKIN, f)),
                                                             cap(el + (hd - el) * .72, el + (hd - el) * .84, 29, shade(AICOL, f))))
        def hand(hd=hd, f=f, s=s):
            disc(hd, 17, shade(SKIN, f))
            if P["thumb"] and s == -1:
                c = X(pr(hd)); d.rounded_rectangle([c[0] - 8 * S2, c[1] - 40 * S2, c[0] + 10 * S2, c[1] - 6 * S2], radius=9 * S2, fill=SKIN)
        add(dp(hd) + .5, hand)
        add(dp(sh) + .6, lambda sh=sh, el=el, f=f: (cap(sh, sh + (el - sh) * .42, 40, shade(TOP, f)), disc(sh, 28, shade(TOP, f)),
                                                    cap(sh + (el - sh) * .36, sh + (el - sh) * .46, 42, shade(TOP_D, f))))

    def head():
        hc = pr(b["head"]); R = HEAD_R; cx, cy = X(hc); rr = lambda v: v * S2; wi = lambda v: max(1, int(round(v * S2)))
        # 波波头后发（偏向背面）
        d.rounded_rectangle([cx - rr(1.12 * R), cy - rr(1.06 * R), cx + rr(0.96 * R), cy + rr(0.66 * R)], radius=rr(0.62 * R), fill=HAIR)
        # 脸
        fx, fy = cx + rr(0.12 * R), cy + rr(0.06 * R)
        d.ellipse([fx - rr(0.92 * R), fy - rr(0.94 * R), fx + rr(0.92 * R), fy + rr(0.98 * R)], fill=SKIN)
        # 近侧耳朵 + 头戴耳机（AI 标识）
        ex_, ey_ = cx - rr(0.74 * R), cy + rr(0.1 * R)
        # 眼睛等面部特征：球面上的点按 YAW 投影
        def face(ex, ey, rad=0.95):
            ez = math.sqrt(max(rad * rad - ex * ex - ey * ey, 0.01))
            return cx + rr((ex * CY_ + ez * SY_) * R), cy + rr(-ey * R), max((-ex * SY_ + ez * CY_), 0.18)
        # 刘海（齐刘海，带轻微弧度）
        top = [(cx + rr(1.03 * R * math.cos(math.radians(a))), cy - rr(0.12 * R) - rr(1.0 * R * math.sin(math.radians(a)))) for a in range(0, 181, 9)]
        bot = [(cx - rr(1.02 * R) + rr(2.04 * R) * k / 10, cy - rr(0.30 * R) + rr(0.07 * R) * math.sin(k * 1.3)) for k in range(0, 11)]
        d.polygon(top + bot[::-1], fill=HAIR)
        for k in range(3): d.line([(cx - rr(0.5 * R) + rr(0.32 * R) * k, cy - rr(0.95 * R) + rr(0.1 * R) * k), (cx - rr(0.3 * R) + rr(0.34 * R) * k, cy - rr(0.46 * R))], fill=HAIR_L, width=wi(3))
        # 近侧发帘遮耳
        d.ellipse([cx - rr(1.0 * R), cy - rr(0.5 * R), cx - rr(0.34 * R), cy + rr(0.62 * R)], fill=HAIR)
        # 耳机
        d.ellipse([ex_ - rr(14), ey_ - rr(14), ex_ + rr(14), ey_ + rr(14)], fill=(70, 72, 100))
        d.ellipse([ex_ - rr(7), ey_ - rr(7), ex_ + rr(7), ey_ + rr(7)], fill=AICOL)
        mx, my, mdep = face(0, -0.52)
        d.line([(ex_, ey_ + rr(10)), (cx - rr(0.34 * R), cy + rr(0.78 * R)), (mx - rr(0.3 * R), my + rr(0.14 * R))], fill=(70, 72, 100), width=wi(4), joint="curve")
        d.ellipse([mx - rr(0.3 * R) - rr(6), my + rr(0.14 * R) - rr(6), mx - rr(0.3 * R) + rr(6), my + rr(0.14 * R) + rr(6)], fill=(70, 72, 100))
        # 发夹
        d.ellipse([cx - rr(0.66 * R) - rr(7), cy - rr(0.55 * R) - rr(7), cx - rr(0.66 * R) + rr(7), cy - rr(0.55 * R) + rr(7)], fill=(255, 170, 190))
        d.ellipse([cx - rr(0.66 * R) - rr(3), cy - rr(0.55 * R) - rr(3), cx - rr(0.66 * R) + rr(3), cy - rr(0.55 * R) + rr(3)], fill=(255, 240, 180))
        # 眉眼
        blink = (t % 3.2) < 0.12
        for s, ex in ((-1, -0.36), (1, 0.36)):
            x, y, dep = face(ex, -0.02); w = rr(11 * dep); hgt = rr(15)
            if P["mood"] == "effort" or blink:
                d.arc([x - w, y - rr(6), x + w, y + rr(10)], 200, 340, fill=INK, width=wi(4))
            else:
                d.ellipse([x - w, y - hgt, x + w, y + hgt], fill=INK)
                d.ellipse([x - w * .1, y - hgt * .7, x + w * .55, y - hgt * .1], fill=(255, 255, 255))
            bx, by, bdep = face(ex, 0.26)
            lift = -rr(4) if P["mood"] == "effort" else 0
            d.line([(bx - rr(13 * bdep), by + rr(3) + lift * (-s)), (bx + rr(13 * bdep), by - rr(3) + lift * s)], fill=HAIR, width=wi(4))
            cxk, cyk, cd = face(ex * 1.55, -0.24); d.ellipse([cxk - rr(14 * cd), cyk - rr(8), cxk + rr(14 * cd), cyk + rr(8)], fill=(255, 160, 160, 130))
        nx, ny, nd = face(0, -0.18, 1.02); d.ellipse([nx - rr(6), ny - rr(5), nx + rr(5), ny + rr(5)], fill=SKIN_D)
        mx, my, md = face(0, -0.5); mw = rr(17 * md)
        if voice > 0.05:
            o = rr(7 + min(voice * 22, 8)); d.ellipse([mx - mw * .7, my - rr(3), mx + mw * .7, my - rr(3) + o], fill=(176, 58, 78))
        elif P["mood"] == "effort":
            d.rounded_rectangle([mx - mw * .8, my, mx + mw * .8, my + rr(10)], radius=rr(5), fill=(176, 58, 78))
        else:
            d.arc([mx - mw, my - rr(12), mx + mw, my + rr(8)], 20, 160, fill=(176, 58, 78), width=wi(4))
    add(8, head)

    # 地面阴影
    sx_, sy_ = pr(V(0, 0, b["P"][2] * 0.6))
    d.ellipse([*X((sx_ - 175, GROUND - 10)), *X((sx_ + 175, GROUND + 30))], fill=(70, 40, 10, 80))
    for dep, _, fn in sorted(prims, key=lambda z: (z[0], z[1])): fn()
    return img.resize((lw, lh), Image.LANCZOS)

# ---------- 背景：金色麦田 + 蓝天 ----------
_bg = None
def background():
    global _bg
    if _bg is not None: return _bg.copy()
    y = np.linspace(0, 1, H)[:, None, None]; hz = 1010 / H
    sky_top, sky_mid, haze = np.array([86, 156, 218]), np.array([170, 212, 238]), np.array([252, 232, 190])
    f_top, f_bot = np.array([232, 184, 84]), np.array([168, 112, 36])
    sky = np.where(y < hz * .7, sky_top + (sky_mid - sky_top) * (y / (hz * .7)), sky_mid + (haze - sky_mid) * np.clip((y - hz * .7) / (hz * .3), 0, 1) ** 1.5)
    fld = f_top + (f_bot - f_top) * np.clip((y - hz) / (1 - hz), 0, 1) ** .8
    im = Image.fromarray(np.repeat(np.where(y < hz, sky, fld), W, axis=1).astype(np.uint8)).convert("RGBA")
    # 太阳光晕 + 云
    g = Image.new("RGBA", (W, H), (255, 244, 205, 0)); gd = ImageDraw.Draw(g)
    gd.ellipse([660, 600, 1200, 1100], fill=(255, 244, 205, 120))
    im.alpha_composite(g.filter(ImageFilter.GaussianBlur(50)))
    d = ImageDraw.Draw(im, "RGBA"); rng = np.random.default_rng(11)
    for _ in range(3400):                                      # 麦穗/麦秆笔触：越近越长越粗
        yy = rng.uniform(1010, H); k = (yy - 1010) / (H - 1010); L = 14 + 90 * k; x = rng.uniform(-20, W + 20)
        c = (int(lerp(240, 150, rng.random() * .8 + k * .2)), int(lerp(200, 100, rng.random() * .8 + k * .2)), int(lerp(110, 30, rng.random())), int(110 + 80 * rng.random()))
        d.line([(x, yy), (x + rng.uniform(-6, 10) * (1 + k), yy - L)], fill=c, width=int(1 + 3 * k))
    # 奶油色瑜伽垫（透视椭圆）
    d.ellipse([CX - 330, GROUND - 40, CX + 330, GROUND + 52], fill=(250, 238, 210, 255)); d.ellipse([CX - 330, GROUND - 40, CX + 330, GROUND + 52], outline=(214, 190, 150, 255), width=5)
    _bg = im.convert("RGB"); return _bg.copy()

# ---------- 文字 / 版式 ----------
def font(sz): return ImageFont.truetype(ZH, sz)
def stext(dr, xy, s, sz, fill, stroke=6, sc=INK, anchor="mm"):
    dr.text(xy, s, font=font(sz), fill=fill, anchor=anchor, stroke_width=stroke, stroke_fill=sc)
def pop(t0, t, dur=0.28):
    u = (t - t0) / dur
    if u < 0: return 0.01
    if u >= 1: return 1.0
    return 0.6 + 0.52 * math.sin(u * math.pi * 0.62) if u < 0.7 else lerp(1.12, 1.0, (u - 0.7) / 0.3)
YEL = (255, 228, 84); PANEL = (52, 36, 70, 205)

def overlay(t, name, lt, timings):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dr = ImageDraw.Draw(ov); beat = (t % 0.5) / 0.5
    titles = {"hook": ("16秒挑战", "你敢接吗？"), "jack": ("① 开合跳", ""), "squat": ("② 深蹲", ""), "knees": ("③ 高抬腿", ""), "outro": ("量力而行", "不舒服就停一停")}
    big, small = titles[name]; a0 = {n: a for a, b, n in SEGS}[name]; sc = pop(a0, t)
    if name == "hook":
        stext(dr, (CX + 20, 285), big, int(190 * sc), YEL, stroke=12); stext(dr, (CX + 20, 408), small, int(70 * sc), (255, 255, 255), stroke=8)
    else:
        stext(dr, (CX + 20, 300), big, int(124 * sc), YEL if name != "outro" else (255, 255, 255), stroke=10)
        if small: stext(dr, (CX + 20, 400), small, int(58 * sc), (255, 255, 255), stroke=7)
    x0, x1, yb = 130, 830, 480
    dr.rounded_rectangle([x0, yb - 11, x1, yb + 11], radius=11, fill=(255, 255, 255, 110))
    dr.rounded_rectangle([x0, yb - 11, x0 + (x1 - x0) * t / DUR, yb + 11], radius=11, fill=YEL)
    for a, b, n in SEGS[:-1]:
        xx = x0 + (x1 - x0) * b / DUR; dr.ellipse([xx - 7, yb - 7, xx + 7, yb + 7], fill=(255, 255, 255, 240))
    rem = max(1, math.ceil(DUR - t - 1e-6)); cxm, cym, R = 150, 640, 74 + int(6 * (1 - beat))
    dr.ellipse([cxm - R, cym - R, cxm + R, cym + R], fill=PANEL, outline=YEL, width=7)
    stext(dr, (cxm, cym - 6), str(rem), 86, (255, 255, 255), stroke=5, sc=(40, 28, 56)); stext(dr, (cxm, cym + 50), "秒", 28, YEL, stroke=3, sc=(40, 28, 56))
    cnt = {"jack": (int(lt) + 1, 4), "squat": (int(lt // 2) + 1, 2), "knees": (min(int(lt * 2) + 1, 6), 6)}.get(name)
    if cnt: stext(dr, (880, 640), f"{cnt[0]}/{cnt[1]}", 76, (255, 255, 255), stroke=7)
    dr.rounded_rectangle([60, 1300, 360, 1364], radius=32, fill=PANEL, outline=(120, 255, 225, 255), width=3)
    stext(dr, (210, 1332), "AI虚拟教练·小燃", 30, (255, 255, 255), stroke=0)
    cur = [x for x in timings if x["start"] - 0.05 <= t <= x["end"] + 0.35]
    if cur: stext(dr, (CX + 20, 1412), cur[0]["text"], 56, (255, 255, 255), stroke=8)
    return ov

def sun_pulse(t, im):
    b = (t % 0.5) / 0.5; g = Image.new("RGBA", (W, H), (255, 244, 205, 0)); gd = ImageDraw.Draw(g)
    gd.ellipse([CX - 300, 640, CX + 300, 1240], fill=(255, 244, 205, int(52 + 34 * (1 - b)))); im.alpha_composite(g.filter(ImageFilter.GaussianBlur(60)))

_voice_env = None
def frame(t, timings):
    global _voice_env
    if _voice_env is None:
        p = os.path.join(TMP, "voice_env.npy"); _voice_env = np.load(p) if os.path.exists(p) else np.zeros(1)
    v = float(_voice_env[min(int(t * 44100), len(_voice_env) - 1)]); P = pose(t); name, lt = seg_of(t)
    im = background().convert("RGBA"); sun_pulse(t, im)
    im.alpha_composite(draw_coach(P, t, v), (LAY[0], LAY[1])); im.alpha_composite(overlay(t, name, lt, timings))
    return im.convert("RGB")

def load_timings():
    p = os.path.join(TMP, "timings.json"); return json.load(open(p)) if os.path.exists(p) else []

def do_stills(ts):
    out = os.path.join(HERE, "preview", "stills"); os.makedirs(out, exist_ok=True)
    for f in os.listdir(out): os.remove(os.path.join(out, f))
    tim = load_timings(); ims = []
    for t in ts:
        im = frame(t, tim); im.save(os.path.join(out, f"t{t:05.2f}.png")); ims.append(im)
    cols = min(4, len(ims)); rows = math.ceil(len(ims) / cols); sw, sh = 405, 720
    sheet = Image.new("RGB", (cols * sw, rows * sh), (0, 0, 0))
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
    if "--still" in sys.argv: do_stills([float(x) for x in sys.argv[sys.argv.index("--still") + 1:]])
    else: do_full()
