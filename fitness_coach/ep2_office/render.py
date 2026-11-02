"""
《小燃 · 办公室「不出汗」3 动作》第二支 16 秒竖屏 1080x1920 · 30fps（坐姿：耸肩 / 扭转 / 抬腿，办公室场景）。
角色是 3D 骨骼（关节三维坐标 + 两节骨 IK），按斜侧面（YAW）正交投影到屏幕，再按深度排序画粗线胶囊/多边形，
这样深蹲、高抬腿的透视缩短自动正确。风格：卡通插画，黑色齐刘海波波头、奶油色运动装、金色麦田蓝天（参考图的配色与造型，非真人）。
时间轴（120BPM，一拍 0.5s）：0-3 钩子（老板路过）| 3-7 耸肩×4 | 7-11 坐姿扭转×2 | 11-14 坐姿抬腿×6 | 14-16 收尾打卡

运行：python3 render.py --still 1 3.5 8 11.25 15   -> preview/stills/*.png + 联系图 preview/sheet.png
     python3 render.py                              -> xiaoran_office_16s.mp4（需先 python3 make_audio.py）
"""
import json, math, os, subprocess, sys
import multiprocessing as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__)); TMP = os.path.join(HERE, ".tmp")
W, H, FPS, DUR = 1080, 1920, 30, 16.0
NF = int(FPS * DUR)
ZH = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
GROUND, CX = 1320, 520
K = 2.2                           # 角色整体放大（以脚底为轴）
SS = 2
LAY = (0, 420, W, 930)
YAW = math.radians(38)             # 机位：角色朝向画面右前方 38°
CY_, SY_ = math.cos(YAW), math.sin(YAW)

# ---------- 配色（Q 版：平涂 + 深棕描边）----------
SKIN, SKIN_D = (255, 230, 134), (238, 188, 70)               # 手脚：与星星头同色系的奶黄
HEADC, HEADC_D = (255, 214, 72), (226, 164, 40)             # 星星头正面 / 侧边厚度
HAIR, HAIR_L = (112, 68, 52), (160, 108, 84)
TOP, TOP_D = (255, 133, 133), (226, 100, 108)         # 珊瑚粉运动上衣
PANTS, PANTS_D = (74, 96, 178), (54, 72, 146)         # 靛蓝运动短裤/紧身裤
SHOE, SHOE_A = (255, 255, 255), (255, 214, 64)
INK = (84, 52, 50)                                    # 描边 / 文字描边
OUT = INK
AICOL = (40, 200, 190)
STAR = (255, 222, 72)
OW = 3.4                                              # 描边宽度（未放大）

# ---------- 身体尺寸（Q 版：大头、短手脚）----------
HEAD_R, TORSO, SH, HIPW = 84, 86, 30, 20
UARM, FARM, THIGH, SHIN = 54, 50, 78, 74
LEG = THIGH + SHIN; ANK = 14

def V(*a): return np.array(a, float)
def lerp(a, b, u): return a + (b - a) * u
def ease(u): u = min(max(u, 0), 1); return u * u * (3 - 2 * u)
def pr(p): return (CX + p[0] * CY_ + p[2] * SY_, GROUND - p[1])
def dp(p): return -p[0] * SY_ + p[2] * CY_

def limb(s, a, f):
    a, f = math.radians(a), math.radians(f)
    return V(s * math.sin(a), -math.cos(a) * math.cos(f), math.cos(a) * math.sin(f))

def ik(a, t, L1, L2, pole):
    d = t - a; dist = float(np.linalg.norm(d)); dc = min(max(dist, 1e-3), (L1 + L2) * 0.998)
    n = d / max(dist, 1e-9); x = (L1 * L1 - L2 * L2 + dc * dc) / (2 * dc); r = math.sqrt(max(L1 * L1 - x * x, 0))
    pv = pole - n * float(np.dot(pole, n)); m = float(np.linalg.norm(pv)); pv = pv / m if m > 1e-6 else V(0, 0, 1)
    return a + n * x + pv * r, a + n * dc


# ---------- 分镜（坐姿）----------
SEAT, DESK_H = ANK + SHIN, 140          # 坐面高度（骨盆）/ 桌面高度
SEGS = [(0, 3, "hook"), (3, 7, "shrug"), (7, 11, "twist"), (11, 14, "legs"), (14, 16.01, "outro")]
def seg_of(t):
    for a, b, n in SEGS:
        if a <= t < b: return n, t - a
    return "outro", t - 14

def rot(v, th):                          # 绕竖直轴旋转（躯干扭转）
    c, s_ = math.cos(th), math.sin(th); return V(v[0] * c + v[2] * s_, v[1], -v[0] * s_ + v[2] * c)

def body(py, pz=0.0, lean=0.0, tw=0.0, sl=0.0):
    P = V(0, py, pz); l = math.radians(lean); u = V(0, math.cos(l), math.sin(l))
    C = P + u * TORSO; hd = V(0, math.cos(l * 0.4), math.sin(l * 0.4))
    return dict(P=P, C=C, tw=tw, sh={s: C + rot(V(s * SHD, 0, 0), tw) + V(0, sl, 0) for s in (-1, 1)}, head=C + hd * (HEAD_R + 2) - V(0, sl * .45, 0), u=u)
SHD = SH

def hip(b, s): return b["P"] + V(s * HIPW, 0, 0)
def sit_leg(b, s, lf=0.0):               # 坐姿腿：lf=0 垂脚，lf=1 伸直抬平
    a0, a1 = V(s * 24, ANK, 100), V(s * 22, SEAT + 8, 150); an = a0 + (a1 - a0) * lf
    return ik(hip(b, s), an, THIGH, SHIN, V(0, 1, .15)) + (an,)
def lap_arm(b, s, sh=None):              # 双手放腿上
    sh = b["sh"][s] if sh is None else sh; e, h = ik(sh, V(s * 30, SEAT + 16, 52), UARM, FARM, V(s * 1, -.3, -.5)); return (sh, e, h)

def pose(t):
    name, lt = seg_of(t)
    mood, thumb, arms, legs, turn = "smile", False, {}, {}, 0.0
    if name == "hook":                   # 敲键盘，老板从背后路过
        b = body(SEAT, 0, 3 + 1.2 * math.sin(t * 6))
        for s in (-1, 1):
            legs[s] = sit_leg(b, s)
            sh = b["sh"][s]; e, h = ik(sh, V(-10 if s == -1 else 34, DESK_H + 2 + 2.5 * math.sin(t * 17 + s * 1.5), 97), UARM, FARM, V(s * 1, -.8, -.3)); arms[s] = (sh, e, h)
        mood = "grin" if t > 2.7 else "smile"; turn = 0.3 * min(ease((t - 1.7) / .35), ease((2.9 - t) / .35))      # 1.7-2.9 秒侧头瞟老板
    elif name == "shrug":                # 耸肩 ×4（每次 1 秒）
        p = lt % 1.0; u = ease(math.sin(math.pi * p)); b = body(SEAT, 0, 2, 0, 20 * u)
        for s in (-1, 1): legs[s] = sit_leg(b, s); arms[s] = lap_arm(b, s)
        mood = "effort" if u > .6 else "grin"
    elif name == "twist":                # 坐姿扭转：2 秒一个来回，×2
        sn = math.sin(math.pi * lt); th = math.radians(42) * sn; b = body(SEAT, 0, 2, th)
        for s in (-1, 1):
            legs[s] = sit_leg(b, s); sh = b["sh"][s]
            e, h = ik(sh, b["C"] + rot(V(0, -26, 40), th), UARM, FARM, rot(V(s * 1, -.5, -.1), th)); arms[s] = (sh, e, h)
        mood = "effort" if abs(sn) > .5 else "smile"; turn = sn
    elif name == "legs":                 # 坐姿抬腿 ×6（每拍换腿）
        q = lt * 2.0; side = -1 if int(q) % 2 == 0 else 1; lf = ease(math.sin(math.pi * (q % 1.0))); b = body(SEAT, 0, 4)
        for s in (-1, 1):
            legs[s] = sit_leg(b, s, lf if s == side else 0.0)
            sh = b["sh"][s]; e, h = ik(sh, V(s * 36, SEAT + 20, 26), UARM, FARM, V(s * .9, -.3, -.7)); arms[s] = (sh, e, h)
        mood = "effort"
    else:                                # 收尾：点赞
        b = body(SEAT, 0, 2)
        for s in (-1, 1): legs[s] = sit_leg(b, s)
        sh = b["sh"][-1]; el = sh + limb(-1, 34, 20) * UARM; arms[-1] = (sh, el, el + limb(-1, 6, 172) * FARM)
        arms[1] = lap_arm(b, 1)
        mood, thumb = "grin", True
    return dict(name=name, b=b, arms=arms, legs=legs, mood=mood, thumb=thumb, turn=turn)

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

def star_pts(cx, cy, ro, ri, rot=-90):
    return [(cx + (ro if i % 2 == 0 else ri) * math.cos(math.radians(rot + 36 * i)), cy + (ro if i % 2 == 0 else ri) * math.sin(math.radians(rot + 36 * i))) for i in range(10)]

def draw_star(d, cx, cy, ro, ri, w, fill=STAR, rot=-90):
    pts = star_pts(cx, cy, ro, ri, rot)
    d.polygon(pts, fill=OUT); d.line(pts + [pts[0]], fill=OUT, width=w, joint="curve")
    for p in pts: d.ellipse([p[0] - w / 2, p[1] - w / 2, p[0] + w / 2, p[1] + w / 2], fill=OUT)
    inner = star_pts(cx, cy, ro - w * .7, ri - w * .4, rot); d.polygon(inner, fill=fill)

def draw_coach(P, t, voice):
    ox, oy, lw, lh = LAY; S2 = SS * K
    img = Image.new("RGBA", (lw * SS, lh * SS), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    X = lambda p: ((CX + (p[0] - CX) * K - ox) * SS, (GROUND + (p[1] - GROUND) * K - oy) * SS)
    rr = lambda v: v * S2; wi = lambda v: max(1, int(round(v * S2)))
    def cap(p3, q3, w, col):
        p, q = X(pr(p3)), X(pr(q3)); d.line([p, q], fill=col, width=wi(w)); r = w * S2 / 2
        for c in (p, q): d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=col)
    def group(segs, col):                                   # 先画整体描边再画填充：关节处没有接缝线
        for p3, q3, w in segs: cap(p3, q3, w + 2 * OW, OUT)
        for p3, q3, w in segs: cap(p3, q3, w, col)
    def disc(p3, r, col, off=(0, 0), outline=True):
        c = X(pr(p3)); c = (c[0] + off[0] * S2, c[1] + off[1] * S2)
        for rad, cc in (((r + OW) * S2, OUT), (r * S2, col)) if outline else ((r * S2, col),):
            d.ellipse([c[0] - rad, c[1] - rad, c[0] + rad, c[1] + rad], fill=cc)
    def poly(pts3, col, round_):
        pts = hull([X(pr(p)) for p in pts3])
        for rd, cc in ((round_ + 2 * OW, OUT), (round_, col)):
            d.polygon(pts, fill=cc); d.line(pts + [pts[0]], fill=cc, width=wi(rd), joint="curve")
            for p in pts: d.ellipse([p[0] - rd * S2 / 2, p[1] - rd * S2 / 2, p[0] + rd * S2 / 2, p[1] + rd * S2 / 2], fill=cc)

    b = P["b"]; prims = []
    far = lambda dep: 0.9 if dep < -6 else 1.0
    add = lambda dep, fn: prims.append((dep, len(prims), fn))

    for s in (-1, 1):
        hp = b["P"] + V(s * HIPW, 0, 0); kn_pt, an_pt = P["legs"][s][0], P["legs"][s][1]; f = far(dp(hp))
        add(dp(hp + an_pt) / 2 - 1, lambda hp=hp, kn_pt=kn_pt, an_pt=an_pt, f=f: (group([(kn_pt, an_pt, 24)], shade(SKIN, f)), group([(hp, kn_pt, 31)], shade(PANTS, f))))
        heel = an_pt + V(0, -4, -8); toe = an_pt + V(s * 4, -4, 24)
        def shoe(heel=heel, toe=toe, f=f):
            group([(heel, toe, 21)], shade(SHOE, f)); disc(toe, 7, shade(SHOE_A, f), (0, 1), outline=False)
        add(dp(an_pt) + 1, shoe)

    def torso():
        C, Pp = b["C"], b["P"]; mid = Pp + b["u"] * TORSO * 0.5
        tw = b["tw"]
        pts = [C + rot(V(sx * (SH + 4), 6, sz * 17), tw) for sx in (-1, 1) for sz in (-1, 1)]
        pts += [mid + rot(V(sx * SH * .7, 0, sz * 15), tw * .5) for sx in (-1, 1) for sz in (-1, 1)]
        pts += [Pp + V(sx * (HIPW + 10), 4, sz * 16) for sx in (-1, 1) for sz in (-1, 1)]
        poly(pts, TOP, 12)
        poly([Pp + V(sx * (HIPW + 10), dy, sz * 17) for sx in (-1, 1) for sz in (-1, 1) for dy in (-14, 5)], PANTS, 16)
        cx_, cy_ = X(pr(Pp + b["u"] * TORSO * .56 + rot(V(0, 0, 15), b["tw"] * .7)))                     # 胸前星星「AI」徽章
        draw_star(d, cx_, cy_, rr(21), rr(9.5), wi(2.6))
        d.text((cx_, cy_ + rr(2)), "AI", font=ImageFont.truetype(ZH, int(12 * S2)), fill=INK, anchor="mm")
    add(0, torso)

    for s in (-1, 1):
        sh, el, hd = P["arms"][s]; f = far(dp(sh))
        add(dp(sh + hd) / 2, lambda sh=sh, el=el, hd=hd, f=f: group([(sh, el, 24), (el, hd, 22)], shade(SKIN, f)))
        def hand(hd=hd, f=f, s=s):
            if P["thumb"] and s == -1:
                c = X(pr(hd)); cap(hd + V(0, 3, 0), hd + V(0, 21, 0), 11 + 2 * OW, OUT); cap(hd + V(0, 3, 0), hd + V(0, 21, 0), 11, SKIN)
            disc(hd, 12, shade(SKIN, f))
        add(dp(hd) + .5, hand)
        add(max(dp(sh) + .6, dp(sh + hd) / 2 + .1), lambda sh=sh, el=el, f=f: (group([(sh, sh + (el - sh) * .42, 29)], shade(TOP, f))))   # 袖子始终盖在本侧手臂上

    def head():
        hc = pr(b["head"]); cx, cy = X(hc); R = HEAD_R; ro = R * 1.22; ri = ro * .56; rnd = 30
        rot = -90 + 5 * math.sin(2 * math.pi * t); T = 50 * SY_
        def star_at(off, col, extra):
            c0 = cx - rr(off)
            pts = [(c0 + (x - c0) * CY_, y) for x, y in star_pts(c0, cy, rr(ro - rnd / 2), rr(ri - rnd * .3), rot)]
            w = wi(rnd + extra); d.polygon(pts, fill=col); d.line(pts + [pts[0]], fill=col, width=w, joint="curve")
            for p in pts: d.ellipse([p[0] - w / 2, p[1] - w / 2, p[0] + w / 2, p[1] + w / 2], fill=col)
        offs = [T * k / 6 for k in range(6, -1, -1)]
        for o in offs: star_at(o, OUT, 2 * OW)
        for o in offs: star_at(o, HEADC_D, 0)
        star_at(0, OUT, 2 * OW * .6); star_at(0, HEADC, 0)
        # 高光（潮玩质感）
        d.ellipse([cx - rr(.62 * R), cy - rr(.52 * R), cx - rr(.22 * R), cy - rr(.30 * R)], fill=(255, 244, 190, 210)); d.ellipse([cx - rr(.74 * R), cy - rr(.28 * R), cx - rr(.66 * R), cy - rr(.20 * R)], fill=(255, 244, 190, 210))
        fx, fy = cx + rr(.07 * R + .30 * R * P["turn"]), cy + rr(.12 * R); k = CY_
        # 耳机（左侧星角）+ 麦克风
        tipx = cx + (-ro * .84 * math.cos(math.radians(18))) * S2 * k; tipy = cy - ro * .84 * math.sin(math.radians(18)) * S2
        tx, ty = tipx - rr(T * .6), tipy + rr(8)
        d.ellipse([tx - rr(16), ty - rr(16), tx + rr(16), ty + rr(16)], fill=OUT); d.ellipse([tx - rr(12), ty - rr(12), tx + rr(12), ty + rr(12)], fill=(70, 72, 100)); d.ellipse([tx - rr(6.5), ty - rr(6.5), tx + rr(6.5), ty + rr(6.5)], fill=AICOL)
        blink = (t % 3.2) < 0.12
        for sgn in (-1, 1):
            ex = fx + sgn * rr(.30 * R * k * 1.15); ey = fy
            if P["mood"] == "effort" or blink: d.arc([ex - rr(10), ey - rr(5), ex + rr(10), ey + rr(9)], 200, 340, fill=INK, width=wi(3.6))
            else:
                d.ellipse([ex - rr(8 * k), ey - rr(12), ex + rr(8 * k), ey + rr(12)], fill=INK); d.ellipse([ex - rr(1.5), ey - rr(9), ex + rr(4.5), ey - rr(2)], fill=(255, 255, 255))
            lift = -rr(3) if P["mood"] == "effort" else 0
            d.line([(ex - rr(9 * k), ey - rr(21) + rr(2) + lift * (-sgn)), (ex + rr(9 * k), ey - rr(23) + lift * sgn)], fill=INK, width=wi(3))
            kx = min(fx + sgn * rr(.58 * R * k), cx + rr(.46 * R)); d.ellipse([kx - rr(13), fy + rr(.12 * R), kx + rr(13), fy + rr(.12 * R) + rr(17)], fill=(255, 130, 140, 190))
        mxx, myy, mw = fx, fy + rr(.30 * R), rr(13)
        if voice > 0.05:
            o = rr(7 + min(voice * 22, 9)); d.ellipse([mxx - mw * .8, myy - rr(2), mxx + mw * .8, myy - rr(2) + o], fill=(190, 62, 80), outline=INK, width=wi(2.4))
        elif P["mood"] == "effort": d.rounded_rectangle([mxx - mw * .8, myy, mxx + mw * .8, myy + rr(10)], radius=rr(5), fill=(190, 62, 80), outline=INK, width=wi(2.4))
        else: d.arc([mxx - mw, myy - rr(10), mxx + mw, myy + rr(7)], 20, 160, fill=INK, width=wi(3.2))
    add(8, head)

    sx_, sy_ = pr(V(0, 0, b["P"][2] * 0.6))
    d.ellipse([*X((sx_ - 100, GROUND - 8)), *X((sx_ + 100, GROUND + 20))], fill=(40, 40, 80, 80))
    for dep, _, fn in sorted(prims, key=lambda z: (z[0], z[1])): fn()
    return img.resize((lw, lh), Image.LANCZOS)

# ---------- 背景：办公室（工位 + 办公椅 + 窗外天际线）----------
def cloud(d, cx, cy, s):
    parts = [(0, 0, 1.0), (-.95, .22, .72), (.95, .18, .8), (.42, -.42, .78), (-.42, -.34, .68)]
    for x, y, r in parts: d.ellipse([cx + x * 60 * s - r * 60 * s, cy + y * 60 * s - r * 60 * s, cx + x * 60 * s + r * 60 * s, cy + y * 60 * s + r * 60 * s], fill=(255, 255, 255, 255))

def vgrad(stops):
    ys = np.arange(H); arr = np.stack([np.interp(ys, [p for p, _ in stops], [c[k] for _, c in stops]) for k in range(3)], 1)
    return Image.fromarray(np.repeat(arr[:, None, :], W, axis=1).astype(np.uint8)).convert("RGBA")

def lay(im, fn):
    g = Image.new("RGBA", (W, H), (0, 0, 0, 0)); fn(ImageDraw.Draw(g)); im.alpha_composite(g)

def leaf(d, cx, cy, ln, ang, col, wd=.28):
    pts = []
    for i in range(24):
        a = i / 24 * 2 * math.pi; x, y = math.cos(a) * ln / 2, math.sin(a) * ln * wd / 2
        pts.append((cx + (x + ln / 2) * math.cos(math.radians(ang)) - y * math.sin(math.radians(ang)), cy - (x + ln / 2) * math.sin(math.radians(ang)) - y * math.cos(math.radians(ang))))
    d.polygon(pts, fill=col)

def S(x, y, z):                         # 世界坐标 -> 屏幕坐标（与角色同一放大）
    px, py = pr(V(x, y, z)); return (CX + (px - CX) * K, GROUND + (py - GROUND) * K)

def box3(d, x0, x1, y0, y1, z0, z1, col, w=4):
    pts = hull([S(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    d.polygon(pts, fill=col); d.line(pts + [pts[0]], fill=INK, width=w, joint="curve")

def bg_office():
    im = vgrad([(0, (206, 226, 250)), (1090, (244, 240, 236)), (1091, (226, 178, 124)), (H, (190, 140, 92))])
    lay(im, lambda g: [g.rectangle([x, 0, x + 45, 1090], fill=(255, 255, 255, 46)) for x in range(0, W, 90)])
    d = ImageDraw.Draw(im); d.rectangle([0, 1090, W, 1112], fill=(255, 255, 255))
    for k in range(0, 12): d.line([(0, 1118 + k * k * 8), (W, 1118 + k * k * 8)], fill=(190, 140, 94), width=3)
    fy = lambda x: 20 + 92 * (1 - (2 * x / W - 1) ** 2)                                                             # 彩旗
    d.line([(x, fy(x)) for x in range(-20, W + 40, 20)], fill=INK, width=4)
    for i, c in zip(range(1, 11), [(255, 170, 170), (255, 224, 120), (150, 220, 200), (160, 190, 255), (230, 170, 255)] * 2):
        x = i * W / 11; y = fy(x); d.polygon([(x - 34, y), (x + 34, y), (x, y + 80)], fill=c, outline=INK)
    wx0, wy0, wx1, wy1 = 70, 770, 340, 1030                                                                         # 窗 + 天际线
    d.rounded_rectangle([wx0 - 16, wy0 - 16, wx1 + 16, wy1 + 16], radius=18, fill=(255, 255, 255), outline=INK, width=4)
    win = vgrad([(wy0, (120, 190, 248)), (wy1, (220, 240, 252))]).crop((wx0, wy0, wx1, wy1)); wd = ImageDraw.Draw(win); cloud(wd, 90, 60, .4); cloud(wd, 200, 120, .3)
    for bx, bw, bh, cc in ((10, 60, 150, (150, 170, 208)), (80, 50, 110, (176, 190, 224)), (140, 70, 180, (130, 152, 196)), (220, 50, 130, (168, 184, 220))):
        wd.rectangle([bx, 260 - bh, bx + bw, 260], fill=cc)
        for yy in range(260 - bh + 12, 250, 22): wd.rectangle([bx + 10, yy, bx + 18, yy + 9], fill=(255, 238, 170)); wd.rectangle([bx + bw - 20, yy, bx + bw - 12, yy + 9], fill=(255, 238, 170))
    im.paste(win, (wx0, wy0)); d = ImageDraw.Draw(im)
    d.rectangle([(wx0 + wx1) // 2 - 6, wy0, (wx0 + wx1) // 2 + 6, wy1], fill=(255, 255, 255)); d.rounded_rectangle([wx0 - 28, wy1 + 12, wx1 + 28, wy1 + 36], radius=10, fill=(255, 255, 255), outline=INK, width=3)
    cx_, cy_, r = 830, 790, 68                                                                                      # 挂钟（快下班了）
    d.ellipse([cx_ - r - 8, cy_ - r - 8, cx_ + r + 8, cy_ + r + 8], fill=(255, 170, 160), outline=INK, width=4); d.ellipse([cx_ - r, cy_ - r, cx_ + r, cy_ + r], fill=(255, 252, 244), outline=INK, width=3)
    for k in range(12): a = k * math.pi / 6; d.line([(cx_ + (r - 14) * math.sin(a), cy_ - (r - 14) * math.cos(a)), (cx_ + (r - 5) * math.sin(a), cy_ - (r - 5) * math.cos(a))], fill=INK, width=4)
    d.line([(cx_, cy_), (cx_ + 32 * math.sin(math.radians(-5)), cy_ - 32 * math.cos(math.radians(-5)))], fill=INK, width=6); d.line([(cx_, cy_), (cx_ + 48 * math.sin(math.radians(-30)), cy_ - 48 * math.cos(math.radians(-30)))], fill=INK, width=4)
    d.ellipse([cx_ - 6, cy_ - 6, cx_ + 6, cy_ + 6], fill=INK)
    d.ellipse([CX - 300, GROUND - 40, CX + 300, GROUND + 56], fill=(150, 168, 214), outline=INK, width=4)             # 椅垫地毯
    dz0 = int(S(0, DESK_H, 0)[1])                                                                                    # 办公桌（只留右侧抽屉柜，腿部空间留给抬腿）
    d.rounded_rectangle([860, dz0 + 44, 1010, 1262], radius=10, fill=(214, 168, 120), outline=INK, width=4)
    for yy in (dz0 + 62, dz0 + 150): d.rounded_rectangle([874, yy, 996, yy + 76], radius=10, fill=(232, 190, 142), outline=INK, width=3); d.rounded_rectangle([914, yy + 30, 956, yy + 40], radius=5, fill=INK)
    d.rounded_rectangle([CX + 90, dz0, W + 40, dz0 + 28], radius=10, fill=(244, 208, 164), outline=INK, width=4); d.rectangle([CX + 100, dz0 + 28, W + 40, dz0 + 46], fill=(214, 168, 120), outline=INK, width=3)
    lx0, lx1 = 670, 830                                                                                              # 笔记本
    d.rounded_rectangle([lx0 - 10, dz0 - 12, lx1 + 10, dz0], radius=5, fill=(176, 182, 198), outline=INK, width=3)
    d.rounded_rectangle([lx0, dz0 - 112, lx1, dz0 - 10], radius=12, fill=(214, 218, 230), outline=INK, width=4); d.polygon(star_pts((lx0 + lx1) / 2, dz0 - 62, 20, 9), fill=STAR, outline=INK)
    d.rounded_rectangle([880, dz0 - 50, 922, dz0], radius=6, fill=(255, 255, 255), outline=INK, width=3); d.arc([910, dz0 - 40, 944, dz0 - 12], -90, 90, fill=INK, width=4); d.rectangle([886, dz0 - 44, 916, dz0 - 36], fill=(150, 90, 70))   # 咖啡杯
    for k, cc in enumerate(((255, 255, 255), (255, 232, 160))): d.rounded_rectangle([950 + k * 6, dz0 - 22 - k * 14, 1040 + k * 6, dz0 - k * 14], radius=4, fill=cc, outline=INK, width=3)                                # 文件
    CH, CH2 = (104, 116, 164), (84, 94, 138)                                                                         # 办公椅
    for ex, ez in ((-52, 0), (52, 0), (-26, -46), (26, 46), (-26, 46)):
        p0, p1 = S(0, 16, 0), S(ex, 10, ez); d.line([p0, p1], fill=INK, width=24); d.line([p0, p1], fill=CH2, width=14); d.ellipse([p1[0] - 11, p1[1] - 11 + 8, p1[0] + 11, p1[1] + 11 + 8], fill=INK)
    a, c = S(0, 62, 0), S(0, 16, 0); d.line([a, c], fill=INK, width=22); d.line([a, c], fill=(176, 180, 200), width=12)
    a, c = S(0, 70, -30), S(0, 118, -38); d.line([a, c], fill=INK, width=22); d.line([a, c], fill=CH2, width=12)
    box3(d, -36, 36, 100, 206, -46, -34, CH); box3(d, -38, 38, 62, 78, -36, 66, CH)
    d.polygon([(200, 1620), (880, 1620), (1060, 1912), (20, 1912)], fill=(176, 200, 238), outline=INK)             # 下方地毯 + 道具
    for k in range(1, 4): d.line([(200 - k * 40, 1620 + k * 70), (880 + k * 45, 1620 + k * 70)], fill=(236, 244, 255), width=6)
    for k, cc in enumerate(((255, 170, 160), (255, 224, 120), (150, 220, 200), (160, 190, 255))): d.rounded_rectangle([70 + k * 8, 1700 - k * 38, 270 + k * 8, 1738 - k * 38], radius=10, fill=cc, outline=INK, width=4)    # 文件夹堆
    d.rounded_rectangle([880, 1620, 990, 1760], radius=18, fill=(255, 170, 160), outline=INK, width=4)
    for ang, ln in ((50, 150), (80, 190), (110, 170), (135, 140), (20, 120), (155, 110)): leaf(d, 935, 1622, ln, ang, (96, 186, 100, 255), .34)
    d.rounded_rectangle([430, 1760, 650, 1820], radius=18, fill=(255, 248, 232), outline=INK, width=4); d.rounded_rectangle([430, 1760, 650, 1790], radius=18, fill=(255, 232, 196), outline=INK, width=4)
    return im

_bgc = {}
def background():
    if "o" not in _bgc: _bgc["o"] = bg_office().convert("RGB")
    return _bgc["o"].copy()

def boss(im, t):                         # 老板：中年上司（秃顶+眼镜+八字胡+啤酒肚+领带+工牌），0.6-3.0 秒从背后走过
    if not 0.6 <= t <= 3.0: return
    u = (t - 0.6) / 2.4; X0 = lerp(-200, 1280, u); ph = (t - 0.6) * 2 * math.pi * 1.6; x, y = 200, 700 - 6 * abs(math.sin(ph))
    g = Image.new("RGBA", (400, 760), (0, 0, 0, 0)); d = ImageDraw.Draw(g); sw = 26 * math.sin(ph)
    SUIT, SUIT_D, PNT, SK, SK_D = (70, 78, 118), (54, 62, 100), (96, 100, 124), (255, 214, 176), (236, 176, 138)
    def cp(p, q, w, col):
        d.line([p, q], fill=col, width=int(w)); r = w / 2
        for c in (p, q): d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=col)
    for sg in (-1, 1):                                                                                  # 腿 + 皮鞋
        a_, b_ = (x + sg * 20, y - 200), (x + sg * 20 + sg * sw, y - 18)
        cp(a_, b_, 40, INK); cp(a_, b_, 32, PNT)
        d.rounded_rectangle([b_[0] - 12, b_[1] - 4, b_[0] + 50, b_[1] + 18], radius=10, fill=(40, 32, 38), outline=INK, width=3); d.ellipse([b_[0] + 20, b_[1], b_[0] + 40, b_[1] + 8], fill=(120, 110, 118))
    d.rounded_rectangle([x - 74, y - 432, x + 74, y - 188], radius=46, fill=SUIT, outline=INK, width=5)       # 西装
    d.ellipse([x + 6, y - 340, x + 96, y - 220], fill=SUIT, outline=INK, width=5); d.rectangle([x - 4, y - 336, x + 40, y - 224], fill=SUIT)   # 啤酒肚
    d.polygon([(x - 6, y - 430), (x + 44, y - 430), (x + 24, y - 336)], fill=(255, 255, 255), outline=INK)  # 衬衫领口
    d.polygon([(x - 8, y - 430), (x + 8, y - 430), (x + 30, y - 330), (x + 4, y - 300)], fill=SUIT_D, outline=INK); d.polygon([(x + 38, y - 430), (x + 54, y - 430), (x + 40, y - 320), (x + 20, y - 336)], fill=SUIT_D, outline=INK)   # 翻领
    d.polygon([(x + 18, y - 424), (x + 32, y - 424), (x + 38, y - 340), (x + 26, y - 296), (x + 14, y - 340)], fill=(220, 60, 64), outline=INK)
    for k in range(3): d.line([(x + 17 + k, y - 380 + k * 22), (x + 34 - k, y - 372 + k * 22)], fill=(255, 150, 150), width=3)             # 领带条纹
    d.line([(x + 6, y - 430), (x + 12, y - 330)], fill=(60, 120, 220), width=4); d.line([(x + 44, y - 430), (x + 38, y - 330)], fill=(60, 120, 220), width=4)    # 工牌挂绳
    d.rounded_rectangle([x - 10, y - 336, x + 30, y - 280], radius=6, fill=(255, 255, 255), outline=INK, width=3); d.rectangle([x - 4, y - 330, x + 12, y - 316], fill=(255, 190, 90)); d.line([(x - 4, y - 306), (x + 24, y - 306)], fill=INK, width=3)
    d.polygon([(x - 52, y - 372), (x - 28, y - 372), (x - 30, y - 356), (x - 52, y - 356)], fill=(255, 255, 255), outline=INK)         # 胸袋巾
    for sg, sw2 in ((-1, -sw), (1, sw)):                                                                # 摆臂：近侧拿公文包，远侧拿报表
        a_, b_ = (x + sg * 70, y - 396), (x + sg * 74 + sw2 * .6, y - 262)
        cp(a_, b_, 40, INK); cp(a_, b_, 32, SUIT_D); d.ellipse([b_[0] - 20, b_[1] - 8, b_[0] + 20, b_[1] + 12], fill=(255, 255, 255), outline=INK, width=3)
        d.ellipse([b_[0] - 16, b_[1] + 4, b_[0] + 16, b_[1] + 34], fill=SK, outline=INK, width=3)
        if sg == 1: d.rounded_rectangle([b_[0] - 40, b_[1] + 22, b_[0] + 40, b_[1] + 84], radius=8, fill=(150, 98, 70), outline=INK, width=4); d.rounded_rectangle([b_[0] - 12, b_[1] + 14, b_[0] + 12, b_[1] + 28], radius=5, outline=INK, width=4)
        else: d.rectangle([b_[0] - 8, b_[1] + 18, b_[0] + 40, b_[1] + 74], fill=(255, 255, 255), outline=INK, width=3); d.text((b_[0] + 16, b_[1] + 46), "报表", font=font(18), fill=INK, anchor="mm")
    hx, hy = x + 6, y - 496                                                                              # 头
    d.ellipse([hx - 66, hy - 62, hx + 66, hy + 66], fill=SK, outline=INK, width=5); d.ellipse([hx - 74, hy - 10, hx - 50, hy + 22], fill=SK, outline=INK, width=4)   # 脸 + 耳
    d.pieslice([hx - 70, hy - 66, hx - 10, hy + 30], 120, 250, fill=(150, 154, 172), outline=INK, width=4)                                      # 两侧灰发
    for k in range(3): d.arc([hx - 40 + k * 20, hy - 76 + k * 3, hx + 24 + k * 20, hy - 30], 200, 340, fill=(150, 154, 172), width=4)               # 地中海梳发
    d.ellipse([hx + 52, hy + 4, hx + 78, hy + 30], fill=SK_D, outline=INK, width=4)                       # 大鼻子
    for ex, sg in ((hx + 12, -1), (hx + 46, 1)):
        d.ellipse([ex - 17, hy - 12, ex + 17, hy + 22], fill=(255, 255, 255), outline=INK, width=4); d.ellipse([ex - 2, hy - 2, ex + 9, hy + 12], fill=INK)             # 圆框眼镜
        d.line([(ex - 20, hy - 30 - sg * 8), (ex + 18, hy - 30 + sg * 8)] if sg == -1 else [(ex - 18, hy - 22), (ex + 20, hy - 38)], fill=INK, width=7)               # 怒眉
    d.line([(hx + 29, hy + 4), (hx + 29, hy + 2)], fill=INK, width=4)
    d.polygon([(hx + 30, hy + 30), (hx + 70, hy + 30), (hx + 66, hy + 44), (hx + 50, hy + 38), (hx + 32, hy + 46)], fill=(92, 70, 62), outline=INK)        # 八字胡
    d.arc([hx + 28, hy + 48, hx + 60, hy + 66], 200, 340, fill=INK, width=4)
    tx, ty = x + 6, y - 590; d.rounded_rectangle([tx - 56, ty - 26, tx + 56, ty + 26], radius=16, fill=(220, 70, 70), outline=INK, width=4); d.text((tx, ty), "老板", font=font(34), fill=(255, 255, 255), anchor="mm")
    if 1.5 <= t <= 2.9:                                                                                  # 老板头顶「？」气泡
        sc = pop(1.5, t); bx, by, r = x + 112, y - 580, 34 * sc
        d.ellipse([bx - r, by - r, bx + r, by + r], fill=(255, 255, 255), outline=INK, width=4); d.polygon([(bx - 22, by + r - 8), (bx - 40, by + r + 18), (bx - 6, by + r - 2)], fill=(255, 255, 255), outline=INK)
        d.ellipse([bx - r + 4, by + r - 14, bx + 6, by + r], fill=(255, 255, 255)); d.text((bx, by), "?", font=font(int(52 * sc)), fill=(220, 60, 64), anchor="mm", stroke_width=2, stroke_fill=(220, 60, 64))
    cv = Image.new("RGBA", (W, H), (0, 0, 0, 0)); cv.paste(g, (int(X0 - 200), 1100 - 700)); im.alpha_composite(cv)

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
    titles = {"hook": ("老板路过", "都看不出你在运动"), "shrug": ("① 坐姿耸肩", ""), "twist": ("② 坐姿扭转", ""), "legs": ("③ 坐姿抬腿", ""), "outro": ("量力而行", "不舒服就停一停")}
    big, small = titles[name]; a0 = {n: a for a, b, n in SEGS}[name]; sc = pop(a0, t)
    if name == "hook":
        stext(dr, (CX + 20, 285), big, int(190 * sc), YEL, stroke=12); stext(dr, (CX + 20, 418), small, int(76 * sc), (255, 255, 255), stroke=8)
    else:
        stext(dr, (CX + 20, 300), big, int(118 * sc), YEL if name != "outro" else (255, 255, 255), stroke=10)
        if small: stext(dr, (CX + 20, 400), small, int(58 * sc), (255, 255, 255), stroke=7)
    x0, x1, yb = 130, 830, 474
    dr.rounded_rectangle([x0, yb - 11, x1, yb + 11], radius=11, fill=(255, 255, 255, 110))
    dr.rounded_rectangle([x0, yb - 11, x0 + (x1 - x0) * t / DUR, yb + 11], radius=11, fill=YEL)
    for a, b, n in SEGS[:-1]:
        xx = x0 + (x1 - x0) * b / DUR; dr.ellipse([xx - 7, yb - 7, xx + 7, yb + 7], fill=(255, 255, 255, 240))
    rem = max(1, math.ceil(DUR - t - 1e-6)); cxm, cym, R = 150, 650, 74 + int(6 * (1 - beat))
    dr.ellipse([cxm - R, cym - R, cxm + R, cym + R], fill=PANEL, outline=YEL, width=7)
    stext(dr, (cxm, cym - 6), str(rem), 86, (255, 255, 255), stroke=5, sc=(40, 28, 56)); stext(dr, (cxm, cym + 50), "秒", 28, YEL, stroke=3, sc=(40, 28, 56))
    cnt = {"shrug": (int(lt) + 1, 4), "twist": (int(lt // 2) + 1, 2), "legs": (min(int(lt * 2) + 1, 6), 6)}.get(name)
    if cnt: stext(dr, (880, 650), f"{cnt[0]}/{cnt[1]}", 76, (255, 255, 255), stroke=7)
    label, f34 = "小燃 · AI虚拟教练", font(36); lw_ = dr.textlength(label, font=f34); w_ = lw_ + 116; nx0 = CX + 20 - w_ / 2; ny0, nh = 1350, 62   # 名牌：星星「AI」图标 + 名称
    dr.rounded_rectangle([nx0, ny0, nx0 + w_, ny0 + nh], radius=29, fill=(255, 248, 232, 255), outline=INK, width=4)
    ix, iy = nx0 + 40, ny0 + nh / 2
    draw_star(dr, ix, iy, 29, 13, 4, fill=STAR); dr.text((ix, iy + 2), "AI", font=font(16), fill=INK, anchor="mm")
    dr.text((nx0 + 76, iy + 1), label, font=f34, fill=INK, anchor="lm")
    cur = [x for x in timings if x["start"] - 0.05 <= t <= x["end"] + 0.35]
    if cur:
        tw = dr.textlength(cur[0]["text"], font=font(56)) / 2 + 36
        dr.rounded_rectangle([CX + 20 - tw, 1420, CX + 20 + tw, 1498], radius=39, fill=(60, 38, 40, 120))
        stext(dr, (CX + 20, 1458), cur[0]["text"], 56, (255, 255, 255), stroke=8)
    return ov

def sun_pulse(t, im):
    b = (t % 0.5) / 0.5; g = Image.new("RGBA", (W, H), (255, 255, 255, 0)); gd = ImageDraw.Draw(g)
    gd.ellipse([CX - 300, 640, CX + 300, 1240], fill=(255, 255, 255, int(40 + 30 * (1 - b)))); im.alpha_composite(g.filter(ImageFilter.GaussianBlur(60)))

_voice_env = None
def frame(t, timings):
    global _voice_env
    if _voice_env is None:
        p = os.path.join(TMP, "voice_env.npy"); _voice_env = np.load(p) if os.path.exists(p) else np.zeros(1)
    v = float(_voice_env[min(int(t * 44100), len(_voice_env) - 1)]); P = pose(t); name, lt = seg_of(t)
    im = background().convert("RGBA"); sun_pulse(t, im)
    boss(im, t); im.alpha_composite(draw_coach(P, t, v), (LAY[0], LAY[1])); im.alpha_composite(overlay(t, name, lt, timings))
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
    out = os.path.join(HERE, "xiaoran_office_16s.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", os.path.join(TMP, "segs.txt"), "-i", os.path.join(TMP, "audio.wav"),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-t", str(DUR), "-movflags", "+faststart", out], check=True)
    print("done", out)

if __name__ == "__main__":
    if "--still" in sys.argv: do_stills([float(x) for x in sys.argv[sys.argv.index("--still") + 1:]])
    else: do_full()