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
K = 1.35                          # 角色整体放大（以脚底为轴）
SS = 2
LAY = (0, 430, W, 900)
YAW = math.radians(38)             # 机位：角色朝向画面右前方 38°
CY_, SY_ = math.cos(YAW), math.sin(YAW)

# ---------- 配色（Q 版：平涂 + 深棕描边）----------
SKIN, SKIN_D = (255, 222, 196), (240, 190, 160)
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
HEAD_R, TORSO, SH, HIPW = 80, 118, 36, 22
UARM, FARM, THIGH, SHIN = 78, 74, 104, 100
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

# ---------- 分镜 ----------
SEGS = [(0, 3, "hook"), (3, 7, "jack"), (7, 11, "squat"), (11, 14, "knees"), (14, 16.01, "outro")]
def seg_of(t):
    for a, b, n in SEGS:
        if a <= t < b: return n, t - a
    return "outro", t - 14

def body(py, pz=0.0, lean=0.0):
    P = V(0, py, pz); l = math.radians(lean); u = V(0, math.cos(l), math.sin(l))
    C = P + u * TORSO; hd = V(0, math.cos(l * 0.4), math.sin(l * 0.4))
    return dict(P=P, C=C, sh={s: C + V(s * SH, 0, 0) for s in (-1, 1)}, head=C + hd * (HEAD_R + 4), u=u)

def pose(t):
    name, lt = seg_of(t)
    mood, thumb, arms, legs = "smile", False, {}, {}
    if name == "hook":
        b = body(ANK + LEG - 5 + 3 * math.sin(t * 4 * math.pi), 0, 2)
        for s in (-1, 1):
            legs[s] = ik(b["P"] + V(s * HIPW, 0, 0), V(s * 32, ANK, 0), THIGH, SHIN, V(s * .3, 0, 1)) + (V(s * 32, ANK, 0),)
        w = math.sin(t * 2 * math.pi * 3.2); sh_n, sh_f = b["sh"][-1], b["sh"][1]
        e, h = ik(sh_n, sh_n + V(-50 + 34 * w, 124, 26), UARM, FARM, V(-1, -.5, -.2)); arms[-1] = (sh_n, e, h)
        e, h = ik(sh_f, b["P"] + V(HIPW + 24, 16, 10), UARM, FARM, V(1, -.1, -.9)); arms[1] = (sh_f, e, h)
        mood = "grin"
    elif name == "jack":
        p = lt % 1.0; u = ease(0.5 - 0.5 * math.cos(2 * math.pi * p)); lift = 20 * abs(math.sin(2 * math.pi * p))
        la = math.radians(lerp(2, 24, u)); b = body(ANK + lift + LEG * math.cos(la) * .995, 0, 0); aa = lerp(12, 162, u)
        for s in (-1, 1):
            sh = b["sh"][s]; e, h = ik(sh, sh + limb(s, aa, 0) * (UARM + FARM) * .985, UARM, FARM, V(0, 0, -1)); arms[s] = (sh, e, h)
            ank = V(s * (HIPW + LEG * math.sin(la)), ANK + lift, 0)
            legs[s] = ik(b["P"] + V(s * HIPW, 0, 0), ank, THIGH, SHIN, V(s * .3, 0, 1)) + (ank,)
        mood = "grin" if u > .5 else "smile"
    elif name == "squat":
        p = (lt % 2.0) / 2.0; d = ease(0.5 - 0.5 * math.cos(2 * math.pi * p))
        b = body(lerp(ANK + LEG - 4, 150, d), lerp(0, -60, d), lerp(2, 32, d))
        for s in (-1, 1):
            ank = V(s * lerp(36, 46, d), ANK, 0)
            legs[s] = ik(b["P"] + V(s * HIPW, 0, 0), ank, THIGH, SHIN, V(s * .25 * d, 0, 1)) + (ank,)
            sh = b["sh"][s]; e, h = ik(sh, sh + V(-s * 10, -6 + 6 * d, 128), UARM, FARM, V(s * .5, -1, 0)); arms[s] = (sh, e, h)
        mood = "effort" if d > .5 else "smile"
    elif name == "knees":
        q = lt * 2.0; side = -1 if int(q) % 2 == 0 else 1; lf = ease(math.sin(math.pi * (q % 1.0)))
        b = body(ANK + LEG + 5 * lf, 0, 6)
        for s in (-1, 1):
            hp = b["P"] + V(s * HIPW, 0, 0)
            if s == side:
                kn = hp + V(s * 2, 0, 0) + limb(0, 0, lerp(0, 82, lf)) * THIGH; an = kn + limb(0, 0, lerp(0, -8, lf)) * SHIN
            else:
                kn = hp + V(s * 1, -THIGH, 4 * lf); an = hp + V(0, -LEG, 0)
            legs[s] = (kn, an, an)
        for s in (-1, 1):
            sh = b["sh"][s]; a_ = lf if s == -side else 1 - lf if s == side else 0
            fu = lerp(-28, 48, a_); el = sh + limb(s, 10, fu) * UARM; arms[s] = (sh, el, el + limb(s, 8, fu + 88) * FARM)
        mood = "effort"
    else:
        b = body(ANK + LEG - 4 + 3 * abs(math.sin(lt * 4 * math.pi)), 0, 2)
        for s in (-1, 1):
            legs[s] = ik(b["P"] + V(s * HIPW, 0, 0), V(s * 32, ANK, 0), THIGH, SHIN, V(s * .3, 0, 1)) + (V(s * 32, ANK, 0),)
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
        add(dp(hp + an_pt) / 2 - 1, lambda hp=hp, kn_pt=kn_pt, an_pt=an_pt, f=f: group([(hp, kn_pt, 36), (kn_pt, an_pt, 31)], shade(PANTS, f)))
        heel = an_pt + V(0, -6, -10); toe = an_pt + V(s * 5, -6, 30)
        def shoe(heel=heel, toe=toe, f=f):
            group([(heel, toe, 25)], shade(SHOE, f)); disc(toe, 8, shade(SHOE_A, f), (0, 1), outline=False)
        add(dp(an_pt) + 1, shoe)

    def torso():
        C, Pp = b["C"], b["P"]; mid = Pp + b["u"] * TORSO * 0.5
        pts = [C + V(sx * (SH + 4), 6, sz * 20) for sx in (-1, 1) for sz in (-1, 1)]
        pts += [mid + V(sx * SH * .7, 0, sz * 18) for sx in (-1, 1) for sz in (-1, 1)]
        pts += [Pp + V(sx * (HIPW + 12), 4, sz * 19) for sx in (-1, 1) for sz in (-1, 1)]
        poly(pts, TOP, 14)
        poly([Pp + V(sx * (HIPW + 12), dy, sz * 20) for sx in (-1, 1) for sz in (-1, 1) for dy in (-18, 6)], PANTS, 20)
        cx_, cy_ = X(pr(Pp + b["u"] * TORSO * .56 + V(0, 0, 18)))                     # 胸前星星「AI」徽章
        draw_star(d, cx_, cy_, rr(26), rr(12), wi(3))
        d.text((cx_, cy_ + rr(2)), "AI", font=ImageFont.truetype(ZH, int(16 * S2)), fill=INK, anchor="mm")
    add(0, torso)

    for s in (-1, 1):
        sh, el, hd = P["arms"][s]; f = far(dp(sh))
        add(dp(sh + hd) / 2, lambda sh=sh, el=el, hd=hd, f=f: group([(sh, el, 30), (el, hd, 27)], shade(SKIN, f)))
        def hand(hd=hd, f=f, s=s):
            if P["thumb"] and s == -1:
                c = X(pr(hd)); cap(hd + V(0, 4, 0), hd + V(0, 26, 0), 13 + 2 * OW, OUT); cap(hd + V(0, 4, 0), hd + V(0, 26, 0), 13, SKIN)
            disc(hd, 14, shade(SKIN, f))
        add(dp(hd) + .5, hand)
        add(dp(sh) + .6, lambda sh=sh, el=el, f=f: (group([(sh, sh + (el - sh) * .4, 36)], shade(TOP, f))))

    def head():
        hc = pr(b["head"]); R = HEAD_R; cx, cy = X(hc); rr_ = rr
        def blob(draw_fn, col):                       # 带描边的形状：先外扩一圈描边色，再画本体
            draw_fn(OW * S2, OUT); draw_fn(0, col)
        def rrect(x0, y0, x1, y1, rad):
            return lambda e, c: d.rounded_rectangle([x0 - e, y0 - e, x1 + e, y1 + e], radius=rad + e, fill=c)
        def ell(x0, y0, x1, y1):
            return lambda e, c: d.ellipse([x0 - e, y0 - e, x1 + e, y1 + e], fill=c)
        blob(rrect(cx - rr_(1.1 * R), cy - rr_(1.04 * R), cx + rr_(0.95 * R), cy + rr_(0.62 * R), rr_(0.6 * R)), HAIR)      # 后发（波波头）
        fx, fy = cx + rr_(0.13 * R), cy + rr_(0.06 * R)
        blob(ell(fx - rr_(0.93 * R), fy - rr_(0.95 * R), fx + rr_(0.93 * R), fy + rr_(0.96 * R)), SKIN)                       # 脸
        def face(ex, ey, rad=0.95):
            ez = math.sqrt(max(rad * rad - ex * ex - ey * ey, 0.01))
            return cx + rr_((ex * CY_ + ez * SY_) * R), cy - rr_(ey * R), max((-ex * SY_ + ez * CY_), 0.2)
        # 齐刘海
        top = [(cx + rr_(1.05 * R * math.cos(math.radians(a))), cy - rr_(0.1 * R) - rr_(1.0 * R * math.sin(math.radians(a)))) for a in range(0, 181, 9)]
        bot = [(cx - rr_(1.04 * R) + rr_(2.09 * R) * k / 12, cy - rr_(0.30 * R) + rr_(0.06 * R) * math.sin(k * 1.4)) for k in range(0, 13)]
        fr = top + bot[::-1]
        d.polygon(fr, fill=OUT); d.line(fr + [fr[0]], fill=OUT, width=wi(2 * OW), joint="curve"); d.polygon(fr, fill=HAIR)
        for k in range(3): d.line([(cx - rr_(0.5 * R) + rr_(0.3 * R) * k, cy - rr_(0.92 * R) + rr_(0.08 * R) * k), (cx - rr_(0.3 * R) + rr_(0.3 * R) * k, cy - rr_(0.5 * R))], fill=HAIR_L, width=wi(3))
        # 近侧发帘
        blob(ell(cx - rr_(1.0 * R), cy - rr_(0.52 * R), cx - rr_(0.3 * R), cy + rr_(0.66 * R)), HAIR)
        # 耳机（AI 标识）
        ex_, ey_ = cx - rr_(0.66 * R), cy + rr_(0.14 * R)
        d.ellipse([ex_ - rr_(15), ey_ - rr_(15), ex_ + rr_(15), ey_ + rr_(15)], fill=OUT); d.ellipse([ex_ - rr_(11), ey_ - rr_(11), ex_ + rr_(11), ey_ + rr_(11)], fill=(70, 72, 100))
        d.ellipse([ex_ - rr_(6), ey_ - rr_(6), ex_ + rr_(6), ey_ + rr_(6)], fill=AICOL)
        mx, my, md = face(0, -0.5)
        d.line([(ex_, ey_ + rr_(10)), (cx - rr_(0.3 * R), cy + rr_(0.86 * R)), (mx - rr_(0.28 * R), my + rr_(0.12 * R))], fill=(70, 72, 100), width=wi(3.5), joint="curve")
        d.ellipse([mx - rr_(0.28 * R) - rr_(5), my + rr_(0.12 * R) - rr_(5), mx - rr_(0.28 * R) + rr_(5), my + rr_(0.12 * R) + rr_(5)], fill=(70, 72, 100))
        # 星星发夹（大小各一）
        draw_star(d, cx - rr_(0.62 * R), cy - rr_(0.62 * R), rr_(16), rr_(7), wi(2.6), rot=-80)
        draw_star(d, cx - rr_(0.30 * R), cy - rr_(0.84 * R), rr_(9), rr_(4), wi(2), rot=-100)
        # 五官：圆点眼 + 小嘴 + 腮红
        blink = (t % 3.2) < 0.12
        for s, ex in ((-1, -0.40), (1, 0.40)):
            x, y, dep = face(ex, -0.10); w = rr_(8.5 * dep); hg = rr_(11.5)
            if P["mood"] == "effort" or blink:
                d.arc([x - w * 1.2, y - rr_(5), x + w * 1.2, y + rr_(9)], 200, 340, fill=INK, width=wi(3.6))
            else:
                d.ellipse([x - w, y - hg, x + w, y + hg], fill=INK); d.ellipse([x - w * .1, y - hg * .65, x + w * .55, y - hg * .05], fill=(255, 255, 255))
            bx, by, bd = face(ex, 0.12); lift = -rr_(3) if P["mood"] == "effort" else 0
            d.line([(bx - rr_(9 * bd), by + rr_(2) + lift * (-s)), (bx + rr_(9 * bd), by - rr_(2) + lift * s)], fill=INK, width=wi(3))
            kx, ky, kd = face(ex * 1.6, -0.34); d.ellipse([kx - rr_(15 * kd), ky - rr_(9), kx + rr_(15 * kd), ky + rr_(9)], fill=(255, 140, 150, 170))
        nx, ny, nd = face(0, -0.24, 1.02); d.ellipse([nx - rr_(3.5), ny - rr_(3), nx + rr_(3.5), ny + rr_(3)], fill=SKIN_D)
        mx, my, md = face(0, -0.5); mw = rr_(15 * md)
        if voice > 0.05:
            o = rr_(7 + min(voice * 22, 8)); d.ellipse([mx - mw * .75, my - rr_(3), mx + mw * .75, my - rr_(3) + o], fill=(190, 62, 80), outline=INK, width=wi(2.4))
        elif P["mood"] == "effort":
            d.rounded_rectangle([mx - mw * .8, my, mx + mw * .8, my + rr_(10)], radius=rr_(5), fill=(190, 62, 80), outline=INK, width=wi(2.4))
        else:
            d.arc([mx - mw, my - rr_(11), mx + mw, my + rr_(7)], 20, 160, fill=INK, width=wi(3.2))
    add(8, head)

    sx_, sy_ = pr(V(0, 0, b["P"][2] * 0.6))
    d.ellipse([*X((sx_ - 130, GROUND - 10)), *X((sx_ + 130, GROUND + 24))], fill=(30, 90, 40, 80))
    for dep, _, fn in sorted(prims, key=lambda z: (z[0], z[1])): fn()
    return img.resize((lw, lh), Image.LANCZOS)

# ---------- 小星星伙伴（原创）----------
def star_buddy(im, t, voice, mood):
    beat = (t % 0.5) / 0.5; sc = 1 + 0.12 * (1 - beat)
    cx, cy = CX + 250, 800 + 16 * math.sin(2 * math.pi * t)
    S = 3; lay = Image.new("RGBA", (220 * S, 220 * S), (0, 0, 0, 0)); d = ImageDraw.Draw(lay); c = 110 * S; r = 62 * S * sc
    draw_star(d, c, c, r, r * .52, int(5 * S), fill=STAR, rot=-90 + 8 * math.sin(2 * math.pi * t))
    for sx in (-1, 1):
        x = c + sx * r * .28; y = c + r * .05
        if mood == "effort": d.arc([x - 7 * S, y - 3 * S, x + 7 * S, y + 8 * S], 200, 340, fill=INK, width=int(3 * S))
        else: d.ellipse([x - 5 * S, y - 7 * S, x + 5 * S, y + 7 * S], fill=INK)
        d.ellipse([x - 12 * S + sx * 8 * S, y + 9 * S, x + 12 * S + sx * 8 * S, y + 19 * S], fill=(255, 150, 150, 180))
    if voice > 0.05: d.ellipse([c - 6 * S, c + r * .24, c + 6 * S, c + r * .24 + 12 * S], fill=(190, 62, 80), outline=INK, width=int(2 * S))
    else: d.arc([c - 9 * S, c + r * .12, c + 9 * S, c + r * .12 + 14 * S], 20, 160, fill=INK, width=int(3 * S))
    for k in range(3):                                                  # 闪光十字
        a = t * 2 + k * 2.1; px, py = c + 90 * S * math.cos(a) * .9, c + 80 * S * math.sin(a * 1.3) * .9
        if abs(px - c) < 105 * S and abs(py - c) < 105 * S:
            q = (4 + 2 * math.sin(t * 8 + k)) * S; d.polygon([(px, py - 3 * q), (px + q, py - q), (px + 3 * q, py), (px + q, py + q), (px, py + 3 * q), (px - q, py + q), (px - 3 * q, py), (px - q, py - q)], fill=(255, 255, 255, 230))
    lay = lay.resize((220, 220), Image.LANCZOS); im.alpha_composite(lay, (int(cx - 110), int(cy - 110)))

# ---------- 背景：蓝天白云绿草（扁平卡通）----------
_bg = None
def cloud(d, cx, cy, s):
    parts = [(0, 0, 1.0), (-.95, .22, .72), (.95, .18, .8), (.42, -.42, .78), (-.42, -.34, .68), (1.6, .3, .5), (-1.55, .32, .5)]
    for dy, col in ((12, (200, 226, 248, 255)), (0, (255, 255, 255, 255))):
        for x, y, r in parts: d.ellipse([cx + x * 60 * s - r * 60 * s, cy + y * 60 * s + dy * s - r * 60 * s, cx + x * 60 * s + r * 60 * s, cy + y * 60 * s + dy * s + r * 60 * s], fill=col)
        d.rounded_rectangle([cx - 1.9 * 60 * s, cy + .2 * 60 * s + dy * s, cx + 1.9 * 60 * s, cy + 1.0 * 60 * s + dy * s], radius=int(30 * s), fill=col)

def background():
    global _bg
    if _bg is not None: return _bg.copy()
    y = np.linspace(0, 1, H)[:, None, None]; hz = 1000 / H
    top, hor = np.array([72, 160, 240]), np.array([190, 228, 252])
    sky = top + (hor - top) * np.clip(y / hz, 0, 1) ** 1.2
    im = Image.fromarray(np.repeat(np.where(y < hz, sky, np.array([120, 205, 100])), W, axis=1).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(im, "RGBA"); rng = np.random.default_rng(21)
    for cx, cy, s in ((170, 640, 1.05), (640, 560, .62), (880, 860, .7), (330, 880, .6)): cloud(d, cx, cy, s)
    d.ellipse([-300, 940, 760, 1340], fill=(150, 216, 130, 255)); d.ellipse([280, 980, 1500, 1400], fill=(122, 204, 108, 255))     # 远处小山
    y0 = 1040; yy = y0; band = 0; h = 26
    while yy < H:                                                                                                           # 透视草坪条纹
        col = [(112, 200, 96), (102, 190, 88)][band % 2]; d.rectangle([0, yy, W, yy + h], fill=col + (255,)); yy += h; h *= 1.16; band += 1
    for _ in range(420):                                                                                                    # 草丛
        gy = rng.uniform(1060, H); k = (gy - 1040) / (H - 1040); gx = rng.uniform(-10, W + 10); sz = 8 + 40 * k
        for dx in (-.5, 0, .5): d.polygon([(gx + dx * sz - sz * .16, gy), (gx + dx * sz + sz * .16, gy), (gx + dx * sz * 1.3, gy - sz * (1.1 - abs(dx) * .4))], fill=(58, 150 - int(30 * k), 62, 255))
    for _ in range(46):                                                                                                     # 小花
        gy = rng.uniform(1080, H); k = (gy - 1040) / (H - 1040); gx = rng.uniform(0, W); r = 4 + 12 * k; col = [(255, 255, 255), (255, 190, 210), (255, 232, 110)][int(rng.integers(0, 3))]
        for a in range(5): d.ellipse([gx + r * 1.2 * math.cos(a * 1.2566) - r, gy + r * 1.2 * math.sin(a * 1.2566) * .8 - r, gx + r * 1.2 * math.cos(a * 1.2566) + r, gy + r * 1.2 * math.sin(a * 1.2566) * .8 + r], fill=col + (255,))
        d.ellipse([gx - r * .7, gy - r * .7, gx + r * .7, gy + r * .7], fill=(255, 170, 60, 255))
    d.ellipse([CX - 300, GROUND - 36, CX + 300, GROUND + 46], fill=(255, 238, 205, 255), outline=OUT, width=4)             # 瑜伽垫
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
    if cur:
        tw = dr.textlength(cur[0]["text"], font=font(56)) / 2 + 36
        dr.rounded_rectangle([CX + 20 - tw, 1374, CX + 20 + tw, 1452], radius=39, fill=(60, 38, 40, 120))
        stext(dr, (CX + 20, 1412), cur[0]["text"], 56, (255, 255, 255), stroke=8)
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
    im.alpha_composite(draw_coach(P, t, v), (LAY[0], LAY[1])); star_buddy(im, t, v, P["mood"]); im.alpha_composite(overlay(t, name, lt, timings))
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
