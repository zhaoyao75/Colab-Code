"""
V2 九镜静帧：明亮的「木刻线刻 + 蜡笔涂鸦」风格（米色纸 + 黑色刻线 + 淡色涂层 + 蜡笔彩色角色）。
  python3 shots.py            # 输出 stills_v2/shot1..9.jpg 和 contact_sheet.jpg
  python3 shots.py 3 6        # 只出指定镜头
字幕取自 timings_v2.json 每镜第一句。
"""
import json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import render as R
import styles_demo as SD
import vertical as V

W, H = R.W, R.H
S2 = 2
PAPER_L, INK_L = SD.PAPER_L, SD.INK_L
CREAM = (250, 246, 236, 255)
WOOD = (224, 180, 128, 235)
YELLOW = (255, 226, 130, 255)
BLUE_W = (170, 205, 225, 170)
CUES = json.load(open(os.path.join(R.HERE, "timings_v2.json"), encoding="utf-8"))


# ------------------------------------------------------------------ 线刻画布
class LA:
    """黑色刻线（L 蒙版）+ 淡色涂层（RGBA），后画的物体会盖掉先画的"""
    def __init__(self, seed=7, wash_all=None):
        self.ink = Image.new("L", (W * S2, H * S2), 0)
        self.d = ImageDraw.Draw(self.ink)
        self.wash = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        self.wd = ImageDraw.Draw(self.wash)
        self.bg = R.paper(W, H, seed, base=PAPER_L, vig=False).convert("RGBA")
        self.rng = np.random.default_rng(seed)
        if wash_all:
            self.wd.rectangle([0, 0, W, H], fill=wash_all)

    def glow(self, cx, cy, r, color, strength=150):
        yy, xx = np.mgrid[0:H, 0:W]
        a = np.clip(1 - np.hypot(xx - cx, yy - cy) / r, 0, 1) ** 1.5 * strength
        g = np.zeros((H, W, 4), np.uint8)
        g[..., :3] = color
        g[..., 3] = a.astype(np.uint8)
        self.bg = Image.alpha_composite(self.bg, Image.fromarray(g, "RGBA"))

    def line(self, pts, w=5):
        self.d.line([(x * S2, y * S2) for x, y in pts], fill=255, width=max(1, int(w * S2)), joint="curve")

    def poly(self, pts, wash=None, w=5, fill_ink=False, outline=True):
        sp = [(x * S2, y * S2) for x, y in pts]
        self.d.polygon(sp, fill=255 if fill_ink else 0)
        self.wd.polygon(pts, fill=(0, 0, 0, 0))
        if wash is not None and not fill_ink:
            self.wd.polygon(pts, fill=wash)
        if outline and not fill_ink:
            self.line(list(pts) + [pts[0]], w)

    def rect(self, x0, y0, x1, y1, **kw):
        self.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], **kw)

    def ell(self, cx, cy, rx, ry, **kw):
        self.poly([(cx + rx * math.cos(t), cy + ry * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 64)], **kw)

    def text(self, xy, s, size=64, font=R.F_BRUSH):
        self.d.text((xy[0] * S2, xy[1] * S2), s, font=R.font(font, size * S2), fill=255, anchor="mm")

    def arcs(self, sx, sy, r0=175, r1=1800, seed=4):
        r = np.random.default_rng(seed)
        for rad in np.arange(r0, r1, 30):
            a0 = r.random() * 360
            wdt = int(max(3, 6 - rad / 500) * S2)
            while a0 < 720:
                seg = r.uniform(10, 46) * 300 / rad + 6
                self.d.arc([(sx - rad) * S2, (sy - rad) * S2, (sx + rad) * S2, (sy + rad) * S2], a0, a0 + seg, fill=255, width=wdt)
                a0 += seg + r.uniform(5, 22) * 300 / rad + 3

    def hills(self, specs):
        for base, amp, seed, col, hatch in specs:
            ridge = base - amp * (0.5 + 0.5 * R.fbm1d(W, 600, 5, seed))
            self.poly([(0, H)] + [(x, ridge[x]) for x in range(0, W, 6)] + [(W, H)], wash=col, outline=False)
            self.line([(x, ridge[x]) for x in range(0, W, 6)], 6)
            for k in range(1, hatch + 1):
                self.line([(x, ridge[x] + k * 18 + 5 * math.sin(x / 90 + k)) for x in range(0, W, 8)], max(2, 4 - k * 0.4))

    def waves(self, y0, y1, step=24, x0=0, x1=W, seed=3):
        r = np.random.default_rng(seed)
        for y in np.arange(y0, y1, step):
            x = x0 + r.random() * 120
            while x < x1:
                ln = r.uniform(50, 170)
                self.line([(x + t, y + 4 * math.sin((x + t) / 40 + y)) for t in range(0, int(ln), 10)], 3)
                x += ln + r.uniform(30, 120)

    def render(self, ink=INK_L):
        a = np.asarray(self.ink.resize((W, H), Image.LANCZOS), np.float32) / 255
        a *= 0.9 + 0.1 * np.random.default_rng(8).random((H, W))
        out = Image.alpha_composite(self.bg, self.wash)
        L = np.zeros((H, W, 4), np.uint8)
        L[..., :3] = ink
        L[..., 3] = np.clip(a * 235, 0, 255).astype(np.uint8)
        return Image.alpha_composite(out, Image.fromarray(L, "RGBA"))


# ------------------------------------------------------------------ 蜡笔层
class CL:
    def __init__(self):
        self.im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)

    def paste(self, spr, cx, cy, scale=1.0, angle=0.0, alpha=1.0):
        im = spr.convert("RGBA")
        if scale != 1.0:
            im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BICUBIC)
        if angle:
            im = im.rotate(angle, Image.BICUBIC, expand=True)
        if alpha < 1.0:
            im = im.copy()
            im.putalpha(im.getchannel("A").point(lambda v: int(v * alpha)))
        x, y = int(cx - im.width / 2), int(cy - im.height / 2)
        if x < 0:                                   # 超出画布的部分裁掉
            im, x = im.crop((-x, 0, im.width, im.height)), 0
        if y < 0:
            im, y = im.crop((0, -y, im.width, im.height)), 0
        im = im.crop((0, 0, min(im.width, W - x), min(im.height, H - y)))
        if im.width > 0 and im.height > 0:
            self.im.alpha_composite(im, (x, y))     # 原地合成，self.d 始终有效

    def finish(self, base, seed=3, wobble=2.2, grain=0.3, halo=5):
        lay = SD.crayon_finish(self.im, seed=seed, wobble=wobble, grain=grain)
        if halo:
            lay = SD.with_halo(lay, halo)
        return Image.alpha_composite(base.convert("RGBA"), lay)


def star(d, x, y, s, fill=(255, 214, 70, 255)):
    pts = [(x + (s if k % 2 == 0 else s * 0.42) * math.sin(k * math.pi / 5),
            y - (s if k % 2 == 0 else s * 0.42) * math.cos(k * math.pi / 5)) for k in range(10)]
    d.polygon(pts, fill=fill, outline=(70, 44, 30, 255))


def dashed(d, pts, color, w, dash=34, gap=22):
    seg = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        n = int(math.hypot(x1 - x0, y1 - y0) / 6) + 1
        seg += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(n)]
    on, run = True, 0.0
    cur = []
    for p, q in zip(seg, seg[1:]):
        run += math.hypot(q[0] - p[0], q[1] - p[1])
        if on:
            cur += [p, q]
        if (on and run >= dash) or (not on and run >= gap):
            if on and len(cur) > 1:
                d.line(cur, fill=color, width=w)
            cur, run, on = [], 0.0, not on


def sub(img, i):
    c = [c for c in CUES if c["scene"] == i][0]
    fz, fe = R.font(R.F_FUN, 52), R.font(R.F_EN, 32)
    zl = V.wrap_zh(c["zh"], fz, 1500)
    el = R.wrap_en(c["en"], fe, 1500)
    hgt = 30 + 62 * len(zl) + 10 + 42 * len(el)
    bw = max([fz.getlength(l) for l in zl] + [fe.getlength(l) for l in el]) + 70
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    y0 = H - 40 - hgt
    d.rounded_rectangle([(W - bw) / 2, y0, (W + bw) / 2, y0 + hgt], radius=22, fill=(20, 16, 12, 130))
    y = y0 + 20 + 31
    for l in zl:
        d.text((W / 2, y), l, font=fz, fill=(255, 252, 240, 255), anchor="mm", stroke_width=3, stroke_fill=(30, 22, 15, 255))
        y += 62
    y += 10 - 31 + 21
    for l in el:
        d.text((W / 2, y), l, font=fe, fill=(235, 228, 210, 255), anchor="mm", stroke_width=2, stroke_fill=(30, 22, 15, 255))
        y += 42
    return Image.alpha_composite(img.convert("RGBA"), lay).convert("RGB")


def finish(base, cl, i, with_sub=True, **kw):
    img = cl.finish(base, **kw).convert("RGB")
    return sub(img, i) if with_sub else img


# ------------------------------------------------------------------ 共用角色
def man_sil(la, cx, top, s=1.0, laugh=False):
    """老先生（黑色剪影上半身，白发）；眼镜用蜡笔层另画（见 glasses）"""
    la.ell(cx, top, 52 * s, 58 * s, fill_ink=True)
    la.poly([(cx - 90 * s, top + 70 * s), (cx + 90 * s, top + 70 * s), (cx + 120 * s, top + 290 * s), (cx - 120 * s, top + 290 * s)], fill_ink=True)
    la.poly([(cx - 52 * s, top - 10 * s), (cx - 34 * s, top - 62 * s), (cx + 30 * s, top - 64 * s), (cx + 54 * s, top - 10 * s),
             (cx + 30 * s, top - 34 * s), (cx - 30 * s, top - 34 * s)], wash=CREAM, w=4 * s)
    if laugh:
        la.ell(cx + 10 * s, top + 30 * s, 20 * s, 13 * s, wash=(0, 0, 0, 0), outline=False)


def glasses(cl, cx, top, s=1.0):
    for dx in (-22, 22):
        cl.d.ellipse([cx + (dx - 16) * s, top + (2 - 16) * s, cx + (dx + 16) * s, top + (2 + 16) * s], outline=CREAM, width=5)
    cl.d.line([(cx - 6 * s, top + 2 * s), (cx + 6 * s, top + 2 * s)], fill=CREAM, width=5)


def old_man(kind="shen"):
    """彩色卡通老人上半身（360×400）。kind='shen' 清瘦儒雅的沈从文；'huang' 圆脸白胡子、笑得张嘴的黄永玉"""
    S = R.SS
    w, h = 360, 400
    im = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    line = (50, 36, 30, 255)
    skin, skin2, white = (246, 208, 168, 255), (236, 186, 148, 255), (250, 248, 244, 255)
    shen = kind == "shen"
    robe = (112, 132, 156, 255) if shen else (214, 88, 66, 255)

    def P(pts):
        return [(x * S, y * S) for x, y in pts]

    def E(cx, cy, rx, ry, **kw):
        d.ellipse([(cx - rx) * S, (cy - ry) * S, (cx + rx) * S, (cy + ry) * S], **kw)

    def L(pts, fill, wd):
        d.line(P(pts), fill=fill, width=int(wd * S), joint="curve")
    # 身体与领口
    d.polygon(P([(30, 400), (50, 270), (120, 222), (240, 222), (310, 270), (330, 400)]), fill=robe, outline=line, width=5 * S)
    d.polygon(P([(150, 222), (180, 300), (210, 222)]), fill=white, outline=line, width=4 * S)
    for by in (322, 358):
        E(180, by, 6, 6, fill=line)
    d.rectangle([160 * S, 190 * S, 200 * S, 234 * S], fill=skin2)
    # 头、耳
    rx, ry = (54, 72) if shen else (64, 70)
    for sx in (-1, 1):
        E(180 + sx * rx, 138, 13, 19, fill=skin, outline=line, width=5 * S)
    E(180, 135, rx, ry, fill=skin, outline=line, width=6 * S)
    # 头发
    if shen:
        outer = [(180 + (rx + 8) * (1 + 0.05 * math.sin(7 * t)) * math.cos(t), 133 + (ry + 10) * (1 + 0.04 * math.sin(7 * t)) * math.sin(t))
                 for t in np.linspace(math.pi * 1.04, math.pi * 1.96, 56)]
        inner = [(x, 98 + ((x - 180) / rx) ** 2 * 30) for x in range(180 + rx - 2, 180 - rx + 1, -4)]
        d.polygon(P(outer + inner), fill=(226, 226, 232, 255), outline=line, width=4 * S)                        # 梳整齐的白发
        for k in (-2, 0, 2):                                                                                         # 发丝纹理
            L([(180 + k * 14, 96 + (k / 3.5) ** 2 * 26), (180 + k * 20, 74 + abs(k) * 5)], (190, 190, 202, 255), 3)
    else:
        d.polygon(P([(118, 122), (102, 86), (128, 94), (124, 58), (152, 78), (160, 44), (182, 72), (204, 44), (212, 78), (238, 58), (236, 94), (258, 86), (242, 122), (222, 98), (180, 88), (140, 98)]),
                  fill=white, outline=line, width=4 * S)
    # 眉、眼镜、眼
    for sx in (-1, 1):
        cx = 180 + sx * 26
        L([(cx - 17, 104 if not shen else 108), (cx + 17, 100 if not shen else 106)] if sx > 0 else [(cx - 17, 100 if not shen else 106), (cx + 17, 104 if not shen else 108)],
          line if shen else line, 11 if not shen else 5)
        if not shen:
            L([(cx - 14, 104), (cx + 14, 102)] if sx > 0 else [(cx - 14, 102), (cx + 14, 104)], white, 7)
        E(cx, 132, 23, 23, fill=(228, 242, 250, 255))
        if shen:
            E(cx, 134, 6, 7, fill=line)
        else:
            d.arc([(cx - 15) * S, 120 * S, (cx + 15) * S, 148 * S], 200, 340, fill=line, width=6 * S)
        E(cx, 132, 23, 23, outline=line, width=5 * S)
    L([(204 - 26, 130), (180 + 4, 130)], line, 5)
    # 鼻、腮红、嘴
    E(180, 156, 10, 14, fill=skin2, outline=line, width=4 * S)
    for sx in (-1, 1):
        E(180 + sx * 42, 168, 14, 9, fill=(242, 172, 156, 255))
    if shen:
        d.arc([150 * S, 160 * S, 210 * S, 200 * S], 20, 160, fill=line, width=5 * S)
    else:
        arc = [(180 + (rx + 3) * math.cos(t), 135 + (ry + 10) * math.sin(t)) for t in np.linspace(0.14 * math.pi, 0.86 * math.pi, 40)]
        d.polygon(P(arc + [(140, 174), (180, 180), (220, 174)]), fill=white, outline=line, width=4 * S)         # 沿下颌的白胡子
        d.polygon(P([(146, 176), (180, 166), (214, 176), (200, 186), (180, 178), (160, 186)]), fill=white, outline=line, width=3 * S)  # 八字须
        E(180, 199, 24, 16, fill=(130, 40, 44, 255), outline=line, width=4 * S)                                  # 张嘴大笑
        E(180, 207, 14, 7, fill=(226, 104, 110, 255))
        d.rectangle([161 * S, 190 * S, 199 * S, 196 * S], fill=white)
    # 双手 + 笔
    for hx in (95, 265):
        E(hx, 386, 24, 20, fill=skin, outline=line, width=5 * S)
    L([(268, 372), (322, 398)], (60, 40, 30, 255), 8)
    return R.fin(im)


def bust(spr, keep=0.66):
    return spr.crop((0, 0, spr.width, int(spr.height * keep)))


def bundle_crayon(cl, x, y, r=46):
    d = cl.d
    d.ellipse([x - r, y - r * 0.9, x + r, y + r * 0.9], fill=(250, 150, 40, 255), outline=(60, 40, 30, 255), width=6)
    for dx, dy in ((-20, -8), (12, -20), (18, 14), (-8, 18)):
        d.ellipse([x + dx - 8, y + dy - 8, x + dx + 8, y + dy + 8], fill=(220, 40, 50, 255))
    d.polygon([(x - 14, y - r * 0.8), (x + 14, y - r * 0.8), (x, y - r * 1.25)], fill=(250, 150, 40, 255), outline=(60, 40, 30, 255))


def house(la, x, base, w, h, wood=WOOD):
    for px in np.linspace(x + 8, x + w - 8, max(3, int(w // 36))):
        la.line([(px, base), (px, base + 56)], 5)
    la.rect(x, base - h, x + w, base, wash=wood, w=5)
    la.line([(x, base - h * 0.34), (x + w, base - h * 0.34)], 4)
    for wx in np.arange(x + 16, x + w - 34, 46):
        la.rect(wx, base - h * 0.8, wx + 26, base - h * 0.5, wash=YELLOW, w=3)
    la.poly([(x - 20, base - h + 8), (x + 10, base - h - 28), (x + w - 10, base - h - 28), (x + w + 20, base - h + 8)], fill_ink=True)


# ------------------------------------------------------------------ 镜 1 钩子
def shot1(sub_on=True):
    img = SD.style_BC2(with_sub=False)
    return sub(img, 0) if sub_on else img


# ------------------------------------------------------------------ 镜 2 凤凰
def shot2(sub_on=True):
    la = LA(11)
    la.glow(1450, 250, 900, (255, 226, 130), 150)
    la.arcs(1450, 250, 175, 1500)
    la.hills([(520, 150, 31, (150, 190, 215, 150), 3), (610, 100, 47, (176, 214, 150, 170), 2)])
    la.rect(0, 690, W, 940, wash=BLUE_W, outline=False)
    r = np.random.default_rng(5)
    x = 40
    while x < W - 120:
        w, h = r.uniform(110, 160), r.uniform(95, 135)
        house(la, x, 690, w, h)
        x += w + r.uniform(20, 60)
    la.waves(770, 930, 26)
    cl = CL()
    d = cl.d
    d.ellipse([1450 - 95, 250 - 95, 1450 + 95, 250 + 95], fill=(255, 190, 50, 255), outline=(70, 44, 30, 255), width=7)
    for rr in (65, 35):
        d.ellipse([1450 - rr, 250 - rr, 1450 + rr, 250 + rr], outline=(235, 110, 40, 255), width=6)
    # 纸船 + 小刻刀（伏笔）
    bx, by = 640, 850
    d.polygon([(bx - 110, by), (bx + 110, by), (bx + 70, by + 46), (bx - 70, by + 46)], fill=CREAM, outline=(60, 40, 30, 255))
    d.polygon([(bx - 100, by + 8), (bx + 100, by + 8), (bx + 92, by + 22), (bx - 92, by + 22)], fill=(232, 50, 52, 255))
    d.rectangle([bx - 20, by - 16, bx + 40, by - 2], fill=(150, 100, 56, 255), outline=(60, 40, 30, 255), width=3)
    d.polygon([(bx + 40, by - 16), (bx + 96, by - 8), (bx + 40, by - 2)], fill=(200, 204, 210, 255), outline=(60, 40, 30, 255))
    for k, (sx, sy) in enumerate(((1000, 160), (1180, 120), (1320, 190), (300, 200))):
        d.line([(sx - 16, sy), (sx, sy - 12), (sx + 16, sy)], fill=(40, 34, 44, 255), width=5)
    cl.paste(R.seal("凤凰", 96), 1780, 170)
    return finish(la.render(), cl, 1, sub_on, seed=4)


# ------------------------------------------------------------------ 镜 3 出走
def shot3(sub_on=True):
    la = LA(13)
    la.glow(1500, 230, 900, (255, 226, 130), 140)
    la.arcs(1500, 230, 175, 1400, seed=9)
    la.hills([(470, 130, 61, (150, 190, 215, 150), 3), (560, 90, 73, (176, 214, 150, 170), 2)])
    la.ell(1000, 560, 840, 150, wash=BLUE_W, w=6)
    la.waves(480, 640, 24, 260, 1660)
    la.rect(810, 520, 1190, 600, wash=CREAM, outline=False)
    la.text((1000, 560), "洞庭湖", 66)
    la.rect(0, 690, W, 940, wash=(190, 222, 160, 170), outline=False)
    la.line([(0, 690), (300, 680), (700, 696), (1100, 680), (1500, 696), (W, 682)], 6)
    cl = CL()
    d = cl.d
    d.ellipse([1500 - 95, 230 - 95, 1500 + 95, 230 + 95], fill=(255, 190, 50, 255), outline=(70, 44, 30, 255), width=7)
    route = [(140, 760), (420, 715), (700, 765), (960, 715), (1220, 755), (1420, 705), (1590, 690)]
    dashed(d, route, (232, 50, 52, 255), 10)
    boy = SD.boy_sprite("sil")
    bx, by, bs = 1650, 540, 0.95
    cl.paste(boy, bx, by, bs)
    hx, hy = SD.boy_bundle_pos(bx, by, bs)
    bundle_crayon(cl, hx, hy, 40)
    for k, (sx, sy, s) in enumerate(((300, 120, 30), (760, 90, 26), (1130, 160, 28))):
        star(d, sx, sy, s)
    for bxx, byy in ((420, 560), (1500, 520)):                                   # 小帆船
        d.polygon([(bxx - 50, byy), (bxx + 50, byy), (bxx + 30, byy + 24), (bxx - 30, byy + 24)], fill=CREAM, outline=(60, 40, 30, 255))
        d.polygon([(bxx, byy - 8), (bxx, byy - 70), (bxx + 40, byy - 10)], fill=(232, 50, 52, 255), outline=(60, 40, 30, 255))
    return finish(la.render(), cl, 2, sub_on, seed=5)


# ------------------------------------------------------------------ 镜 4 表叔沈从文
def shot4(sub_on=True):
    la = LA(17, wash_all=(0, 0, 0, 0))
    la.rect(0, 0, W, 810, wash=(253, 240, 208, 210), outline=False)
    la.rect(0, 810, W, H, wash=(232, 206, 168, 190), outline=False)
    la.line([(0, 810), (W, 810)], 6)
    la.rect(790, 110, 1130, 440, wash=(196, 224, 244, 220), w=7)                  # 窗
    la.line([(960, 110), (960, 440)], 5); la.line([(790, 275), (1130, 275)], 5)
    for (x0, x1) in ((150, 810), (1110, 1770)):                                   # 书桌
        la.rect(x0, 700, x1, 735, wash=WOOD, w=6)
        la.line([(x0 + 40, 735), (x0 + 40, 810)], 7); la.line([(x1 - 40, 735), (x1 - 40, 810)], 7)
    la.line([(1700, 700), (1700, 560)], 6); la.poly([(1650, 550), (1750, 550), (1728, 490), (1672, 490)], wash=YELLOW, w=5)   # 台灯
    la.wd.polygon([(1700, 500), (1900, 706), (1500, 706)], fill=(255, 226, 120, 85))
    la.text((960, 70), "一九四六", 64)
    la.text((1420, 310), "表叔 沈从文", 52)
    la.text((470, 310), "少年", 52)
    cl = CL()
    d = cl.d
    # 左桌：木板 + 刻刀；右桌：稿纸 + 灯光
    d.rectangle([400, 676, 580, 702], fill=(190, 130, 70, 255), outline=(60, 40, 30, 255), width=5)
    for k in range(5):
        d.line([(420 + k * 32, 684), (440 + k * 32, 698)], fill=(250, 240, 220, 255), width=4)
    d.rectangle([610, 684, 680, 700], fill=(150, 100, 56, 255), outline=(60, 40, 30, 255), width=4)
    d.polygon([(680, 684), (736, 692), (680, 700)], fill=(200, 204, 210, 255), outline=(60, 40, 30, 255))
    d.polygon([(1250, 692), (1560, 692), (1580, 712), (1230, 712)], fill=CREAM, outline=(60, 40, 30, 255))
    boy = bust(SD.boy_sprite("color", False), 0.7)
    cl.paste(boy, 470, 700 - boy.height * 1.45 / 2 + 4, 1.45)
    man = old_man("shen")
    cl.paste(man, 1420, 700 - man.height / 2 + 6, 1.0)
    def envelope(cx, cy, ang, s=1.0):
        e = Image.new("RGBA", (130, 90), (0, 0, 0, 0))
        ed = ImageDraw.Draw(e)
        ed.rectangle([4, 4, 126, 86], fill=CREAM, outline=(60, 40, 30, 255), width=4)
        ed.line([(4, 4), (65, 50), (126, 4)], fill=(60, 40, 30, 255), width=4)
        for k in range(0, 126, 22):
            ed.rectangle([k + 4, 80, k + 14, 86], fill=(232, 50, 52, 255))
            ed.rectangle([k + 15, 80, k + 25, 86], fill=(60, 110, 200, 255))
        cl.paste(e, cx, cy, s, ang)
    for k, t in enumerate(np.linspace(0.08, 0.92, 6)):
        envelope(1180 - 420 * t, 640 - 330 * math.sin(math.pi * t), 18 - 36 * t, 0.9)
    for k in range(6):
        envelope(300 + k * 4, 668 - k * 14, (k % 3 - 1) * 4, 0.9)
    cl.paste(R.seal("书信", 86), 1760, 170)
    return finish(la.render(), cl, 3, sub_on, seed=6)


# ------------------------------------------------------------------ 镜 5 北京
def shot5(sub_on=True):
    la = LA(19)
    la.rect(0, 0, W, 810, wash=(214, 226, 236, 200), outline=False)
    la.rect(0, 810, W, H, wash=(226, 204, 170, 190), outline=False)
    la.line([(0, 810), (W, 810)], 6)
    la.rect(200, 140, 1100, 640, wash=(84, 126, 104, 240), w=9)                     # 黑板
    la.rect(180, 640, 1120, 664, wash=WOOD, w=6)
    la.text((650, 100), "中央美术学院", 60)
    # 北京城楼
    la.rect(1380, 600, 1800, 810, wash=(204, 62, 52, 240), w=6)
    for cx in (1450, 1590, 1730):
        la.rect(cx - 34, 710, cx + 34, 810, wash=(70, 34, 30, 255), w=4)
    la.poly([(1340, 604), (1410, 548), (1770, 548), (1840, 604)], wash=(238, 190, 76, 250), w=6)
    la.rect(1450, 450, 1730, 548, wash=(204, 62, 52, 240), w=6)
    la.poly([(1410, 452), (1480, 392), (1700, 392), (1770, 452)], wash=(238, 190, 76, 250), w=6)
    la.text((1590, 310), "北 京", 84)
    cl = CL()
    d = cl.d
    # 黑板上的粉笔涂鸦
    d.ellipse([860, 220, 980, 340], outline=(250, 240, 200, 255), width=7)
    for k in range(8):
        a = k * math.pi / 4
        d.line([(920 + 70 * math.cos(a), 280 + 70 * math.sin(a)), (920 + 100 * math.cos(a), 280 + 100 * math.sin(a))], fill=(250, 240, 200, 255), width=6)
    d.line([(280, 560), (400, 470), (520, 560)], fill=(250, 240, 200, 255), width=7)
    d.rectangle([320, 520, 480, 580], outline=(250, 240, 200, 255), width=6)
    # 老师：高大的黑色剪影；少年：矮小、彩色、背着小包袱
    teacher = SD.boy_sprite("sil", False)
    cl.paste(teacher, 760, 540, 1.9)
    kid = SD.boy_sprite("color")
    cl.paste(kid, 400, 690, 0.9)
    cl.paste(R.seal("讲台", 86), 1100, 700)
    return finish(la.render(), cl, 4, sub_on, seed=7)


# ------------------------------------------------------------------ 镜 6 猫头鹰（夜色）
def shot6(sub_on=True):
    la = LA(21, wash_all=(46, 64, 134, 246))
    la.arcs(1100, 230, 150, 1600, seed=5)
    la.poly([(W, 280), (1650, 350), (1720, 520), (1600, 640), (1740, 800), (W, 870)], wash=(18, 22, 54, 255), outline=False)
    la.poly([(W, 160), (1800, 220), (1760, 330), (W, 400)], wash=(18, 22, 54, 255), outline=False)
    base = la.render(ink=(250, 240, 210))
    cl = CL()
    d = cl.d
    d.ellipse([1100 - 110, 230 - 110, 1100 + 110, 230 + 110], fill=(255, 238, 150, 255), outline=(255, 250, 220, 255), width=6)
    d.ellipse([1060, 190, 1100, 230], fill=(240, 220, 130, 255)); d.ellipse([1130, 250, 1156, 276], fill=(240, 220, 130, 255))
    # 树枝
    d.line([(0, 790), (400, 770), (900, 780), (1500, 740)], fill=(110, 72, 44, 255), width=54)
    d.line([(1100, 765), (1300, 640), (1420, 610)], fill=(110, 72, 44, 255), width=24)
    for lx, ly in ((1300, 640), (1420, 610), (200, 770), (1000, 770)):
        d.ellipse([lx - 34, ly - 16, lx + 34, ly + 16], fill=(90, 170, 100, 255), outline=(30, 70, 40, 255), width=4)
    owl = R.owl_sprite("open", "closed")
    cl.paste(owl, 620, 520, 1.0)
    for sx, sy, s in ((260, 150, 26), (520, 90, 30), (880, 130, 24), (1500, 120, 30), (180, 400, 22)):
        star(d, sx, sy, s)
    return finish(base, cl, 5, sub_on, seed=8, halo=0)


# ------------------------------------------------------------------ 镜 7 猴票
def shot7(sub_on=True):
    la = LA(23)
    la.glow(960, 470, 900, (255, 214, 120), 150)
    r = np.random.default_rng(6)
    for k in range(60):                                                              # 放射刻线
        a = k * 2 * math.pi / 60 + r.normal(0, 0.01)
        r0, r1 = 360 + r.random() * 40, 520 + r.random() * 380
        la.line([(960 + r0 * math.cos(a), 470 + r0 * 0.82 * math.sin(a)), (960 + r1 * math.cos(a), 470 + r1 * 0.82 * math.sin(a))], 5)
    cl = CL()
    d = cl.d
    stamp = R.compose_monkey(*R.monkey_stamp())
    for x, y, ang in ((300, 260, -12), (1620, 250, 10), (270, 650, 8), (1650, 650, -9)):
        cl.paste(stamp, x, y, 0.36, ang)
    cl.paste(stamp, 960, 470, 0.95)
    for sx, sy, s in ((600, 130, 30), (1330, 120, 34), (520, 820, 26), (1420, 820, 30), (960, 90, 24)):
        star(d, sx, sy, s)
    return finish(la.render(), cl, 6, sub_on, seed=9)


# ------------------------------------------------------------------ 镜 8 无愁河
def shot8(sub_on=True):
    la = LA(29)
    la.rect(0, 0, W, 810, wash=(253, 240, 208, 210), outline=False)
    la.rect(0, 810, W, H, wash=(232, 206, 168, 190), outline=False)
    la.line([(0, 810), (W, 810)], 6)
    la.rect(150, 720, 1020, 758, wash=WOOD, w=6)
    la.line([(190, 758), (190, 810)], 7); la.line([(980, 758), (980, 810)], 7)
    la.line([(900, 720), (900, 570)], 6); la.poly([(850, 560), (950, 560), (928, 500), (872, 500)], wash=YELLOW, w=5)
    la.wd.polygon([(900, 505), (1020, 722), (780, 722)], fill=(255, 226, 120, 85))
    base = la.render()
    mini = shot2(False).resize((900, 506), Image.LANCZOS).convert("RGBA")           # 云朵里的凤凰小景（回到镜 2）
    mask = Image.new("L", mini.size, 0)
    ImageDraw.Draw(mask).ellipse([20, 14, 880, 492], fill=255)
    mini.putalpha(mask.filter(ImageFilter.GaussianBlur(10)))
    tmp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    tmp.paste(mini, (1390 - 450, 400 - 253))
    base = Image.alpha_composite(base, tmp)
    cl = CL()
    d = cl.d
    d.polygon([(620, 712), (900, 704), (920, 724), (600, 732)], fill=CREAM, outline=(60, 40, 30, 255))
    man = old_man("huang")
    cl.paste(man, 560, 720 - man.height * 1.3 / 2 + 6, 1.3)
    for (x, y, r_) in ((850, 520, 16), (905, 470, 24), (975, 420, 34)):             # 思绪泡泡
        d.ellipse([x - r_, y - r_, x + r_, y + r_], fill=CREAM, outline=(60, 40, 30, 255), width=5)
    mag = Image.new("RGBA", (190, 250), (0, 0, 0, 0))
    md = ImageDraw.Draw(mag)
    md.rectangle([4, 4, 186, 246], fill=(210, 50, 48, 255), outline=(60, 40, 30, 255), width=5)
    md.text((95, 90), "收获", font=R.font(R.F_BRUSH, 74), fill=(255, 240, 220, 255), anchor="mm")
    md.rectangle([22, 160, 168, 172], fill=(255, 226, 120, 255))
    cl.paste(mag, 255, 650, 0.7, 8)
    d.text((300, 250), "哈 哈 哈", font=R.font(R.F_FUN, 84), fill=(250, 140, 40, 255), anchor="mm", stroke_width=3, stroke_fill=(70, 44, 30, 255))
    for sx, sy, s in ((120, 120, 28), (760, 110, 30)):
        star(d, sx, sy, s)
    return finish(base, cl, 7, sub_on, seed=10)


# ------------------------------------------------------------------ 镜 9 结尾
def shot9(sub_on=True):
    la = LA(31)
    la.glow(960, 650, 1000, (255, 190, 110), 170)
    la.hills([(560, 120, 81, (170, 198, 220, 150), 3), (650, 70, 93, (190, 214, 170, 160), 2)])
    la.rect(0, 660, W, 940, wash=BLUE_W, outline=False)
    la.waves(700, 930, 26)
    for k in range(5):                                                               # 石阶
        y = 700 + k * 52
        la.poly([(0, y), (620 - k * 70, y), (620 - k * 70 + 26, y + 52), (0, y + 52)], wash=(206, 200, 188, 245), w=5)
    base = la.render()
    cl = CL()
    d = cl.d
    d.pieslice([960 - 130, 660 - 130, 960 + 130, 660 + 130], 180, 360, fill=(255, 176, 60, 255), outline=(70, 44, 30, 255), width=7)
    for k in range(5):                                                               # 水中倒影
        d.line([(960 - 90 + k * 8, 700 + k * 44), (960 + 90 - k * 8, 700 + k * 44)], fill=(255, 190, 80, 255), width=8)
    # 小包袱放在石阶上
    d.line([(250, 770), (440, 690)], fill=(150, 100, 56, 255), width=10)
    bundle_crayon(cl, 330, 726, 52)
    # 猴子（去掉邮票底）和猫头鹰从水面浮现、眨眼
    parts = R.monkey_stamp()
    monkey = Image.alpha_composite(parts[2], parts[1])
    monkey = monkey.crop(monkey.getbbox())
    cl.paste(monkey, 1200, 480, 0.62)
    cl.paste(R.owl_sprite("closed", "open"), 1580, 500, 0.62)
    for sx, sy, s in ((900, 160, 30), (1380, 140, 34), (700, 280, 24), (1750, 260, 28), (1090, 250, 22)):
        star(d, sx, sy, s)
    return finish(base, cl, 8, sub_on, seed=11)


SHOTS = [shot1, shot2, shot3, shot4, shot5, shot6, shot7, shot8, shot9]

if __name__ == "__main__":
    os.chdir(R.HERE)
    os.makedirs("stills_v2", exist_ok=True)
    pick = [int(a) for a in sys.argv[1:]] or list(range(1, 10))
    for i in pick:
        SHOTS[i - 1]().save(f"stills_v2/shot{i}.jpg", quality=90)
        print("->", f"stills_v2/shot{i}.jpg", flush=True)
    if len(pick) == 9:
        sheet = Image.new("RGB", (1920, 1080 * 3 // 2 * 1 + 0))
        sheet = Image.new("RGB", (1920, 1080 * 3 // 2 + 0), (255, 255, 255))
        sheet = Image.new("RGB", (3 * 640, 3 * 360))
        for k in range(9):
            sheet.paste(Image.open(f"stills_v2/shot{k + 1}.jpg").resize((640, 360), Image.LANCZOS), ((k % 3) * 640, (k // 3) * 360))
        sheet.save("stills_v2/contact_sheet.jpg", quality=88)
