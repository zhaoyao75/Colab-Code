"""
V2 动态渲染：明亮木刻线刻 + 蜡笔涂鸦，1920×1080，30fps，128 秒，9 镜。
背景（线刻）静止，会动的角色是单独的蜡笔精灵，每个角色预生成 3 个轻微抖动的版本、每秒轮换约 6 次（手绘「逐帧抖动」感）。

  python3 animate.py --still 5 20 35        # 指定秒数出静帧 -> anim_stills/
  python3 animate.py --demo 42              # 前 42 秒 demo（镜 1–3 有动画，镜 4 起为静帧占位）-> demo_v2.mp4
  python3 animate.py                        # 完整 128 秒 -> huang_yongyu_v2.mp4
依赖：script_v2.json、timings_v2.json、mix_v2.wav（先跑 tts.py script_v2.json _v2 与 mix.py _v2）。
"""
import json, math, os, subprocess, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter, map_coordinates
import render as R
import styles_demo as SD
import shots as SH

W, H, FPS, LENGTH = 1920, 1080, 30, 128.0
HERE = R.HERE
SCRIPT = json.load(open(os.path.join(HERE, "script_v2.json"), encoding="utf-8"))
CUES = json.load(open(os.path.join(HERE, "timings_v2.json"), encoding="utf-8"))
STARTS = SCRIPT["starts"]
ENDS = STARTS[1:] + [LENGTH]
XF = 0.5                      # 镜头间交叉淡化的半宽（秒）
BOIL = 6                      # 蜡笔抖动：每秒轮换次数
NVAR = 3


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def ease_back(x, k=1.7):
    x = min(1.0, max(0.0, x)) - 1
    return 1 + (k + 1) * x ** 3 + k * x ** 2


def seg(t, a, b):
    return smooth((t - a) / (b - a))


# ------------------------------------------------------------------ 蜡笔精灵
def crayon_rgba(spr, seed=1, wobble=2.0, grain=0.3, halo=4):
    """给小图加手绘抖动 + 蜡笔颗粒 + 白边（不同 seed 抖动不同）"""
    pad = halo * 3 + 10
    w, h = spr.size
    cv = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    cv.paste(spr, (pad, pad))
    a = np.asarray(cv, np.float32)
    h2, w2 = a.shape[:2]
    r = np.random.default_rng(seed)
    dx = gaussian_filter(r.standard_normal((h2, w2)), 14)
    dy = gaussian_filter(r.standard_normal((h2, w2)), 14)
    dx, dy = dx / dx.std() * wobble, dy / dy.std() * wobble
    yy, xx = np.mgrid[0:h2, 0:w2].astype(np.float32)
    out = np.stack([map_coordinates(a[..., c], [yy + dy, xx + dx], order=1, mode="nearest") for c in range(4)], -1)
    g = gaussian_filter(r.random((h2, w2)), (1.0, 3.0))
    g = (g - g.min()) / (g.max() - g.min() + 1e-6)
    out[..., 3] *= (1 - grain) + grain * np.clip((g - 0.25) * 2.2, 0, 1)
    im = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")
    return SD.with_halo(im, halo) if halo else im


def sprite(w, h, fn):
    """用 2 倍超采样画一个 w×h 的精灵；fn(d, S) 里的坐标都要乘 S"""
    S = 2
    im = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    fn(ImageDraw.Draw(im), S)
    return im.resize((w, h), Image.LANCZOS)


class Actor:
    """fn(tl) -> dict(x, y, scale, angle, alpha) 或 None；x、y 是精灵中心在画面上的位置"""
    def __init__(self, spr, fn, halo=4, boil=True, wobble=2.0, crayon=True):
        n = NVAR if boil else 1
        if crayon:
            self.vars = [crayon_rgba(spr, seed=11 + k * 7, wobble=wobble, halo=halo) for k in range(n)]
        else:
            self.vars = [spr.convert("RGBA")]
        self.fn = fn

    def draw(self, canvas, tl, t):
        st = self.fn(tl)
        if not st:
            return
        al = st.get("alpha", 1.0)
        if al <= 0.01:
            return
        im = self.vars[int(t * BOIL) % len(self.vars)]
        sc, ang = st.get("scale", 1.0), st.get("angle", 0.0)
        if sc != 1.0:
            im = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))), Image.BICUBIC)
        if ang:
            im = im.rotate(ang, Image.BICUBIC, expand=True)
        if al < 0.99:
            im = im.copy()
            im.putalpha(im.getchannel("A").point(lambda v: int(v * al)))
        canvas.paste(im, (int(st["x"] - im.width / 2), int(st["y"] - im.height / 2)), im)


class Scene:
    def __init__(self, base, actors=(), zoom=1.035, pan=(0, 0)):
        self.base, self.actors, self.zoom, self.pan = base.convert("RGB"), list(actors), zoom, pan

    def frame(self, tl, dur, t):
        tl = max(0.0, tl)
        img = self.base.copy()
        for a in self.actors:
            a.draw(img, tl, t)
        z = 1 + (self.zoom - 1) * min(1.0, tl / dur)
        if z > 1.0005:
            cw, ch = W / z, H / z
            ox = (W - cw) / 2 + self.pan[0] * (z - 1) * 10
            oy = (H - ch) / 2 + self.pan[1] * (z - 1) * 10
            img = img.crop((int(ox), int(oy), int(ox + cw), int(oy + ch))).resize((W, H), Image.BICUBIC)
        return img


# ------------------------------------------------------------------ 镜 1：钩子
def build_scene1():
    bg = SD.light_woodcut_bg().convert("RGBA")
    cl = SH.CL()
    d = cl.d
    sx, sy = 1380, 300
    d.ellipse([sx - 125, sy - 125, sx + 125, sy + 125], fill=(255, 190, 50, 255), outline=(70, 44, 30, 255), width=7)
    for rr in (92, 60, 28):
        d.ellipse([sx - rr, sy - rr, sx + rr, sy + rr], outline=(235, 110, 40, 255), width=7)
    for stx, sty, s in ((300, 90, 30), (700, 150, 26), (1700, 120, 34), (1750, 420, 26)):
        SH.star(d, stx, sty, s)
    base = cl.finish(bg, seed=3, wobble=2.2, grain=0.3, halo=5).convert("RGB")
    R.place(base, R.seal("童心", 96), 1780, 960)

    def car_sprite(S_=None):
        def fn(d, S):
            SD.draw_car(d, 320 * S, 232 * S, fill=(232, 50, 52, 255), line=(40, 34, 44, 255), s=S, off=7, lw=10)
        return sprite(640, 260, fn)

    def speed_sprite():
        def fn(d, S):
            for k in range(4):
                d.line([(170 * S, (20 + k * 28) * S), (20 * S, (20 + k * 28) * S)], fill=(40, 34, 44, 255), width=6 * S)
        return sprite(190, 130, fn)

    def beam_sprite():
        def fn(d, S):
            for k in range(18):
                d.line([(0, 95 * S), (240 * S, (4 + k * 10.5) * S)], fill=(255, 214, 70, 255), width=6 * S)
        return sprite(250, 200, fn)

    def boy_sprite_img():
        b = SD.boy_sprite("sil")
        d = ImageDraw.Draw(b)
        hx, hy = 223, 34                                                   # 包袱
        d.ellipse([hx - 30, hy - 27, hx + 30, hy + 27], fill=(250, 150, 40, 255), outline=(60, 40, 30, 255), width=4)
        for dx, dy in ((-13, -5), (8, -13), (12, 9), (-5, 12)):
            d.ellipse([hx + dx - 5, hy + dy - 5, hx + dx + 5, hy + dy + 5], fill=(220, 40, 50, 255))
        return b

    def title_sprite():
        def fn(d, S):
            d.rounded_rectangle([6 * S, 6 * S, 514 * S, 254 * S], radius=24 * S, fill=(250, 245, 230, 240))
            d.text((260 * S, 112 * S), "黄永玉", font=R.font(R.F_BRUSH, 150 * S), fill=(30, 26, 28, 255), anchor="mm")
            d.line([(70 * S, 205 * S), (450 * S, 201 * S), (430 * S, 217 * S), (80 * S, 219 * S)], fill=(232, 50, 52, 255), width=10 * S)
        return sprite(520, 260, fn)

    CAR_BASE = 900

    def car_x(tl):
        return -360 + 800 * ease_out((tl - 0.3) / 3.9)

    def car_fn(tl):
        p = (tl - 0.3) / 3.9
        bob = 2.5 * math.sin(tl * 11) if p < 1 else 1.2 * math.sin(tl * 3)
        return dict(x=car_x(tl), y=CAR_BASE - 102 + bob)

    def speed_fn(tl):
        p = (tl - 0.3) / 3.9
        if not (0 < p < 0.85):
            return None
        return dict(x=car_x(tl) - 250, y=CAR_BASE - 150 + 40, alpha=1 - p)

    def beam_fn(tl):
        p = smooth((tl - 3.2) / 1.2)
        return dict(x=car_x(tl) + 258 + 125, y=CAR_BASE - 72 - 6, alpha=p * (0.85 + 0.15 * math.sin(tl * 9)))

    def boy_fn(tl):
        walk = seg(tl, 6.4, 13.4)
        sc = 1.5 - 0.45 * walk
        x = 1000 + 55 * walk
        y = 700 - 127 * walk
        ph = tl * 5.5
        bob = (6 if walk < 1 else 3) * abs(math.sin(ph)) * (0.6 + 0.4 * (1 - walk))
        return dict(x=x, y=y - bob, scale=sc, angle=2 * math.sin(ph), alpha=seg(tl, 0.2, 1.0))

    def title_fn(tl):
        return dict(x=345, y=343, alpha=seg(tl, 0.4, 1.2) * (1 - seg(tl, 11.8, 12.8)))
    actors = [Actor(title_sprite(), title_fn, halo=0), Actor(speed_sprite(), speed_fn, halo=0), Actor(car_sprite(), car_fn),
              Actor(beam_sprite(), beam_fn, halo=0), Actor(boy_sprite_img(), boy_fn)]
    return Scene(base, actors, zoom=1.04)


# ------------------------------------------------------------------ 镜 2：凤凰
def build_scene2():
    SH.SKIP = {"boat", "birds"}
    base = SH.shot2(False)
    SH.SKIP = set()

    def boat_fn_sprite():
        def fn(d, S):
            bx, by = 150 * S, 70 * S
            d.polygon([(bx - 110 * S, by), (bx + 110 * S, by), (bx + 70 * S, by + 46 * S), (bx - 70 * S, by + 46 * S)], fill=SH.CREAM, outline=(60, 40, 30, 255))
            d.polygon([(bx - 100 * S, by + 8 * S), (bx + 100 * S, by + 8 * S), (bx + 92 * S, by + 22 * S), (bx - 92 * S, by + 22 * S)], fill=(232, 50, 52, 255))
            d.rectangle([bx - 20 * S, by - 16 * S, bx + 40 * S, by - 2 * S], fill=(150, 100, 56, 255), outline=(60, 40, 30, 255), width=3 * S)
            d.polygon([(bx + 40 * S, by - 16 * S), (bx + 96 * S, by - 8 * S), (bx + 40 * S, by - 2 * S)], fill=(200, 204, 210, 255), outline=(60, 40, 30, 255))
        return sprite(300, 150, fn)

    def bird_sprite(up):
        def fn(d, S):
            wy = 4 if up else 26
            d.line([(4 * S, wy * S), (32 * S, 24 * S), (60 * S, wy * S)], fill=(40, 34, 44, 255), width=5 * S, joint="curve")
        return sprite(64, 40, fn)

    def boat_fn(tl):
        p = tl / 14.0
        return dict(x=330 + 900 * p, y=745 + 6 * math.sin(tl * 1.4), angle=2.5 * math.sin(tl * 1.1), alpha=seg(tl, 0.0, 0.8))

    class Bird(Actor):
        def __init__(self, k):
            self.k = k
            self.up, self.down = bird_sprite(True), bird_sprite(False)
            self.vars_up = [crayon_rgba(self.up, 5 + k, 1.2, halo=0)]
            self.vars_dn = [crayon_rgba(self.down, 9 + k, 1.2, halo=0)]
            self.vars = self.vars_up

        def draw(self, canvas, tl, t):
            k = self.k
            x = -80 + ((tl * 85 + k * 230) % 2100)
            y = 140 + k * 26 + 12 * math.sin(tl * 1.3 + k)
            im = (self.vars_up if int((t + k * 0.07) * 4) % 2 == 0 else self.vars_dn)[0]
            canvas.paste(im, (int(x - im.width / 2), int(y - im.height / 2)), im)
    actors = [Actor(boat_fn_sprite(), boat_fn)] + [Bird(k) for k in range(5)]
    return Scene(base, actors, zoom=1.035, pan=(1, 0))


# ------------------------------------------------------------------ 镜 3：出走
def build_scene3():
    SH.SKIP = {"route", "boy", "sails"}
    base = SH.shot3(False)
    SH.SKIP = set()
    route = SH.ROUTE3
    y0 = 600

    def route_img():
        band = Image.new("RGBA", (W, 260), (0, 0, 0, 0))
        SH.dashed(ImageDraw.Draw(band), [(x, y - y0) for x, y in route], (232, 50, 52, 255), 10)
        return [crayon_rgba(band, 3 + 5 * k, 1.8, halo=0) for k in range(NVAR)]
    ROUTE = route_img()
    xs = [p[0] for p in route]
    ys = [p[1] for p in route]
    X0, X1 = 140, 1590

    def boy_x(tl):
        return X0 + (X1 - X0) * seg(tl, 1.5, 12.5)

    class Route(Actor):
        def __init__(self):
            pass

        def draw(self, canvas, tl, t):
            front = int(boy_x(tl) + 40)
            if front <= 20:
                return
            im = ROUTE[int(t * BOIL) % NVAR]
            front = min(front, im.width)
            piece = im.crop((0, 0, front, im.height))
            canvas.paste(piece, (0, y0 - 30), piece)

    def boy_sprite_img():
        b = SD.boy_sprite("sil")
        d = ImageDraw.Draw(b)
        hx, hy = 223, 34
        d.ellipse([hx - 30, hy - 27, hx + 30, hy + 27], fill=(250, 150, 40, 255), outline=(60, 40, 30, 255), width=4)
        for dx, dy in ((-13, -5), (8, -13), (12, 9), (-5, 12)):
            d.ellipse([hx + dx - 5, hy + dy - 5, hx + dx + 5, hy + dy + 5], fill=(220, 40, 50, 255))
        return b

    def boy_fn(tl):
        x = boy_x(tl)
        y = float(np.interp(x, xs, ys)) - 133
        ph = tl * 4.6
        walking = 0 < seg(tl, 1.5, 12.5) < 1
        bob = (7 if walking else 2) * abs(math.sin(ph))
        return dict(x=x, y=y - bob, scale=0.95, angle=2.5 * math.sin(ph), alpha=seg(tl, 0.3, 1.2))

    def sail_sprite():
        def fn(d, S):
            d.polygon([(10 * S, 78 * S), (110 * S, 78 * S), (90 * S, 102 * S), (30 * S, 102 * S)], fill=SH.CREAM, outline=(60, 40, 30, 255))
            d.polygon([(60 * S, 70 * S), (60 * S, 8 * S), (100 * S, 68 * S)], fill=(232, 50, 52, 255), outline=(60, 40, 30, 255))
        return sprite(120, 110, fn)
    sail = sail_sprite()
    actors = [Route(), Actor(boy_sprite_img(), boy_fn)]
    for k, (sx, sy) in enumerate(SH.SAILS3):
        def fn(tl, sx=sx, sy=sy, k=k):
            return dict(x=sx + (60 if k == 0 else -80) * tl / 14, y=sy + 6 * math.sin(tl * 1.5 + k) + 12, angle=2 * math.sin(tl * 1.2 + k))
        actors.append(Actor(sail, fn))
    return Scene(base, actors, zoom=1.035, pan=(-1, 0))


# ------------------------------------------------------------------ 镜 4：表叔沈从文
def envelope_sprite():
    def fn(d, S):
        d.rectangle([4 * S, 4 * S, 126 * S, 86 * S], fill=SH.CREAM, outline=(60, 40, 30, 255), width=4 * S)
        d.line([(4 * S, 4 * S), (65 * S, 50 * S), (126 * S, 4 * S)], fill=(60, 40, 30, 255), width=4 * S)
        for k in range(0, 126, 22):
            d.rectangle([(k + 4) * S, 80 * S, (k + 14) * S, 86 * S], fill=(232, 50, 52, 255))
            d.rectangle([(k + 15) * S, 80 * S, (k + 25) * S, 86 * S], fill=(60, 110, 200, 255))
    return sprite(130, 90, fn)


def build_scene4():
    SH.SKIP = {"people", "env"}
    base = SH.shot4(False)
    SH.SKIP = set()
    boy = SH.bust(SD.boy_sprite("color", False), 0.7)
    man = SH.old_man_alt("shen", SH.STYLE) if SH.STYLE else SH.old_man("shen")
    boy_y = 700 - boy.height * 1.45 / 2 + 4
    man_y = 700 - man.height / 2 + 6

    def boy_fn(tl):                                                  # 少年低头刻木板
        return dict(x=470, y=boy_y + 3 * abs(math.sin(tl * 4.2)), scale=1.45, angle=1.5 * math.sin(tl * 2.1), alpha=seg(tl, 0, 0.6))

    def man_fn(tl):                                                  # 老先生写信
        return dict(x=1420, y=man_y + 2.5 * math.sin(tl * 3.0), angle=0.8 * math.sin(tl * 1.5), alpha=seg(tl, 0.2, 0.8))
    launches = [1.8, 3.4] + [5.5 + 0.85 * k for k in range(6)]
    FLY = 2.0
    env = envelope_sprite()

    class Letters(Actor):
        def __init__(self):
            self.vars = [crayon_rgba(env, 21 + 7 * k, 1.5, halo=3) for k in range(NVAR)]

        def draw(self, canvas, tl, t):
            im0 = self.vars[int(t * BOIL) % NVAR]
            for k, l0 in enumerate(launches):
                p = (tl - l0) / FLY
                if p < 0:
                    continue
                ex, ey = 310 + k * 4, 668 - k * 14
                if p < 1:
                    q = smooth(p)
                    x, y = 1230 + (ex - 1230) * q, 660 + (ey - 660) * q - 300 * math.sin(math.pi * p)
                    ang = 18 - 36 * q
                else:
                    x, y, ang = ex, ey, (k % 3 - 1) * 4
                im = im0.resize((int(im0.width * 0.9), int(im0.height * 0.9)), Image.BICUBIC).rotate(ang, Image.BICUBIC, expand=True)
                canvas.paste(im, (int(x - im.width / 2), int(y - im.height / 2)), im)
    return Scene(base, [Actor(boy, boy_fn), Actor(man, man_fn), Letters()], zoom=1.03)


# ------------------------------------------------------------------ 镜 5：北京
def build_scene5():
    SH.SKIP = {"people"}
    base = SH.shot5(False)
    SH.SKIP = set()
    teacher = SD.boy_sprite("sil", False)
    kid = SD.boy_sprite("color")

    def teacher_fn(tl):                                              # 老师讲课：轻轻摇摆
        return dict(x=760, y=540 + 3 * math.sin(tl * 2.2), scale=1.9, angle=1.6 * math.sin(tl * 1.3), alpha=seg(tl, 0, 0.7))

    def kid_fn(tl):                                                  # 「心里的少年」从左边蹦进来
        p = seg(tl, 4.2, 5.6)
        if p <= 0:
            return None
        hop = 26 * abs(math.sin(math.pi * min(1.0, (tl - 4.2) / 1.4) * 3)) * (1 - p) if p < 1 else 0
        bob = 4 * abs(math.sin(tl * 3.4)) if p >= 1 else 0
        return dict(x=150 + 250 * p, y=690 - hop - bob, scale=0.9, angle=4 * math.sin(tl * 3.4), alpha=min(1.0, p * 3))
    return Scene(base, [Actor(teacher, teacher_fn), Actor(kid, kid_fn)], zoom=1.03, pan=(-1, 0))


# ------------------------------------------------------------------ 镜 6：猫头鹰
def star_sprite(s=30):
    def fn(d, S):
        pts = [(s + (s if k % 2 == 0 else s * 0.42) * math.sin(k * math.pi / 5), s - (s if k % 2 == 0 else s * 0.42) * math.cos(k * math.pi / 5)) for k in range(10)]
        d.polygon([(x * S, y * S) for x, y in pts], fill=(255, 214, 70, 255), outline=(70, 44, 30, 255))
    return sprite(int(2 * s + 4), int(2 * s + 4), fn)


def twinkle_actors(stars, ph0=0.0):
    out = []
    for k, (sx, sy, s) in enumerate(stars):
        spr = star_sprite(s)

        def fn(tl, sx=sx, sy=sy, k=k):
            return dict(x=sx, y=sy, alpha=0.55 + 0.45 * math.sin(tl * 2.2 + k * 1.7 + ph0), scale=0.92 + 0.1 * math.sin(tl * 2.2 + k))
        out.append(Actor(spr, fn, halo=0, boil=False))
    return out


def build_scene6():
    SH.SKIP = {"block", "owl", "stars"}
    base = SH.shot6(False)
    SH.SKIP = set()
    states = {"oc": R.owl_sprite("open", "closed"), "cc": R.owl_sprite("closed", "closed"), "co": R.owl_sprite("closed", "open")}

    def state(tl):
        for b0 in (1.6, 7.3, 12.4):
            if b0 < tl < b0 + 0.22:
                return "cc"
        return "co" if 7.5 <= tl < 11.5 else "oc"

    def mk(key):
        def fn(tl):
            if state(tl) != key:
                return None
            return dict(x=620, y=520 + 6 * math.sin(tl * 2.2), angle=3 * math.sin(tl * 1.3), alpha=seg(tl, 0, 0.6))
        return Actor(states[key], fn, halo=0)
    ox, oy, bw, bh = 1460, 120, 470, 780

    def block_sprite():
        def fn(d, S):
            for poly in SH.BLOCK6:
                d.polygon([((x - ox) * S, (y - oy) * S) for x, y in poly], fill=(18, 22, 54, 255))
        return sprite(bw, bh, fn)

    def block_fn(tl):                                               # 压力：「风波」从右边压过来，后来退去
        a = seg(tl, 3.0, 5.0) - seg(tl, 5.6, 7.4)
        if a <= 0.01:
            return None
        return dict(x=ox + bw / 2 + (1 - a) * 560 + 8 * math.sin(tl * 3) * a, y=oy + bh / 2)
    actors = [mk("oc"), mk("cc"), mk("co"), Actor(block_sprite(), block_fn, halo=0)] + twinkle_actors(SH.STARS6)
    return Scene(base, actors, zoom=1.035)


# ------------------------------------------------------------------ 镜 7：猴票
def build_scene7():
    SH.SKIP = {"stamps", "stars"}
    base = SH.shot7(False)
    SH.SKIP = set()
    stamp = R.compose_monkey(*R.monkey_stamp())

    def main_fn(tl):
        g = ease_back((tl - 0.4) / 1.0, 1.6)
        if g <= 0.02:
            return None
        return dict(x=960, y=470 + 5 * math.sin(tl * 1.8), scale=0.95 * g * (1 + 0.012 * math.sin(tl * 2.4)), angle=2.2 * math.sin(tl * 1.3))
    actors = [Actor(stamp, main_fn, halo=5)]
    starts = [(-260, 220), (2180, 200), (-260, 760), (2180, 760)]
    for k, ((tx, ty, ang), (sx, sy)) in enumerate(zip(SH.SMALL7, starts)):
        def fn(tl, tx=tx, ty=ty, ang=ang, sx=sx, sy=sy, k=k):
            p = ease_back((tl - (5.6 + 0.5 * k)) / 1.2, 1.2)
            if p <= 0:
                return None
            return dict(x=sx + (tx - sx) * p, y=sy + (ty - sy) * p + 6 * math.sin(tl * 1.7 + k), scale=0.36, angle=ang + (1 - min(p, 1.0)) * 180 + 3 * math.sin(tl * 1.4 + k))
        actors.append(Actor(stamp, fn, halo=4))
    actors += twinkle_actors(SH.STARS7)
    return Scene(base, actors, zoom=1.03)


# ------------------------------------------------------------------ 镜 8：无愁河
def build_scene8():
    SH.SKIP = {"man", "bubbles", "haha", "cloud"}
    base = SH.shot8(False)
    SH.SKIP = set()
    man = SH.old_man_alt("huang", SH.STYLE) if SH.STYLE else SH.old_man("huang")
    man_y = 720 - man.height * 1.3 / 2 + 6

    def man_fn(tl):                                                  # 写字，后半段哈哈大笑，抖得更厉害
        laugh = seg(tl, 9.6, 10.2)
        return dict(x=560, y=man_y + (2 + 7 * laugh) * math.sin(tl * (3 + 5 * laugh)), scale=1.3 * (1 + 0.02 * laugh * math.sin(tl * 8)),
                    angle=(0.8 + 2.2 * laugh) * math.sin(tl * (2 + 4 * laugh)), alpha=seg(tl, 0, 0.6))

    def cloud_fn(tl):
        return dict(x=1390, y=400 + 6 * math.sin(tl * 1.1), alpha=seg(tl, 6.4, 7.8))
    actors = [Actor(man, man_fn), Actor(SH.cloud8(), cloud_fn, crayon=False)]
    for k, (bx, by, br) in enumerate(SH.BUBBLES8):
        def bsp(br=br):
            def fn(d, S):
                d.ellipse([5 * S, 5 * S, (2 * br + 5) * S, (2 * br + 5) * S], fill=SH.CREAM, outline=(60, 40, 30, 255), width=5 * S)
            return sprite(2 * br + 10, 2 * br + 10, fn)

        def fn(tl, bx=bx, by=by, k=k):
            p = seg(tl, 5.8 + 0.35 * k, 6.4 + 0.35 * k)
            return dict(x=bx, y=by - 8 * (1 - p) + 3 * math.sin(tl * 2 + k), scale=0.5 + 0.5 * p, alpha=p * seg(7.8 - tl, 0, 0.3) if tl > 7.8 else p)
        actors.append(Actor(bsp(), fn, halo=3))

    def haha_sprite():
        def fn(d, S):
            d.text((210 * S, 65 * S), "哈 哈 哈", font=R.font(R.F_FUN, 84 * S), fill=(250, 140, 40, 255), anchor="mm", stroke_width=3 * S, stroke_fill=(70, 44, 30, 255))
        return sprite(420, 130, fn)

    def haha_fn(tl):
        p = seg(tl, 9.6, 10.1)
        if p <= 0:
            return None
        return dict(x=300, y=250 - 12 * abs(math.sin(tl * 5)), scale=(0.7 + 0.3 * p) * (1 + 0.06 * math.sin(tl * 7)), angle=4 * math.sin(tl * 4), alpha=p)
    actors.append(Actor(haha_sprite(), haha_fn, halo=3))
    actors += twinkle_actors(((120, 120, 28), (760, 110, 30)))
    return Scene(base, actors, zoom=1.03, pan=(1, 0))


# ------------------------------------------------------------------ 镜 9：结尾
def build_scene9():
    SH.SKIP = {"creatures", "stars"}
    base = SH.shot9(False)
    SH.SKIP = set()
    parts = R.monkey_stamp()
    monkey = Image.alpha_composite(parts[2], parts[1])
    monkey = monkey.crop(monkey.getbbox())
    monkey = monkey.resize((int(monkey.width * 0.62), int(monkey.height * 0.62)), Image.LANCZOS)
    own = {k: R.owl_sprite(*v) for k, v in {"oc": ("closed", "open"), "cc": ("closed", "closed"), "co": ("open", "closed")}.items()}
    own = {k: v.resize((int(v.width * 0.62), int(v.height * 0.62)), Image.LANCZOS) for k, v in own.items()}
    T_M, T_O = 8.6, 9.4                                               # 对齐旁白「可他的猴子，他的猫头鹰……」

    def rise(tl, t0, y_end):
        p = ease_out((tl - t0) / 1.8)
        return p, y_end + (1 - p) * 220

    def monkey_fn(tl):
        p, y = rise(tl, T_M, 480)
        if p <= 0.01:
            return None
        wink = 1 if 12.0 < tl < 12.25 or 15.0 < tl < 15.25 else 0
        return dict(x=1200, y=y + 6 * math.sin(tl * 1.9), alpha=p, angle=3 * math.sin(tl * 1.5))

    def owl_state(tl):
        return "cc" if (11.6 < tl < 11.85 or 14.6 < tl < 14.85) else ("co" if tl > 12.6 and tl < 14.4 else "oc")

    def mk_owl(key):
        def fn(tl):
            p, y = rise(tl, T_O, 500)
            if p <= 0.01 or owl_state(tl) != key:
                return None
            return dict(x=1580, y=y + 6 * math.sin(tl * 1.9 + 1), alpha=p, angle=-3 * math.sin(tl * 1.5))
        return Actor(own[key], fn, halo=0)

    def card():
        im = R.paper(W, H, 53, base=(247, 240, 222), vig=False).convert("RGBA")
        d = ImageDraw.Draw(im)
        d.text((W / 2, 380), "黄永玉", font=R.font(R.F_BRUSH, 210), fill=(34, 30, 32, 255), anchor="mm")
        d.text((W / 2, 545), "1924 — 2023", font=R.font(R.F_EN, 52), fill=(80, 70, 60, 255), anchor="mm")
        d.text((W / 2, 670), "永远长不大的老头", font=R.font(R.F_BRUSH, 84), fill=(200, 38, 42, 255), anchor="mm")
        d.text((W - 50, H - 50), "画面与配乐由代码生成，AI 配音", font=R.font(R.F_FUN, 28), fill=(130, 120, 110, 255), anchor="rs")
        return im
    actors = [Actor(monkey, monkey_fn, halo=4)] + [mk_owl(k) for k in own] + twinkle_actors(SH.STARS9)
    actors.append(Actor(card(), lambda tl: dict(x=W / 2, y=H / 2, alpha=seg(tl, 17.2, 18.6)), crayon=False))
    return Scene(base, actors, zoom=1.03, pan=(0, 1))


# ------------------------------------------------------------------ 其余镜头：先用静帧占位
def static_scene(i):
    fn = SH.SHOTS[i]
    return Scene(fn(False), [], zoom=1.03)


BUILDERS = {0: build_scene1, 1: build_scene2, 2: build_scene3, 3: build_scene4, 4: build_scene5, 5: build_scene6,
            6: build_scene7, 7: build_scene8, 8: build_scene9}
SCENES = []
SUBS = []


def build_subs():
    """只要中文、单行：每条字幕来自 subs.py 的切分（≤30 字）"""
    import subs
    fz = R.font(R.F_FUN, 56)
    out = []
    for sg in subs.load():
        tw = fz.getlength(sg["text"])
        hgt = 84
        im = Image.new("RGBA", (W, hgt + 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([(W - tw) / 2 - 36, 0, (W + tw) / 2 + 36, hgt], radius=24, fill=(20, 16, 12, 130))
        d.text((W / 2, hgt / 2 + 2), sg["text"], font=fz, fill=(255, 252, 240, 255), anchor="mm", stroke_width=3, stroke_fill=(30, 22, 15, 255))
        out.append((sg["start"] - 0.05, sg["end"] + 0.2, im, H - 56 - hgt))
    return out


def setup(n_scenes):
    global SCENES, SUBS
    for i in range(n_scenes):
        SCENES.append(BUILDERS[i]() if i in BUILDERS else static_scene(i))
        print("  scene", i + 1, "ready", flush=True)
    SUBS = build_subs()


def scene_frame(i, t):
    return SCENES[i].frame(t - STARTS[i], ENDS[i] - STARTS[i], t)


def render(t, with_subs=True):
    n = len(SCENES)
    i = max(k for k in range(n) if STARTS[k] <= t)
    img = scene_frame(i, t)
    if i + 1 < n and t > STARTS[i + 1] - XF:
        img = Image.blend(img, scene_frame(i + 1, t), smooth((t - (STARTS[i + 1] - XF)) / (2 * XF)))
    elif i > 0 and t < STARTS[i] + XF:
        img = Image.blend(scene_frame(i - 1, t), img, smooth((t - (STARTS[i] - XF)) / (2 * XF)))
    if t < 0.5:
        img = Image.blend(Image.new("RGB", (W, H), (247, 240, 222)), img, t / 0.5)
    for s0, s1, im, y in (SUBS if with_subs else []):
        if s0 <= t <= s1:
            a = min(1.0, (t - s0) / 0.15, (s1 - t) / 0.15)
            if a < 0.99:
                im = im.copy()
                im.putalpha(im.getchannel("A").point(lambda v: int(v * a)))
            img.paste(im, (0, y), im)
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


def make_video(t1, out, procs=4):
    f1 = int(t1 * FPS)
    step = math.ceil(f1 / procs)
    jobs = [(a, min(f1, a + step), f".seg_a{k}.mp4") for k, a in enumerate(range(0, f1, step))]
    with Pool(len(jobs)) as pool:
        segs = pool.map(encode, jobs)
    with open(".segs_a.txt", "w") as fh:
        fh.writelines(f"file '{s}'\n" for s in segs)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", ".segs_a.txt",
                    "-t", str(t1), "-i", "mix_v2.wav", "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", out], check=True)
    for s in segs:
        os.remove(s)
    os.remove(".segs_a.txt")
    print("->", out)


if __name__ == "__main__":
    os.chdir(HERE)
    if len(sys.argv) > 1 and sys.argv[1] == "--still":
        setup(9)
        os.makedirs("anim_stills", exist_ok=True)
        for a in sys.argv[2:]:
            render(float(a)).save(f"anim_stills/t{float(a):05.1f}.jpg", quality=90)
            print("anim_stills/", a)
    elif len(sys.argv) > 1 and sys.argv[1] == "--demo":
        setup(4)
        make_video(float(sys.argv[2]) if len(sys.argv) > 2 else 42.0, "demo_v2.mp4")
    else:
        setup(9)
        make_video(LENGTH, "huang_yongyu_v2.mp4")
