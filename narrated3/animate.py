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
    def __init__(self, spr, fn, halo=4, boil=True, wobble=2.0):
        n = NVAR if boil else 1
        self.vars = [crayon_rgba(spr, seed=11 + k * 7, wobble=wobble, halo=halo) for k in range(n)]
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


# ------------------------------------------------------------------ 其余镜头：先用静帧占位
def static_scene(i):
    fn = SH.SHOTS[i]
    return Scene(fn(False), [], zoom=1.03)


BUILDERS = {0: build_scene1, 1: build_scene2, 2: build_scene3}
SCENES = []
SUBS = []


def build_subs():
    import vertical as V
    fz, fe = R.font(R.F_FUN, 52), R.font(R.F_EN, 32)
    out = []
    for c in CUES:
        zl = V.wrap_zh(c["zh"], fz, 1500)
        el = R.wrap_en(c["en"], fe, 1500)
        hgt = 30 + 62 * len(zl) + 10 + 42 * len(el)
        bw = max([fz.getlength(l) for l in zl] + [fe.getlength(l) for l in el]) + 70
        im = Image.new("RGBA", (W, hgt + 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([(W - bw) / 2, 0, (W + bw) / 2, hgt], radius=22, fill=(20, 16, 12, 130))
        y = 20 + 31
        for l in zl:
            d.text((W / 2, y), l, font=fz, fill=(255, 252, 240, 255), anchor="mm", stroke_width=3, stroke_fill=(30, 22, 15, 255))
            y += 62
        y += 10 - 31 + 21
        for l in el:
            d.text((W / 2, y), l, font=fe, fill=(235, 228, 210, 255), anchor="mm", stroke_width=2, stroke_fill=(30, 22, 15, 255))
            y += 42
        out.append((c["start"] - 0.05, c["end"] + 0.35, im, H - 40 - hgt))
    return out


def setup(n_scenes):
    global SCENES, SUBS
    for i in range(n_scenes):
        SCENES.append(BUILDERS[i]() if i in BUILDERS else static_scene(i))
        print("  scene", i + 1, "ready", flush=True)
    SUBS = build_subs()


def scene_frame(i, t):
    return SCENES[i].frame(t - STARTS[i], ENDS[i] - STARTS[i], t)


def render(t):
    n = len(SCENES)
    i = max(k for k in range(n) if STARTS[k] <= t)
    img = scene_frame(i, t)
    if i + 1 < n and t > STARTS[i + 1] - XF:
        img = Image.blend(img, scene_frame(i + 1, t), smooth((t - (STARTS[i + 1] - XF)) / (2 * XF)))
    elif i > 0 and t < STARTS[i] + XF:
        img = Image.blend(scene_frame(i - 1, t), img, smooth((t - (STARTS[i] - XF)) / (2 * XF)))
    if t < 0.5:
        img = Image.blend(Image.new("RGB", (W, H), (247, 240, 222)), img, t / 0.5)
    for s0, s1, im, y in SUBS:
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
        setup(4)
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
