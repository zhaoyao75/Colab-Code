"""
《红线》16 秒抖音预告片 —— 极简字幕后期（1080×1920）
全程不放字幕，只保留两处：开头的 F1 RACING（粗斜体，从左滑入）和结尾的 WIN（粗斜体，弹入）。

输入：redline_teaser_720.mp4（race.py --teaser 渲出，已带音乐）
输出：redline_teaser.mp4

字体：Barlow Condensed Black Italic（默认，F1 转播风格的窄体粗斜体）；--font racing 用 Racing Sans One，
     --font montserrat 用 Montserrat Black Italic。字体不入库，先运行 fonts/get_fonts.sh 下载（SIL OFL，可商用）。
运行：python3 -I teaser_text.py                                  # 完整编码
     python3 -I teaser_text.py --still 1 14.5                     # 取预告片底片的指定时刻（秒）预览
     python3 -I teaser_text.py --still 1 14.5 --base 底帧.png     # 用一张 PNG 当底帧预览
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(HERE, "fonts")
W, H, FPS = 1080, 1920, 30
SRC = os.path.join(HERE, "redline_teaser_720.mp4")
OUT = os.path.join(HERE, "redline_teaser.mp4")

IVORY, GOLD, RED = (246, 241, 232), (236, 176, 128), (255, 104, 72)


def font(name, size, wght=None):
    f = ImageFont.truetype(os.path.join(FD, name), int(size))
    if wght is not None:
        f.set_variation_by_axes([wght])
    return f


def ease(x):
    x = float(np.clip(x, 0, 1))
    return 1 - (1 - x) ** 3


def smooth(t, a, b):
    x = float(np.clip((t - a) / (b - a), 0, 1))
    return x * x * (3 - 2 * x)


def blur(m, r):
    return np.asarray(Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r))).astype(np.float32) / 255


def make_sprite(mask, color, glow_r, glow_a=0.30, shadow_dy=0, shadow_r=0, shadow_a=0.5, glow_color=None):
    """mask：0–1 的字形；返回预乘 RGBA（float32）：阴影 → 光晕 → 字形"""
    col = np.array(color, np.float32) / 255
    sa = blur(np.roll(mask, shadow_dy, axis=0), shadow_r) * shadow_a if shadow_r else np.zeros_like(mask)
    ga = blur(mask, glow_r) * glow_a if glow_r else np.zeros_like(mask)
    gcol = col if glow_color is None else np.array(glow_color, np.float32) / 255
    rgb = ga[..., None] * gcol                                           # 阴影是黑色，rgb 为 0
    a = ga + sa * (1 - ga)
    rgb = mask[..., None] * col + rgb * (1 - mask)[..., None]
    a = mask + a * (1 - mask)
    return np.concatenate([rgb, a[..., None]], axis=2)


class Item:
    def __init__(self, sprite, x, y, t_in):
        self.sp, self.x, self.y, self.t_in = sprite, int(x), int(y), t_in


class Element:
    """一组文字：逐字淡入上浮，整体在 t_out0–t_out1 淡出"""

    def __init__(self, items, t_out0=None, t_out1=None, stagger=0.07, rise=26, fade=0.55, slide=0):
        self.items, self.t_out0, self.t_out1 = items, t_out0, t_out1
        self.stagger, self.rise, self.fade, self.slide = stagger, rise, fade, slide

    def draw(self, fr, t):
        out = 1.0 if self.t_out0 is None else 1 - smooth(t, self.t_out0, self.t_out1)
        if out <= 0.002:
            return
        for k, it in enumerate(self.items):
            e = ease((t - it.t_in) / self.fade)
            a = e * out
            if a <= 0.002:
                continue
            h, w = it.sp.shape[:2]
            x0, y0 = it.x + int((1 - e) ** 2 * self.slide), it.y + int((1 - e) * self.rise)
            xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
            if xb <= xa or yb <= ya:
                continue
            sp = it.sp[ya - y0:yb - y0, xa - x0:xb - x0]
            reg = fr[ya:yb, xa:xb].astype(np.float32)
            reg = reg * (1 - a * sp[..., 3:4]) + a * sp[..., :3] * 255
            fr[ya:yb, xa:xb] = np.clip(reg, 0, 255).astype(np.uint8)


def text_element(txt, fnt, color, cx, cy, tracking, t_in, t_out0=None, t_out1=None, stagger=0.07, vertical=False,
                 glow_r=None, glow_a=0.30, rise=26, fade=0.55, slide=0, glow_color=None):
    """cx, cy：整行（或竖排整列）的中心。tracking：字距（像素）。"""
    size = fnt.size
    pad = int(size * 0.45)
    glow_r = size * 0.05 if glow_r is None else glow_r
    ws = [fnt.getlength(c) for c in txt]
    items = []
    if vertical:
        total = size * len(txt) + tracking * (len(txt) - 1)
        pos = [(cx, cy - total / 2 + size / 2 + i * (size + tracking)) for i in range(len(txt))]
    else:
        total = sum(ws) + tracking * (len(txt) - 1)
        xs, x = [], cx - total / 2
        for w_ in ws:
            xs.append(x + w_ / 2)
            x += w_ + tracking
        pos = [(px, cy) for px in xs]
    for i, (ch, (px, py)) in enumerate(zip(txt, pos)):
        if ch == " ":
            continue
        cw, chh = int(ws[i] + 2 * pad) + 2, int(size * 1.5 + 2 * pad)
        im = Image.new("L", (cw, chh), 0)
        ImageDraw.Draw(im).text((cw / 2, chh / 2), ch, font=fnt, fill=255, anchor="mm")
        sp = make_sprite(np.asarray(im).astype(np.float32) / 255, color, glow_r, glow_a,
                         shadow_dy=int(size * 0.03), shadow_r=size * 0.04, glow_color=glow_color)
        items.append(Item(sp, px - cw / 2, py - chh / 2, t_in + i * stagger))
    return Element(items, t_out0, t_out1, stagger, rise, fade, slide)


def rule_element(cx, cy, length, t_in, t_out0=None, t_out1=None, color=GOLD, thick=3):
    pad = 12
    m = np.zeros((thick + 2 * pad, length + 2 * pad), np.float32)
    m[pad:pad + thick, pad:pad + length] = 1
    sp = make_sprite(m, color, 4, 0.5)
    return Element([Item(sp, cx - length / 2 - pad, cy - thick / 2 - pad, t_in)], t_out0, t_out1, rise=0, fade=0.6)


class PopElement:
    """整体弹入：从放大 1.6 倍收回到原尺寸并淡入（WIN）；每帧按当前尺寸重画字形，只在前 0.3 秒需要"""

    def __init__(self, txt, fname, base_size, color, cx, cy, tracking, t_in, t_out0, t_out1, glow_color=None, wght=None):
        self.a = (txt, fname, base_size, color, cx, cy, tracking, glow_color, wght)
        self.t_in, self.t_out0, self.t_out1 = t_in, t_out0, t_out1
        self.cache = {}

    def _el(self, scale):
        key = round(scale, 2)
        if key not in self.cache:
            txt, fname, size, color, cx, cy, trk, gc, wg = self.a
            self.cache[key] = text_element(txt, font(fname, size * key, wg), color, cx, cy, trk * key, -99.0, stagger=0,
                                           glow_r=size * key * 0.07, glow_a=0.45, rise=0, fade=0.001, glow_color=gc)
        return self.cache[key]

    def draw(self, fr, t):
        tau = t - self.t_in
        if tau < 0:
            return
        out = 1 - smooth(t, self.t_out0, self.t_out1)
        if out <= 0.002:
            return
        scale = 1 + 0.6 * (1 - ease(tau / 0.3))
        el = self._el(scale)
        a = ease(tau / 0.12) * out
        saved = [(it.sp) for it in el.items]
        for it in el.items:
            it.sp = it.sp * np.array([a, a, a, a], np.float32)           # 预乘 RGBA，整体乘透明度
        el.draw(fr, 1e9)
        for it, sp in zip(el.items, saved):
            it.sp = sp


FONTS = {"barlow": "BarlowCondensed-BlackItalic.ttf", "racing": "RacingSansOne-Regular.ttf", "montserrat": "Montserrat-Italic.ttf"}


def build(variant="barlow"):
    fname = FONTS[variant]
    wg = 900 if variant == "montserrat" else None
    size = {"barlow": 250, "racing": 190, "montserrat": 170}[variant]
    wsize = {"barlow": 340, "racing": 260, "montserrat": 260}[variant]
    trk = {"barlow": 8, "racing": 6, "montserrat": 4}[variant]
    f = font(fname, size, wg)
    # 开头：F1（红）+ RACING（白），整行居中，从左滑入，灯灭（3.2 秒）前淡出
    sp_w = f.getlength(" ")
    w1, w2 = f.getlength("F1") + trk, sum(f.getlength(c) + trk for c in "RACING") - trk
    total = w1 + sp_w + trk + w2
    x0 = 540 - total / 2
    E = [text_element("F1", f, (255, 66, 44), x0 + (f.getlength("F1") + trk) / 2 - trk / 2, 520, trk, 0.15, 2.9, 3.3,
                      stagger=0.05, glow_r=size * 0.05, glow_a=0.35, rise=0, fade=0.45, slide=-120),
         text_element("RACING", f, (250, 248, 244), x0 + w1 + sp_w + trk + w2 / 2, 520, trk, 0.3, 2.9, 3.3,
                      stagger=0.05, glow_r=size * 0.05, glow_a=0.35, rise=0, fade=0.45, slide=-120)]
    # 结尾：WIN，15.25 秒（赛道边分镜里两辆车都冲出画面之后）弹入，停到结束，最后 0.4 秒随画面淡出
    E.append(PopElement("WIN", fname, wsize, (255, 255, 255), 540, 640, trk, 15.25, 15.6, 16.0, glow_color=(255, 70, 40), wght=wg))
    return E


def build_old():
    serif_xl = font("NotoSerifSC-ExtraLight.otf", 300)
    serif_l = font("NotoSerifSC-Light.otf", 44)
    serif_v = font("NotoSerifSC-Light.otf", 68)
    en_i = lambda size: font("CormorantGaramond-Italic.ttf", size, 400)
    en_r = lambda size: font("CormorantGaramond.ttf", size, 400)
    E = []
    # 0.0–3.2 片名：红线 / REDLINE / 细线 / 一句话
    E.append(text_element("红线", serif_xl, IVORY, 540, 430, 120, 0.15, 2.9, 3.3, stagger=0.18))
    E.append(text_element("REDLINE", en_r(66), GOLD, 540, 625, 34, 0.7, 2.9, 3.3, stagger=0.06, glow_r=3))
    E.append(rule_element(540, 676, 120, 1.1, 2.9, 3.3))
    E.append(text_element("原创配乐 × 代码生成", serif_l, IVORY, 540, 735, 20, 1.2, 2.9, 3.3, stagger=0.05, glow_r=2, glow_a=0.2))
    # 3.2 灯灭
    E.append(text_element("灯灭", font("NotoSerifSC-ExtraLight.otf", 400), IVORY, 540, 470, 90, 3.2, 4.9, 5.5, stagger=0.14))
    E.append(text_element("LIGHTS OUT", en_i(74), GOLD, 540, 735, 30, 3.55, 4.9, 5.5, stagger=0.05, glow_r=3))
    # 6.4 竖排：把转速推到极限
    E.append(text_element("把转速推到极限", serif_v, IVORY, 130, 1010, 34, 6.4, 9.0, 9.5, stagger=0.1, vertical=True, glow_a=0.22))
    E.append(rule_element(205, 1010, 3, 6.4, 9.0, 9.5, thick=3, color=GOLD))
    # 9.6 座舱特写：逼近红线
    E.append(text_element("逼近红线", font("NotoSerifSC-ExtraLight.otf", 170), IVORY, 540, 440, 38, 9.6, 12.45, 12.75, stagger=0.2))
    E.append(text_element("APPROACHING THE LIMIT", en_i(54), GOLD, 540, 570, 20, 10.3, 12.45, 12.75, stagger=0.035, glow_r=3))
    # 12.8 高潮：冲破 / 红线
    E.append(text_element("冲破", font("NotoSerifSC-ExtraLight.otf", 250), IVORY, 540, 600, 70, 12.8, 14.9, 15.5, stagger=0.1, rise=34))
    E.append(text_element("红线", font("NotoSerifSC-ExtraLight.otf", 250), RED, 540, 870, 70, 13.05, 14.9, 15.5, stagger=0.1, rise=34))
    E.append(text_element("BREAK THROUGH", en_i(66), GOLD, 540, 420, 28, 13.3, 14.9, 15.5, stagger=0.04, glow_r=3))
    # 15.0 收尾（和开头呼应，循环播放时衔接）
    E.append(text_element("REDLINE", en_r(70), IVORY, 540, 500, 40, 15.0, stagger=0.05, glow_r=3))
    E.append(text_element("原创配乐 × 代码生成", serif_l, IVORY, 540, 575, 20, 15.2, stagger=0.04, glow_r=2, glow_a=0.2))
    return E


def decode(path, scale=True):
    cmd = ["ffmpeg", "-loglevel", "error", "-i", path, "-vf", f"scale={W}:{H}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE)


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    variant = arg("--font", "barlow")
    els = build(variant)
    if "--still" in sys.argv:
        k = sys.argv.index("--still") + 1
        ts = []
        while k < len(sys.argv) and not sys.argv[k].startswith("--"):
            ts.append(float(sys.argv[k]))
            k += 1
        bases = arg("--base")
        for n, t in enumerate(ts):
            if bases:
                bp = bases.split(",")[min(n, len(bases.split(",")) - 1)]
                fr = np.asarray(Image.open(bp).convert("RGB").resize((W, H), Image.LANCZOS)).copy()
            else:
                p = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", SRC, "-frames:v", "1", "-vf",
                                    f"scale={W}:{H}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                   capture_output=True, check=True)
                fr = np.frombuffer(p.stdout, np.uint8).reshape(H, W, 3).copy()
            for e in els:
                e.draw(fr, t)
            name = os.path.join(arg("--outdir", HERE), "teaser_still_%s_%g.png" % (variant, t))
            Image.fromarray(fr).save(name)
            print("saved", name)
        return
    dec = decode(SRC)
    sfx = os.path.join(HERE, ".driveby_sfx.wav")
    sys.path.insert(0, HERE)
    import driveby_sfx
    driveby_sfx.make(sfx)
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                            "-i", "-", "-i", SRC, "-i", sfx,
                            "-filter_complex", "[1:a][2:a]amix=inputs=2:duration=first:normalize=0[aout]",
                            "-map", "0:v", "-map", "[aout]", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
                            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-t", "16", "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
    i, n = 0, W * H * 3
    while True:
        buf = dec.stdout.read(n)
        if len(buf) < n:
            break
        fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
        t = i / FPS
        for e in els:
            e.draw(fr, t)
        enc.stdin.write(fr.tobytes())
        i += 1
    enc.stdin.close()
    enc.wait()
    dec.wait()
    print("frames", i, "->", OUT)


if __name__ == "__main__":
    main()
