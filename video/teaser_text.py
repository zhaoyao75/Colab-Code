"""
《红线》16 秒抖音预告片 —— 高级感字幕后期（1080×1920）
输入：redline_teaser_720.mp4（race.py --teaser 渲出，已带音乐）
输出：redline_teaser.mp4

排版风格（「高级感」字幕的通用做法）：
  · 细体宋体大字（思源宋体 ExtraLight/Light），笔画极细、对比强
  · 很宽的字距，少量文字，大量留白
  · 英文副标用细斜体衬线（Cormorant Garamond），暖金色
  · 逐字淡入 + 轻微上浮，退场整体淡出；加一点柔和光晕和阴影，保证压在亮处也看得清
  · 关键字（红线）用暖红点缀，其余象牙白

字体不入库，先运行 fonts/get_fonts.sh 下载（SIL OFL 许可，可商用）。
运行：python3 teaser_text.py                 # 完整编码
     python3 teaser_text.py --still 1 3.6 11  # 只出指定时刻的静帧（秒）预览
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


def make_sprite(mask, color, glow_r, glow_a=0.30, shadow_dy=0, shadow_r=0, shadow_a=0.5):
    """mask：0–1 的字形；返回预乘 RGBA（float32）：阴影 → 光晕 → 字形"""
    col = np.array(color, np.float32) / 255
    sa = blur(np.roll(mask, shadow_dy, axis=0), shadow_r) * shadow_a if shadow_r else np.zeros_like(mask)
    ga = blur(mask, glow_r) * glow_a if glow_r else np.zeros_like(mask)
    rgb = ga[..., None] * col + 0.0 * (1 - ga)[..., None]               # 阴影是黑色，rgb 为 0
    a = ga + sa * (1 - ga)
    rgb = mask[..., None] * col + rgb * (1 - mask)[..., None]
    a = mask + a * (1 - mask)
    return np.concatenate([rgb, a[..., None]], axis=2)


class Item:
    def __init__(self, sprite, x, y, t_in):
        self.sp, self.x, self.y, self.t_in = sprite, int(x), int(y), t_in


class Element:
    """一组文字：逐字淡入上浮，整体在 t_out0–t_out1 淡出"""

    def __init__(self, items, t_out0=None, t_out1=None, stagger=0.07, rise=26, fade=0.55):
        self.items, self.t_out0, self.t_out1 = items, t_out0, t_out1
        self.stagger, self.rise, self.fade = stagger, rise, fade

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
            x0, y0 = it.x, it.y + int((1 - e) * self.rise)
            xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
            if xb <= xa or yb <= ya:
                continue
            sp = it.sp[ya - y0:yb - y0, xa - x0:xb - x0]
            reg = fr[ya:yb, xa:xb].astype(np.float32)
            reg = reg * (1 - a * sp[..., 3:4]) + a * sp[..., :3] * 255
            fr[ya:yb, xa:xb] = np.clip(reg, 0, 255).astype(np.uint8)


def text_element(txt, fnt, color, cx, cy, tracking, t_in, t_out0=None, t_out1=None, stagger=0.07, vertical=False,
                 glow_r=None, glow_a=0.30, rise=26, fade=0.55):
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
                         shadow_dy=int(size * 0.03), shadow_r=size * 0.04)
        items.append(Item(sp, px - cw / 2, py - chh / 2, t_in + i * stagger))
    return Element(items, t_out0, t_out1, stagger, rise, fade)


def rule_element(cx, cy, length, t_in, t_out0=None, t_out1=None, color=GOLD, thick=3):
    pad = 12
    m = np.zeros((thick + 2 * pad, length + 2 * pad), np.float32)
    m[pad:pad + thick, pad:pad + length] = 1
    sp = make_sprite(m, color, 4, 0.5)
    return Element([Item(sp, cx - length / 2 - pad, cy - thick / 2 - pad, t_in)], t_out0, t_out1, rise=0, fade=0.6)


def build():
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


def main():
    els = build()
    if "--still" in sys.argv:
        ts = [float(x) for x in sys.argv[sys.argv.index("--still") + 1:]]
        for t in ts:
            p = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", SRC, "-frames:v", "1", "-vf",
                                f"scale={W}:{H}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                               capture_output=True, check=True)
            fr = np.frombuffer(p.stdout, np.uint8).reshape(H, W, 3).copy()
            for e in els:
                e.draw(fr, t)
            Image.fromarray(fr).save(os.path.join(HERE, "teaser_still_%g.png" % t))
            print("saved teaser_still_%g.png" % t)
        return
    dec = decode(SRC)
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                            "-i", "-", "-i", SRC, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
                            "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
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
