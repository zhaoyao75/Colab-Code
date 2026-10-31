"""
抖音竖版（1080×1920，9:16）：横版画面整幅居中（不裁内容），上方标题 + 分段小标题，下方大号双语字幕。
避开抖音界面：顶部约 200px（搜索栏）、底部约 380px（文案/音乐条）、右侧约 140px（点赞等按钮）只放背景。

  python3 vertical.py --still 4 24 70 120 127   # 静帧 -> stills_v/
  python3 vertical.py                            # 完整竖版 -> huang_yongyu_douyin.mp4
依赖 render.py 的场景代码，以及 timings.json、mix.wav。
"""
import math, os, subprocess, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import render as R

VW, VH = 1080, 1920
VID_W = 1080
VID_H = VID_W * R.H // R.W          # 607
VID_Y = 560                         # 画面窗口顶边
SUB_Y = VID_Y + VID_H + 50          # 字幕起始
CHAPTERS = ["凤凰童年", "离家学艺", "北京任教", "猫头鹰风波", "猴年邮票", "荷花与写作", "老顽童", "永远的童心"]
CH_EN = ["Childhood in Fenghuang", "Self-taught woodcut", "Teaching in Beijing", "The winking owl",
         "The monkey stamp", "Lotus & letters", "The Old Rascal", "Forever young at heart"]


def build_bg():
    bg = R.paper(VW, VH, 77)
    d = ImageDraw.Draw(bg, "RGBA")
    # 画面窗口投影
    sh = Image.new("RGBA", (VW, VH), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rectangle([0, VID_Y + 10, VW, VID_Y + VID_H + 22], fill=(60, 40, 20, 90))
    sh = sh.filter(ImageFilter.GaussianBlur(14))
    bg.paste(sh, (0, 0), sh)
    # 顶部标题
    d.text((VW / 2, 300), "黄永玉", font=R.font(R.F_BRUSH, 150), fill=R.INK + (255,), anchor="mm")
    d.text((VW / 2, 415), "永远长不大的老头", font=R.font(R.F_BRUSH, 56), fill=R.RED + (255,), anchor="mm")
    d.line([(VW / 2 - 330, 470), (VW / 2 + 330, 470)], fill=(150, 130, 110, 120), width=2)
    return bg


def chapter_layers():
    out = []
    for k in range(8):
        im = Image.new("RGBA", (VW, 80), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.text((VW / 2, 26), f"{k + 1:02d}  {CHAPTERS[k]}", font=R.font(R.F_FUN, 40), fill=(90, 70, 55, 255), anchor="mm")
        d.text((VW / 2, 66), CH_EN[k], font=R.font(R.F_EN, 22), fill=(130, 115, 100, 255), anchor="mm")
        out.append(im)
    return out


PUNCT = "，。、；：！？"


def wrap_zh(text, f, maxw):
    """先按标点切成小句，再把小句装进行里；单个小句太长才硬断"""
    clauses, cur = [], ""
    for ch in text:
        cur += ch
        if ch in PUNCT:
            clauses.append(cur)
            cur = ""
    if cur:
        clauses.append(cur)
    lines, line = [], ""
    for cl in clauses:
        if f.getlength(line + cl) <= maxw:
            line += cl
            continue
        if line:
            lines.append(line)
        line = ""
        for ch in cl:
            if f.getlength(line + ch) > maxw and ch not in PUNCT:
                lines.append(line)
                line = ""
            line += ch
    if line:
        lines.append(line)
    return lines


def build_subs():
    fz, fe = R.font(R.F_FUN, 60), R.font(R.F_EN, 34)
    subs = []
    for c in R.CUES:
        zl = wrap_zh(c["zh"], fz, 900)
        el = R.wrap_en(c["en"], fe, 900)
        hgt = 40 + 78 * len(zl) + 12 + 46 * len(el)
        im = Image.new("RGBA", (VW, hgt + 10), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        bw = max([fz.getlength(l) for l in zl] + [fe.getlength(l) for l in el]) + 70
        d.rounded_rectangle([(VW - bw) / 2, 0, (VW + bw) / 2, hgt], radius=26, fill=(20, 16, 12, 150))
        y = 20 + 39
        for l in zl:
            d.text((VW / 2, y), l, font=fz, fill=(255, 252, 240, 255), anchor="mm", stroke_width=3, stroke_fill=(30, 22, 15, 255))
            y += 78
        y += 12 - 78 + 39 + 23
        for l in el:
            d.text((VW / 2, y), l, font=fe, fill=(235, 228, 210, 255), anchor="mm", stroke_width=2, stroke_fill=(30, 22, 15, 255))
            y += 46
        subs.append((c["start"] - 0.05, c["end"] + 0.35, im))
    return subs


BG = CH = VSUBS = None


def setup():
    global BG, CH, VSUBS
    R.setup()
    R.SUBS = []                     # 横版画面里不烧字幕，竖版另排
    BG = build_bg()
    CH = chapter_layers()
    VSUBS = build_subs()


def vframe(t):
    img = BG.copy()
    land = R.render(t).resize((VID_W, VID_H), Image.LANCZOS)
    img.paste(land, (0, VID_Y))
    # 分段小标题（随场景切换淡入淡出）
    i = max(k for k in range(len(R.STARTS)) if R.STARTS[k] <= t)
    a = R.smoothstep(R.STARTS[i] + 0.2, R.STARTS[i] + 0.9, t)
    if i + 1 < len(R.STARTS):
        a *= 1 - R.smoothstep(R.STARTS[i + 1] - 0.6, R.STARTS[i + 1], t)
    if i == 0:
        a = R.smoothstep(0.5, 1.5, t) * (1 - R.smoothstep(R.STARTS[1] - 0.6, R.STARTS[1], t))
    R.place(img, CH[i], VW / 2, 510, alpha=a)
    for s0, s1, im in VSUBS:
        if s0 <= t <= s1:
            al = min(1, (t - s0) / 0.15, (s1 - t) / 0.15)
            R.place(img, im, VW / 2, SUB_Y + im.height / 2, alpha=al)
    if t < 0.6:
        img = Image.blend(Image.new("RGB", (VW, VH), R.PAPER), img, t / 0.6)
    return img


def encode(args):
    f0, f1, path = args
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{VW}x{VH}", "-r", str(R.FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                          "-crf", "19", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for f in range(f0, f1):
        p.stdin.write(vframe(f / R.FPS).tobytes())
        if (f - f0) % 150 == 0:
            print(f"  {path}: {f - f0}/{f1 - f0}", flush=True)
    p.stdin.close()
    p.wait()
    return path


def make_video(t0, t1, out, procs=4):
    f0, f1 = int(t0 * R.FPS), int(t1 * R.FPS)
    step = math.ceil((f1 - f0) / procs)
    jobs = [(a, min(f1, a + step), f".seg_v{k}.mp4") for k, a in enumerate(range(f0, f1, step))]
    with Pool(len(jobs)) as pool:
        segs = pool.map(encode, jobs)
    with open(".segs.txt", "w") as fh:
        fh.writelines(f"file '{s}'\n" for s in segs)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", ".segs.txt",
                    "-ss", str(t0), "-t", str(t1 - t0), "-i", "mix.wav", "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", out], check=True)
    for s in segs:
        os.remove(s)
    os.remove(".segs.txt")
    print("->", out)


if __name__ == "__main__":
    os.chdir(R.HERE)
    setup()
    if len(sys.argv) > 1 and sys.argv[1] == "--still":
        os.makedirs("stills_v", exist_ok=True)
        for a in sys.argv[2:]:
            vframe(float(a)).save(f"stills_v/t{float(a):05.1f}.jpg", quality=90)
            print("stills_v/", a)
    else:
        make_video(0, R.LENGTH, "huang_yongyu_douyin.mp4")
