"""
抖音竖版（1080×1920，9:16），复用横版动画帧：横版画面整幅居中（不裁内容），上方标题 + 章节名，下方大号中文单行字幕。
避开抖音界面：顶部约 200px、底部约 380px、右侧约 140px 只放背景。

  python3 vertical_v2.py --still 5 60 101   # 静帧 -> stills_v3/
  python3 vertical_v2.py                    # 完整竖版 -> huang_yongyu_v2_douyin.mp4
"""
import math, os, subprocess, sys
from multiprocessing import Pool
from PIL import Image, ImageDraw
import render as R
import animate as A
import subs

VW, VH = 1080, 1920
VID_W, VID_H = 1080, 607
VID_Y = 560
SUB_CY = VID_Y + VID_H + 120                     # 字幕中心 y
CHAPTERS = ["开场", "凤凰童年", "离家出走", "表叔沈从文", "北京任教", "猫头鹰", "猴年邮票", "无愁河", "永远的少年"]
BG = LAB = VS = None


def build_bg():
    bg = R.paper(VW, VH, 77, base=(247, 240, 222), vig=False)
    d = ImageDraw.Draw(bg, "RGBA")
    d.text((VW / 2, 300), "黄永玉", font=R.font(R.F_BRUSH, 150), fill=(34, 30, 32, 255), anchor="mm")
    d.text((VW / 2, 415), "永远长不大的老头", font=R.font(R.F_BRUSH, 56), fill=(200, 38, 42, 255), anchor="mm")
    d.line([(VW / 2 - 330, 470), (VW / 2 + 330, 470)], fill=(150, 130, 110, 120), width=2)
    return bg


def build_labels():
    out = []
    for k, n in enumerate(CHAPTERS):
        im = Image.new("RGBA", (VW, 60), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((VW / 2, 30), f"{k + 1:02d}  {n}", font=R.font(R.F_FUN, 40), fill=(90, 70, 55, 255), anchor="mm")
        out.append(im)
    return out


def split_long(sg, fz, maxw=960):
    """单条字幕超宽时拆成两条（优先在书名号前或标点后断开），时间按字数分配"""
    t = sg["text"]
    if fz.getlength(t) <= maxw:
        return [sg]
    n = len(t)
    cand = [i for i in range(n // 3, 2 * n // 3 + 1) if t[i:i + 1] == "《" or t[i - 1:i] in "》，"]
    i = min(cand, key=lambda k: abs(k - n / 2)) if cand else n // 2
    d = sg["end"] - sg["start"]
    mid = sg["start"] + d * i / n
    return [dict(sg, text=t[:i], end=mid), dict(sg, text=t[i:], start=mid)]


def build_vsubs():
    """竖版字幕：单行，每条约 ≤18 字，字号 52（约 940px 宽）"""
    fz = R.font(R.F_FUN, 52)
    out = []
    for sg0 in subs.load(maxc=14, hard=18, minc=4):
        for sg in split_long(sg0, fz):
            tw = fz.getlength(sg["text"])
            im = Image.new("RGBA", (VW, 84), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            d.rounded_rectangle([(VW - tw) / 2 - 30, 0, (VW + tw) / 2 + 30, 80], radius=24, fill=(20, 16, 12, 140))
            d.text((VW / 2, 42), sg["text"], font=fz, fill=(255, 252, 240, 255), anchor="mm", stroke_width=3, stroke_fill=(30, 22, 15, 255))
            out.append((sg["start"] - 0.05, sg["end"] + 0.2, im))
    return out


def setup():
    global BG, LAB, VS
    A.setup(9)
    BG, LAB, VS = build_bg(), build_labels(), build_vsubs()


def vframe(t):
    img = BG.copy().convert("RGB")
    img.paste(A.render(t, with_subs=False).resize((VID_W, VID_H), Image.LANCZOS), (0, VID_Y))
    i = max(k for k in range(len(A.STARTS)) if A.STARTS[k] <= t)
    a = A.seg(t, A.STARTS[i] + 0.1, A.STARTS[i] + 0.7)
    if i + 1 < len(A.STARTS):
        a *= 1 - A.seg(t, A.STARTS[i + 1] - 0.5, A.STARTS[i + 1])
    if i == 0:
        a = A.seg(t, 0.4, 1.2) * (1 - A.seg(t, A.STARTS[1] - 0.5, A.STARTS[1]))
    lab = LAB[i].copy()
    lab.putalpha(lab.getchannel("A").point(lambda v: int(v * a)))
    img.paste(lab, (0, 480), lab)
    for s0, s1, im in VS:
        if s0 <= t <= s1:
            al = min(1.0, (t - s0) / 0.15, (s1 - t) / 0.15)
            if al < 0.99:
                im = im.copy()
                im.putalpha(im.getchannel("A").point(lambda v: int(v * al)))
            img.paste(im, (0, SUB_CY - 42), im)
    return img


def encode(args):
    f0, f1, path = args
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VW}x{VH}", "-r", "30",
                          "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for f in range(f0, f1):
        p.stdin.write(vframe(f / 30).tobytes())
        if (f - f0) % 150 == 0:
            print(f"  {path}: {f - f0}/{f1 - f0}", flush=True)
    p.stdin.close()
    p.wait()
    return path


def make_video(out, procs=4):
    f1 = int(A.LENGTH * 30)
    step = math.ceil(f1 / procs)
    jobs = [(a, min(f1, a + step), f".seg_d{k}.mp4") for k, a in enumerate(range(0, f1, step))]
    with Pool(len(jobs)) as pool:
        segs = pool.map(encode, jobs)
    with open(".segs_d.txt", "w") as fh:
        fh.writelines(f"file '{s}'\n" for s in segs)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", ".segs_d.txt", "-i", "mix_v2.wav", "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", out], check=True)
    for s in segs:
        os.remove(s)
    os.remove(".segs_d.txt")
    print("->", out)


if __name__ == "__main__":
    os.chdir(R.HERE)
    setup()
    if len(sys.argv) > 1 and sys.argv[1] == "--still":
        os.makedirs("stills_v3", exist_ok=True)
        for a in sys.argv[2:]:
            vframe(float(a)).save(f"stills_v3/t{float(a):05.1f}.jpg", quality=90)
            print("stills_v3/", a)
    else:
        make_video("huang_yongyu_v2_douyin.mp4")
