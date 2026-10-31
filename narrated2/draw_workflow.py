"""画 PLAYBOOK 第 8 节的流程图：上半是制作流程（9 步），下半是文件数据流。输出 workflow.png"""
import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FD = "/usr/share/fonts/opentype/noto/"
F = lambda name, size: ImageFont.truetype(FD + name + ".ttc", size, index=2)
TITLE, H2, B, S, XS = F("NotoSerifCJK-Bold", 46), F("NotoSansCJK-Bold", 30), F("NotoSansCJK-Bold", 24), \
    F("NotoSansCJK-Regular", 20), F("NotoSansCJK-Regular", 17)

W, Hh = 1920, 1270
BG, INK, MUTED, LINE = "#f6f1e7", "#2b2723", "#7a7266", "#b9ae9c"
RED = "#b2261f"
TONE = {"audio": ("#e3edf5", "#3d6a8c"), "visual": ("#f3e6d6", "#9a5b24"), "text": ("#e8efe1", "#4e7139"),
        "out": ("#efe3ef", "#7a4a7a")}

img = Image.new("RGB", (W, Hh), BG)
d = ImageDraw.Draw(img)


def box(x, y, w, h, title, sub, tone, step=None, confirm=None):
    fill, edge = TONE[tone]
    d.rounded_rectangle([x, y, x + w, y + h], radius=16, fill=fill, outline=edge, width=3)
    if step:
        d.ellipse([x - 18, y - 18, x + 22, y + 22], fill=edge)
        d.text((x + 2, y + 2), step, font=B, fill="white", anchor="mm")
    d.text((x + w / 2, y + 36), title, font=B, fill=INK, anchor="mm")
    for i, t in enumerate(sub):
        d.text((x + w / 2, y + 70 + i * 26), t, font=S, fill=MUTED, anchor="mm")
    if confirm:
        tw = d.textlength(confirm, font=XS) + 24
        cx = x + w / 2
        d.rounded_rectangle([cx - tw / 2, y + h - 16, cx + tw / 2, y + h + 14], radius=14, fill=RED)
        d.text((cx, y + h - 1), confirm, font=XS, fill="white", anchor="mm")


def arrow(x0, y0, x1, y1, col=LINE, w=4, dash=False, head=True):
    if dash:
        n = int(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 // 14)
        for i in range(0, n, 2):
            a, b = i / n, min((i + 1) / n, 1)
            d.line([x0 + (x1 - x0) * a, y0 + (y1 - y0) * a, x0 + (x1 - x0) * b, y0 + (y1 - y0) * b], fill=col, width=w)
    else:
        d.line([x0, y0, x1, y1], fill=col, width=w)
    if not head:
        return
    import math
    ang = math.atan2(y1 - y0, x1 - x0)
    L = 14
    d.polygon([(x1, y1), (x1 - L * math.cos(ang - 0.45), y1 - L * math.sin(ang - 0.45)),
               (x1 - L * math.cos(ang + 0.45), y1 - L * math.sin(ang + 0.45))], fill=col)


# ---------------- 标题
d.text((70, 52), "AI 旁白解说短片 · 制作流程", font=TITLE, fill=INK)
d.text((72, 118), "PLAYBOOK 第 8 节  ·  以《青城 · 呼和浩特》为例（128 秒，1080p）", font=S, fill=MUTED)
legend = [("text", "文案 / 数据"), ("audio", "声音"), ("visual", "画面"), ("out", "成品")]
lx = 1080
for tone, name in legend:
    fill, edge = TONE[tone]
    d.rounded_rectangle([lx, 66, lx + 30, 92], radius=6, fill=fill, outline=edge, width=2)
    d.text((lx + 40, 79), name, font=S, fill=INK, anchor="lm")
    lx += 40 + d.textlength(name, font=S) + 36
d.rounded_rectangle([lx, 66, lx + 30, 92], radius=12, fill=RED)
d.text((lx + 40, 79), "用户确认", font=S, fill=INK, anchor="lm")

# ---------------- 上半：9 步流程（两行，蛇形）
d.text((70, 170), "一、制作流程", font=H2, fill=INK)
bw, bh, gap = 300, 140, 62
row1 = [("开工前问清楚", ["主题·时长·音色·字幕", "画面·配乐·平台·署名"], "text", "问清需求"),
        ("TTS 选型试听", ["Edge / Piper / gTTS / espeak", "10 个音色样本"], "audio", "挑音色"),
        ("写文案", ["script.json", "每章 16 秒约 50 字"], "text", "审文案史实"),
        ("旁白 + 配乐 + 混音", ["tts · compose_bg · mix", "约 1 分钟"], "audio", "听 audio_preview"),
        ("静帧九宫格", ["render.py --still …", "约 40 秒"], "visual", "看构图")]
row2 = [("10 秒 demo", ["render.py --demo 26 10", "约 3 分钟"], "visual", "看动态与转场"),
        ("完整版", ["render.py（后台 4 进程）", "约 12 分钟 · 63MB"], "out", "看成片"),
        ("反馈修改", ["只重跑受影响的步骤", "见下方「改动重跑」"], "visual", None),
        ("压缩版（可选）", ["ffmpeg 两遍编码", "按平台大小限制"], "out", None),
        ("推送 + 更新 PLAYBOOK", ["git push", "新经验写进第 8 节"], "text", None)]
x0, y1, y2 = 90, 240, 470
pos = []
for i, (t, sub, tone, conf) in enumerate(row1):
    x = x0 + i * (bw + gap)
    box(x, y1, bw, bh, t, sub, tone, str(i + 1), conf)
    pos.append((x, y1))
    if i < 4:
        arrow(x + bw + 6, y1 + bh / 2, x + bw + gap - 8, y1 + bh / 2)
for i, (t, sub, tone, conf) in enumerate(row2):
    x = x0 + (4 - i) * (bw + gap)                       # 第二行从右往左
    box(x, y2, bw, bh, t, sub, tone, str(i + 6), conf)
    pos.append((x, y2))
    if i < 4:
        arrow(x - 6, y2 + bh / 2, x - gap + 8, y2 + bh / 2)
# 第 5 → 6 拐弯
xr = x0 + 4 * (bw + gap) + bw / 2
arrow(xr, y1 + bh + 22, xr, y2 - 8)
# 反馈回路：8 → 5（虚线向上）
xf = x0 + 2 * (bw + gap) + bw / 2
ym = (y1 + bh + y2) / 2 + 6
xt = xr - 70
for a, b in [((xf, y2 - 6), (xf, ym)), ((xf, ym), (xt, ym))]:
    arrow(*a, *b, col=RED, w=3, dash=True, head=False)
arrow(xt, ym, xt, y1 + bh + 24, col=RED, w=3, dash=True)
d.text(((xf + xt) / 2, ym - 16), "改画面 → 回到第 5 步重出静帧", font=XS, fill=RED, anchor="mm")

# ---------------- 下半：数据流
yb = 690
d.text((70, yb), "二、文件数据流（全部在 narrated2/ 下）", font=H2, fill=INK)
fw, fh = 270, 110


def fbox(x, y, title, sub, tone):
    box(x, y, fw, fh, title, sub, tone)
    return x, y


r0 = yb + 110
sj = fbox(90, r0 + 120, "script.json", ["文案 · 锚点 · 印文"], "text")
tts = fbox(470, r0, "tts.py", ["逐句合成、去静音"], "audio")
cmp_ = fbox(470, r0 + 240, "compose_bg.py", ["五声音阶 · 马头琴"], "audio")
tl = fbox(850, r0, "timeline.json", ["每句起止 · 转场点"], "text")
nar = fbox(850, r0 + 120, ".cache/narration.wav", ["旁白整轨"], "audio")
mus = fbox(850, r0 + 240, ".cache/music.wav", ["配乐 128.00 秒"], "audio")
mix = fbox(1230, r0 + 180, "mix.py", ["按句子时间压低音乐"], "audio")
ren = fbox(1230, r0, "render.py", ["9 个场景 · 字幕 · 印章"], "visual")
out = fbox(1590, r0 + 90, "hohhot_full.mp4", ["1080p · 双语字幕"], "out")


def link(a, b, ya=0.5, yb_=0.5, col=LINE):
    arrow(a[0] + fw + 4, a[1] + fh * ya, b[0] - 6, b[1] + fh * yb_, col=col)


link(sj, tts, 0.3, 0.6)
# script.json → render.py：从上方绕过去（印文、标题）
pts = [(sj[0] + fw / 2, sj[1] - 6), (sj[0] + fw / 2, r0 - 30), (ren[0] + fw / 2, r0 - 30)]
d.line(pts, fill=LINE, width=4)
arrow(ren[0] + fw / 2, r0 - 30, ren[0] + fw / 2, r0 - 6)
d.text((sj[0] + fw / 2 + 14, r0 - 46), "印文、标题、英文时代名", font=XS, fill=MUTED, anchor="lm")
link(tts, tl)
link(tts, nar, 0.75, 0.3)
link(cmp_, mus)
link(tl, ren)
link(tl, mix, 0.85, 0.15)
link(nar, mix, 0.6, 0.35)
link(mus, mix, 0.5, 0.7)
link(mix, out, 0.4, 0.75)
link(ren, out, 0.6, 0.25)
d.text((1240, r0 + 300 + 4), "也输出 audio_preview.mp3（先给用户听）", font=XS, fill=MUTED)

# ---------------- 改动重跑速查
yr = Hh - 66
d.rounded_rectangle([70, yr - 26, W - 70, yr + 30], radius=14, fill="#ece4d6")
d.text((96, yr + 2), "改动重跑：改文案 → tts · mix · 静帧 · 完整版　｜　改配乐 → compose_bg · mix · 完整版　｜　"
       "只改画面或标题 → 静帧 · 完整版", font=S, fill=INK, anchor="lm")

out_path = os.path.join(HERE, "workflow.png")
img.save(out_path)
print(out_path)
