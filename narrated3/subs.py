"""
字幕切分：只要中文、单行不换行。把每句旁白按标点拆成短句（每条 ≤ MAXC 个字），按字数分配时间。
  python3 subs.py   ->  打印切分结果，并写出 subtitles_v2_zh.srt
"""
import json, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
PUN = "，。；：！？、"
MAXC = 24            # 单条字幕最多字数（56px 字体约 1350px 宽，单行放得下）
MINC = 6             # 少于这个字数的碎片并入相邻字幕
HARD = 30            # 并入后的上限


def clauses(zh):
    return [x for x in re.findall(r"[^，。；：！？]+[，。；：！？]?", zh) if x]


def segments(cues, maxc=MAXC):
    out = []
    for c in cues:
        groups, cur = [], ""
        for x in clauses(c["zh"]):
            if cur and len((cur + x).rstrip(PUN)) > maxc:
                groups.append(cur)
                cur = ""
            cur += x
        if cur:
            groups.append(cur)
        i = 0
        while i < len(groups):                                   # 太短的碎片并入相邻的一条
            n = len(groups[i].rstrip(PUN))
            if n < MINC and len(groups) > 1:
                j = i + 1 if i + 1 < len(groups) else i - 1
                if len((groups[min(i, j)] + groups[max(i, j)]).rstrip(PUN)) <= HARD:
                    a, b = min(i, j), max(i, j)
                    groups[a:b + 1] = [groups[a] + groups[b]]
                    i = 0
                    continue
            i += 1
        w = [max(1, len(re.sub("[" + PUN + "]", "", g))) for g in groups]
        t, tot = c["start"], sum(w)
        for g, wi in zip(groups, w):
            d = (c["end"] - c["start"]) * wi / tot
            out.append({"scene": c["scene"], "start": round(t, 3), "end": round(t + d, 3), "text": g.rstrip("，。；")})
            t += d
    return out


def load(path="timings_v2.json"):
    return segments(json.load(open(os.path.join(HERE, path), encoding="utf-8")))


def ts(x):
    ms = int(round(x * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


if __name__ == "__main__":
    segs = load()
    with open(os.path.join(HERE, "subtitles_v2_zh.srt"), "w", encoding="utf-8") as fh:
        for n, s in enumerate(segs, 1):
            fh.write(f"{n}\n{ts(s['start'])} --> {ts(s['end'] + 0.2)}\n{s['text']}\n\n")
    for s in segs:
        print(f"{s['start']:7.2f}–{s['end']:7.2f} [{len(s['text']):2d}字] {s['text']}")
