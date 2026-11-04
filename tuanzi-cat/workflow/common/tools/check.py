# 一致性检查。用法: check.py <集目录> <out目录>   （交付目录存在时一并检查文件名与文案）
import json,sys,re,os,glob
src,out=sys.argv[1:3]; cfg=json.load(open(f"{src}/episode.json")); tail=open(f"{src}/tail.html",encoding='utf8').read()
no,name,total=cfg['no'],cfg['name'],int(cfg['total'])
norm=lambda s:re.sub(r"[。，、！？…,.!?：:\s·「」]","",s)
res=[]
def chk(ok,msg): res.append((ok,msg))
# 1 每句配音都有对应字幕
for v in cfg['vo']:
    if v['id']=='next': continue
    chk(norm(v['text']) in norm(tail), f"配音「{v['text']}」在场景字幕里能找到")
# 2 结尾卡
chk(f"第{no}集" in tail, f"结尾卡标签含「第{no}集」")
nt=cfg['publish']['next_teaser']; chk(nt in tail, f"结尾卡含下集题材「{nt}」")
chk("下集预告" in tail, "结尾卡含「下集预告」"); chk("关注" not in tail, "结尾卡不含「关注」按钮")
nv=[v for v in cfg['vo'] if v['id']=='next']
if nv: chk(nt in nv[0]['text'], f"预告配音含下集题材「{nt}」")
# 3 封面文字与开场台词一致
chk(norm(cfg['vo'][0]['text']) in norm(''.join(cfg['cover']['lines'])), "封面标题包含开场台词")
# 4 禁用旧词
files={'场景tail.html':tail,'episode.json 台词':'\n'.join(v['text'] for v in cfg['vo'])+''.join(cfg['cover']['lines'])}
dl=os.path.join(out,'交付')
if os.path.isdir(dl):
    for f in glob.glob(dl+'/*.md'): files['交付/'+os.path.basename(f)]=open(f,encoding='utf8').read()
    files['交付文件名']='\n'.join(os.listdir(dl))
for w in cfg.get('banned',[]):
    hit=[k for k,t in files.items() if w in t]; chk(not hit, f"旧词「{w}」未残留"+(f"（出现在：{', '.join(hit)}）" if hit else ""))
# 5 文案与文件名
if os.path.isdir(dl):
    names=os.listdir(dl)
    chk(f"第{no}集_{name}_{total}s.mp4" in names, f"成片文件名 第{no}集_{name}_{total}s.mp4")
    for k in ['竖版9x16','主页3x4','合集方形1x1','横版4x3']:
        chk(f"第{no}集_封面_{k}.png" in names, f"封面文件 第{no}集_封面_{k}.png")
    copy=[f for f in names if f.endswith('发布文案.md')]
    if copy:
        t=open(f"{dl}/{copy[0]}",encoding='utf8').read()
        for s in cfg.get('copy_must_include',[]): chk(s in t, f"发布文案含「{s}」")
        chk(f"第{no}集" in t, f"发布文案含「第{no}集」")
        ft=os.path.join(os.path.dirname(os.path.abspath(src)),'..','common','fixed_tags.txt')
        ft=os.path.normpath(ft)
        if os.path.exists(ft):
            fixed=open(ft,encoding='utf8').read().strip(); m=re.search(r"## 话题\n(.*?)\n\n##",t,re.S)
            chk(bool(m) and m.group(1).strip()==fixed, f"发布文案话题固定为：{fixed}")
    else: chk(False,"缺少发布文案")
bad=0
for ok,m in res:
    print(('✓ ' if ok else '✗ ')+m); bad+=not ok
print(f"一致性检查: {len(res)-bad}/{len(res)} 通过"); sys.exit(1 if bad else 0)
