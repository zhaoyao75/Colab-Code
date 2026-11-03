# 自动质检。用法: qa.py <集目录> <out目录> <common目录>   不通过则退出码1，并写 out/qa报告.md
import json,sys,subprocess,os,re
src,out,common=sys.argv[1:4]; cfg=json.load(open(f"{src}/episode.json")); tl=json.load(open(f"{out}/timeline.json"))
res=[]
def chk(ok,msg): res.append((ok,msg))
def sh(*a): return subprocess.run(a,capture_output=True,text=True)
v=f"{out}/video.mp4"
info=json.loads(sh('ffprobe','-v','error','-show_entries','stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels,duration','-of','json',v).stdout)['streams']
vs=[s for s in info if s['codec_type']=='video'][0]; au=[s for s in info if s['codec_type']=='audio']
dur=float(vs['duration'])
chk((vs['width'],vs['height'])==(1080,1920),f"分辨率 {vs['width']}×{vs['height']}（要求 1080×1920）")
chk(vs['r_frame_rate']=='30/1',f"帧率 {vs['r_frame_rate']}（要求 30）"); chk(10<=dur<=12.05,f"时长 {dur:.2f}s（要求 10–12s）")
chk(bool(au) and au[0]['codec_name']=='aac' and au[0]['sample_rate']=='48000' and au[0]['channels']==2, "音频 AAC 48kHz 立体声")
# 响度、峰值
e=sh('ffmpeg','-i',v,'-af','ebur128=peak=true','-f','null','-').stderr
I=float(re.findall(r"\n\s+I:\s+(-?[\d.]+) LUFS",e)[-1]); P=float(re.findall(r"Peak:\s+(-?[\d.]+) dBFS",e)[-1])
chk(abs(I+14)<=0.5,f"响度 {I} LUFS（要求 -14±0.5）"); chk(P<=-3.0,f"真峰值 {P} dBFS（要求 ≤ -3.0）")
# 音轨非静音：逐秒音量
quiet=[]
for s in range(int(dur)):
    o=sh('ffmpeg','-v','info','-ss',str(s),'-t','1','-i',v,'-vn','-af','volumedetect','-f','null','-').stderr
    m=re.findall(r"mean_volume:\s+(-?[\d.inf]+) dB",o); 
    if not m or m[0]=='-inf' or float(m[0])<-50: quiet.append(s)
chk(len(quiet)==0,"每一秒都有声音"+(f"（静音秒：{quiet}）" if quiet else ""))
# 配音是否放得下
for k,x in tl.items():
    if k.startswith('_'): continue
    chk(x['end']<=cfg['total']-0.05,f"配音 {k} 结束于 {x['end']:.2f}s（总长 {cfg['total']}s）")
# 字幕不超画面
r=sh('node',f"{common}/tools/qa_text.js",out,str(cfg['total']),str(cfg['fps']))
try: bad=json.loads(r.stdout.strip().splitlines()[-1])
except Exception: bad=[{'text':'检查脚本出错','t':0,'left':0,'right':0}]; print(r.stderr)
chk(not bad,"字幕/文字未超出画面（左右留 24px）"+(''.join(f"；「{x['text']}」{x['t']}s 位置 {x['left']}–{x['right']}" for x in bad)))
# 封面尺寸
exp={'竖版9x16':(1080,1920),'主页3x4':(1080,1440),'合集方形1x1':(1080,1080),'横版4x3':(1440,1080)}
dl=f"{out}/交付"
for k,(w,h) in exp.items():
    f=f"{dl}/第{cfg['no']}集_封面_{k}.png"
    if not os.path.exists(f): chk(False,f"封面缺失：{k}"); continue
    s=sh('ffprobe','-v','error','-show_entries','stream=width,height','-of','csv=p=0',f).stdout.strip().split(',')
    W,H=int(s[0]),int(s[1]); chk((W,H)==(w,h) and min(W,H)>=640 and max(W,H)>=854,f"封面 {k} {W}×{H}（最小 640×854）")
bad=sum(not ok for ok,_ in res)
lines=[f"# 第{cfg['no']}集《{cfg['name']}》质检报告",""]+[('- ✓ ' if ok else '- ✗ ')+m for ok,m in res]+["",f"结果：{len(res)-bad}/{len(res)} 通过","","**提醒：以上不能替代人工——声音没人听过，只抽查关键帧；猫叫 CC BY-NC，配音为合成语音，商用需自行确认。**"]
open(f"{out}/qa报告.md",'w',encoding='utf8').write('\n'.join(lines)); print('\n'.join(lines[2:-3]))
print(f"质检: {len(res)-bad}/{len(res)} 通过"); sys.exit(1 if bad else 0)
