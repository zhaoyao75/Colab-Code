# 配音 + 台词时间自动排布。用法: plan.py <episode.json> <out/vo> <tts.py> <timeline.json>
# episode.json 的 vo 每句可写: "at"(固定起点) 或 "after":"前一句id" + "gap"(间隔,默认0.15)
#   "fit":true → 若会超出总时长，自动加速（最多1.5倍，超过则报错）
import json,sys,subprocess,math
cfg_p,out,tts,tl_p=sys.argv[1:5]; cfg=json.load(open(cfg_p)); total=cfg['total']; MARGIN=0.05
def dur(p): return float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',p]).decode())
TRIM="silenceremove=start_periods=1:start_threshold=-45dB:stop_periods=-1:stop_duration=0.15:stop_threshold=-45dB"
def run(*a): subprocess.run(a,check=True)
raw={}
for v in cfg['vo']:
    mp3=f"{out}/{v['id']}.mp3"; wav=f"{out}/{v['id']}.raw.wav"
    run('python3',tts,mp3,v['text']); run('ffmpeg','-v','error','-y','-i',mp3,'-af',TRIM,'-ar','44100','-ac','1',wav); raw[v['id']]=dur(wav)
tl={}; errors=[]; prev=None
for v in cfg['vo']:
    i=v['id']; d=raw[i]; tempo=v.get('tempo',1.0)
    at=v['at'] if 'at' in v else tl[v['after']]['end']+v.get('gap',0.15)
    if v.get("fit") and at+d/tempo>total-MARGIN-0.06:
        need=d/(total-MARGIN-0.06-at)
        tempo=math.ceil(need*20)/20
        if tempo>1.5: errors.append(f"{i}: 要加速到 {tempo:.2f} 倍才放得下（>1.5），请缩短台词或提前起点")
    final=f"{out}/{i}.wav"
    if tempo!=1.0: run('ffmpeg','-v','error','-y','-i',f"{out}/{i}.raw.wav",'-af',f"atempo={tempo}",final)
    else: run('ffmpeg','-v','error','-y','-i',f"{out}/{i}.raw.wav",'-c','copy',final)
    d2=dur(final); end=at+d2
    if end>total-MARGIN+1e-6: errors.append(f"{i}: 结束于 {end:.2f}s，超出总时长 {total}s")
    if prev and at<tl[prev]['end']+0.05: errors.append(f"{i}: 与上一句 {prev} 重叠或间隔<0.05s（{tl[prev]['end']:.2f}→{at:.2f}）")
    tl[i]={'at':round(at,3),'end':round(end,3),'tempo':tempo,'text':v['text']}; prev=i
last=cfg['vo'][-1]['id']
tl['_card']=round(tl[last]['at']-0.1,3)   # 结尾卡在最后一句预告配音前0.1秒出现
print(f"{'id':8}{'起点':>7}{'结束':>7}{'倍速':>6}  文本")
for k,v in tl.items():
    if k.startswith('_'): continue
    print(f"{k:8}{v['at']:7.2f}{v['end']:7.2f}{v['tempo']:6.2f}  {v['text']}")
print(f"结尾卡出现: {tl['_card']}s   总时长: {total}s")
json.dump(tl,open(tl_p,'w'),ensure_ascii=False,indent=1)
if errors:
    print('\n'.join('✗ '+e for e in errors)); sys.exit(1)
