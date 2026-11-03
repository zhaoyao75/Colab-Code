import numpy as np, wave, sys, json
E=sys.argv[1]; CFG=json.load(open(sys.argv[2])); COMMON=sys.argv[3];sr=44100;T=CFG['total'];n=int(sr*T)
rng=np.random.default_rng(57)
def rd(p):
    w=wave.open(p);a=np.frombuffer(w.readframes(w.getnframes()),np.int16).astype(float)/32768
    return a[::w.getnchannels()]
def mk():return np.zeros(n)
bgm=mk();sfx=mk();vo=mk()
def add(buf,t0,sig,g=1.0):
    i=int(t0*sr);j=min(n,i+len(sig))
    if 0<=i<n:buf[i:j]+=sig[:j-i]*g
def tone(f,d,dec=8,h=0.3):
    t=np.arange(int(d*sr))/sr;e=np.exp(-dec*t)
    return (np.sin(2*np.pi*f*t)+h*np.sin(2*np.pi*2*f*t)+0.1*np.sin(2*np.pi*3*f*t))*e
def sweep(f0,f1,d,dec=6):
    t=np.arange(int(d*sr))/sr;f=f0+(f1-f0)*t/d;return np.sin(2*np.pi*np.cumsum(f)/sr)*np.exp(-dec*t)
def noise(d,dec=10,lp=0.2):
    x=rng.standard_normal(int(d*sr))*np.exp(-dec*np.arange(int(d*sr))/sr)
    y=np.zeros_like(x)
    for i in range(1,len(x)):y[i]=y[i-1]+lp*(x[i]-y[i-1])
    return y
def bloop(f0,f1,d=0.16):
    t=np.arange(int(d*sr))/sr;f=f0+(f1-f0)*t/d;return np.sin(2*np.pi*np.cumsum(f)/sr)*np.exp(-t*20)
b=60/118
pent=[261.6,293.7,329.6,392.0,440.0,523.3,587.3,659.3]
mel=[0,4,2,4,5,7,5,4, 3,2,0,2,4,3,1,0, 2,4,5,4,2,0,1,2, 4,5,7,5,4,2,3,1]
TL=json.load(open(f"{E}/timeline.json"))
T0=CFG['bgm']['start'];T1=CFG['bgm'].get('stop') or TL['_card']-0.1
for i,m in enumerate(mel):
    t=T0+i*b/2
    if t<T1-0.2:
        fade=min(1,(t-T0)/0.3)*min(1,(T1-0.15-t)/0.3);add(bgm,t,tone(pent[m],0.4,7),0.15*max(0,fade))
roots=[130.8,174.6,196,174.6]
for i in range(int((T1-T0)/b)):
    t=T0+i*b;add(bgm,t,tone(roots[(i//2)%4] if i%2==0 else roots[(i//2)%4]*0.75,0.35,6,0.1),0.16)
add(sfx,0.0,sweep(190,80,0.35,7),0.5)                      # deadpan low hit
add(sfx,1.45,noise(0.3,6,0.05),0.28)                       # pull-back whoosh
for k,f in enumerate([1760,2217]):add(sfx,1.98+k*0.06,tone(f,0.35,8),0.12)  # bubble glint
for tt in [3.6,4.8,6.3]:add(sfx,tt,bloop(500,1000),0.3);add(sfx,tt,noise(0.2,14,0.3),0.12)
add(sfx,3.68,noise(0.5,4,0.08),0.12)                       # hiss (hot)
add(sfx,3.85,tone(1800,0.05,60,0),0.2)                     # quick yank
for tt in [4.0,4.25,5.35,5.6]:add(sfx,tt,tone(1500,0.04,70,0),0.18)   # knob clicks
add(sfx,4.8,tone(2400,0.25,10),0.1);add(sfx,4.87,tone(3000,0.25,10),0.08)  # cold glint
add(sfx,5.8,sweep(700,300,0.1,20),0.12)                    # sweat drop
for k in range(5):add(sfx,7.3+k*0.22,bloop(300+k*40,700+k*60,0.14),0.18)  # sinking bubbles
add(sfx,9.3,noise(0.06,50,0.8),0.3);add(sfx,9.31,tone(1568,0.1,40,0),0.15)  # bubble hat pop
add(sfx,TL['_card'],tone(1047,0.9,4),0.26);add(sfx,TL['_card'],tone(1319,0.9,4),0.22)      # end ding
add(sfx,0,np.zeros(1))
for v in CFG['vo']:
    a=rd(f"{E}/vo/{v['id']}.wav");add(vo,TL[v['id']]['at'],a,1.0)
M=CFG['meow'];m=rd(f"{COMMON}/meow/{M['file']}");add(sfx,M['at'],m,M['gain'])
def save(x,name):
    with wave.open(name,'w') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(sr);w.writeframes((np.clip(x,-1,1)*32767).astype(np.int16).tobytes())
save(bgm/np.max(np.abs(bgm)),f"{E}/bgm.wav");save(sfx/np.max(np.abs(sfx)),f"{E}/sfx.wav");save(vo/np.max(np.abs(vo)),f"{E}/vo.wav")
