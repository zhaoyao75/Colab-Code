import numpy as np, wave
sr=44100;T=11.2;n=int(sr*T);out=np.zeros(n)
rng=np.random.default_rng(57)
def add(t0,sig,g=1.0):
    i=int(t0*sr);j=min(n,i+len(sig))
    if 0<=i<n:out[i:j]+=sig[:j-i]*g
def tone(f,d,dec=8,h=0.3):
    t=np.arange(int(d*sr))/sr;e=np.exp(-dec*t)
    return (np.sin(2*np.pi*f*t)+h*np.sin(2*np.pi*2*f*t)+0.1*np.sin(2*np.pi*3*f*t))*e
def sweep(f0,f1,d,dec=6):
    t=np.arange(int(d*sr))/sr;f=f0+(f1-f0)*t/d;ph=2*np.pi*np.cumsum(f)/sr
    return np.sin(ph)*np.exp(-dec*t)
def noise(d,dec=10,lp=0.2):
    x=rng.standard_normal(int(d*sr))*np.exp(-dec*np.arange(int(d*sr))/sr)
    y=np.zeros_like(x);a=lp
    for i in range(1,len(x)):y[i]=y[i-1]+a*(x[i]-y[i-1])
    return y
def crinkle(d=0.25,g=1.0):
    x=rng.standard_normal(int(d*sr));env=np.abs(np.sin(np.arange(len(x))/sr*np.pi*28))*np.exp(-9*np.arange(len(x))/sr)
    hp=x-np.convolve(x,np.ones(8)/8,'same');return hp*env*g
# hook: nearly silent for the cold stare, cute BGM only after the reveal; BGM stops dead when the owner leaves
bpm=122;b=60/bpm
pent=[261.6,293.7,329.6,392.0,440.0,523.3,587.3,659.3]
mel=[0,4,2,4,5,7,5,4, 3,2,0,2,4,3,1,0, 2,4,5,4,2,0,1,2, 4,5,7,5,4,2,3,1]
T0,T1=1.95,8.3
for i,m in enumerate(mel):
    t=T0+i*b/2
    if t<T1-0.2:
        fade=min(1,(t-T0)/0.3)*min(1,(T1-0.15-t)/0.3)
        add(t,tone(pent[m],0.4,7),0.15*max(0,fade))
roots=[130.8,174.6,196,174.6]
for i in range(int((T1-T0)/b)):
    t=T0+i*b
    add(t,tone(roots[(i//2)%4] if i%2==0 else roots[(i//2)%4]*0.75,0.35,6,0.1),0.16)
# ---- sfx ----
add(0.0,sweep(190,80,0.35,7),0.5)                                              # deadpan low hit
add(1.45,noise(0.3,6,0.05),0.28)                                               # arm whoosh
for k,f in enumerate([1760,2217]):add(1.98+k*0.06,tone(f,0.35,8),0.14)         # price tag glint
add(2.5,sweep(130,60,0.25,10),0.5);add(2.5,noise(0.12,25,0.3),0.2)             # bed set down
for ts in [2.72,2.8,2.88]:add(ts,noise(0.05,40,0.8),0.22)                      # sniff x3
add(3.5,tone(2093,0.2,9),0.12);add(3.56,tone(2637,0.25,9),0.12)                # toy jingle
add(4.0,noise(0.25,7,0.05),0.35);add(4.12,sweep(200,90,0.12,14),0.45)          # swat
add(4.15,noise(0.35,8,0.2),0.18);add(4.5,tone(1500,0.04,70,0),0.2)             # toy skids / stops
add(4.9,sweep(500,950,0.18,5),0.3)                                             # "oh?" sting
for ts in [5.05,5.22,5.39]:add(ts,tone(1250,0.05,55,0),0.2)                    # quick steps
add(5.55,noise(0.35,5,0.05),0.36)                                              # leap
add(5.95,sweep(140,70,0.3,8),0.6);add(5.95,noise(0.14,22,0.35),0.35);add(5.97,crinkle(0.3,0.35))  # into the box
for ts,g in [(6.1,.3),(6.3,.25),(6.55,.3),(6.8,.22)]:add(ts,crinkle(0.22,g))    # wriggling
for k,f in enumerate([330,311,294,262]):add(6.3+k*0.17,tone(f,0.28 if k<3 else 0.6,3.5,0.5),0.17)  # tiny "wah-wah" for the owner
add(6.45,sweep(700,300,0.1,20),0.12)                                           # sweat drop
add(7.0,noise(0.4,5,0.2),0.18);add(7.5,noise(0.6,3,0.2),0.15)                  # picks up the bed
add(8.35,sweep(160,70,0.25,10),0.55);add(8.55,tone(700,0.06,60,0),0.4)         # door shuts
add(9.5,tone(1047,0.9,4),0.26);add(9.5,tone(1319,0.9,4),0.22)                  # end ding
out/=np.max(np.abs(out))*1.1
with wave.open('a_ep2f.wav','w') as w:
    w.setnchannels(1);w.setsampwidth(2);w.setframerate(sr);w.writeframes((out*32767).astype(np.int16).tobytes())
