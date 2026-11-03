const {chromium}=require('playwright');const fs=require('fs');
const [dir,cfgPath,outDir]=process.argv.slice(2);const CFG=JSON.parse(fs.readFileSync(cfgPath,'utf8')).cover;
const INK='#3b2040';
const sizes={
 '9x16':{w:1080,h:1920,sx:540,sy:1500,k:1.68,l1:[470,148],l2:[690,196]},
 '3x4':{w:1080,h:1440,sx:540,sy:1040,k:1.4,l1:[300,136],l2:[505,180]},
 '4x3':{w:1440,h:1080,sx:720,sy:850,k:1.15,l1:[135,115],l2:[290,150]},
 '1x1':{w:1080,h:1080,sx:540,sy:790,k:1.2,pill:true}
};
(async()=>{const b=await chromium.launch();
for(const [k,c] of Object.entries(sizes)){c.key=k;
 const p=await b.newPage({viewport:{width:c.w,height:c.h},deviceScaleFactor:2});
 p.on('pageerror',x=>console.log('ERR',x.message));
 await p.goto('file://'+dir+'/scene.html');await p.evaluate(()=>document.fonts.load('48px KuaiLe'));await p.evaluate(()=>document.fonts.ready);
 await p.evaluate(([c,INK,CFG])=>{
  window.NOARM=true;
  const svg=$('s');svg.setAttribute('width',c.w);svg.setAttribute('height',c.h);svg.setAttribute('viewBox',`0 0 ${c.w} ${c.h}`);
  const v=document.querySelector('rect[fill="url(#vg)"]');v.setAttribute('width',c.w);v.setAttribute('height',c.h);
  R(CFG.scene_t);$('door').innerHTML='';
  const pick=(v,d)=>(v&&typeof v==='object')?(v[c.key]??d):(v??d);const F0=CFG.focus||[665,1250],Zm=pick(CFG.zoom,1),SY=pick(CFG.sy,c.sy);$('cam').setAttribute('transform',`translate(${c.sx} ${SY}) scale(${c.k*Zm}) translate(${-F0[0]} ${-F0[1]})`);
  const T=(txt,x,y,sz,fill,rot)=>`<g transform="translate(${x} ${y}) rotate(${rot})"><text text-anchor="middle" y="${sz*0.07}" font-size="${sz}" fill="${INK}" opacity=".35" stroke="${INK}" stroke-width="${sz*0.2}" stroke-linejoin="round">${txt}</text><text text-anchor="middle" font-size="${sz}" fill="${fill}" stroke="${INK}" stroke-width="${sz*0.2}" stroke-linejoin="round" paint-order="stroke">${txt}</text></g>`;
  const bub=[[110,150,26],[c.w-105,130,34],[c.w-150,300,18],[170,320,16]].map(([x,y,r])=>`<g transform="translate(${x} ${y})"><circle r="${r*1.4}" fill="#cfe9ff" fill-opacity=".6" stroke="#7fb8e6" stroke-width="3"/><circle cx="${-r*0.45}" cy="${-r*0.45}" r="${r*0.26}" fill="#fff" opacity=".85"/></g>`).join('');
  let h=bub;
  if(c.pill){
   h+=T('团子猫日常',540,150,138,'#fff6e3',-2)+`<g transform="translate(540 290) rotate(-1.5)"><rect x="-236" y="-52" width="472" height="104" rx="52" fill="#ffd66b" stroke="${INK}" stroke-width="8"/><text text-anchor="middle" y="24" font-size="68" fill="${INK}">动画合集</text></g>`+
   `<g transform="translate(540 1030)"><rect x="-340" y="-30" width="680" height="60" rx="30" fill="${INK}" opacity=".78"/><text text-anchor="middle" y="14" font-size="36" fill="#fff6e3">${CFG.strip}</text></g>`;
  } else {
   const cx=c.w/2;
   h+=T(CFG.lines[0],cx,c.l1[0],c.l1[1]*0.72,'#fff6e3',-2)+T(CFG.lines[1],cx,c.l2[0],c.l2[1]*0.82,'#ffd66b',2);
  }
  $('hud').innerHTML=h;
 },[c,INK,CFG]);
 await p.screenshot({path:`${outDir}/cover_${k}_2x.png`,clip:{x:0,y:0,width:c.w,height:c.h}});
 await p.close();}
await b.close();})();
