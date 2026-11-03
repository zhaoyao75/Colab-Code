const {chromium}=require('playwright');
(async()=>{const b=await chromium.launch();const p=await b.newPage({viewport:{width:1080,height:1920},deviceScaleFactor:2});
const e=[];p.on('pageerror',x=>e.push(x.message));
await p.goto('file://'+process.cwd()+'/scene6.html');
await p.evaluate(()=>document.fonts.load('48px KuaiLe'));await p.evaluate(()=>document.fonts.ready);
await p.evaluate(()=>{
  R(0.5);                                   // cold stare frame of the hook
  $('fx').innerHTML=`<g transform="translate(395 ${F-24})">${bedSVG()}</g>`;   // the expensive bed, rejected
  $('door').innerHTML='';                    // no door in the cover: bed | cat | box
  $('cam').setAttribute('transform','translate(540 1625) scale(1.45) translate(-640 -1350)');
  const INK='#3b2040';
  const t=(txt,y,sz,fill,rot)=>`<g transform="translate(540 ${y}) rotate(${rot})">
    <text text-anchor="middle" y="${sz*0.07}" font-size="${sz}" fill="${INK}" opacity=".35" stroke="${INK}" stroke-width="${sz*0.2}" stroke-linejoin="round">${txt}</text>
    <text text-anchor="middle" font-size="${sz}" fill="${fill}" stroke="${INK}" stroke-width="${sz*0.2}" stroke-linejoin="round" paint-order="stroke">${txt}</text></g>`;
  $('hud').innerHTML=t('两百块的窝，',540,148,'#fff6e3',-2)+t('不稀罕。',758,200,'#ffd66b',2)+'';
});
await p.screenshot({path:'cover6_full_2x.png'});
console.log(e);await b.close();})();
