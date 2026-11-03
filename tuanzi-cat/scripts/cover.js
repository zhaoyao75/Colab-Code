const {chromium}=require('playwright');
(async()=>{const b=await chromium.launch();const p=await b.newPage({viewport:{width:1080,height:1920}});
const e=[];p.on('pageerror',x=>e.push(x.message));
await p.goto('file://'+process.cwd()+'/scene.html');
await p.evaluate(()=>document.fonts.load('48px KuaiLe'));await p.evaluate(()=>document.fonts.ready);
await p.evaluate(()=>{
  R(9.2);
  // camera: big close-up, cat lower in frame so the title has room
  $('cam').setAttribute('transform','translate(540 1265) scale(1.65) translate(-690 -830)');
  // sparkles around the cake hat + keep sweat drop already in fx
  let f=$('fx').innerHTML;
  [[560,640,1.1],[820,610,0.9],[690,540,0.7],[500,760,0.6],[870,740,0.75]].forEach(([x,y,s])=>{f+=star(x,y,s,0.95,'#ffe79a');});
  $('fx').innerHTML=f;
  const INK='#3b2040';
  const t=(txt,y,sz,fill,rot,dx=0)=>`<g transform="translate(${540+dx} ${y}) rotate(${rot})">
    <text text-anchor="middle" y="${sz*0.07}" font-size="${sz}" fill="${INK}" opacity=".35" stroke="${INK}" stroke-width="${sz*0.2}" stroke-linejoin="round">${txt}</text>
    <text text-anchor="middle" font-size="${sz}" fill="${fill}" stroke="${INK}" stroke-width="${sz*0.2}" stroke-linejoin="round" paint-order="stroke">${txt}</text></g>`;
  $('hud').innerHTML=t('团子猫',440,175,'#fff6e3',-3)+t('偷蛋糕！',640,200,'#ffd66b',2);
});
await p.screenshot({path:'cover_full.png'});
console.log(e);await b.close();})();
