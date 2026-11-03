const {chromium}=require('playwright');
(async()=>{const b=await chromium.launch();const p=await b.newPage({viewport:{width:1080,height:1920},deviceScaleFactor:2});
const e=[];p.on('pageerror',x=>e.push(x.message));
await p.goto('file://'+process.cwd()+'/scene6.html');
await p.evaluate(()=>document.fonts.load('48px KuaiLe'));await p.evaluate(()=>document.fonts.ready);
await p.evaluate(()=>{
  const INK='#3b2040',BX=900;
  $('shelf').setAttribute('display','none');$('chair').setAttribute('transform','translate(-3000 0)');$('glow').innerHTML='';$('door').innerHTML='';
  // cat: deadpan, wearing the cupcake from episode 1
  const o={x:700,y:F,sx:1,sy:1,rot:-1.5,eye:'half',dx:0,dy:-4,mouth:'n',blush:0,tail:6,fishMouth:false,fishSq:0,shs:1,shy:F,phi:0,pw:0};
  drawCat(o);
  // props behind the cat: the 299元 bed (ep2) on the left and the cardboard box (ep2) on the right
  const box=`<g transform="translate(60 0)"><path d="M${BX-155} ${F-150} L${BX-205} ${F-188} L${BX-90} ${F-162} Z" fill="#b98757" stroke="${INK}" stroke-width="5" stroke-linejoin="round"/><path d="M${BX+155} ${F-150} L${BX+205} ${F-188} L${BX+90} ${F-162} Z" fill="#b98757" stroke="${INK}" stroke-width="5" stroke-linejoin="round"/><rect x="${BX-155}" y="${F-164}" width="310" height="30" fill="#8f6538" stroke="${INK}" stroke-width="5"/><rect x="${BX-155}" y="${F-150}" width="310" height="150" rx="6" fill="#d4a674" stroke="${INK}" stroke-width="6"/><rect x="${BX-30}" y="${F-150}" width="60" height="150" fill="#e8cfa0" opacity=".85"/><rect x="${BX+50}" y="${F-106}" width="84" height="56" rx="4" fill="#fff6e3" stroke="${INK}" stroke-width="3"/><path d="M${BX+62} ${F-88} H${BX+122} M${BX+62} ${F-74} H${BX+108} M${BX+62} ${F-60} H${BX+116}" stroke="${INK}" stroke-width="3" stroke-linecap="round"/></g>`;
  $('fishW').innerHTML=box+`<g transform="translate(440 ${F-22}) scale(1.05)">${bedSVG()}</g>`;
  // cupcake hat + a few sparkles (in front of the cat)
  let fx=`<g transform="translate(702 ${F-262}) rotate(-5) scale(0.95)">${cake(1,0)}</g>`;
  [[590,F-400,0.8],[820,F-420,0.6]].forEach(([x,y,s])=>{fx+=star(x,y,s,0.95,'#ffe79a');});
  $('fx').innerHTML=fx;
  $('cam').setAttribute('transform','translate(540 930) scale(1.2) translate(-700 -1350)');
  // ---- text layout (screen space, 1080x1080 square) ----
  const T=(txt,y,sz,fill,rot,sw=0.2)=>`<g transform="translate(540 ${y}) rotate(${rot})">
    <text text-anchor="middle" y="${sz*0.07}" font-size="${sz}" fill="${INK}" opacity=".35" stroke="${INK}" stroke-width="${sz*sw}" stroke-linejoin="round">${txt}</text>
    <text text-anchor="middle" font-size="${sz}" fill="${fill}" stroke="${INK}" stroke-width="${sz*sw}" stroke-linejoin="round" paint-order="stroke">${txt}</text></g>`;
  const pill=`<g transform="translate(540 268) rotate(-1.5)"><rect x="-236" y="-52" width="472" height="104" rx="52" fill="#ffd66b" stroke="${INK}" stroke-width="8"/><text text-anchor="middle" y="24" font-size="68" fill="${INK}">动画合集</text></g>`;
  const bub=[[110,150,26,0],[975,120,34,1.7],[930,270,18,3.1],[170,300,16,2.3],[1010,400,22,4.4]].map(([x,y,r])=>`<g transform="translate(${x} ${y})"><circle r="${r*1.4}" fill="#cfe9ff" fill-opacity=".6" stroke="#7fb8e6" stroke-width="3"/><circle cx="${-r*0.45}" cy="${-r*0.45}" r="${r*0.26}" fill="#fff" opacity=".85"/></g>`).join('');
  const strip=`<g transform="translate(540 1030)"><rect x="-300" y="-30" width="600" height="60" rx="30" fill="${INK}" opacity=".78"/><text text-anchor="middle" y="14" font-size="36" fill="#fff6e3">第1集 · 第2集 · 持续更新中</text></g>`;
  $('hud').innerHTML=bub+T('团子猫日常',140,138,'#fff6e3',-2)+pill+strip;
});
await p.screenshot({path:'coll_2x.png',clip:{x:0,y:0,width:1080,height:1080}});
console.log(e);await b.close();})();
