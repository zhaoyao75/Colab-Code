const {chromium}=require('playwright');
(async()=>{const b=await chromium.launch();const p=await b.newPage({viewport:{width:1440,height:1080},deviceScaleFactor:2});
const e=[];p.on('pageerror',x=>e.push(x.message));
await p.goto('file://'+process.cwd()+'/scene6.html');
await p.evaluate(()=>document.fonts.load('48px KuaiLe'));await p.evaluate(()=>document.fonts.ready);
await p.evaluate(()=>{
  const svg=$('s');svg.setAttribute('width',1440);svg.setAttribute('height',1080);svg.setAttribute('viewBox','0 0 1440 1080');
  document.querySelector('rect[fill="url(#vg)"]').setAttribute('width',1440);document.querySelector('rect[fill="url(#vg)"]').setAttribute('height',1080);
  R(0.5);$('fx').innerHTML='';$('door').innerHTML='';
  const INK='#3b2040',BX=900;
  const box=`<g transform="translate(20 0)"><path d="M${BX-155} ${F-150} L${BX-205} ${F-188} L${BX-90} ${F-162} Z" fill="#b98757" stroke="${INK}" stroke-width="5" stroke-linejoin="round"/><path d="M${BX+155} ${F-150} L${BX+205} ${F-188} L${BX+90} ${F-162} Z" fill="#b98757" stroke="${INK}" stroke-width="5" stroke-linejoin="round"/><rect x="${BX-155}" y="${F-164}" width="310" height="30" fill="#8f6538" stroke="${INK}" stroke-width="5"/><rect x="${BX-155}" y="${F-150}" width="310" height="150" rx="6" fill="#d4a674" stroke="${INK}" stroke-width="6"/><rect x="${BX-30}" y="${F-150}" width="60" height="150" fill="#e8cfa0" opacity=".85"/><rect x="${BX+50}" y="${F-106}" width="84" height="56" rx="4" fill="#fff6e3" stroke="${INK}" stroke-width="3"/><path d="M${BX+62} ${F-88} H${BX+122} M${BX+62} ${F-74} H${BX+108} M${BX+62} ${F-60} H${BX+116}" stroke="${INK}" stroke-width="3" stroke-linecap="round"/></g>`;
  $('fishW').innerHTML=box+`<g transform="translate(385 ${F-22}) scale(1.05)">${bedSVG()}</g>`;
  $('cam').setAttribute('transform','translate(720 850) scale(1.3) translate(-670 -1350)');
  const T=(txt,x,y,sz,fill,rot)=>`<g transform="translate(${x} ${y}) rotate(${rot})"><text text-anchor="middle" y="${sz*0.07}" font-size="${sz}" fill="${INK}" opacity=".35" stroke="${INK}" stroke-width="${sz*0.2}" stroke-linejoin="round">${txt}</text><text text-anchor="middle" font-size="${sz}" fill="${fill}" stroke="${INK}" stroke-width="${sz*0.2}" stroke-linejoin="round" paint-order="stroke">${txt}</text></g>`;
  const tag=`<g transform="translate(720 120)"><rect x="-210" y="-34" width="420" height="68" rx="34" fill="#ffd66b" stroke="${INK}" stroke-width="6"/><text text-anchor="middle" y="14" font-size="40" fill="${INK}">团子猫日常 · 第2集</text></g>`;
  $('hud').innerHTML=tag+T('两百块的窝，不稀罕。',720,290,118,'#fff6e3',-1.5);
});
await p.screenshot({path:'land_2x.png',clip:{x:0,y:0,width:1440,height:1080}});
console.log(e);await b.close();})();
