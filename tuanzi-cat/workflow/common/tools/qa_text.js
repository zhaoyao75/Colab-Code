// 字幕是否超出画面。用法: node qa_text.js <scene目录> <总时长> <fps>
const {chromium}=require('playwright');const [dir,total,fps]=process.argv.slice(2);
(async()=>{const b=await chromium.launch();const p=await b.newPage({viewport:{width:1080,height:1920}});
await p.goto('file://'+dir+'/scene.html');await p.evaluate(()=>document.fonts.load('48px KuaiLe'));await p.evaluate(()=>document.fonts.ready);
const r=await p.evaluate(([total])=>{const bad=[];for(let t=0;t<total;t+=0.1){R(t);
 document.querySelectorAll('#hud text').forEach(e=>{const g=e.getBoundingClientRect();if(g.width<2)return;
  if(g.left<24||g.right>1056)bad.push({t:+t.toFixed(1),text:e.textContent,left:Math.round(g.left),right:Math.round(g.right)});});}
 return bad;},[+total]);
const seen={};r.forEach(x=>{if(!seen[x.text])seen[x.text]=x});console.log(JSON.stringify(Object.values(seen)));await b.close();})();
