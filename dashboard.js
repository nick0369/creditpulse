/* Same portable model artifact as FastAPI. No API key, remote text upload, or fake scores. */
'use strict';
const assetBase=typeof document!=='undefined'?new URL('.',document.currentScript.src):null;
let model;
const sigmoid=x=>1/(1+Math.exp(-Math.max(-700,Math.min(700,x))));
function treeScore(node,values){
 if(Object.hasOwn(node,'leaf'))return node.leaf;
 const value=values[Number(node.split.replace('f',''))];
 const child=value==null?node.missing:(value<node.split_condition?node.yes:node.no);
 return treeScore(node.children.find(n=>n.nodeid===child),values);
}
function scoreCredit(input,m){
 const values=m.features.map(n=>input[n]==null?m.medians[n]:Number(input[n]));
 if(values.some(v=>!Number.isFinite(v))||input.limit_bal<=0||input.latest_status>input.max_status)throw Error('Check numeric inputs, positive credit limit, and latest status ≤ worst status.');
 if(input.delinquent_months<0||input.delinquent_months>6||input.latest_status< -2||input.max_status>9)throw Error('Repayment history is outside the supported range.');
 let margin=m.kind==='xgboost'?m.base_margin+m.trees.reduce((s,t)=>s+treeScore(t,values),0):m.intercept+m.coef.reduce((s,c,i)=>s+c*(values[i]-m.means[i])/m.scales[i],0);
 const p=sigmoid(m.calibration[0]*margin+m.calibration[1]);
 return {confidence:p,needs_review:p>=m.threshold,route:p>=m.threshold?'Human review candidate':'Standard monitoring'};
}
function scoreText(input,m){
 const text=input.text.normalize('NFKC').toLowerCase().trim();
 if(text.length<20||text.length>20000)throw Error('Enter 20–20,000 characters.');
 const words=text.match(/[a-z0-9]{2,}/g)||[];
 const terms=[...words,...words.slice(1).map((w,i)=>words[i]+' '+w)];
 const counts=new Map();for(const t of terms)counts.set(t,(counts.get(t)||0)+1);
 const values=[];for(const [term,n]of counts){const i=m.vocabulary[term];if(i!==undefined)values.push([i,(1+Math.log(n))*m.idf[i]]);}
 const norm=Math.sqrt(values.reduce((s,[i,v])=>s+v*v,0))||1;
 const scores=m.intercepts.map((v,c)=>v+values.reduce((s,[i,w])=>s+m.coefficients[c][i]*w/norm,0));
 const exps=scores.map(s=>Math.exp(s-Math.max(...scores)));const total=exps.reduce((a,b)=>a+b,0);
 const probabilities=exps.map(x=>x/total),index=scores.indexOf(Math.max(...scores)),confidence=probabilities[index];
 return {route:m.classes[index],confidence,needs_review:confidence<m.threshold||!values.length,probabilities};
}
// Export also supports a Node parity check without creating a browser dependency.
if(typeof module!=='undefined')module.exports={scoreCredit,scoreText};
if(typeof document!=='undefined'){
 document.getElementById('score-form').addEventListener('submit',e=>{
  e.preventDefault();const out=document.getElementById('prediction');
  try{
   if(!model)throw Error('Model is still loading.');
   const data=Object.fromEntries(new FormData(e.target));
   const result=model.task==='credit'?scoreCredit(data,model):scoreText(data,model);
   out.textContent=`${result.route} · ${(100*result.confidence).toFixed(1)}% ${model.task==='credit'?'default probability':'top-route score'} · ${result.needs_review?'Human review required':'Accepted by the review policy'}`;
   if(model.task==='text'){
    const bars=document.getElementById('route-bars');bars.replaceChildren();
    model.classes.forEach((name,i)=>{const row=document.createElement('div');row.className='bar-row';row.textContent=`${name}: ${(100*result.probabilities[i]).toFixed(1)}%`;const meter=document.createElement('meter');meter.min=0;meter.max=1;meter.value=result.probabilities[i];row.append(meter);bars.append(row);});
   }
  }catch(error){out.textContent=error.message;}
 });
 Promise.all([fetch(new URL('model.json',assetBase)).then(r=>{if(!r.ok)throw Error('Model failed to load');return r.json();}),fetch(new URL('metrics.json',assetBase)).then(r=>r.json())])
 .then(([m,metrics])=>{model=m;document.getElementById('model-status').textContent=`Trained model ready · ${m.version}`;document.getElementById('run-model').disabled=false;})
 .catch(error=>{document.getElementById('model-status').textContent=error.message;});
 document.getElementById('roi-form').addEventListener('input',()=>{
  const d=Object.fromEntries(new FormData(document.getElementById('roi-form')));
  const volume=Number(d.volume),unit=Number(d.unit),rate=Number(d.rate)/100;
  document.getElementById('roi-output').textContent=`Scenario value: ${(volume*unit*rate).toLocaleString(undefined,{maximumFractionDigits:0})} value units/month before operating costs. Assumptions only; not measured ROI.`;
 });
}
