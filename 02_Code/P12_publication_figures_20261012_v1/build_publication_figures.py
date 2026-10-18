"""Reproducible publication figures, not model training or manuscript-editing code.

python build_publication_figures.py --source-dir PATH --output-dir NEW_PATH
Input: frozen endpoint_cluster_CI.csv and an open-licensed Chinese font.
Outputs: schematic decision graphs and unchanged endpoint operating points.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor,Color,black,white

MM=72/25.4;WIDTH=150*MM;FS=8.5
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def text(c,x,y,s,size=FS,font='Times-Roman',align='left',color=black):
 c.setFillColor(color);c.setFont(font,size)
 if align=='center':c.drawCentredString(x,y,s)
 elif align=='right':c.drawRightString(x,y,s)
 else:c.drawString(x,y,s)
def wrap(s,width,font,size=FS):
 units=list(s) if font.startswith('Noto') else s.split(' ');lines=[];line=''
 for u in units:
  sep='' if font.startswith('Noto') else ' '
  trial=(line+sep+u) if line else u
  if pdfmetrics.stringWidth(trial,font,size)>width and line:lines.append(line);line=u
  else:line=trial
 if line:lines.append(line)
 return lines
def box(c,x,y,w,h,s,font='Times-Roman',fill='#F4F4F4'):
 c.setStrokeColor(black);c.setFillColor(HexColor(fill));c.setLineWidth(.65);c.roundRect(x,y,w,h,3,stroke=1,fill=1)
 lines=[]
 for line in s.split('\n'):lines+=wrap(line,w-10,font)
 assert len(lines)*10<=h-6,(s,len(lines),h)
 yy=y+h/2+(len(lines)-1)*5-3
 for line in lines:text(c,x+w/2,yy,line,font=font,align='center');yy-=10
def arrow(c,points,dashed=False):
 c.setStrokeColor(black);c.setFillColor(black);c.setLineWidth(.65)
 c.setDash(3,2) if dashed else c.setDash()
 p=c.beginPath();p.moveTo(*points[0])
 for point in points[1:]:p.lineTo(*point)
 c.drawPath(p);c.setDash()
 x,y=points[-1];px,py=points[-2];a=math.atan2(y-py,x-px);h=c.beginPath();h.moveTo(x,y)
 h.lineTo(x-5*math.cos(a-.45),y-5*math.sin(a-.45));h.lineTo(x-5*math.cos(a+.45),y-5*math.sin(a+.45));h.close();c.drawPath(h,stroke=0,fill=1)
def workflow(src,out,lang):
 zh=lang=='ZH';font='Noto' if zh else 'Times-Roman'
 path=out/f'Figure_Workflow_{lang}.pdf';c=canvas.Canvas(str(path),pagesize=(WIDTH,365),invariant=1)
 text(c,10,350,'(a) 人员隔离的开发与评价角色' if zh else '(a) Subject-disjoint development and evaluation roles',font=font)
 labels=['训练：拟合模型与质量阈值','策略：估计收益与时延','校准：固定策略的集合阈值','评价：只评分，不调参'] if zh else ['Training\nModels and quality thresholds','Policy fitting\nBenefit and latency','Calibration\nFrozen-strategy quantiles','Evaluation\nScoring only; no tuning']
 for j,label in enumerate(labels):
  box(c,10+j*104,287,94,50,label,font)
  if j<3:arrow(c,[(104+j*104,312),(113+j*104,312)])
 text(c,WIDTH/2,270,'评价前冻结学习量；评价标签不进入调用状态' if zh else 'Freeze learned quantities; evaluation labels are not calling inputs',font=font,align='center')
 text(c,10,245,'(b) 可用主工具的逐窗门控执行' if zh else '(b) Window-wise gated execution with an available primary',font=font)
 box(c,125,201,175,32,'已完成窗口：主概率与质量' if zh else 'Completed window\nPrimary probability and quality',font)
 box(c,125,152,175,34,'收益与完整预算是否支持调用？' if zh else 'Conditional benefit and exact\nremaining budget support a call?',font)
 arrow(c,[(212.5,201),(212.5,186)])
 box(c,12,92,155,36,'不调用：保留主证据' if zh else 'No call\nKeep primary evidence',font)
 box(c,258,82,155,56,'调用次工具；付追加成本\n技术失败则弃答\n不退还调用成本' if zh else 'Call secondary and pay full cost\nTechnical failure: abstain\nNo cost refund',font)
 arrow(c,[(155,152),(89.5,142),(89.5,128)]);arrow(c,[(270,152),(335.5,146),(335.5,138)])
 text(c,108,130,'否' if zh else 'No',font=font);text(c,310,151,'是' if zh else 'Yes',font=font)
 box(c,82,46,261,28,'最终概率：主工具或概率融合\n采用最终工具的质量' if zh else 'Final primary or fused probability\nUse quality of the final tool',font)
 arrow(c,[(89.5,92),(89.5,84),(150,74)]);arrow(c,[(335.5,82),(335.5,78),(276,74)])
 box(c,82,4,261,28,'固定策略校准集合：单标签且质量通过则报告\n否则弃答并记录原因' if zh else 'Frozen-strategy set: singleton and quality pass -> report\nOtherwise abstain; record reason',font)
 arrow(c,[(212.5,46),(212.5,32)])
 c.showPage();c.save();return path

COLORS=['#444444','#0072B2','#D55E00','#6B4C9A']
def marker(c,x,y,kind,col,size=3):
 c.setStrokeColor(HexColor(col));c.setFillColor(white);c.setLineWidth(.9)
 if kind==0:c.circle(x,y,size,stroke=1,fill=1)
 elif kind==1:c.rect(x-size,y-size,2*size,2*size,stroke=1,fill=1)
 else:
  p=c.beginPath();p.moveTo(x,y+size*1.3)
  if kind==2:p.lineTo(x-size*1.2,y-size);p.lineTo(x+size*1.2,y-size)
  else:p.lineTo(x-size*1.2,y);p.lineTo(x,y-size*1.3);p.lineTo(x+size*1.2,y)
  p.close();c.drawPath(p,stroke=1,fill=1)
def loss_report(src,out):
 inputfile=src/'endpoint_cluster_CI.csv';records=list(csv.DictReader(inputfile.open(encoding='utf-8-sig')))
 sources=['VITALDB','PPGSPORT','GALAXYPPG','Stress-Predict'];strategies=['attempt','quality','fixed_fusion','adaptive']
 selected=[];points={}
 for source in sources:
  for strategy in strategies:
   d={}
   for metric in ['mean_decision_loss','report_fraction','secondary_call_fraction']:
    rr=[r for r in records if r['source']==source and r['model']=='classical' and r['strategy']==strategy and r['metric']==metric and int(r['budget_hundredths'])==(270 if source=='Stress-Predict' else 420)]
    assert len(rr)==1,(source,strategy,metric,len(rr));r=rr[0]
    assert r['valid_replicates']=='10000' and not r['undefined_reason'];selected.append(r);d[metric]=r
   points[(source,strategy)]=d
 with (out/'report_loss_points.csv').open('w',encoding='utf8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(selected[0])+['source_CSV_SHA256']);writer.writeheader()
  writer.writerows([dict(r,source_CSV_SHA256=sha(inputfile)) for r in selected])
 ymax=math.ceil(max(float(p['mean_decision_loss']['CI95_high']) for p in points.values())/.1)*.1
 ymax=max(.3,ymax);p=out/'Figure_ReportLoss.pdf';c=canvas.Canvas(str(p),pagesize=(WIDTH,480),invariant=1)
 names=['(a) AF/AFL: VitalDB','(b) HR: PPG-Sport','(c) HR: GalaxyPPG','(d) Stress-Predict']
 for n,source in enumerate(sources):
  x=43+(n%2)*211;y=265-(n//2)*210;pw=159;ph=144
  tx=lambda v:x+v/100*pw;ty=lambda v:y+v/ymax*ph
  text(c,x,y+ph+15,names[n],size=FS,font='Times-Bold')
  c.setLineWidth(.5);c.setStrokeColor(black);c.line(x,y,x+pw,y);c.line(x,y,x,y+ph)
  for tick in [0,20,40,60,80,100]:
   c.line(tx(tick),y,tx(tick),y-3);text(c,tx(tick),y-13,str(tick),align='center')
  for j in range(int(round(ymax/.1))+1):
   tick=j*.1;c.line(x-3,ty(tick),x,ty(tick));text(c,x-6,ty(tick)-3,f'{tick:.1f}',align='right')
  c.setStrokeColor(HexColor('#777777'));c.setDash(3,2);c.line(x,ty(.2),x+pw,ty(.2));c.setDash()
  text(c,x+pw/2,y-28,'Reported windows (%)',align='center')
  c.saveState();c.translate(x-29,y+ph/2);c.rotate(90);text(c,0,0,'Decision loss',align='center');c.restoreState()
  for k,strategy in enumerate(strategies):
   d=points[source,strategy];xr=d['report_fraction'];yr=d['mean_decision_loss']
   xx=tx(float(xr['estimate'])*100);yy=ty(float(yr['estimate']))
   c.setStrokeColor(HexColor(COLORS[k]));c.setLineWidth(.6)
   xl=tx(float(xr['CI95_low'])*100);xh=tx(float(xr['CI95_high'])*100);yl=ty(float(yr['CI95_low']));yh=ty(float(yr['CI95_high']))
   c.line(xl,yy,xh,yy);c.line(xx,yl,xx,yh)
   for ex in [xl,xh]:c.line(ex,yy-2,ex,yy+2)
   for ey in [yl,yh]:c.line(xx-2,ey,xx+2,ey)
   marker(c,xx,yy,k,COLORS[k],size=3 if k!=3 else 2.3)
 # Legend and analytic line use no observations from a fabricated abstain experiment.
 for k,label in enumerate(['Attempt','Quality gate','Fixed fusion / ref.','Adaptive']):
  xx=17+k*105;marker(c,xx,461,k,COLORS[k]);text(c,xx+8,458,label,size=FS)
 text(c,WIDTH/2,9,'Dashed: analytic all-abstain loss = 0.2; bars: separate marginal 95% CIs',size=FS,align='center')
 c.showPage();c.save();return p,selected

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--source-dir',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
 src=args.source_dir.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 pdfmetrics.registerFont(TTFont('Noto',str(src/'NotoSerifSC_Regular.ttf')))
 outputs=[workflow(src,out,l) for l in ['EN','ZH']];p,selected=loss_report(src,out);outputs.append(p)
 manifest={'input':'endpoint_cluster_CI.csv','input_SHA256':sha(src/'endpoint_cluster_CI.csv'),'scientific_results_version':'P10_statistics_20261011_v1','selected_rows':len(selected),'observed_endpoints_changed':False,'new_experiments':False,'workflow_is_schematic_not_performance_data':True,'outputs':[{'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in outputs]}
 (out/'new_figure_provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
 print('Rendered workflow language variants and 16 unchanged classical operating points.')
if __name__=='__main__':main()
