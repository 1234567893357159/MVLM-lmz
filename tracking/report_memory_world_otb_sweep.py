"""Write a local report after the exploratory sweep completes."""
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/memory_world_b_otb_sweep_20261112_run01'
REPORT = ROOT/'reports/reproduction/MEMORY_WORLD_OTB_SWEEP_20261112.md'
while True:
    if (OUT/'status.json').exists():
        status=json.loads((OUT/'status.json').read_text())
        if status['status'] in ('completed','failed'): break
    time.sleep(20)
if status['status'] != 'completed':
    print('Sweep failed:',status.get('error'),flush=True)
    raise SystemExit(1)
manifest=json.loads((OUT/'manifest.json').read_text());summary=json.loads((OUT/'summary.json').read_text())
anchor=json.loads((ROOT/'output/memory_world_b_otb_historical_params_20261112_run01/summary.json').read_text())
lines=['# OTB 参数搜索（探索性）','',f"完成时间：{status['completed_at']}。两组 B 权重各测试12组参数，每组 48 条完整视频；以下均为 OTB 测试集参数搜索，不是独立最终成绩，也不保证参数空间全局最优。",'', '上轮历史参数作为参考。每组改变的参数及完整数值保存在实验目录 manifest；原始模型及其他设置不变。','', '| 组别 | 模型 | AUC | PRE | NPR |','|---|---|---:|---:|---:|']
for model in ('b1025','b4000'):
    ref=anchor[model]['otb99_lang'];lines.append('| 历史参数 | {} | {:.2f} | {:.2f} | {:.2f} |'.format(model,ref['AUC'],ref['PRE'],ref['NPR']))
    for variant,entry in summary.items():
        x=entry[model];lines.append('| {} | {} | {:.2f} | {:.2f} | {:.2f} |'.format(variant,model,x['AUC'],x['PRE'],x['NPR']))
    candidates=[('历史参数',ref,manifest['anchor'])]+[(v,summary[v][model],manifest['variants'][v]) for v in summary]
    name,best,params=max(candidates,key=lambda x:x[1]['AUC'])
    lines.extend(['',f"{model} 在上述有限尝试中最高 AUC：{best['AUC']:.2f}（{name}）；参数：`{json.dumps(params,ensure_ascii=False,sort_keys=True)}`。",''])
lines.extend(['同参数原始基线仅有上轮历史参数结果；其他参数组合未对基线重测，不能直接当作公平同参数对照。候选权重和参数都曾使用 OTB 测试结果进行探索，需在独立留出视频上复核。','',f"实验输出：`{OUT}`，含 manifest、status、summary、逐视频结果和日志。",''])
REPORT.write_text('\n'.join(lines));print('report',REPORT,flush=True)
