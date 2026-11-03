"""Persistent two-GPU train/validation queue, W&B uploader and experiment archive."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.train.memory_run_utils import write_json


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


class Supervisor:
    def __init__(self, out):
        self.out = out
        self.spec = json.loads((out/'run_spec.json').read_text())
        self.source = Path(self.spec['source'])
        self.config = out/'config.yaml'
        self.run = None
        self.child = None
        self.offset = int(json.loads((out/'upload_cursor.json').read_text()).get('bytes',0)) if (out/'upload_cursor.json').exists() else 0
        self.state = dict(status='initializing',supervisor_pid=os.getpid(),started_at=now(),
                          global_step=0,target_steps=4000,physical_gpus=[4,5],
                          batch_per_rank=self.spec['batch_per_rank'],global_batch=self.spec['global_batch'],
                          wandb_url=self.spec['wandb']['url'])
        if (out/'latest_checkpoint.json').exists():
            self.state['global_step']=json.loads((out/'latest_checkpoint.json').read_text())['step']
        self.env = dict(os.environ,CUDA_VISIBLE_DEVICES='4,5',MVLM_DEVICE='cuda',OMP_NUM_THREADS='1',
            MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',WANDB_CONSOLE='off',
            WANDB_INIT_TIMEOUT='60',WANDB_HTTP_TIMEOUT='20',PYTHONPATH=str(self.source))
        self.archive = out/'EXPERIMENT_RECORD.md'

    def record(self, event, **values):
        payload = dict(time=now(),event=event,**values)
        with (self.out/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(payload,ensure_ascii=False)+'\n')

    def status(self, **values):
        self.state.update(values,updated_at=now())
        write_json(self.out/'training_status.json',self.state)
        text = ['# 记忆拼接 CENTER 双卡实验档案','',
            '状态：'+self.state['status']+'；更新时间：'+self.state['updated_at'],
            '','W&B：'+self.spec['wandb']['url'],'',
            'GPU4、5；每卡%d个四帧片段，全局%d；固定4,000次更新。'%(self.spec['batch_per_rank'],self.spec['global_batch']),
            '冻结视觉/文本编码器与CENTER BN运行统计，训练记忆读取与CENTER。',
            '学习率峰值记忆1e-4、CENTER1e-5；前200步预热，第3000步后乘0.1。',
            '','## 当前记录','',json.dumps(self.state,ensure_ascii=False,indent=2),
            '','## 档案入口','',
            '- run_spec.json：初始化、数据/源码哈希及资源预算',
            '- config.yaml / split/：冻结配置和训练/验证清单',
            '- training_metrics.jsonl：跨卡平均训练指标',
            '- events.jsonl / resource_metrics.jsonl：阶段事件与资源',
            '- checkpoints/：每500步检查点、优化器、随机状态及步数日程',
            '- rank*_start_audit.json / rank*_final_audit.json：逐段冻结、BN与更新审计',
            '- validation/：基线、固定30条子集及完整150条验证结果',
            '- validation_summary.json / best.json：完整验证的候选模型',
            '- logs/ / wandb/：训练验证日志与完整W&B本地文件','',
            '验证集来自本轮训练部分，官方初始化可能已见过。子集与完整验证分数分开记录。',
            '未评估标准测试集；原GPU6、7评测保持独立。',
            '检查点保存各rank随机状态；DataLoader预取/worker状态未完整捕获，不保证位级恢复。']
        if self.state.get('error'):
            text += ['', '错误：'+self.state['error']]
        self.archive.write_text('\n'.join(text)+'\n')

    def open_wandb(self):
        import wandb
        os.environ.update({k:v for k,v in self.env.items() if k.startswith('WANDB_')})
        options = dict(project=self.spec['wandb']['project'],entity=self.spec['wandb']['entity'],
            id=self.spec['wandb']['id'],name=self.out.name,dir=str(self.out),resume='allow',
            config=self.spec,settings=wandb.Settings(init_timeout=60))
        try:
            self.run = wandb.init(mode='online',**options)
            self.state['wandb_mode']='online'
        except Exception as error:
            self.record('wandb_online_init_failed',error=str(error))
            # Keep a complete W&B artifact even if the proxy is temporarily down.
            options['resume']=None
            self.run = wandb.init(mode='offline',**options)
            self.state['wandb_mode']='offline'
        self.run.define_metric('global_step')
        self.run.define_metric('*',step_metric='global_step')
        self.run.define_metric('train/*',step_metric='global_step')
        self.run.define_metric('lr/*',step_metric='global_step')
        self.run.define_metric('val/global_step')
        self.run.define_metric('val/*',step_metric='val/global_step')
        self.run.define_metric('baseline/*',step_metric='val/global_step')
        self.run.summary['status']='training'
        write_json(self.out/'wandb_status.json',dict(mode=self.state['wandb_mode'],
            id=self.run.id,url=self.spec['wandb']['url'],local_dir=self.run.dir,updated_at=now()))

    def upload(self):
        metrics = self.out/'training_metrics.jsonl'
        if metrics.exists():
            with metrics.open() as stream:
                stream.seek(self.offset)
                while True:
                    position=stream.tell()
                    line=stream.readline()
                    if not line or not line.endswith('\n'):
                        stream.seek(position)
                        break
                    payload = json.loads(line)
                    self.state['global_step']=payload['global_step']
                    self.run.log({k:v for k,v in payload.items() if k!='updated_at'})
                self.offset=stream.tell()
            write_json(self.out/'upload_cursor.json',dict(bytes=self.offset,updated_at=now()))
        try:
            result=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True,timeout=10)
            payload={'global_step':self.state['global_step']}
            for row in result.splitlines():
                index,memory,util=map(int,row.split(','))
                if index in (4,5):
                    payload.update({'resource/gpu%d_memory_mib'%index:memory,'resource/gpu%d_utilization'%index:util})
            with (self.out/'resource_metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(time=now(),**payload))+'\n')
            self.run.log(payload)
        except subprocess.SubprocessError:
            pass

    def execute(self, command, label):
        for filename, expected in self.spec.get('input_hashes', {}).items():
            actual=hashlib.sha256(Path(filename).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError('Frozen experiment input changed: '+filename)
        logs=self.out/'logs';logs.mkdir(exist_ok=True)
        self.record('start',stage=label,command=command)
        with (logs/(label+'.log')).open('a',buffering=1) as log:
            self.child=subprocess.Popen(command,cwd=self.source,env=self.env,stdout=log,
                                        stderr=subprocess.STDOUT,start_new_session=True)
            self.status(status=label,worker_pid=self.child.pid)
            while self.child.poll() is None:
                self.upload();self.status()
                time.sleep(5)
            result=self.child.returncode
        self.upload()
        self.record('end',stage=label,exit_code=result)
        self.child=None
        if result:
            raise RuntimeError('%s exited %d; inspect logs/%s.log'%(label,result,label))

    def validation(self, step, scope, baseline=False):
        label=('baseline_' if baseline else 'step%06d_'%step)+scope
        folder=self.out/'validation'/label
        if not (folder/'metrics.json').exists():
            command=[sys.executable,str(self.source/'tracking/evaluate_memory_concat_run.py'),
                '--config',str(self.config),'--manifest',str(self.out/'split'/(
                    'quick_manifest.json' if scope=='quick30' else 'validation_manifest.json')),
                '--weights',str(Path(self.spec['baseline_checkpoint']) if baseline else self.out/'checkpoints'/('MVLM_step%06d.pth.tar'%step)),
                '--out',str(folder),'--scope',scope,'--workers','8']
            if baseline:command.append('--baseline')
            self.execute(command,label)
        result=json.loads((folder/'metrics.json').read_text())
        prefix='baseline' if baseline else 'val'
        payload={'val/global_step':step,prefix+'/'+scope+'/mean_AUC':result['mean_AUC']}
        for dataset,values in result['metrics'].items():
            payload.update({prefix+'/'+scope+'/'+dataset+'/'+key:value for key,value in values.items()})
        self.run.log(payload)
        self.record('validation_result',step=step,scope=scope,baseline=baseline,mean_AUC=result['mean_AUC'])
        return result

    def main(self):
        self.status()
        self.open_wandb()
        quick, full = {}, {}
        # The baseline is fixed before training; evaluate it after releasing
        # the first training segment so the two GPUs can train immediately.
        # No result is used to alter data, parameters or the 4,000-step budget.
        for end in (1000,2000,3000,4000):
            checkpoint=self.out/'checkpoints'/('MVLM_step%06d.pth.tar'%end)
            if not checkpoint.exists():
                previous=self.out/'latest_checkpoint.json'
                resume=json.loads(previous.read_text())['path'] if previous.exists() else None
                command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                    str(self.source/'tracking/train_memory_concat_run.py'),'--config',str(self.config),
                    '--out',str(self.out),'--stop-step',str(end)]
                if resume:command+=['--resume',resume]
                self.execute(command,'training_to_%04d'%end)
            if end==1000:self.validation(0,'quick30',baseline=True)
            quick[end]=self.validation(end,'quick30')
            if end in (2000,4000):
                if end==2000:self.validation(0,'full150',baseline=True)
                full[end]=self.validation(end,'full150')
                write_json(self.out/'validation_summary.json',dict(quick=quick,full=full))
        candidate=max(quick,key=lambda step:quick[step]['mean_AUC'])
        if candidate not in full:
            full[candidate]=self.validation(candidate,'full150')
        best=max(full,key=lambda step:full[step]['mean_AUC'])
        write_json(self.out/'validation_summary.json',dict(quick=quick,full=full,best_step=best))
        write_json(self.out/'best.json',dict(step=best,metrics=full[best],
            checkpoint=str(self.out/'checkpoints'/('MVLM_step%06d.pth.tar'%best))))
        self.run.summary.update(status='completed',best_val_step=best,best_val_mean_AUC=full[best]['mean_AUC'])
        self.status(status='completed',global_step=4000,best_step=best)
        self.run.finish()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out=args.out.resolve()
    lock=(args.out/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    supervisor=Supervisor(args.out)
    try:
        supervisor.main()
    except BaseException as error:
        if supervisor.child and supervisor.child.poll() is None:
            os.killpg(supervisor.child.pid,signal.SIGTERM)
        supervisor.record('failure',error=repr(error),traceback=traceback.format_exc())
        supervisor.status(status='failed',error=repr(error))
        if supervisor.run:
            supervisor.run.summary['status']='failed';supervisor.run.finish(exit_code=1)
        raise
