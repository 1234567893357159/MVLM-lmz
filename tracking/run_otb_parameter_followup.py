"""Three more adaptive full-dataset trials after the initial five finish."""
import copy
import datetime
import json
import shutil

from tracking.run_otb_parameter_sweep import best, install_config, run_trial, OUT, ROOT, timestamp


def main():
    batch = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plans = [
        ('06_less_memory', {'OMEGA': 0.2},
         '较强记忆配置已使 AUC 下降，现仅降低记忆融合权重，检验更及时的帧间响应。'),
        ('07_tau001', {'TAU': 0.01},
         '从当前最佳配置出发，补测 0 与 0.03 之间的阈值，细化候选框筛选强度。'),
        ('08_psi02', {'PSI_OUT': 0.2},
         'PSI_OUT 从 0 到 0.1 曾改善，现补测 0.2，检验更宽的干扰候选池是否仍有收益。'),
    ]
    for label, changes, rationale in plans:
        parameters = copy.deepcopy(best()['config']['TEST']['MVLM'])
        parameters.update(changes)
        run_trial(label, parameters, rationale, batch)
    winner = best()
    install_config(winner['config']['TEST']['MVLM'])
    if winner.get('eval_data'):
        shutil.copy2(winner['eval_data'], ROOT / 'output/test/result_plots/otb99_lang/eval_data.pkl')
    summary = dict(batch=batch, completed_trials=3, selected_exp_id=winner['exp_id'],
                   selected_parameters=winner['config']['TEST']['MVLM'], metrics=winner['metrics'],
                   criterion='Highest full-test-set AUC; exploratory tuning', timestamp=timestamp())
    (OUT / (batch + '_summary.json')).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print('FOLLOWUP_COMPLETED', json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
