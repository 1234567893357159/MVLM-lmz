import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tracking'))
import sweep_mvlm_parallel as parallel


class ParallelSweepTests(unittest.TestCase):
    def test_jobs_are_isolated_and_model_variant_paths_do_not_collide(self):
        spec = {'data': {'kind': 'benchmark'}, 'models': {'a': {}, 'a_b': {}},
                'variants': {'b_c': {}, 'c': {}}}
        jobs = parallel.make_jobs(spec)
        self.assertEqual(len(jobs), 4)
        self.assertEqual(len({parallel.job_dir(Path('/tmp/x'), job) for job in jobs}), 4)
        self.assertEqual(list(jobs[0]['spec']['models']), ['a'])
        self.assertEqual(len(spec['models']), 2)
        with self.assertRaises(ValueError):
            parallel.make_jobs({'models': {'../bad': {}}, 'variants': {'v': {}}})

    def test_report_shows_only_complete_metrics(self):
        spec = {'models': {'model': {}}, 'variants': {'done': {}, 'partial': {}}}
        jobs = parallel.make_jobs(spec)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            done = parallel.job_dir(out, jobs[0]); partial = parallel.job_dir(out, jobs[1])
            parallel.write_json(done / 'status.json', {'completed': 2, 'total': 2, 'status': 'completed'})
            parallel.write_json(done / 'summary.json', {'model': {'done': {'per_dataset': {'ds': {'AUC': 50, 'PRE': 60, 'NPR': 70}}}}})
            parallel.write_json(partial / 'status.json', {'completed': 1, 'total': 2, 'status': 'running'})
            parallel.render(out, jobs, {('5', 0): (None, jobs[1], None)}, {('model', 'done'): '完成'}, '运行中', '留出验证')
            report = (out / 'LIVE.md').read_text()
            self.assertIn('ds: 50.000 / 60.000 / 70.000', report)
            self.assertIn('| model | partial | 5 | 运行中 | 1/2 | — |', report)


if __name__ == '__main__':
    unittest.main()
