"""Check that training exclusions preserve source data and reach the sampler."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lib.config.mvlm.config import cfg
from lib.train.base_functions import names2datasets, update_settings
from lib.train.dataset import Tnl2k


class TrainingExclusionsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.names = {'healthy_a', 'healthy_b', 'Fight_game_004-Done'}
        for name in self.names:
            folder = self.root / 'train' / name
            folder.mkdir(parents=True)
            (folder / 'groundtruth.txt').write_text('1,2,3,4\n2,3,4,5\n')
            (folder / 'language.txt').write_text('the target\n')
        self.environment = SimpleNamespace(tnl2k_dir=str(self.root))
        self.mock_env = patch('lib.train.dataset.tnl2k.env_settings', return_value=self.environment)
        self.mock_env.start()
        self.addCleanup(self.mock_env.stop)

    def test_configured_training_factory_excludes_only_bad_clip(self):
        settings = SimpleNamespace(env=self.environment)
        update_settings(settings, cfg)
        dataset, = names2datasets(['TNL2K_train'], settings, image_loader=lambda path: path)
        self.assertEqual(set(dataset.sequence_list), self.names - {'Fight_game_004-Done'})
        self.assertEqual(dataset.excluded_sequence_list, ['Fight_game_004-Done'])
        self.assertEqual(dataset.get_num_sequences(), 2)
        self.assertEqual(dataset.get_sequence_info(0)['bbox'].shape, (2, 4))
        self.assertEqual({p.name for p in (self.root / 'train').iterdir()}, self.names)
        self.assertTrue((self.root / 'train/Fight_game_004-Done/groundtruth.txt').is_file())

    def test_exclusion_precedes_fraction_sampling(self):
        dataset = Tnl2k(data_fraction=1.0, exclude_sequences=['Fight_game_004-Done'])
        self.assertEqual(set(dataset.sequence_list), self.names - {'Fight_game_004-Done'})

    def test_explicit_empty_exclusions_preserve_all_source_sequences(self):
        self.assertEqual(set(Tnl2k(exclude_sequences=[]).sequence_list), self.names)

    def test_reject_empty_training_set_and_mistyped_exclusions(self):
        with self.assertRaises(ValueError):
            Tnl2k(exclude_sequences=self.names)
        with self.assertRaises(TypeError):
            Tnl2k(exclude_sequences='Fight_game_004-Done')


if __name__ == '__main__':
    unittest.main()
