"""Verify faster enumeration preserves annotation fingerprints and alias rejection."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tracking'))
from lib.train.dataset import world_clips
from world_manifest_fast import describe_sequence, read_manifest


class FastManifestTests(unittest.TestCase):
    def test_three_sources_match_including_symlinks_and_visibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for dataset in ('lasot', 'vasttrack', 'tnl2k'):
                name = 'target-1'
                folder = root / ('train' if dataset != 'lasot' else '')
                if dataset != 'tnl2k':
                    folder /= 'target'
                folder /= name
                images = folder / ('img' if dataset == 'lasot' else 'imgs')
                images.mkdir(parents=True)
                for i in range(5):
                    (images / ('%04d.jpg' % i)).write_bytes(b'fixture')
                outside = root / ('outside_' + dataset + '.jpg')
                outside.write_bytes(b'fixture')
                (images / '0005.jpg').symlink_to(outside)
                annotation = folder / ('Groundtruth.txt' if dataset == 'vasttrack' else 'groundtruth.txt')
                annotation.write_text('1,2,3,4\n' * 6)
                (folder / ('language.txt' if dataset == 'tnl2k' else 'nlp.txt')).write_text('the target')
                if dataset == 'lasot':
                    (folder / 'full_occlusion.txt').write_text('0,0,1,0,0,0')
                    (folder / 'out_of_view.txt').write_text('0,0,0,0,1,0')
                with mock.patch.object(world_clips, 'official_names', return_value={name}):
                    old = world_clips.describe_sequence(dataset, root, name)
                    new = describe_sequence(dataset, root, name)
                for key in old:
                    if isinstance(old[key], np.ndarray):
                        np.testing.assert_array_equal(old[key], new[key])
                    else:
                        self.assertEqual(old[key], new[key])

    def test_shared_image_alias_across_videos_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = root / 'shared.jpg'
            shared.write_bytes(b'fixture')
            for name in ('train_video', 'validation_video'):
                folder = root / 'train' / name
                (folder / 'imgs').mkdir(parents=True)
                for i in range(5):
                    (folder / 'imgs' / ('%04d.jpg' % i)).symlink_to(shared)
                (folder / 'groundtruth.txt').write_text('1,2,3,4\n' * 5)
                (folder / 'language.txt').write_text('the target')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(dict(protocol='official-train-video-split-v1', roots={'tnl2k': str(root)},
                train=[dict(dataset='tnl2k', name='train_video')], val=[dict(dataset='tnl2k', name='validation_video')])))
            for reader in (world_clips.read_manifest, read_manifest):
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    reader(manifest, 5)


if __name__ == '__main__':
    unittest.main()
