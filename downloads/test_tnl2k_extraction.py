"""Protect the train/test boundary while recovering a reviewed name collision."""
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import hashlib
import zipfile

import prepare_official_training as prep

from prepare_official_training import copy_member, tnl2k_member_path


class Tnl2kExtractionTests(unittest.TestCase):
    def member(self, name, kind=tarfile.REGTYPE):
        member = tarfile.TarInfo(name)
        member.type = kind
        return member

    def test_training_files_are_preserved(self):
        for suffix in ('firstFrame_bbox.jpg', 'imgs/00001.jpg', 'groundtruth.txt', 'language.txt'):
            with self.subTest(suffix=suffix):
                self.assertEqual(tnl2k_member_path(self.member('tnl2k_train/train_seq/' + suffix), {'test_seq'}),
                                 Path('train_seq') / suffix)

    def test_reviewed_name_collision_preserves_full_training_sequence(self):
        for suffix in ('firstFrame_bbox.jpg', 'imgs/000001.jpg', 'groundtruth.txt', 'language.txt'):
            self.assertEqual(tnl2k_member_path(self.member('tnl2k_train/monitor_manypeople/' + suffix),
                                             {'monitor_manypeople'}),
                             Path('monitor_manypeople_train') / suffix)
        with self.assertRaisesRegex(RuntimeError, 'train/test overlap'):
            tnl2k_member_path(self.member('tnl2k_train/monitor_manypeople/imgs/000001.jpg'),
                             {'monitor_manypeople_train'})

    def test_real_test_data_is_rejected(self):
        for suffix in ('firstFrame_bbox.jpg', 'imgs/00001.jpg', 'groundtruth.txt', 'language.txt', 'imgs/firstFrame_bbox.jpg', 'other.jpg'):
            with self.subTest(suffix=suffix), self.assertRaisesRegex(RuntimeError, 'train/test overlap'):
                tnl2k_member_path(self.member('tnl2k_train/test_seq/' + suffix), {'test_seq'})

    def test_unsafe_or_invalid_paths_are_rejected(self):
        for name in ('/tnl2k_train/train_seq/file', 'tnl2k_train/../file', 'other/train_seq/file', 'tnl2k_train/file'):
            with self.subTest(name=name), self.assertRaises((ValueError, RuntimeError)):
                tnl2k_member_path(self.member(name), {'test_seq'})

    def test_links_and_special_members_are_rejected(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE):
            with self.subTest(kind=kind), self.assertRaisesRegex(RuntimeError, 'Unsupported'):
                tnl2k_member_path(self.member('tnl2k_train/test_seq/firstFrame_bbox.jpg', kind), {'test_seq'})

    def test_renamed_sequence_cannot_share_test_frame_content(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root / 'train/monitor_manypeople_train/imgs'
            test = root / 'test/monitor_manypeople/imgs'
            train.mkdir(parents=True)
            test.mkdir(parents=True)
            (train / '1.jpg').write_bytes(b'training frame')
            (test / 'different_name.jpg').write_bytes(b'test frame')
            prep.check_tnl2k_renamed_frames(root / 'train', root / 'test')
            (test / 'different_name.jpg').write_bytes(b'training frame')
            with self.assertRaisesRegex(RuntimeError, 'shares test frames'):
                prep.check_tnl2k_renamed_frames(root / 'train', root / 'test')

    def test_directories_are_ignored(self):
        for name in ('tnl2k_train', 'tnl2k_train/test_seq', 'tnl2k_train/train_seq/imgs'):
            self.assertIsNone(tnl2k_member_path(self.member(name, tarfile.DIRTYPE), {'test_seq'}))

    def test_stream_renames_collision_and_resumes_existing_training_files(self):
        stream = io.BytesIO()
        files = [('tnl2k_train/train_seq/language.txt', b'object'),
                 ('tnl2k_train/monitor_manypeople/firstFrame_bbox.jpg', b'preview'),
                 ('tnl2k_train/train_seq/imgs/00001.jpg', b'image')]
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            for name, data in files:
                member = self.member(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            existing = root / 'train_seq/language.txt'
            existing.parent.mkdir()
            existing.write_bytes(b'object')
            stamp = existing.stat().st_mtime_ns
            stream.seek(0)
            with tarfile.open(fileobj=stream, mode='r|gz') as archive:
                for member in archive:
                    relative = tnl2k_member_path(member, {'monitor_manypeople'})
                    if relative is not None:
                        copy_member(archive.extractfile(member), root / relative, member.size)
            self.assertFalse((root / 'monitor_manypeople').exists())
            self.assertEqual((root / 'monitor_manypeople_train/firstFrame_bbox.jpg').read_bytes(), b'preview')
            self.assertEqual(existing.stat().st_mtime_ns, stamp)
            self.assertEqual((root / 'train_seq/imgs/00001.jpg').read_bytes(), b'image')


class Tnl2kRevisionTests(unittest.TestCase):
    def test_official_correction_is_backed_up_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'downloads'
            out.mkdir()
            root = Path(temp) / 'train'
            target = root / 'seq/groundtruth.txt'
            target.parent.mkdir(parents=True)
            target.write_bytes(b'old annotation')
            archive = out / 'tnl2k_revised_annotations.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('seq/groundtruth.txt', b'correct annotation')
                z.writestr('seq/language.txt', b'unrelated text')
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            with patch.object(prep, 'OUT', out), patch.object(prep, 'TNL2K_REVISIONS_SHA256', digest):
                prep.apply_tnl2k_revisions(root)
                stamp = target.stat().st_mtime_ns
                prep.apply_tnl2k_revisions(root)
            self.assertEqual(target.read_bytes(), b'correct annotation')
            self.assertEqual(target.stat().st_mtime_ns, stamp)
            self.assertEqual((out / 'tnl2k_original_annotations/seq/groundtruth.txt').read_bytes(), b'old annotation')
            self.assertFalse((root / 'seq/language.txt').exists())

    def test_modified_revision_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / 'tnl2k_revised_annotations.zip').write_bytes(b'corrupt')
            with patch.object(prep, 'OUT', out), self.assertRaisesRegex(RuntimeError, 'SHA256 mismatch'):
                prep.apply_tnl2k_revisions(out / 'train')


if __name__ == '__main__':
    unittest.main()
