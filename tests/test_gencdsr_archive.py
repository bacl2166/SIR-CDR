import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from experiments.prepare_gencdsr_assets import extract_reference_archive, EXPECTED_FILES, REVISION


class ArchiveTests(unittest.TestCase):
    def test_only_verified_data_files_are_written(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); archive=root/'reference.zip'
            payload=b'{}'; digest=hashlib.sha256(payload).hexdigest()
            expected={name:digest for name in EXPECTED_FILES}
            with ZipFile(archive,'w') as z:
                for name in expected:
                    z.writestr(f'RecSys2026_GenCDSR-{REVISION}/data/Book_Movie/{name}',payload)
                z.writestr('../../outside.txt',b'not extracted')
            with patch.dict(EXPECTED_FILES,expected,clear=True):
                out=extract_reference_archive(archive,root/'data')
                self.assertTrue((out/'data/Book_Movie/Book_Movie.inter.json').exists())
                self.assertFalse((root/'outside.txt').exists())
                self.assertEqual(extract_reference_archive(archive,root/'data'),out)

    def test_corrupt_or_wrong_revision_archive_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); archive=root/'reference.zip'
            with ZipFile(archive,'w') as z:
                for name in EXPECTED_FILES:
                    z.writestr(f'RecSys2026_GenCDSR-{REVISION}/data/Book_Movie/{name}',b'bad')
            with self.assertRaisesRegex(ValueError,'hash'):
                extract_reference_archive(archive,root/'data')
            self.assertFalse((root/'data').exists())
