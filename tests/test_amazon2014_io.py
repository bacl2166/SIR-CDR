import gzip
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cdr_framework.datasets.amazon2014 import (
    AmazonRecordParseError,
    category_files,
    download_category_files,
    iter_amazon_records,
    sha256_file,
)


class FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=8192):
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset : offset + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def gzip_bytes(payload: bytes) -> bytes:
    return gzip.compress(payload)


class Amazon2014IOTests(unittest.TestCase):
    def test_parses_json_and_python_literal_lines_without_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write('{"asin":"A1","title":"JSON"}\n')
                stream.write("\n")
                stream.write("{'asin': 'A2', 'title': 'Literal'}\n")

            records = list(iter_amazon_records(path))

            self.assertEqual([record["asin"] for record in records], ["A1", "A2"])

    def test_parse_error_contains_path_and_one_based_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write('{"asin":"A1"}\n')
                stream.write("not a record\n")

            with self.assertRaises(AmazonRecordParseError) as raised:
                list(iter_amazon_records(path))

            message = str(raised.exception)
            self.assertIn(str(path), message)
            self.assertIn("line 2", message)

    def test_rejects_a_parsed_value_that_is_not_a_dictionary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "list.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write('["A1"]\n')

            with self.assertRaises(AmazonRecordParseError) as raised:
                list(iter_amazon_records(path))

            self.assertIn("line 1", str(raised.exception))

    def test_category_files_and_sha256_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = category_files("Sports_and_Outdoors", root)
            payload = b"amazon-2014"
            files.reviews.write_bytes(payload)

            self.assertEqual(files.reviews, root / "reviews_Sports_and_Outdoors_5.json.gz")
            self.assertEqual(files.metadata, root / "meta_Sports_and_Outdoors.json.gz")
            self.assertEqual(sha256_file(files.reviews), hashlib.sha256(payload).hexdigest())

    @patch("requests.get")
    def test_existing_valid_files_are_returned_without_network_requests(self, get):
        with tempfile.TemporaryDirectory() as directory:
            files = category_files("Sports_and_Outdoors", directory)
            files.reviews.write_bytes(gzip_bytes(b'{"asin":"A1"}\n'))
            files.metadata.write_bytes(gzip_bytes(b'{"asin":"A1"}\n'))

            result = download_category_files("Sports_and_Outdoors", directory)

            self.assertEqual(result, files)
            get.assert_not_called()

    @patch("requests.get")
    def test_download_uses_part_files_and_atomically_installs_valid_gzip(self, get):
        review_gzip = gzip_bytes(b'{"asin":"R1"}\n')
        metadata_gzip = gzip_bytes(b'{"asin":"M1"}\n')
        get.side_effect = [FakeResponse(review_gzip), FakeResponse(metadata_gzip)]

        with tempfile.TemporaryDirectory() as directory:
            files = download_category_files("Sports_and_Outdoors", directory)

            self.assertEqual(files.reviews.read_bytes(), review_gzip)
            self.assertEqual(files.metadata.read_bytes(), metadata_gzip)
            self.assertFalse(files.reviews.with_name(files.reviews.name + ".part").exists())
            self.assertFalse(files.metadata.with_name(files.metadata.name + ".part").exists())

    @patch("requests.get")
    def test_partial_download_sends_range_and_appends_a_partial_response(self, get):
        complete = gzip_bytes(b'{"asin":"R1"}\n')
        split_at = len(complete) // 2
        get.side_effect = [
            FakeResponse(complete[split_at:], status_code=206),
            FakeResponse(gzip_bytes(b'{"asin":"M1"}\n')),
        ]

        with tempfile.TemporaryDirectory() as directory:
            files = category_files("Sports_and_Outdoors", directory)
            part = files.reviews.with_name(files.reviews.name + ".part")
            part.parent.mkdir(parents=True, exist_ok=True)
            part.write_bytes(complete[:split_at])

            download_category_files("Sports_and_Outdoors", directory)

            self.assertEqual(files.reviews.read_bytes(), complete)
            self.assertEqual(get.call_args_list[0].kwargs["headers"], {"Range": f"bytes={split_at}-"})

    @patch("requests.get")
    def test_invalid_gzip_is_rejected_without_installing_destination(self, get):
        get.return_value = FakeResponse(b"not gzip")

        with tempfile.TemporaryDirectory() as directory:
            files = category_files("Sports_and_Outdoors", directory)

            with self.assertRaises(OSError):
                download_category_files("Sports_and_Outdoors", directory)

            self.assertFalse(files.reviews.exists())
            self.assertTrue(files.reviews.with_name(files.reviews.name + ".part").exists())


if __name__ == "__main__":
    unittest.main()
