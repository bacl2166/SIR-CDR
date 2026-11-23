import gzip
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import requests

from experiments.amazon_metadata_download import download_metadata


class Response:
    def __init__(self, body=b"", status=200, headers=None, error=None):
        self.body, self.status_code = body, status
        self.headers = headers or {}
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size):
        yield self.body
        if self.error:
            raise self.error


class MetadataDownloadTests(unittest.TestCase):
    def test_handshake_timeout_retries_with_longer_timeouts(self):
        data = gzip.compress(b"metadata")
        get = Mock(side_effect=[requests.ReadTimeout("handshake"), Response(data)])
        sleep = Mock()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "meta.json.gz"
            download_metadata("https://example.test/meta", path, get=get, sleep=sleep)
            self.assertEqual(path.read_bytes(), data)
            self.assertFalse(path.with_name(path.name + ".part").exists())
        self.assertEqual(get.call_count, 2)
        self.assertEqual(get.call_args.kwargs["timeout"], (60, 300))
        sleep.assert_called_once()

    def test_stream_failure_resumes_at_saved_byte_offset(self):
        data = gzip.compress(b"metadata" * 100)
        get = Mock(side_effect=[Response(data[:8], error=requests.ReadTimeout("stream")),
                                Response(data[8:], 206, {"Content-Range": f"bytes 8-{len(data)-1}/{len(data)}"})])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "meta.json.gz"
            download_metadata("https://example.test/meta", path, get=get, sleep=Mock())
            self.assertEqual(path.read_bytes(), data)
        self.assertEqual(get.call_args.kwargs["headers"]["Range"], "bytes=8-")

    def test_server_ignoring_range_restarts_instead_of_appending(self):
        data = gzip.compress(b"metadata")
        get = Mock(return_value=Response(data))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "meta.json.gz"
            path.with_name(path.name + ".part").write_bytes(data[:8])
            download_metadata("https://example.test/meta", path, get=get, sleep=Mock())
            self.assertEqual(path.read_bytes(), data)

    def test_exhausted_retries_keep_partial_file_for_later_run(self):
        get = Mock(side_effect=requests.ReadTimeout("unreachable"))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "meta.json.gz"
            part = path.with_name(path.name + ".part")
            part.write_bytes(b"saved")
            with self.assertRaises(requests.ReadTimeout):
                download_metadata("https://example.test/meta", path, attempts=2, get=get, sleep=Mock())
            self.assertEqual(part.read_bytes(), b"saved")
            self.assertFalse(path.exists())

    def test_completed_partial_file_can_finish_after_http_416(self):
        data = gzip.compress(b"metadata")
        get = Mock(return_value=Response(status=416, headers={"Content-Range": f"bytes */{len(data)}"}))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "meta.json.gz"
            path.with_name(path.name + ".part").write_bytes(data)
            download_metadata("https://example.test/meta", path, get=get, sleep=Mock())
            self.assertEqual(path.read_bytes(), data)

    def test_wrong_range_is_rejected_without_changing_saved_bytes(self):
        get = Mock(return_value=Response(b"bad", 206, {"Content-Range": "bytes 0-2/10"}))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "meta.json.gz"
            part = path.with_name(path.name + ".part")
            part.write_bytes(b"saved")
            with self.assertRaisesRegex(requests.RequestException, "Content-Range"):
                download_metadata("https://example.test/meta", path, attempts=1, get=get, sleep=Mock())
            self.assertEqual(part.read_bytes(), b"saved")
            self.assertFalse(path.exists())

    def test_invalid_gzip_restarts_then_publishes_only_valid_archive(self):
        data = gzip.compress(b"metadata")
        get = Mock(side_effect=[Response(b"invalid gzip"), Response(data)])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "meta.json.gz"
            download_metadata("https://example.test/meta", path, get=get, sleep=Mock())
            self.assertEqual(path.read_bytes(), data)
        self.assertNotIn("Range", get.call_args.kwargs["headers"])
