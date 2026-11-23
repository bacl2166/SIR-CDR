"""Resumable metadata acquisition for the new Amazon preparation entry point."""
from pathlib import Path
import re
import time

import requests

from cdr_framework.datasets.amazon2014 import _validate_gzip


def download_metadata(url, destination, *, attempts=6, get=requests.get, sleep=time.sleep):
    destination = Path(destination)
    if destination.exists():
        _validate_gzip(destination)
        return
    if attempts < 1:
        raise ValueError("attempts must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    for attempt in range(1, attempts + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        print(f"METADATA_DOWNLOAD {destination.name} attempt={attempt}/{attempts} offset={offset}", flush=True)
        try:
            with get(url, stream=True, headers=headers, timeout=(60, 300)) as response:
                if response.status_code == 416 and offset:
                    if response.headers.get("Content-Range") != f"bytes */{offset}":
                        raise requests.RequestException("Unsatisfied range differs from local file size")
                else:
                    response.raise_for_status()
                    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                        raise requests.RequestException("Compressed HTTP transfer cannot preserve range offsets")
                    total = None
                    if response.status_code == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                        if not match or int(match[1]) != offset or not offset <= int(match[2]) < int(match[3]):
                            raise requests.RequestException("Invalid Content-Range")
                        total = int(match[3])
                    elif response.status_code != 200:
                        raise requests.RequestException(f"Unexpected download status {response.status_code}")
                    append = response.status_code == 206 and offset > 0
                    written = offset if append else 0
                    next_progress = written + 16 * 1024**2
                    with part.open("ab" if append else "wb") as stream:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                stream.write(chunk)
                                written += len(chunk)
                                if written >= next_progress:
                                    print(f"METADATA_BYTES {destination.name} {written}", flush=True)
                                    next_progress = written + 16 * 1024**2
                    if total is not None and written != total:
                        raise requests.RequestException(f"Incomplete range download: {written}/{total}")
            try:
                _validate_gzip(part)
            except (EOFError, OSError):
                # A complete HTTP response with an invalid archive must restart.
                part.write_bytes(b"")
                raise
            part.replace(destination)
            print(f"METADATA_READY {destination.name} bytes={destination.stat().st_size}", flush=True)
            return
        except (requests.RequestException, EOFError, OSError) as error:
            print(f"METADATA_RETRY {destination.name} {type(error).__name__}: {error}", flush=True)
            if attempt == attempts:
                raise
            sleep(min(10 * attempt, 60))
