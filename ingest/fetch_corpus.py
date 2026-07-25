"""Fetches the Spark docs corpus at a pinned ref.

Writes markdown into data/corpus/ and a manifest with a hash per file, so the
corpus is reproducible and eval scores stay comparable between runs.

    python -m ingest.fetch_corpus
    python -m ingest.fetch_corpus --ref v3.5.1 --verify
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = "apache/spark"
DOCS_PATH = "docs"
DEFAULT_REF = "v3.5.1"

ROOT = Path(__file__).parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
MANIFEST = ROOT / "data" / "manifest.json"


def get(url, retries=3):
    # Unauthenticated GitHub API allows 60 requests/hour. We use one listing call
    # and then raw.githubusercontent for the files, which is not rate limited the
    # same way, so this stays well inside the limit.
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rag-eval-harness"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as err:
            if err.code == 403 and attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  rate limited, waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError(f"gave up on {url}")


def list_markdown(ref):
    url = f"https://api.github.com/repos/{REPO}/contents/{DOCS_PATH}?ref={ref}"
    entries = json.loads(get(url))
    return [e for e in entries if e["name"].endswith(".md") and e["type"] == "file"]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(ref=DEFAULT_REF, verify=False):
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    entries = list_markdown(ref)
    print(f"{len(entries)} markdown files in {REPO}/{DOCS_PATH} at {ref}")

    manifest = {"repo": REPO, "ref": ref, "path": DOCS_PATH, "files": {}}
    total = 0

    for i, entry in enumerate(entries, 1):
        name = entry["name"]
        target = CORPUS_DIR / name

        if target.exists() and not verify:
            data = target.read_bytes()
        else:
            data = get(entry["download_url"])
            target.write_bytes(data)

        digest = sha256(data)
        # GitHub's blob sha is a git object hash, not a content hash, so we keep our
        # own. Storing theirs too makes it easy to spot an upstream change.
        manifest["files"][name] = {"sha256": digest, "bytes": len(data), "blob_sha": entry["sha"]}
        total += len(data)

        if i % 50 == 0:
            print(f"  {i}/{len(entries)}")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"wrote {len(entries)} files, {total / 1024:.0f} KB, to {CORPUS_DIR}")
    print(f"manifest at {MANIFEST}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default=DEFAULT_REF, help="git tag or sha to pin to")
    parser.add_argument("--verify", action="store_true",
                        help="re-download everything and rewrite hashes")
    args = parser.parse_args()
    fetch(ref=args.ref, verify=args.verify)


if __name__ == "__main__":
    main()
