#!/usr/bin/env python3
"""
Check a built package before anything is uploaded.

    python -m build
    python tools/check_dist.py dist

A wheel or a source archive is permanent once it is on a package index, so the build is
checked against a list written by hand, tools/dist_files.txt:

- the wheel holds exactly the files the list names for it, and the source archive exactly
  the files named for it: nothing more, nothing missing;
- every packaged file has the same bytes as the file in this repository;
- the packaged question set matches the hashes in its own manifest;
- the metadata says what pyproject.toml says: the name, the version in
  parsica_recall/recall_common.py, the license, the Python floor, exactly the optional
  extras in EXTRAS, and no dependency but each extra's own SDK.

Standard library only. Exit status: 0 when everything holds, 1 when anything does not,
2 when the folder does not hold exactly one wheel and one source archive.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tarfile
import zipfile
from email.parser import Parser
from typing import Dict, List, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIST = os.path.join(REPO, "tools", "dist_files.txt")
NAME = "parsica-recall"
DIST = "parsica_recall"
MAX_BYTES = 12 * 1024 * 1024      # the wheel is about 2 MB; a much larger one holds something it should not
# The optional extras and the one SDK each installs: generate is the anthropic SDK (both
# generators), openai and google are the question builder's other providers, and bedrock is
# the anthropic SDK again with its AWS signing. Nothing is installed without an extra.
EXTRAS = {"generate": "anthropic", "openai": "openai", "google": "google-genai", "bedrock": "anthropic"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def version() -> str:
    with open(os.path.join(REPO, "parsica_recall", "recall_common.py"), encoding="utf-8") as f:
        return re.search(r'^HARNESS_VERSION = "([^"]+)"', f.read(), re.M).group(1)


def read_list() -> Dict[str, List[Tuple[str, str]]]:
    """tools/dist_files.txt: a [wheel] and an [sdist] section, one line per packaged file,
    as `path in the archive <- path in the repository`, or the path alone for a file the
    build writes itself (the metadata). {version} stands for the version."""
    out: Dict[str, List[Tuple[str, str]]] = {}
    section = None
    with open(LIST, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                out[section] = []
                continue
            if section is None:
                raise SystemExit(f"{LIST}: a file is listed before any [section]")
            inside, _, source = (part.strip() for part in line.partition("<-"))
            out[section].append((inside, source))
    return out


def wheel_files(path: str) -> Dict[str, bytes]:
    with zipfile.ZipFile(path) as z:
        return {i.filename: z.read(i) for i in z.infolist() if not i.is_dir()}


def sdist_files(path: str) -> Dict[str, bytes]:
    out = {}
    with tarfile.open(path, "r:gz") as t:
        for m in t.getmembers():
            if m.isfile():
                out[m.name] = t.extractfile(m).read()
            elif not m.isdir():
                out[m.name] = b""  # a link or a device has no place in a source archive; reported below
    return out


def check_files(kind: str, found: Dict[str, bytes], expected: List[Tuple[str, str]], ver: str,
                problems: List[str]) -> None:
    names = {inside.replace("{version}", ver): source for inside, source in expected}
    extra = sorted(set(found) - set(names))
    missing = sorted(set(names) - set(found))
    for name in extra:
        problems.append(f"{kind}: holds a file the list does not name: {name}")
    for name in missing:
        problems.append(f"{kind}: a listed file is missing: {name}")
    for name, source in sorted(names.items()):
        if not source or name not in found:
            continue
        try:
            with open(os.path.join(REPO, source), "rb") as f:
                repo_bytes = f.read()
        except OSError as exc:
            problems.append(f"{kind}: {name} is listed as {source}, which cannot be read here ({exc})")
            continue
        if found[name].replace(b"\r\n", b"\n") != repo_bytes.replace(b"\r\n", b"\n"):
            problems.append(f"{kind}: {name} does not have the bytes of {source} in this repository")


def check_metadata(text: str, ver: str, where: str, problems: List[str]) -> None:
    meta = Parser().parsestr(text)
    want = {"Name": NAME, "Version": ver, "License-Expression": "MIT", "Requires-Python": ">=3.11"}
    for key, value in want.items():
        if meta.get(key) != value:
            problems.append(f"{where}: {key} is {meta.get(key)!r}, expected {value!r}")
    for req in meta.get_all("Requires-Dist") or []:
        extra = re.search(r"""extra\s*==\s*['"]([^'"]+)['"]""", req)
        package = re.sub(r"[-_.]+", "-", re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req).group(1)).lower()
        if not extra or extra.group(1) not in EXTRAS:
            problems.append(f"{where}: a dependency outside the optional extras: {req}")
        elif package != EXTRAS[extra.group(1)]:
            problems.append(f"{where}: the extra {extra.group(1)} brings {package}, expected {EXTRAS[extra.group(1)]}: {req}")
    if sorted(meta.get_all("Provides-Extra") or []) != sorted(EXTRAS):
        problems.append(f"{where}: extras are {meta.get_all('Provides-Extra')}, expected {sorted(EXTRAS)}")


def check_question_set(found: Dict[str, bytes], prefix: str, where: str, problems: List[str]) -> None:
    """The packaged current set against its own manifest: the hashes a run checks."""
    manifest_name = next((n for n in found if n.startswith(prefix) and n.endswith("/manifest.json")), None)
    if manifest_name is None:
        problems.append(f"{where}: no question-set manifest under {prefix}")
        return
    folder = manifest_name[:-len("manifest.json")]
    manifest = json.loads(found[manifest_name].decode("utf-8"))
    for name, digest in manifest["files"].items():
        data = found.get(folder + name)
        if data is None:
            problems.append(f"{where}: the manifest names {name}, which is not packaged")
        elif sha256(data) != digest:
            problems.append(f"{where}: {folder + name} does not match the SHA-256 in its manifest")


def main(argv: List[str]) -> int:
    if len(argv) != 1 or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    dist = argv[0]
    wheels = sorted(n for n in os.listdir(dist) if n.endswith(".whl"))
    sdists = sorted(n for n in os.listdir(dist) if n.endswith(".tar.gz"))
    others = sorted(n for n in os.listdir(dist) if not n.endswith((".whl", ".tar.gz")))
    if len(wheels) != 1 or len(sdists) != 1 or others:
        print(f"ERROR: {dist} must hold exactly one wheel and one source archive; found {wheels + sdists + others}",
              file=sys.stderr)
        return 2
    ver = version()
    expected = read_list()
    problems: List[str] = []

    want_wheel = f"{DIST}-{ver}-py3-none-any.whl"
    want_sdist = f"{DIST}-{ver}.tar.gz"
    if wheels[0] != want_wheel:
        problems.append(f"the wheel is named {wheels[0]}, expected {want_wheel}")
    if sdists[0] != want_sdist:
        problems.append(f"the source archive is named {sdists[0]}, expected {want_sdist}")
    for name in (wheels[0], sdists[0]):
        size = os.path.getsize(os.path.join(dist, name))
        if size > MAX_BYTES:
            problems.append(f"{name} is {size:,} bytes, over the limit of {MAX_BYTES:,}")

    wheel = wheel_files(os.path.join(dist, wheels[0]))
    check_files("wheel", wheel, expected["wheel"], ver, problems)
    info = f"{DIST}-{ver}.dist-info/"
    if info + "METADATA" in wheel:
        check_metadata(wheel[info + "METADATA"].decode("utf-8"), ver, "wheel METADATA", problems)
    entry = wheel.get(info + "entry_points.txt", b"").decode("utf-8")
    if "parsica-recall = parsica_recall.cli:main" not in entry:
        problems.append("wheel: entry_points.txt does not define the parsica-recall command")
    check_question_set(wheel, "parsica_recall/data/", "wheel", problems)

    sdist = sdist_files(os.path.join(dist, sdists[0]))
    check_files("sdist", sdist, expected["sdist"], ver, problems)
    root = f"{DIST}-{ver}/"
    if root + "PKG-INFO" in sdist:
        check_metadata(sdist[root + "PKG-INFO"].decode("utf-8"), ver, "sdist PKG-INFO", problems)
    check_question_set(sdist, root + "data/", "sdist", problems)

    print(f"wheel  {wheels[0]}: {len(wheel)} files, {os.path.getsize(os.path.join(dist, wheels[0])):,} bytes, "
          f"sha256 {sha256(open(os.path.join(dist, wheels[0]), 'rb').read())}")
    print(f"sdist  {sdists[0]}: {len(sdist)} files, {os.path.getsize(os.path.join(dist, sdists[0])):,} bytes, "
          f"sha256 {sha256(open(os.path.join(dist, sdists[0]), 'rb').read())}")
    if problems:
        print(f"\nFAILED: {len(problems)} problem(s)")
        for p in problems:
            print("  " + p)
        return 1
    print("\nOK: both archives hold exactly the listed files, with this repository's bytes, and the metadata matches")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
