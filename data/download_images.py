#!/usr/bin/env python3
"""
Download and assemble images for VLM-GeoPrivacyBench from source datasets.

Due to copyright considerations, images are not hosted directly. Run this script
to populate data/images/ from the source datasets.

Prerequisites:
- For Flickr-yfcc_openai_*: pip install datasets pillow
- For Flickr-yfcc4k and Flickr-im2gps3k: Download and unzip from
  https://github.com/lugiavn/revisiting-im2gps (MediaFire links):
  - im2gps3k: http://www.mediafire.com/file/7ht7sn78q27o9we/im2gps3ktest.zip
  - yfcc4k:   https://www.mediafire.com/file/3og8y3o6c9de3ye/yfcc4k.zip
  Place unzipped im2gps3ktest/ and yfcc4k/ in data/ (or set paths via --im2gps3k-dir, --yfcc4k-dir)
- For Flickr-yfcc26k: Clone https://github.com/TIBHannover/GeoEstimation and run:
  python download_images.py --output resources/images/yfcc25600 --url_csv resources/yfcc25600_urls.csv --shuffle --size_suffix ""
  Then set --yfcc26k-dir to the yfcc25600/images/ folder
- For ShutterStock-GPTGeoChat: Download https://www.mediafire.com/file/luwlv2p9ofgxdb5/human.zip/file
  Unzip to get human/ folder (human/train/images/, human/test/images/, human/val/images/)
  Set --gptgeochat-dir to the human/ folder
"""

import argparse
import csv
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_metadata(metadata_path: Path) -> dict[str, list[tuple[str, str]]]:
    """Load images_metadata.csv and group (image_id, numeric_id) by image_source."""
    by_source: dict[str, list[tuple[str, str]]] = {}
    with open(metadata_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"].strip()
            source = row["image_source"].strip()
            # Extract numeric id: "yfcc-1008954785" -> "1008954785", "im2gps3k-1439764921" -> "1439764921"
            parts = image_id.split("-", 1)
            numeric_id = parts[1] if len(parts) > 1 else image_id
            if source not in by_source:
                by_source[source] = []
            by_source[source].append((image_id, numeric_id))
    return by_source


def download_yfcc_openai(
    needed: list[tuple[str, str]],
    output_dir: Path,
    split: str,
    skip_existing: bool,
    ds_split=None,
) -> int:
    """Download from HuggingFace YFCC100M_OpenAI_subset. ds_split can be dataset[split] if pre-loaded, or IterableDataset when streaming."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets pillow required for YFCC OpenAI subset", file=sys.stderr)
        return 0

    needed_ids = {nid for _, nid in needed}
    needed_map = {nid: iid for iid, nid in needed}

    if ds_split is None:
        ds_split = load_dataset(
            "dalle-mini/YFCC100M_OpenAI_subset",
            split=split,
            streaming=True,
            trust_remote_code=True,
        )

    # IterableDataset (streaming): iterate and save matches. Regular Dataset: use select(indices).
    from datasets import IterableDataset

    if isinstance(ds_split, IterableDataset):
        saved = 0
        scanned = 0
        write_futures = []
        max_workers = 4

        def _write_img(path: Path, data: bytes) -> None:
            with open(path, "wb") as f:
                f.write(data)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for row in ds_split:
                scanned += 1
                if scanned % 15000 == 0:
                    print(f"    streamed {scanned} rows, found {saved}...", flush=True)
                photoid = str(row.get("photoid", ""))
                if photoid not in needed_ids:
                    continue
                image_id = needed_map[photoid]
                out_path = output_dir / f"{image_id}.jpg"
                if skip_existing and out_path.exists():
                    saved += 1
                else:
                    img_data = row.get("img")
                    if img_data is not None:
                        if isinstance(img_data, dict) and "bytes" in img_data:
                            img_data = img_data["bytes"]
                        if isinstance(img_data, list):
                            img_data = bytes(img_data)
                        fut = ex.submit(_write_img, out_path, img_data)
                        write_futures.append(fut)
                        saved += 1
                if saved >= len(needed_ids):
                    break
            for f in as_completed(write_futures):
                f.result()
        return saved

    # Map-style Dataset: find indices, select, then save
    photoids = ds_split["photoid"]
    indices = [i for i, pid in enumerate(photoids) if str(pid) in needed_ids]
    selected = ds_split.select(indices)

    saved = 0
    for row in selected:
        photoid = str(row.get("photoid", ""))
        image_id = needed_map.get(photoid)
        if image_id is None:
            continue
        out_path = output_dir / f"{image_id}.jpg"
        if skip_existing and out_path.exists():
            saved += 1
            continue
        img_data = row.get("img")
        if img_data is None:
            continue
        if isinstance(img_data, dict) and "bytes" in img_data:
            img_data = img_data["bytes"]
        with open(out_path, "wb") as f:
            f.write(img_data)
        saved += 1
    return saved


def copy_from_im2gps3k(
    needed: list[tuple[str, str]],
    source_dir: Path,
    output_dir: Path,
    skip_existing: bool,
) -> int:
    """Copy from im2gps3k (filenames: 31700873_d7c4159106_22_25159586@N00.jpg, id=first part)."""
    saved = 0
    # Build mapping: numeric_id -> image_id
    needed_map = {nid: iid for iid, nid in needed}
    # Scan source for files matching our ids (filename starts with {id}_)
    for f in source_dir.glob("*_*_*_*@N00.jpg"):
        prefix = f.stem.split("_")[0]
        if prefix not in needed_map:
            continue
        image_id = needed_map[prefix]
        out_path = output_dir / f"{image_id}.jpg"
        if skip_existing and out_path.exists():
            saved += 1
            continue
        shutil.copy2(f, out_path)
        saved += 1
    return saved


def copy_from_yfcc4k(
    needed: list[tuple[str, str]],
    source_dir: Path,
    output_dir: Path,
    skip_existing: bool,
) -> int:
    """Copy from yfcc4k (filenames: 10003206806.jpg, id=stem)."""
    saved = 0
    needed_map = {nid: iid for iid, nid in needed}
    for ext in (".jpg", ".jpeg", ".png"):
        for f in source_dir.glob(f"*{ext}"):
            stem = f.stem
            if stem not in needed_map:
                continue
            image_id = needed_map[stem]
            out_path = output_dir / f"{image_id}.jpg"
            if skip_existing and out_path.exists():
                saved += 1
                continue
            shutil.copy2(f, out_path)
            saved += 1
    return saved


def copy_from_yfcc26k(
    needed: list[tuple[str, str]],
    source_dir: Path,
    output_dir: Path,
    skip_existing: bool,
) -> int:
    """Copy from yfcc25600 (filenames: 00_4c_2283161368, id=last part after underscore)."""
    saved = 0
    needed_map = {nid: iid for iid, nid in needed}
    for f in source_dir.glob("*_*_*"):
        parts = f.stem.split("_")
        if len(parts) < 3:
            continue
        numeric_id = parts[-1]
        if numeric_id not in needed_map:
            continue
        image_id = needed_map[numeric_id]
        out_path = output_dir / f"{image_id}.jpg"
        if skip_existing and out_path.exists():
            saved += 1
            continue
        shutil.copy2(f, out_path)
        saved += 1
    return saved


def copy_from_gptgeochat(
    needed: list[tuple[str, str]],
    human_dir: Path,
    output_dir: Path,
    skip_existing: bool,
) -> int:
    """Copy from GPTGeoChat human/ (images in train/images/, test/images/, val/images/ as {id}.jpg)."""
    saved = 0
    needed_map = {nid: iid for iid, nid in needed}
    for sub in ("train", "test", "val"):
        img_dir = human_dir / sub / "images"
        if not img_dir.is_dir():
            continue
        for ext in (".jpg", ".jpeg", ".png"):
            for f in img_dir.glob(f"*{ext}"):
                stem = f.stem
                if stem not in needed_map:
                    continue
                image_id = needed_map[stem]
                out_path = output_dir / f"{image_id}.jpg"
                if skip_existing and out_path.exists():
                    saved += 1
                    continue
                shutil.copy2(f, out_path)
                saved += 1
    return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--metadata",
        type=Path,
        default=Path(__file__).resolve().parent / "images_metadata.csv",
        help="Path to images_metadata.csv",
    )
    _project_root = Path(__file__).resolve().parent.parent
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=_project_root / "data" / "images",
        help="Output directory for images (default: <project_root>/data/images)",
    )
    ap.add_argument(
        "--im2gps3k-dir",
        type=Path,
        default=None,
        help="Path to unzipped im2gps3ktest folder (or data/im2gps3ktest)",
    )
    ap.add_argument(
        "--yfcc4k-dir",
        type=Path,
        default=None,
        help="Path to unzipped yfcc4k folder (or data/yfcc4k)",
    )
    ap.add_argument(
        "--yfcc26k-dir",
        type=Path,
        default=None,
        help="Path to yfcc25600/images/ from GeoEstimation download",
    )
    ap.add_argument(
        "--gptgeochat-dir",
        type=Path,
        default=None,
        help="Path to unzipped human/ folder from GPTGeoChat",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already exist in output",
    )
    ap.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Only process these image sources (default: all)",
    )
    ap.add_argument(
        "--unzip-if-needed",
        action="store_true",
        help="Unzip im2gps3ktest.zip and yfcc4k.zip if folders are missing",
    )
    args = ap.parse_args()

    data_dir = Path(__file__).resolve().parent
    args.im2gps3k_dir = args.im2gps3k_dir or data_dir / "im2gps3ktest"
    args.yfcc4k_dir = args.yfcc4k_dir or data_dir / "yfcc4k"
    args.yfcc26k_dir = args.yfcc26k_dir or data_dir / "yfcc25600" / "images"
    args.gptgeochat_dir = args.gptgeochat_dir or data_dir / "human"

    if args.unzip_if_needed:
        for zippath, outdir in [
            (data_dir / "im2gps3ktest.zip", args.im2gps3k_dir),
            (data_dir / "yfcc4k.zip", args.yfcc4k_dir),
        ]:
            if zippath.exists() and not outdir.is_dir():
                print(f"Unzipping {zippath.name} ... ", end="", flush=True)
                with zipfile.ZipFile(zippath) as zf:
                    zf.extractall(data_dir)
                print("done")

    by_source = load_metadata(args.metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.sources:
        by_source = {k: v for k, v in by_source.items() if k in args.sources}

    # Pre-load YFCC OpenAI dataset once if needed. Use streaming=True to avoid full download
    # before iterating; we stream and save matches on-the-fly.
    yfcc_ds = None
    if "Flickr-yfcc_openai_train" in by_source or "Flickr-yfcc_openai_valid" in by_source:
        try:
            from datasets import load_dataset

            print("Loading YFCC100M_OpenAI_subset (streaming=True, fetches on-the-fly)...", flush=True)
            yfcc_ds = load_dataset(
                "dalle-mini/YFCC100M_OpenAI_subset",
                streaming=True,
                trust_remote_code=True,
            )
            print("Stream ready.", flush=True)
        except ImportError:
            pass

    total = 0
    for source, items in by_source.items():
        n = len(items)
        print(f"[{source}] {n} images ... ", end="", flush=True)
        saved = 0

        if source == "Flickr-yfcc_openai_train":
            ds_split = yfcc_ds["train"] if yfcc_ds else None
            saved = download_yfcc_openai(
                items, args.output_dir, "train", args.skip_existing, ds_split=ds_split
            )
        elif source == "Flickr-yfcc_openai_valid":
            ds_split = yfcc_ds["validation"] if yfcc_ds else None
            saved = download_yfcc_openai(
                items, args.output_dir, "validation", args.skip_existing, ds_split=ds_split
            )
        elif source == "Flickr-im2gps3k":
            if not args.im2gps3k_dir.is_dir():
                print(f"SKIP (dir not found: {args.im2gps3k_dir})")
                continue
            saved = copy_from_im2gps3k(items, args.im2gps3k_dir, args.output_dir, args.skip_existing)
        elif source == "Flickr-yfcc4k":
            if not args.yfcc4k_dir.is_dir():
                print(f"SKIP (dir not found: {args.yfcc4k_dir})")
                continue
            saved = copy_from_yfcc4k(items, args.yfcc4k_dir, args.output_dir, args.skip_existing)
        elif source == "Flickr-yfcc26k":
            if not args.yfcc26k_dir.is_dir():
                print(f"SKIP (dir not found: {args.yfcc26k_dir})")
                continue
            saved = copy_from_yfcc26k(items, args.yfcc26k_dir, args.output_dir, args.skip_existing)
        elif source == "ShutterStock-GPTGeoChat":
            if not args.gptgeochat_dir.is_dir():
                print(f"SKIP (dir not found: {args.gptgeochat_dir})")
                continue
            saved = copy_from_gptgeochat(items, args.gptgeochat_dir, args.output_dir, args.skip_existing)
        else:
            print(f"UNKNOWN SOURCE")
            continue

        total += saved
        print(f"{saved}/{n}")

    print(f"\nTotal: {total} images in {args.output_dir}")


if __name__ == "__main__":
    main()
