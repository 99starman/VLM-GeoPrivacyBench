Images are not hosted in this repository for copyright reasons. To obtain them, run the download script.

## Quick Start

```bash
# From project root
python data/download_images.py
```

This populates `data/images/` with images from all sources. Use `--sources` to process only specific sources. All images must be in `data/images/` for the benchmark to work. The download script saves there by default. If you use a custom `--output-dir` or obtain images from manual downloads, move them into `data/images/`.

## Source Datasets

### yfcc_openai (train + validation)
- **Source:** [dalle-mini/YFCC100M_OpenAI_subset](https://huggingface.co/datasets/dalle-mini/YFCC100M_OpenAI_subset) on HuggingFace
- Downloaded automatically by the script

### yfcc4k and im2gps3k
- **Source:** [revisiting-im2gps](https://github.com/lugiavn/revisiting-im2gps)
- **Links:**
  - im2gps3k: http://www.mediafire.com/file/7ht7sn78q27o9we/im2gps3ktest.zip
  - yfcc4k: https://www.mediafire.com/file/3og8y3o6c9de3ye/yfcc4k.zip
- Place the zip files in `data/`, then either:
  - Unzip manually, or
  - Run `python data/download_images.py --unzip-if-needed`

### yfcc26k
- **Source:** [TIBHannover/GeoEstimation](https://github.com/TIBHannover/GeoEstimation)
- Run their download script:
  ```bash
  python download_images.py --output resources/images/yfcc25600 --url_csv resources/yfcc25600_urls.csv --shuffle --size_suffix ""
  ```
- Copy the `yfcc25600` folder into `data/`, or set `--yfcc26k-dir` to point to `yfcc25600/images/`

### GPTGeoChat
- **Source:** [GPTGeoChat](https://github.com/ethanm88/GPTGeoChat)
- **Link:** https://www.mediafire.com/file/luwlv2p9ofgxdb5/human.zip/file
- Unzip `human.zip` and place the `human/` folder in `data/`, or set `--gptgeochat-dir`

## Script Options

```
python data/download_images.py --help
```

- `--output-dir`: Output directory (default: `data/images/`). Use default to keep all images in the expected location.
- `--im2gps3k-dir`, `--yfcc4k-dir`, `--yfcc26k-dir`, `--gptgeochat-dir`: Override source paths
- `--skip-existing`: Skip images that already exist
- `--sources`: Only process listed sources (e.g. `--sources Flickr-yfcc4k Flickr-im2gps3k`)
- `--unzip-if-needed`: Unzip im2gps3ktest.zip and yfcc4k.zip if folders are missing
