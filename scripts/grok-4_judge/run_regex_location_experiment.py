#!/usr/bin/env python3
"""Run regex-based location extraction + geocoding experiment on label_inspection samples."""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:
    def load_dotenv(dotenv_path):
        """Minimal dotenv loader fallback when python-dotenv is unavailable."""
        if not dotenv_path:
            return False
        path_obj = Path(dotenv_path) if not isinstance(dotenv_path, Path) else dotenv_path
        if not path_obj.exists():
            return False
        loaded = False
        for raw_line in path_obj.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
                loaded = True
        return loaded


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from location_regex import extract_location_name_regex  # noqa: E402

DEFAULT_INPUT_FILE = Path("/nethome/ryang396/flash/GeoPrivGuard/benchmark/experiments/results/all/label_inspection_gpt-4.1-mini.json")
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env_new"
LOG_DIR = PROJECT_ROOT / "evaluation" / "logs"
REPORT_DIR = PROJECT_ROOT / "evaluation" / "logs"


def configure_logging() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"regex_geocode_experiment_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Logging initialized: %s", log_path)
    return log_path


def geocode_location(location: str, api_key: str) -> Optional[Dict[str, Any]]:
    params = {"address": location, "key": api_key}
    try:
        response = requests.get("https://maps.googleapis.com/maps/api/geocode/json", params=params, timeout=30)
    except Exception as exc:
        logging.warning("Geocoding request failed for %s: %s", location, exc)
        return None
    if response.status_code != 200:
        logging.warning("Geocoding HTTP %s for %s", response.status_code, location)
        return None
    payload = response.json()
    if payload.get("status") != "OK" or not payload.get("results"):
        logging.warning("Geocoding API returned status %s for %s", payload.get('status'), location)
        return None
    best = payload["results"][0]
    loc = best.get("geometry", {}).get("location", {})
    if not loc:
        return None
    return {
        "lat": loc.get("lat"),
        "lng": loc.get("lng"),
        "formatted_address": best.get("formatted_address"),
    }


def load_samples(input_path: Path) -> List[Dict]:
    logging.info("Loading samples from %s", input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", [])
    logging.info("Loaded %d total samples", len(samples))
    return samples


def resolve_api_key(args_api_key: Optional[str], env_file: Optional[Path]) -> str:
    if args_api_key:
        return args_api_key
    if env_file and env_file.exists():
        load_dotenv(env_file)
        logging.info("Loaded environment variables from %s", env_file)
    api_key = os.getenv("GOOGLE_GEOCODING_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GOOGLE_GEOCODING_API_KEY is required for geocoding experiment")
    return api_key


def run_experiment(samples: List[Dict], api_key: str, sleep_seconds: float = 0.1, max_samples: Optional[int] = None) -> Dict:
    label_c_samples = [s for s in samples if str(s.get("Q7-label", "")).upper().startswith("C")]
    if max_samples is not None:
        label_c_samples = label_c_samples[:max_samples]
    total_label_c = len(label_c_samples)

    logging.info("Running regex extraction on %d label-C samples", total_label_c)

    extraction_success = 0
    geocode_success = 0
    extraction_failures: List[Dict] = []
    geocode_failures: List[Dict] = []
    geocode_results: List[Dict] = []

    for idx, sample in enumerate(label_c_samples, start=1):
        sample_id = str(sample.get("id"))
        q7_gen = sample.get("Q7-gen", "")
        candidate = extract_location_name_regex(q7_gen or "")

        if not candidate:
            extraction_failures.append({
                "id": sample_id,
                "reason": "no_regex_match",
                "text_preview": (q7_gen or "").strip().replace("\n", " ")[:200],
            })
            continue

        extraction_success += 1
        geocoded = geocode_location(candidate, api_key)
        if geocoded:
            geocode_success += 1
            geocode_results.append({
                "id": sample_id,
                "location": candidate,
                "lat": geocoded.get("lat"),
                "lng": geocoded.get("lng"),
                "formatted_address": geocoded.get("formatted_address"),
            })
        else:
            geocode_failures.append({
                "id": sample_id,
                "location": candidate,
            })

        if sleep_seconds:
            time.sleep(sleep_seconds)

        if idx % 25 == 0:
            logging.info(
                "Processed %d/%d samples (extraction success=%d, geocode success=%d)",
                idx,
                total_label_c,
                extraction_success,
                geocode_success,
            )

    extraction_error_rate = 1.0 - (extraction_success / total_label_c) if total_label_c else 0.0
    geocode_attempts = extraction_success
    geocode_error_rate = (
        1.0 - (geocode_success / geocode_attempts)
        if geocode_attempts
        else 0.0
    )

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total_samples": len(samples),
        "label_c_samples": total_label_c,
        "regex_extraction_success": extraction_success,
        "regex_extraction_error_rate": extraction_error_rate,
        "regex_extraction_failure_count": len(extraction_failures),
        "geocode_attempts": geocode_attempts,
        "geocode_success": geocode_success,
        "geocode_error_rate": geocode_error_rate,
        "geocode_failure_count": len(geocode_failures),
        "geocode_success_rate_over_label_c": geocode_success / total_label_c if total_label_c else 0.0,
        "sample_sleep_seconds": sleep_seconds,
        "max_samples": max_samples,
        "extraction_failures": extraction_failures,
        "geocode_failures": geocode_failures,
        "geocode_results": geocode_results,
    }

    return report


def save_report(report: Dict, output_path: Path) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logging.info("Saved experiment report to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regex-based location extraction experiment")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE, help="Path to label_inspection JSON file")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE, help="Path to .env file with API keys")
    parser.add_argument("--api-key", type=str, default=None, help="Override Google Geocoding API key")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional explicit output path for report")
    parser.add_argument("--sleep-seconds", type=float, default=0.1, help="Delay between geocode requests to avoid rate limits")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of label-C samples to process")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = configure_logging()
    try:
        api_key = resolve_api_key(args.api_key, args.env_file)
    except Exception as exc:
        logging.error("Failed to resolve API key: %s", exc)
        sys.exit(1)

    try:
        samples = load_samples(args.input_file)
    except Exception as exc:
        logging.error("Failed to load input file: %s", exc)
        sys.exit(1)

    report = run_experiment(samples, api_key, sleep_seconds=args.sleep_seconds, max_samples=args.max_samples)

    if args.output_json:
        output_path = args.output_json
    else:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = REPORT_DIR / f"regex_geocode_experiment_{timestamp}.json"

    save_report(report, output_path)

    extraction_rate = 1.0 - report["regex_extraction_error_rate"]
    geocode_rate = 1.0 - report["geocode_error_rate"] if report["geocode_attempts"] else 0.0

    print("\n=== Regex Location Extraction Experiment Summary ===")
    print(f"Input file:       {args.input_file}")
    print(f"Log file:         {log_path}")
    print(f"Report file:      {output_path}")
    print(f"Label-C samples:  {report['label_c_samples']} / {report['total_samples']} total")
    print(f"Extraction succ.: {report['regex_extraction_success']} ({extraction_rate:.2%} success, {report['regex_extraction_error_rate']:.2%} error)")
    print(f"Geocode attempts: {report['geocode_attempts']}")
    print(f"Geocode succ.:    {report['geocode_success']} ({geocode_rate:.2%} success, {report['geocode_error_rate']:.2%} error)")
    print(f"Overall success over label-C: {report['geocode_success_rate_over_label_c']:.2%}")


if __name__ == "__main__":
    main()
