#!/usr/bin/env python3
"""
Combine temperature=0 safety evaluation results:
- Original 400-sample results (already evaluated)
- New 800-sample results (full1200 run, excluding the 400)
- Output: Combined 1200-sample results
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d-%(name)s-%(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


def load_json_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSON file and return list of entries."""
    if not file_path.exists():
        logging.warning(f"File not found: {file_path}")
        return []
    with open(file_path, "r") as f:
        data = json.load(f)
        if not isinstance(data, list):
            logging.error(f"Expected list, got {type(data)}: {file_path}")
            return []
        return data


def save_json_file(file_path: Path, data: List[Dict[str, Any]]):
    """Save data to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved {len(data)} entries to {file_path}")


def combine_results(
    original_file: Path,
    new_file: Path,
    output_file: Path,
):
    """Combine original and new results, ensuring no duplicates."""
    logging.info(f"Loading original results: {original_file}")
    original_data = load_json_file(original_file)
    
    logging.info(f"Loading new results: {new_file}")
    new_data = load_json_file(new_file)
    
    if not original_data and not new_data:
        logging.error("Both files are empty or not found")
        return
    
    # Create a set of IDs from original data to avoid duplicates
    original_ids = {entry.get("id") for entry in original_data if entry.get("id")}
    
    # Combine data, prioritizing original (already-evaluated) entries
    combined_data = original_data.copy()
    
    # Add new entries that aren't in original
    added_count = 0
    for entry in new_data:
        entry_id = entry.get("id")
        if entry_id and entry_id not in original_ids:
            combined_data.append(entry)
            added_count += 1
        elif entry_id in original_ids:
            logging.debug(f"Skipping duplicate ID: {entry_id}")
    
    logging.info(f"Combined: {len(original_data)} original + {added_count} new = {len(combined_data)} total")
    
    # Sort by ID for consistency
    combined_data.sort(key=lambda x: x.get("id", ""))
    
    # Save combined results
    save_json_file(output_file, combined_data)
    
    return len(combined_data)


def main():
    parser = argparse.ArgumentParser(
        description="Combine temperature=0 safety evaluation results from original 400 samples and new 800 samples"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="evaluation/temp0_safety",
        help="Base directory for temp0_safety results",
    )
    parser.add_argument(
        "--sample-tag",
        type=str,
        default="sample_20251117-111925",
        help="Sample directory tag (e.g., sample_20251117-111925)",
    )
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    sample_tag = args.sample_tag
    
    models = ["gemini-2.5-flash", "o4-mini", "gpt-4.1-mini"]
    methods = ["zs", "iter-cot", "malicious"]
    
    total_combined = 0
    
    for model in models:
        for method in methods:
            # Original 400-sample file (already evaluated)
            original_file = (
                base_dir / method / "responses" / 
                f"api_gen_{method}_free-form_{model}_temp0_safety_{method}_{sample_tag}.json"
            )
            
            # New 800-sample file (full1200 run)
            new_file = (
                base_dir / method / "responses" / 
                f"api_gen_{method}_free-form_{model}_temp0_safety_{method}_full1200_{sample_tag}.json"
            )
            
            # Output combined file
            output_file = (
                base_dir / method / "responses" / 
                f"api_gen_{method}_free-form_{model}_temp0_safety_{method}_combined1200_{sample_tag}.json"
            )
            
            print(f"\n{'='*60}")
            print(f"Model: {model} | Method: {method}")
            print(f"{'='*60}")
            
            if not original_file.exists() and not new_file.exists():
                logging.warning(f"Neither file exists for {model}/{method}, skipping")
                continue
            
            count = combine_results(original_file, new_file, output_file)
            if count > 0:
                total_combined += count
                print(f"[OK] Combined {count} entries -> {output_file.name}")
            else:
                print(f"[FAIL] Failed to combine results")
    
    print(f"\n{'='*60}")
    print(f"Combination complete!")
    print(f"Total entries combined: {total_combined}")
    print(f"Expected: {len(models) * len(methods) * 1200} (if all successful)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

