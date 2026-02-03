#!/usr/bin/env python3
"""
Re-run granularity label mapping using grok-4 on existing free-form results.
Reads existing JSON files with Q7-gen text and re-maps granularity labels using grok-4.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root and src directory to Python path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
from openai import AzureOpenAI

from src.utils import extract_granularity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d-%(name)s-%(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


def load_json_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSON file and return list of entries."""
    with open(file_path, "r") as f:
        return json.load(f)


def save_json_file(file_path: Path, data: List[Dict[str, Any]]):
    """Save data to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved {len(data)} entries to {file_path}")


def rerun_granularity_mapping(
    input_file: Path,
    output_file: Path,
    azure_endpoint: str,
    azure_api_key: str,
    judge_model: str = "grok-4-fast-reasoning",
):
    """Re-run granularity mapping on existing free-form results."""
    logging.info(f"Loading input file: {input_file}")
    data = load_json_file(input_file)
    
    if not data:
        logging.warning(f"Input file {input_file} is empty or invalid")
        return
    
    logging.info(f"Processing {len(data)} entries with {judge_model} for granularity mapping")
    
    # Create Azure OpenAI client for granularity mapping
    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version="2025-01-01-preview",
    )
    
    # Process each entry
    updated_data = []
    for i, entry in enumerate(data):
        entry_id = entry.get("id", f"unknown_{i}")
        q7_gen = entry.get("Q7-gen", "")
        
        if not q7_gen:
            logging.warning(f"Entry {entry_id} has no Q7-gen text, skipping")
            updated_data.append(entry)
            continue
        
        # Re-extract granularity with grok-4
        try:
            new_label = extract_granularity(
                q7_gen,
                api_key=azure_api_key,
                api_endpoint=azure_endpoint,
                client=client,
                model_name=judge_model,
            )
            entry["Q7-label"] = new_label
            logging.debug(f"Entry {entry_id}: mapped to {new_label}")
        except Exception as e:
            logging.error(f"Failed to extract granularity for entry {entry_id}: {e}")
            # Keep original label if extraction fails
            if "Q7-label" not in entry:
                entry["Q7-label"] = "N/A"
        
        updated_data.append(entry)
        
        if (i + 1) % 50 == 0:
            logging.info(f"Processed {i + 1}/{len(data)} entries")
    
    # Save updated results
    save_json_file(output_file, updated_data)
    logging.info(f"Completed: {len(updated_data)} entries processed")


def main():
    parser = argparse.ArgumentParser(
        description="Re-run granularity label mapping using grok-4 on existing free-form results"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to input JSON file with free-form results (must have Q7-gen field)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to output JSON file with updated Q7-label mappings",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="grok-4-fast-reasoning",
        help="Model to use for granularity mapping",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="Path to .env file",
    )
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv(args.env_file, override=True)
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_API_KEY")
    
    if not azure_endpoint or not azure_api_key:
        raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_API_KEY must be set in .env file")
    
    input_file = Path(args.input_file)
    if not input_file.exists():
        raise ValueError(f"Input file not found: {input_file}")
    
    output_file = Path(args.output_file)
    
    rerun_granularity_mapping(
        input_file,
        output_file,
        azure_endpoint,
        azure_api_key,
        args.judge_model,
    )


if __name__ == "__main__":
    main()

