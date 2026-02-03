#!/usr/bin/env python3
"""
Re-run granularity label mapping using grok-4-fast-reasoning on label_inspection samples.
Processes the 640 samples from label_inspection_gpt-4.1-mini.json.
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


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load JSON file and return data."""
    with open(file_path, "r") as f:
        return json.load(f)


def save_json_file(file_path: Path, data: Dict[str, Any]):
    """Save data to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved {len(data.get('samples', []))} entries to {file_path}")


def rerun_granularity_mapping(
    input_file: Path,
    output_file: Path,
    azure_endpoint: str,
    azure_api_key: str,
    judge_model: str = "grok-4-fast-reasoning",
):
    """Re-run granularity mapping on label_inspection samples."""
    logging.info(f"Loading input file: {input_file}")
    data = load_json_file(input_file)
    
    samples = data.get("samples", [])
    if not samples:
        logging.warning(f"Input file {input_file} has no samples")
        return
    
    logging.info(f"Processing {len(samples)} samples with {judge_model} for granularity mapping")
    
    # Create Azure OpenAI client for granularity mapping
    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version="2025-01-01-preview",
    )
    
    # Process each sample
    updated_samples = []
    for i, sample in enumerate(samples):
        sample_id = sample.get("id", f"unknown_{i}")
        q7_gen = sample.get("Q7-gen", "")
        
        if not q7_gen:
            logging.warning(f"Sample {sample_id} has no Q7-gen text, skipping")
            updated_samples.append(sample)
            continue
        
        # Re-extract granularity with grok-4-fast-reasoning
        try:
            new_label = extract_granularity(
                q7_gen,
                api_key=azure_api_key,
                api_endpoint=azure_endpoint,
                client=client,
                model_name=judge_model,
            )
            # Update Q7-label with grok-4 result
            sample["Q7-label"] = new_label
            # Add metadata to track which judge model was used
            sample["judge_model"] = judge_model
            logging.debug(f"Sample {sample_id}: mapped to {new_label}")
        except Exception as e:
            logging.error(f"Failed to extract granularity for sample {sample_id}: {e}")
            # Keep original label if extraction fails
            if "Q7-label" not in sample:
                sample["Q7-label"] = "N/A"
        
        updated_samples.append(sample)
        
        if (i + 1) % 50 == 0:
            logging.info(f"Processed {i + 1}/{len(samples)} samples")
    
    # Update data structure
    data["samples"] = updated_samples
    data["judge_model"] = judge_model
    
    # Save updated results
    save_json_file(output_file, data)
    logging.info(f"Completed: {len(updated_samples)} samples processed")


def main():
    parser = argparse.ArgumentParser(
        description="Re-run granularity label mapping using grok-4-fast-reasoning on label_inspection samples"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to input label_inspection JSON file (e.g., label_inspection_gpt-4.1-mini.json)",
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
        help="Model to use for granularity mapping (default: grok-4-fast-reasoning)",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="Path to .env file (default: .env)",
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

