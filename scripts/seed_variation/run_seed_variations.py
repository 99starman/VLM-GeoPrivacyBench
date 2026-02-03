#!/usr/bin/env python3
"""
Run inference with multiple seeds for reproducibility analysis.

This script:
1. Runs inference with 3 different seeds (1, 2, 3) for each model
2. Runs for both MCQ and free-form (Vanilla Prompting) settings
3. Computes mean±std across seeds for key metrics
4. Generates a summary report
"""

import argparse
import json
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 9 API models to test
MODELS = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "o3",
    "o4-mini",
    "gpt-4o",
    "gpt-5",
    "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "gemini-2.5-flash",
    "claude-sonnet-4-20250514",
]

SEEDS = [1, 2, 3]
LABEL_ORDER = ["A", "B", "C"]


def extract_first_char_or_none(label):
    """Extract first character from label, return None if invalid."""
    if not label or not isinstance(label, str):
        return None
    first = label.strip()[0].upper()
    return first if first in LABEL_ORDER else None


def compute_mcq_metrics(pred_file: Path, gold_file: Path) -> Dict[str, float]:
    """Compute MCQ metrics (Q7 accuracy and F1)."""
    try:
        # Load predictions
        with open(pred_file, 'r') as f:
            pred_data = json.load(f)
        df_pred = pd.DataFrame(pred_data)
        df_pred["id"] = df_pred["id"].astype(str)
        
        # Load ground truth
        df_gold = pd.read_csv(gold_file)
        df_gold["id"] = df_gold["id"].astype(str)
        
        # Merge
        df = pd.merge(df_pred, df_gold, on="id", suffixes=("_pred", "_true"), how="inner")
        
        if len(df) == 0:
            logger.warning(f"No matching samples found between {pred_file} and {gold_file}")
            return {"q7_accuracy": 0.0, "q7_f1": 0.0, "n_samples": 0}
        
        # Extract Q7 labels
        y_true = df["Q7_true"].apply(extract_first_char_or_none)
        y_pred = df["Q7_pred"].apply(extract_first_char_or_none)
        
        # Filter out None values
        valid_mask = y_true.notna() & y_pred.notna()
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        if len(y_true) == 0:
            logger.warning(f"No valid Q7 labels found in {pred_file}")
            return {"q7_accuracy": 0.0, "q7_f1": 0.0, "n_samples": 0}
        
        # Compute metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)
        
        return {
            "q7_accuracy": float(acc),
            "q7_f1": float(f1),
            "n_samples": len(y_true)
        }
    except Exception as e:
        logger.error(f"Error computing MCQ metrics for {pred_file}: {e}")
        return {"q7_accuracy": 0.0, "q7_f1": 0.0, "n_samples": 0}


def compute_freeform_metrics(pred_file: Path, gold_file: Path) -> Dict[str, float]:
    """Compute free-form metrics (Q7-label accuracy and F1)."""
    try:
        # Load predictions
        with open(pred_file, 'r') as f:
            pred_data = json.load(f)
        df_pred = pd.DataFrame(pred_data)
        df_pred["id"] = df_pred["id"].astype(str)
        
        # Load ground truth
        df_gold = pd.read_csv(gold_file)
        df_gold["id"] = df_gold["id"].astype(str)
        
        # Merge
        df = pd.merge(df_pred, df_gold, on="id", suffixes=("_pred", "_true"), how="inner")
        
        if len(df) == 0:
            logger.warning(f"No matching samples found between {pred_file} and {gold_file}")
            return {"q7_label_accuracy": 0.0, "q7_label_f1": 0.0, "n_samples": 0}
        
        # Extract Q7-label (free-form granularity label)
        if "Q7-label" not in df.columns:
            logger.warning(f"Q7-label column not found in {pred_file}")
            return {"q7_label_accuracy": 0.0, "q7_label_f1": 0.0, "n_samples": 0}
        
        y_true = df["Q7"].apply(extract_first_char_or_none)  # Ground truth Q7
        y_pred = df["Q7-label"].apply(extract_first_char_or_none)  # Predicted granularity
        
        # Filter out None values
        valid_mask = y_true.notna() & y_pred.notna()
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        if len(y_true) == 0:
            logger.warning(f"No valid Q7-label found in {pred_file}")
            return {"q7_label_accuracy": 0.0, "q7_label_f1": 0.0, "n_samples": 0}
        
        # Compute metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)
        
        return {
            "q7_label_accuracy": float(acc),
            "q7_label_f1": float(f1),
            "n_samples": len(y_true)
        }
    except Exception as e:
        logger.error(f"Error computing free-form metrics for {pred_file}: {e}")
        return {"q7_label_accuracy": 0.0, "q7_label_f1": 0.0, "n_samples": 0}


def run_inference(
    model: str,
    seed: int,
    image_dir: str,
    task_name: str,
    is_free_form: bool,
    out_dir: str,
    max_examples: int = None,
    use_azure: bool = True,
) -> Path:
    """Run inference and return path to output file."""
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "src" / "api_gen.py"
    
    # Determine output directory structure
    if is_free_form:
        method_folder = "zs"  # Vanilla prompting
        dest_dir = Path(out_dir) / method_folder / "responses"
    else:
        dest_dir = Path(out_dir) / "mcq"
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        "-m", model,
        "--image-dir", image_dir,
        "--task-name", f"{task_name}_seed{seed}",
        "--prompting-method", "zs",
        "--n-threads", "2",
        "--seed", str(seed),
        "--out-dir", out_dir,
    ]
    
    if use_azure:
        cmd.append("--use-azure")
    
    if is_free_form:
        cmd.append("--free-form")
    
    if max_examples:
        cmd.extend(["--max-examples", str(max_examples)])
    
    # Claude batch processing
    if model.startswith("claude"):
        cmd.extend(["--claude-batch", "--claude-batch-size", "50"])
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Successfully ran inference for {model} seed {seed}")
        
        # Determine output file path
        mode = "api_gen_zs"
        if is_free_form:
            mode += "_free-form"
        output_file = dest_dir / f"{mode}_{model}_{task_name}_seed{seed}.json"
        
        if not output_file.exists():
            logger.warning(f"Expected output file {output_file} not found")
        
        return output_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running inference for {model} seed {seed}: {e.stderr}")
        raise


def aggregate_results(
    results_by_seed: List[Dict[str, float]]
) -> Dict[str, Tuple[float, float]]:
    """Compute mean±std from results across seeds."""
    aggregated = {}
    
    # Get all metric keys
    all_keys = set()
    for result in results_by_seed:
        all_keys.update(result.keys())
    
    for key in all_keys:
        values = [r.get(key, 0.0) for r in results_by_seed if key in r]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            aggregated[key] = (mean, std)
        else:
            aggregated[key] = (0.0, 0.0)
    
    return aggregated


def generate_report(
    mcq_results: Dict[str, Dict[str, Tuple[float, float]]],
    freeform_results: Dict[str, Dict[str, Tuple[float, float]]],
    output_file: Path
):
    """Generate a summary report."""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Seed Variation Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Models tested: {len(MODELS)}\n")
        f.write(f"Seeds used: {SEEDS}\n")
        f.write(f"Temperature: 0.7 (default)\n\n")
        
        # MCQ Results
        f.write("=" * 80 + "\n")
        f.write("MCQ Setting Results (Mean ± Std across 3 seeds)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Model':<35} {'Q7 Accuracy':<20} {'Q7 F1 (macro)':<20}\n")
        f.write("-" * 80 + "\n")
        
        for model in MODELS:
            if model in mcq_results:
                metrics = mcq_results[model]
                acc_mean, acc_std = metrics.get("q7_accuracy", (0.0, 0.0))
                f1_mean, f1_std = metrics.get("q7_f1", (0.0, 0.0))
                f.write(f"{model:<35} {acc_mean:.4f}±{acc_std:.4f}     {f1_mean:.4f}±{f1_std:.4f}\n")
            else:
                f.write(f"{model:<35} {'N/A':<20} {'N/A':<20}\n")
        
        f.write("\n")
        
        # Free-form Results
        f.write("=" * 80 + "\n")
        f.write("Free-form (Vanilla Prompting) Setting Results (Mean ± Std across 3 seeds)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Model':<35} {'Q7-label Accuracy':<20} {'Q7-label F1 (macro)':<20}\n")
        f.write("-" * 80 + "\n")
        
        for model in MODELS:
            if model in freeform_results:
                metrics = freeform_results[model]
                acc_mean, acc_std = metrics.get("q7_label_accuracy", (0.0, 0.0))
                f1_mean, f1_std = metrics.get("q7_label_f1", (0.0, 0.0))
                f.write(f"{model:<35} {acc_mean:.4f}±{acc_std:.4f}     {f1_mean:.4f}±{f1_std:.4f}\n")
            else:
                f.write(f"{model:<35} {'N/A':<20} {'N/A':<20}\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("End of Report\n")
        f.write("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with multiple seeds and compute mean±std"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/images",
        help="Directory containing input images"
    )
    parser.add_argument(
        "--gold-file",
        type=str,
        default="data/annotation_labels.csv",
        help="Path to ground truth CSV file"
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="seed_variation",
        help="Task name for output files"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="evaluation/seed_variation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to process (for testing)"
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference and only compute metrics from existing results"
    )
    parser.add_argument(
        "--use-azure",
        action="store_true",
        default=True,
        help="Use Azure OpenAI (default: True)"
    )
    parser.add_argument(
        "--no-azure",
        action="store_false",
        dest="use_azure",
        help="Don't use Azure OpenAI"
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[2]
    gold_file = project_root / args.gold_file
    
    if not gold_file.exists():
        logger.error(f"Ground truth file not found: {gold_file}")
        sys.exit(1)
    
    # Storage for results
    mcq_results_by_model = defaultdict(list)
    freeform_results_by_model = defaultdict(list)
    
    # Run inference for each model and seed
    if not args.skip_inference:
        logger.info("Starting inference runs...")
        total_runs = len(MODELS) * len(SEEDS) * 2  # 2 settings (MCQ + free-form)
        current_run = 0
        
        for model in MODELS:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing model: {model}")
            logger.info(f"{'='*80}\n")
            
            for seed in SEEDS:
                current_run += 1
                logger.info(f"\nRun {current_run}/{total_runs}: {model} seed {seed}")
                
                # MCQ setting
                try:
                    logger.info(f"  Running MCQ inference...")
                    mcq_output = run_inference(
                        model=model,
                        seed=seed,
                        image_dir=str(project_root / args.image_dir),
                        task_name=args.task_name,
                        is_free_form=False,
                        out_dir=args.out_dir,
                        max_examples=args.max_examples,
                        use_azure=args.use_azure,
                    )
                    logger.info(f"  MCQ output: {mcq_output}")
                except Exception as e:
                    logger.error(f"  Failed MCQ inference: {e}")
                    continue
                
                # Free-form setting
                try:
                    logger.info(f"  Running free-form inference...")
                    freeform_output = run_inference(
                        model=model,
                        seed=seed,
                        image_dir=str(project_root / args.image_dir),
                        task_name=args.task_name,
                        is_free_form=True,
                        out_dir=args.out_dir,
                        max_examples=args.max_examples,
                        use_azure=args.use_azure,
                    )
                    logger.info(f"  Free-form output: {freeform_output}")
                except Exception as e:
                    logger.error(f"  Failed free-form inference: {e}")
                    continue
    
    # Compute metrics for all runs
    logger.info("\n" + "="*80)
    logger.info("Computing metrics...")
    logger.info("="*80 + "\n")
    
    for model in MODELS:
        logger.info(f"Processing metrics for {model}...")
        
        # Collect MCQ metrics across seeds
        mcq_metrics_list = []
        for seed in SEEDS:
            mode = "api_gen_zs"
            # File name matches what api_gen.py creates: {mode}_{model}_{task_name}.json
            # where task_name was passed as "{args.task_name}_seed{seed}"
            output_file = (
                Path(args.out_dir) / "mcq" / 
                f"{mode}_{model}_{args.task_name}_seed{seed}.json"
            )
            
            if output_file.exists():
                metrics = compute_mcq_metrics(output_file, gold_file)
                mcq_metrics_list.append(metrics)
                logger.info(f"  Seed {seed} MCQ: Q7 Acc={metrics['q7_accuracy']:.4f}, "
                          f"F1={metrics['q7_f1']:.4f}, N={metrics['n_samples']}")
            else:
                logger.warning(f"  MCQ output file not found: {output_file}")
        
        # Collect free-form metrics across seeds
        freeform_metrics_list = []
        for seed in SEEDS:
            mode = "api_gen_zs_free-form"
            # File name matches what api_gen.py creates: {mode}_{model}_{task_name}.json
            # where task_name was passed as "{args.task_name}_seed{seed}"
            output_file = (
                Path(args.out_dir) / "zs" / "responses" /
                f"{mode}_{model}_{args.task_name}_seed{seed}.json"
            )
            
            if output_file.exists():
                metrics = compute_freeform_metrics(output_file, gold_file)
                freeform_metrics_list.append(metrics)
                logger.info(f"  Seed {seed} Free-form: Q7-label Acc={metrics['q7_label_accuracy']:.4f}, "
                          f"F1={metrics['q7_label_f1']:.4f}, N={metrics['n_samples']}")
            else:
                logger.warning(f"  Free-form output file not found: {output_file}")
        
        # Aggregate results
        if mcq_metrics_list:
            mcq_results_by_model[model] = aggregate_results(mcq_metrics_list)
        if freeform_metrics_list:
            freeform_results_by_model[model] = aggregate_results(freeform_metrics_list)
    
    # Generate report
    report_file = Path(args.out_dir) / "seed_variation_report.txt"
    logger.info(f"\nGenerating report: {report_file}")
    generate_report(mcq_results_by_model, freeform_results_by_model, report_file)
    
    logger.info("\n" + "="*80)
    logger.info("Analysis complete!")
    logger.info(f"Report saved to: {report_file}")
    logger.info("="*80)


if __name__ == "__main__":
    main()

