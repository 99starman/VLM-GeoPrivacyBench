#!/usr/bin/env python3
"""
Compute mean±std metrics from seed variation results.

This script reads the JSON output files from gen_api_seed_variations.sh
and computes mean±std across seeds for each model.
"""

import argparse
import json
import logging
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

# 8 API models (Claude excluded - does not support seed parameter)
# Order: Gemini 2.5 Flash, GPT-5, o3, o4-mini, GPT-4.1, GPT-4.1-mini, GPT-4o, Llama-4-Maverick
MODELS = [
    "gemini-2.5-flash",
    "gpt-5",
    "o3",
    "o4-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "Llama-4-Maverick-17B-128E-Instruct-FP8",
    # "claude-sonnet-4-20250514",  # Skipped: Claude API does not support seed parameter
]

DEFAULT_SEEDS = [1, 2, 3]
LABEL_ORDER = ["A", "B", "C"]
LABEL_TO_INT = {lbl: idx for idx, lbl in enumerate(LABEL_ORDER)}


def extract_first_char_or_none(label):
    """Extract first character from label, return None if invalid."""
    if not label or not isinstance(label, str):
        return None
    first = label.strip()[0].upper()
    return first if first in LABEL_ORDER else None


def directionality_stats(y_true, y_pred):
    """Compute over/under disclosure rates.
    
    Returns:
        over_disclosure_rate: fraction of predictions that are more specific than ground truth
        under_disclosure_rate: fraction of predictions that are more conservative than ground truth
    """
    y_true_int = [LABEL_TO_INT.get(extract_first_char_or_none(y), None) for y in y_true]
    y_pred_int = [LABEL_TO_INT.get(extract_first_char_or_none(y), None) for y in y_pred]
    
    idx_valid = [i for i, (t, p) in enumerate(zip(y_true_int, y_pred_int)) if t is not None and p is not None]
    
    if not idx_valid:
        return 0.0, 0.0
    
    errors = [y_pred_int[i] - y_true_int[i] for i in idx_valid]
    n = len(errors)
    over_disclosure_rate = sum(e > 0 for e in errors) / n  # More specific (e.g., C when truth is B)
    under_disclosure_rate = sum(e < 0 for e in errors) / n  # More conservative (e.g., A when truth is B)
    
    return over_disclosure_rate, under_disclosure_rate


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
            return {"q7_label_accuracy": 0.0, "q7_label_f1": 0.0, "over_disclosure_rate": 0.0, "under_disclosure_rate": 0.0, "n_samples": 0}
        
        # Extract Q7 labels
        y_true = df["Q7_true"].apply(extract_first_char_or_none)
        y_pred = df["Q7_pred"].apply(extract_first_char_or_none)
        
        # Filter out None values
        valid_mask = y_true.notna() & y_pred.notna()
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        if len(y_true) == 0:
            logger.warning(f"No valid Q7 labels found in {pred_file}")
            return {"q7_accuracy": 0.0, "q7_f1": 0.0, "over_disclosure_rate": 0.0, "under_disclosure_rate": 0.0, "n_samples": 0}
        
        # Compute metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)
        over_disclosure, under_disclosure = directionality_stats(y_true, y_pred)
        
        return {
            "q7_accuracy": float(acc),
            "q7_f1": float(f1),
            "over_disclosure_rate": float(over_disclosure),
            "under_disclosure_rate": float(under_disclosure),
            "n_samples": len(y_true)
        }
    except Exception as e:
        logger.error(f"Error computing MCQ metrics for {pred_file}: {e}")
        return {"q7_accuracy": 0.0, "q7_f1": 0.0, "over_disclosure_rate": 0.0, "under_disclosure_rate": 0.0, "n_samples": 0}


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
            return {"q7_label_accuracy": 0.0, "q7_label_f1": 0.0, "over_disclosure_rate": 0.0, "under_disclosure_rate": 0.0, "n_samples": 0}
        
        # Extract Q7-label (free-form granularity label)
        if "Q7-label" not in df.columns:
            logger.warning(f"Q7-label column not found in {pred_file}")
            return {"q7_label_accuracy": 0.0, "q7_label_f1": 0.0, "over_disclosure_rate": 0.0, "under_disclosure_rate": 0.0, "n_samples": 0}
        
        y_true = df["Q7"].apply(extract_first_char_or_none)  # Ground truth Q7
        y_pred = df["Q7-label"].apply(extract_first_char_or_none)  # Predicted granularity
        
        # Filter out None values
        valid_mask = y_true.notna() & y_pred.notna()
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        if len(y_true) == 0:
            logger.warning(f"No valid Q7-label found in {pred_file}")
            return {"q7_label_accuracy": 0.0, "q7_label_f1": 0.0, "over_disclosure_rate": 0.0, "under_disclosure_rate": 0.0, "n_samples": 0}
        
        # Compute metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)
        over_disclosure, under_disclosure = directionality_stats(y_true, y_pred)
        
        return {
            "q7_label_accuracy": float(acc),
            "q7_label_f1": float(f1),
            "over_disclosure_rate": float(over_disclosure),
            "under_disclosure_rate": float(under_disclosure),
            "n_samples": len(y_true)
        }
    except Exception as e:
        logger.error(f"Error computing free-form metrics for {pred_file}: {e}")
        return {"q7_label_accuracy": 0.0, "q7_label_f1": 0.0, "over_disclosure_rate": 0.0, "under_disclosure_rate": 0.0, "n_samples": 0}


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
    output_file: Path,
    seeds: List[int],
    temperature: float = 0.7,
) -> None:
    """Generate a summary report."""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Seed Variation Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Models tested: {len(MODELS)}\n")
        f.write(f"Seeds used: {seeds}\n")
        f.write(f"Temperature: {temperature} (default)\n\n")
        
        # MCQ Results
        f.write("=" * 80 + "\n")
        f.write(f"MCQ Setting Results (Mean ± Std across {len(seeds)} seeds)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Model':<35} {'Q7 Accuracy':<20} {'Q7 F1 (macro)':<20} {'Over-Disclosure (%)':<20} {'Under-Disclosure (%)':<20}\n")
        f.write("-" * 115 + "\n")
        
        for model in MODELS:
            if model in mcq_results:
                metrics = mcq_results[model]
                acc_mean, acc_std = metrics.get("q7_accuracy", (0.0, 0.0))
                f1_mean, f1_std = metrics.get("q7_f1", (0.0, 0.0))
                over_mean, over_std = metrics.get("over_disclosure_rate", (0.0, 0.0))
                under_mean, under_std = metrics.get("under_disclosure_rate", (0.0, 0.0))
                # Convert to percentages
                over_mean_pct = over_mean * 100
                over_std_pct = over_std * 100
                under_mean_pct = under_mean * 100
                under_std_pct = under_std * 100
                f.write(f"{model:<35} {acc_mean:.4f}±{acc_std:.4f}     {f1_mean:.4f}±{f1_std:.4f}     {over_mean_pct:.2f}±{over_std_pct:.2f}%     {under_mean_pct:.2f}±{under_std_pct:.2f}%\n")
            else:
                f.write(f"{model:<35} {'N/A':<20} {'N/A':<20} {'N/A':<20} {'N/A':<20}\n")
        
        f.write("\n")
        
        # Free-form Results
        f.write("=" * 80 + "\n")
        f.write(f"Free-form (Vanilla Prompting) Setting Results (Mean ± Std across {len(seeds)} seeds)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Model':<35} {'Q7-label Accuracy':<20} {'Q7-label F1 (macro)':<20} {'Over-Disclosure (%)':<20} {'Under-Disclosure (%)':<20}\n")
        f.write("-" * 115 + "\n")
        
        for model in MODELS:
            if model in freeform_results:
                metrics = freeform_results[model]
                acc_mean, acc_std = metrics.get("q7_label_accuracy", (0.0, 0.0))
                f1_mean, f1_std = metrics.get("q7_label_f1", (0.0, 0.0))
                over_mean, over_std = metrics.get("over_disclosure_rate", (0.0, 0.0))
                under_mean, under_std = metrics.get("under_disclosure_rate", (0.0, 0.0))
                # Convert to percentages
                over_mean_pct = over_mean * 100
                over_std_pct = over_std * 100
                under_mean_pct = under_mean * 100
                under_std_pct = under_std * 100
                f.write(f"{model:<35} {acc_mean:.4f}±{acc_std:.4f}     {f1_mean:.4f}±{f1_std:.4f}     {over_mean_pct:.2f}±{over_std_pct:.2f}%     {under_mean_pct:.2f}±{under_std_pct:.2f}%\n")
            else:
                f.write(f"{model:<35} {'N/A':<20} {'N/A':<20} {'N/A':<20} {'N/A':<20}\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("End of Report\n")
        f.write("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean±std metrics from seed variation results"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="evaluation/seed_variation",
        help="Output directory where results are stored"
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="seed_variation",
        help="Task name used in output files"
    )
    parser.add_argument(
        "--gold-file",
        type=str,
        default="data/annotation_labels.csv",
        help="Path to ground truth CSV file"
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default=None,
        help="Path to output report file (default: {out_dir}/seed_variation_report.txt)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="List of seeds to aggregate (default: 1 2 3)"
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir)
    gold_file = project_root / args.gold_file
    seeds = args.seeds
    
    if not gold_file.exists():
        logger.error(f"Ground truth file not found: {gold_file}")
        return 1
    
    # Storage for results
    mcq_results_by_model = {}
    freeform_results_by_model = {}
    
    # Compute metrics for all runs
    logger.info("Computing metrics from seed variation results...")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Task name: {args.task_name}")
    logger.info("")
    
    for model in MODELS:
        logger.info(f"Processing metrics for {model}...")
        
        # Collect MCQ metrics across seeds
        mcq_metrics_list = []
        for seed in seeds:
            mode = "api_gen_zs_heuristics"
            output_file = (
                out_dir / "mcq" / 
                f"{mode}_{model}_{args.task_name}_seed{seed}.json"
            )
            
            if output_file.exists():
                metrics = compute_mcq_metrics(output_file, gold_file)
                mcq_metrics_list.append(metrics)
                logger.info(f"  Seed {seed} MCQ: Q7 Acc={metrics['q7_accuracy']:.4f}, "
                          f"F1={metrics['q7_f1']:.4f}, Over={metrics['over_disclosure_rate']:.4f}, "
                          f"Under={metrics['under_disclosure_rate']:.4f}, N={metrics['n_samples']}")
            else:
                logger.warning(f"  MCQ output file not found: {output_file}")
        
        # Collect free-form metrics across seeds
        freeform_metrics_list = []
        for seed in seeds:
            mode = "api_gen_zs_free-form"
            output_file = (
                out_dir / "zs" / "responses" /
                f"{mode}_{model}_{args.task_name}_seed{seed}.json"
            )
            
            if output_file.exists():
                metrics = compute_freeform_metrics(output_file, gold_file)
                freeform_metrics_list.append(metrics)
                logger.info(f"  Seed {seed} Free-form: Q7-label Acc={metrics['q7_label_accuracy']:.4f}, "
                          f"F1={metrics['q7_label_f1']:.4f}, Over={metrics['over_disclosure_rate']:.4f}, "
                          f"Under={metrics['under_disclosure_rate']:.4f}, N={metrics['n_samples']}")
            else:
                logger.warning(f"  Free-form output file not found: {output_file}")
        
        # Aggregate results
        if mcq_metrics_list:
            mcq_results_by_model[model] = aggregate_results(mcq_metrics_list)
        if freeform_metrics_list:
            freeform_results_by_model[model] = aggregate_results(freeform_metrics_list)
    
    # Generate report
    if args.report_file:
        report_file = Path(args.report_file)
    else:
        report_file = out_dir / "seed_variation_report.txt"
    
    logger.info(f"\nGenerating report: {report_file}")
    generate_report(mcq_results_by_model, freeform_results_by_model, report_file, seeds, temperature=0.7)
    
    logger.info("\n" + "="*80)
    logger.info("Analysis complete!")
    logger.info(f"Report saved to: {report_file}")
    logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    exit(main())

