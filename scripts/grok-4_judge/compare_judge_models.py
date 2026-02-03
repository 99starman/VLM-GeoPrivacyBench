#!/usr/bin/env python3
"""
Compare Q7-label agreement between two judge models.
Calculates percentage agreement between gpt-4.1-mini and grok-4-fast-reasoning labels.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def load_json_file(file_path: Path) -> Dict:
    """Load JSON file and return data."""
    with open(file_path, "r") as f:
        return json.load(f)


def calculate_agreement(
    file1: Path,
    file2: Path,
    model1_name: str = "gpt-4.1-mini",
    model2_name: str = "grok-4-fast-reasoning",
):
    """Calculate label agreement between two judge models."""
    print(f"Loading {model1_name} labels from: {file1}")
    data1 = load_json_file(file1)
    samples1 = {s["id"]: s for s in data1.get("samples", [])}
    
    print(f"Loading {model2_name} labels from: {file2}")
    data2 = load_json_file(file2)
    samples2 = {s["id"]: s for s in data2.get("samples", [])}
    
    # Find common IDs
    common_ids = set(samples1.keys()) & set(samples2.keys())
    print(f"\nFound {len(common_ids)} common samples")
    print(f"  - {model1_name}: {len(samples1)} samples")
    print(f"  - {model2_name}: {len(samples2)} samples")
    
    if not common_ids:
        print("ERROR: No common samples found!")
        return
    
    # Compare labels
    agreements = 0
    disagreements = 0
    label_pairs = []
    disagreement_details = []
    
    for sample_id in sorted(common_ids):
        label1 = samples1[sample_id].get("Q7-label", "N/A")
        label2 = samples2[sample_id].get("Q7-label", "N/A")
        
        label_pairs.append((label1, label2))
        
        if label1 == label2:
            agreements += 1
        else:
            disagreements += 1
            disagreement_details.append({
                "id": sample_id,
                f"{model1_name}": label1,
                f"{model2_name}": label2,
                "Q7-gen": samples1[sample_id].get("Q7-gen", "")[:100] + "..." if len(samples1[sample_id].get("Q7-gen", "")) > 100 else samples1[sample_id].get("Q7-gen", ""),
            })
    
    total = agreements + disagreements
    agreement_pct = (agreements / total * 100) if total > 0 else 0
    
    # Count label distributions
    label1_dist = Counter([p[0] for p in label_pairs])
    label2_dist = Counter([p[1] for p in label_pairs])
    
    # Count agreement by label
    agreement_by_label = Counter()
    disagreement_by_label1 = Counter()
    disagreement_by_label2 = Counter()
    
    for label1, label2 in label_pairs:
        if label1 == label2:
            agreement_by_label[label1] += 1
        else:
            disagreement_by_label1[label1] += 1
            disagreement_by_label2[label2] += 1
    
    # Print results
    print("\n" + "="*60)
    print("LABEL AGREEMENT ANALYSIS")
    print("="*60)
    print(f"\nTotal samples compared: {total}")
    print(f"Agreements: {agreements} ({agreement_pct:.2f}%)")
    print(f"Disagreements: {disagreements} ({100 - agreement_pct:.2f}%)")
    
    print(f"\n{model1_name} label distribution:")
    for label, count in sorted(label1_dist.items()):
        pct = count / total * 100
        print(f"  {label}: {count} ({pct:.2f}%)")
    
    print(f"\n{model2_name} label distribution:")
    for label, count in sorted(label2_dist.items()):
        pct = count / total * 100
        print(f"  {label}: {count} ({pct:.2f}%)")
    
    print(f"\nAgreement by label (when both models agree):")
    for label, count in sorted(agreement_by_label.items()):
        pct = count / agreements * 100 if agreements > 0 else 0
        print(f"  {label}: {count} ({pct:.2f}% of agreements)")
    
    print(f"\n{model1_name} labels in disagreements:")
    for label, count in sorted(disagreement_by_label1.items()):
        pct = count / disagreements * 100 if disagreements > 0 else 0
        print(f"  {label}: {count} ({pct:.2f}% of disagreements)")
    
    print(f"\n{model2_name} labels in disagreements:")
    for label, count in sorted(disagreement_by_label2.items()):
        pct = count / disagreements * 100 if disagreements > 0 else 0
        print(f"  {label}: {count} ({pct:.2f}% of disagreements)")
    
    # Show disagreement transition matrix
    print(f"\nDisagreement transition matrix ({model1_name} -> {model2_name}):")
    transition_matrix = Counter()
    for label1, label2 in label_pairs:
        if label1 != label2:
            transition_matrix[(label1, label2)] += 1
    
    if transition_matrix:
        print("  From -> To:")
        for (label1, label2), count in sorted(transition_matrix.items()):
            print(f"    {label1} -> {label2}: {count}")
    
    # Save detailed disagreement report
    if disagreement_details:
        output_file = file1.parent / f"disagreement_report_{model1_name}_vs_{model2_name}.json"
        with open(output_file, "w") as f:
            json.dump({
                "summary": {
                    "total_samples": total,
                    "agreements": agreements,
                    "disagreements": disagreements,
                    "agreement_percentage": agreement_pct,
                },
                "disagreements": disagreement_details[:50],  # First 50 for brevity
            }, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed disagreement report (first 50) saved to: {output_file}")
    
    print("\n" + "="*60)
    print(f"FINAL RESULT: {agreement_pct:.2f}% agreement")
    print("="*60)
    
    return agreement_pct


def main():
    parser = argparse.ArgumentParser(
        description="Compare Q7-label agreement between two judge models"
    )
    parser.add_argument(
        "--file1",
        type=str,
        required=True,
        help="Path to first JSON file (e.g., label_inspection_gpt-4.1-mini.json)",
    )
    parser.add_argument(
        "--file2",
        type=str,
        required=True,
        help="Path to second JSON file (e.g., label_inspection_grok4-fast-reasoning.json)",
    )
    parser.add_argument(
        "--model1-name",
        type=str,
        default="gpt-4.1-mini",
        help="Name of first judge model (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--model2-name",
        type=str,
        default="grok-4-fast-reasoning",
        help="Name of second judge model (default: grok-4-fast-reasoning)",
    )
    args = parser.parse_args()
    
    file1 = Path(args.file1)
    file2 = Path(args.file2)
    
    if not file1.exists():
        raise ValueError(f"File not found: {file1}")
    if not file2.exists():
        raise ValueError(f"File not found: {file2}")
    
    calculate_agreement(
        file1,
        file2,
        args.model1_name,
        args.model2_name,
    )


if __name__ == "__main__":
    main()

