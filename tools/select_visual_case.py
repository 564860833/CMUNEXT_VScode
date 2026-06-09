import argparse
import csv
import os
import re


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank visualization candidates using a proposed model and one or more baselines."
    )
    parser.add_argument("--proposed_csv", required=True, help="Per-case metrics CSV for the proposed model")
    parser.add_argument(
        "--baseline_csv",
        action="append",
        required=True,
        help="Per-case metrics CSV for a baseline model; repeat for multiple baselines",
    )
    parser.add_argument("--metric", choices=["dice", "iou", "f1"], default="dice", help="Ranking metric")
    parser.add_argument("--min_proposed", type=float, default=0.80, help="Minimum proposed-model score")
    parser.add_argument("--min_baseline", type=float, default=0.30, help="Minimum best-baseline score")
    parser.add_argument("--max_baseline", type=float, default=0.75, help="Maximum best-baseline score")
    parser.add_argument("--top_k", type=int, default=20, help="Number of candidates to save; 0 saves all")
    parser.add_argument(
        "--output",
        default="candidate_visual_cases.csv",
        help="Output candidate ranking CSV",
    )
    return parser.parse_args()


def model_label(csv_path):
    parent_name = os.path.basename(os.path.dirname(os.path.abspath(csv_path)))
    label = re.sub(r"^predictions_", "", parent_name, flags=re.IGNORECASE)
    label = re.sub(r"[^A-Za-z0-9_]+", "_", label).strip("_")
    return label or "model"


def unique_labels(csv_paths):
    labels = []
    counts = {}
    for csv_path in csv_paths:
        label = model_label(csv_path)
        counts[label] = counts.get(label, 0) + 1
        labels.append(label if counts[label] == 1 else f"{label}_{counts[label]}")
    return labels


def read_metric(csv_path, metric):
    values = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"case_name", metric}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        for row in reader:
            case_name = row["case_name"].strip()
            if case_name in values:
                raise ValueError(f"{csv_path} contains duplicate case_name: {case_name}")
            values[case_name] = float(row[metric])
    if not values:
        raise ValueError(f"{csv_path} contains no metric rows.")
    return values


def rank_candidates(args):
    baseline_labels = unique_labels(args.baseline_csv)
    proposed_values = read_metric(args.proposed_csv, args.metric)
    baseline_values = [
        read_metric(csv_path, args.metric)
        for csv_path in args.baseline_csv
    ]

    shared_cases = set(proposed_values)
    for values in baseline_values:
        shared_cases.intersection_update(values)
    if not shared_cases:
        raise ValueError("The proposed model and baselines have no shared case_name values.")

    candidates = []
    for case_name in shared_cases:
        scores = [values[case_name] for values in baseline_values]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        proposed_score = proposed_values[case_name]
        best_baseline_score = scores[best_index]
        if not (
                proposed_score >= args.min_proposed
                and args.min_baseline <= best_baseline_score <= args.max_baseline):
            continue

        row = {
            "case_name": case_name,
            f"proposed_{args.metric}": round(proposed_score, 6),
            "best_baseline_model": baseline_labels[best_index],
            f"best_baseline_{args.metric}": round(best_baseline_score, 6),
            "improvement": round(proposed_score - best_baseline_score, 6),
        }
        for label, score in zip(baseline_labels, scores):
            row[f"{label}_{args.metric}"] = round(score, 6)
        candidates.append(row)

    candidates.sort(
        key=lambda row: (row["improvement"], row[f"proposed_{args.metric}"]),
        reverse=True,
    )
    if args.top_k > 0:
        candidates = candidates[:args.top_k]
    return candidates, baseline_labels, len(shared_cases)


def write_candidates(candidates, baseline_labels, metric, output_path):
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    fieldnames = [
        "rank",
        "case_name",
        f"proposed_{metric}",
        "best_baseline_model",
        f"best_baseline_{metric}",
        "improvement",
        *[f"{label}_{metric}" for label in baseline_labels],
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, candidate in enumerate(candidates, start=1):
            writer.writerow({"rank": rank, **candidate})


def main():
    args = parse_args()
    candidates, baseline_labels, shared_count = rank_candidates(args)
    write_candidates(candidates, baseline_labels, args.metric, args.output)
    print(f"Shared cases: {shared_count}")
    print(f"Eligible candidates saved: {len(candidates)}")
    print(f"Candidate ranking: {os.path.abspath(args.output)}")
    if candidates:
        print(f"Top candidate: {candidates[0]['case_name']} (improvement={candidates[0]['improvement']:.6f})")


if __name__ == "__main__":
    main()
