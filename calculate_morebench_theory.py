import argparse

from utils import (calculate_all_metrics, calculate_task_scores,
                   format_results_row, group_criteria_by_task,
                   load_judgement_data, load_valid_task_ids)

parser = argparse.ArgumentParser(description='Calculate MoReBench-Theory scores from judgement data')
parser.add_argument("--input_file", "-i", required=True, help="Path to judgement JSONL file")
parser.add_argument("--format", "-f", default="human", choices=["latex", "human"], 
                    help="Output format: latex (table row) or human (readable)")
parser.add_argument("--expected_samples", "-es", type=int, default=1547,
                    help="Expected number of judgement entries (default: 1547)")
parser.add_argument("--pipeline_results", "-p", required=False, default=None,
                    help="Path to pipeline_results.jsonl to filter valid tasks only")

args = parser.parse_args()

# Load and validate data
data = load_judgement_data(args.input_file)
assert len(data) == args.expected_samples, \
    f"Expected {args.expected_samples} entries, got {len(data)}"

# Group criteria by task
task_id_to_criteria = group_criteria_by_task(data)

# Calculate scores
task_id_to_score = calculate_task_scores(task_id_to_criteria)

# If pipeline results provided, filter to valid tasks only
if args.pipeline_results:
    valid_task_ids = load_valid_task_ids(args.pipeline_results)
    task_id_to_score = {k: v for k, v in task_id_to_score.items() if str(k) in valid_task_ids}
    task_id_to_criteria = {k: v for k, v in task_id_to_criteria.items() if str(k) in valid_task_ids}
    data = [dp for dp in data if str(dp["task_id"]) in valid_task_ids]

# Define analysis categories (different from main benchmark)
TASK_CATEGORIES = [None, "theory", "dilemma_source"]
CRITERION_CATEGORIES = ["criterion_dimension", "criterion_weight"]
TOKEN_FIELDS = ["input_tokens", "output_tokens", "len"]

# Calculate all metrics
all_results = calculate_all_metrics(
    data=data,
    task_id_to_criteria=task_id_to_criteria,
    task_id_to_score=task_id_to_score,
    task_categories=TASK_CATEGORIES,
    criterion_categories=CRITERION_CATEGORIES,
    token_fields=TOKEN_FIELDS,
    human_readable=(args.format == "human")
)

# Output formatted results (different fields from main benchmark)
RESULT_FIELDS = [
    'Gauthierian Contractarianism',
    'Scanlonian Contractualism',
    'Act Utilitarianism',
    'Aristotelian Virtue Ethics',
    'Kantian Deontology',
    "overall",
    "len"
]

if args.format == "latex":
    print(format_results_row(all_results, RESULT_FIELDS))
else:
    # Human-readable format already printed by calculate_all_metrics
    print("\n=== Summary ===")
    print(f"Overall Score: {all_results['overall']}")
    print(f"Average Response Length: {all_results['len']} chars")
    if 'overall' in all_results and 'len' in all_results:
        normalized = round(all_results["overall"] / all_results["len"] * 1000, 1)
        print(f"Normalized Score (per 1k chars): {normalized}")
