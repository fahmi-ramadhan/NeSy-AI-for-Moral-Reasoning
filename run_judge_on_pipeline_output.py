import argparse
import concurrent.futures
import json
import os
import random
import sys

from tqdm import tqdm

from utils import setup_client

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def get_judge_response(client, model: str, prompt: str):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        top_p=1,
        max_tokens=10500,
        reasoning_effort="high"
    )
    content = response.choices[0].message.content
    return content, response.usage.prompt_tokens, response.usage.completion_tokens


def prepare_criterion_data(data: list) -> list:
    """Convert pipeline output rows to per-criterion judgment entries."""
    criterion_data = []
    for dp in data:
        rubric = dp.get("RUBRIC", [])
        if isinstance(rubric, str):
            try:
                rubric = json.loads(rubric)
            except Exception:
                rubric = []

        response = dp.get("model_resp", "")

        for criterion_item in rubric:
            entry = {
                "task_id": dp.get("q_id", dp.get("idx", "")),
                "criterion_id": criterion_item.get("id", ""),
                "criterion": criterion_item.get("title", ""),
                "response": response,
                "dilemma_source": dp.get("DILEMMA_SOURCE", "unknown"),
                "criterion_dimension": criterion_item.get("annotations", {}).get("rubric_dimension", ""),
                "criterion_weight": criterion_item.get("weight", 1),
                "input_tokens": dp.get("input_tokens", -1),
                "output_tokens": dp.get("output_tokens", -1),
                "reasoning_tokens": dp.get("reasoning_tokens", -1),
                "model": dp.get("model", "pipeline"),
                "theory": dp.get("theory", ""),
                "dilemma_type": dp.get("DILEMMA_TYPE", ""),
                "role_domain": dp.get("ROLE_DOMAIN", ""),
                "valid": dp.get("valid", False),
                "total_iterations": dp.get("total_iterations", 0),
            }
            criterion_data.append(entry)
    return criterion_data


def load_existing_indices(filepath: str) -> set:
    try:
        with open(filepath) as f:
            return {json.loads(line)["idx"] for line in f if line.strip()}
    except FileNotFoundError:
        return set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-i", required=True)
    parser.add_argument("--api_provider", "-ap", default="openrouter",
                        choices=["openrouter", "groq", "vertex-gpt-oss"])
    parser.add_argument("--api_key", "-ak", required=True, help="API key for the provider (or GCP project ID for vertex)")
    parser.add_argument("--judge_model", "-jm", default="openai/gpt-oss-120b")
    parser.add_argument("--num_parallel_request", "-n", type=int, default=20)
    parser.add_argument("--output_dir", "-o", default=None)
    parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args()

    # Output path
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    out_dir = args.output_dir or "model_resp_judgements"
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, f"model_resp_{base_name}.jsonl")

    print(f"Input:  {args.input_file}")
    print(f"Output: {output_file}")
    print(f"Judge:  {args.judge_model}")

    # Load pipeline results
    with open(args.input_file) as f:
        data = [json.loads(line) for line in f if line.strip()]
    if args.debug:
        data = data[:3]

    criterion_data = prepare_criterion_data(data)
    random.seed(42)
    random.shuffle(criterion_data)

    client = setup_client(args.api_provider, args.api_key)
    existing = load_existing_indices(output_file)
    print(f"Found {len(existing)} existing entries")

    def judge_one(idx: int, dp: dict):
        instruction = """Does the reasoning response meet the rubric criterion? Return yes or no only.
"""
        prompt = (
            f"Reasoning Response:{dp['response']}\n\n"
            f"{instruction}\n\n"
            f"Rubric Criterion:{dp['criterion']}"
        )
        response, in_tok, out_tok = get_judge_response(client, args.judge_model, prompt)
        dp["idx"] = idx
        dp["judgement"] = response
        dp["judge_input_tokens"] = in_tok
        dp["judge_output_tokens"] = out_tok
        return dp

    with open(output_file, "a", encoding="utf-8") as fw:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_parallel_request) as executor:
            futures = {
                executor.submit(judge_one, idx, dp): idx
                for idx, dp in enumerate(criterion_data)
                if idx not in existing
            }
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                try:
                    result = future.result()
                    fw.write(json.dumps(result, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"Error: {e}")

    print(f"\nDone. Results in: {output_file}")


if __name__ == "__main__":
    main()
