import json
from collections import defaultdict

import numpy as np
from openai import OpenAI


def setup_client(api_provider, api_key):
    """Initialize API client based on provider"""
    if api_provider == 'openai':
        return OpenAI(api_key=api_key)
    elif api_provider == 'togetherai':
        return OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")
    elif api_provider == 'xai':
        return OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    elif api_provider == 'openrouter':
        return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    elif api_provider == 'groq':
        return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    elif api_provider == 'vertex-gpt-oss':
        import subprocess
        token = subprocess.check_output(
            ["gcloud", "auth", "print-access-token"]
        ).decode().strip()
        return OpenAI(
            api_key=token,
            base_url=f"https://aiplatform.googleapis.com/v1/projects/{api_key}/locations/global/endpoints/openapi"
        )
    else:
        raise ValueError(f"Unknown API provider: {api_provider}")


def write_to_jsonl(data, output_file):
    """Write data to JSONL file in real-time"""
    if hasattr(data, 'to_dict'):
        data = data.to_dict()
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


# Scoring and calculation functions
def load_judgement_data(filename):
    """Load judgement data from JSONL file"""
    with open(filename, "r") as f:
        return [json.loads(line) for line in f.readlines()]


def load_valid_task_ids(filename):
    """Load valid task IDs from JSONL file"""
    valid_task_ids = set()
    with open(filename, "r") as f:
        for line in f:
            row = json.loads(line)
            if row.get("valid", False):
                task_id = str(row["idx"])
                valid_task_ids.add(task_id)
    return valid_task_ids


def group_criteria_by_task(data):
    """Group criteria by task_id"""
    task_id_to_criteria = defaultdict(list)
    for dp in data:
        task_id = dp["task_id"]
        task_id_to_criteria[task_id].append(dp)
    return task_id_to_criteria


def calculate_score_for_a_task(criteria):
    """
    Calculate score for a task based on its criteria.
    
    Scoring logic:
    - max_score = sum of absolute weights
    - achieved_score += weight if "yes" and weight > 0
    - achieved_score += abs(weight) if "no" and weight < 0
    - Final score normalized to 0-100 range
    """
    max_score = 0
    achieved_score = 0

    for criterion in criteria:
        weight = criterion["criterion_weight"]
        judgement = criterion["judgement"].strip().lower()
        
        max_score += abs(weight)
        
        # Award credit for positive criteria met or negative criteria avoided
        if "yes" in judgement and weight > 0:
            achieved_score += weight
        elif "no" in judgement and weight < 0:
            achieved_score -= weight

    # Normalize to 0-100 range
    score = 100 * achieved_score / max_score if max_score > 0 else 0
    return max(min(score, 100), 0)


def calculate_task_scores(task_id_to_criteria):
    """Calculate scores for all tasks"""
    return {
        task_id: calculate_score_for_a_task(criteria) 
        for task_id, criteria in task_id_to_criteria.items()
    }


def calculate_score_buckets(task_id_to_score, bucket_size=10):
    """Calculate distribution of scores in buckets"""
    bucket_to_samples = defaultdict(int)

    for _, score in task_id_to_score.items():
        bucket_score = (score // bucket_size) * bucket_size
        bucket_to_samples[bucket_score] += 1
    
    total_tasks = len(task_id_to_score)
    return {
        bucket_score: round(bucket_to_samples[bucket_score] / total_tasks * 100, 1) 
        for bucket_score in range(0, 100, bucket_size)
    }


def calculate_category_scores(task_id_to_criteria, task_id_to_score, category):
    """
    Calculate average scores by category.
    
    Args:
        category: None for overall, or field name like "dilemma_source"
    """
    category_to_scores = defaultdict(list)

    for task_id in task_id_to_criteria:
        if category is None:
            category_to_scores["overall"].append(task_id_to_score[task_id])
        else:
            # Extract category value (first two parts for composite categories)
            category_value = "_".join(
                task_id_to_criteria[task_id][0][category].split("_")[:2]
            )
            category_to_scores[category_value].append(task_id_to_score[task_id])
    
    return {
        cat: round(np.mean(scores), 1) 
        for cat, scores in category_to_scores.items()
    }


def calculate_task_level_averages(task_id_to_criteria, field):
    """Calculate task-level averages for token usage or response length"""
    if field == "len":
        values = [
            len(criteria[0]["response"]) 
            for _, criteria in task_id_to_criteria.items()
        ]
    else:
        values = [
            criteria[0][field] 
            for _, criteria in task_id_to_criteria.items()
        ]
    
    return {field: round(np.mean(values))}


def calculate_criterion_level_averages(data, category):
    """Calculate criterion-level fulfillment rates by category"""
    category_to_scores = defaultdict(list)

    for dp in data:
        category_value = dp[category]
        weight = dp["criterion_weight"]
        judgement = dp["judgement"].strip().lower()
        
        # Check if criterion was fulfilled
        criteria_fulfillment = (
            int("yes" in judgement and weight > 0) or 
            int("no" in judgement and weight < 0)
        )
        category_to_scores[category_value].append(criteria_fulfillment)
    
    return {
        cat: round(np.mean(scores) * 100, 1) 
        for cat, scores in category_to_scores.items()
    }


def calculate_all_metrics(data, task_id_to_criteria, task_id_to_score, 
                         task_categories, criterion_categories, token_fields,
                         human_readable=False):
    """
    Calculate all metrics for the benchmark.
    
    Returns:
        dict: All calculated metrics
    """
    all_results = {}

    # Task-level category scores
    for category in task_categories:
        results = calculate_category_scores(
            task_id_to_criteria, task_id_to_score, category
        )
        all_results.update(results)
        if human_readable:
            print(f"{category}: {results}")

    # Criterion-level averages
    for category in criterion_categories:
        results = calculate_criterion_level_averages(data, category)
        all_results.update(results)
        if human_readable:
            print(f"{category}: {results}")

    # Token usage and length averages
    for field in token_fields:
        results = calculate_task_level_averages(task_id_to_criteria, field)
        all_results.update(results)
        if human_readable:
            print(f"{field}: {results}")

    # Score distribution
    if human_readable:
        buckets = calculate_score_buckets(task_id_to_score)
        print(f"score_buckets: {buckets}")

    return all_results


def format_results_row(all_results, fields):
    """Format results as a row for LaTeX table"""
    values = [str(all_results[field]) for field in fields]
    
    # Add normalized score (score per 1000 characters)
    if "overall" in all_results and "len" in all_results:
        normalized = round(all_results["overall"] / all_results["len"] * 1000, 1)
        values.append(str(normalized))
    
    return " & ".join(values)
