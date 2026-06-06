# Evaluating a Neuro-Symbolic AI Approach Based on Iterative Symbolic Refinement for Logical Process in Moral Reasoning

As language models become increasingly involved in ethical decision-making, the ability to reason logically in moral contexts becomes ever more important to evaluate. This study aims to evaluate whether a neuro-symbolic AI approach can improve scores on the logical process dimension of MoReBench-Theory, a benchmark that measures the quality of language model moral reasoning based on normative ethics theories. As a secondary objective, this study also examines whether the approach can mitigate the performance disparity between Benthamite Act Utilitarianism and Gauthierian Contractarianism. Experiments were conducted on 60 moral dilemma scenarios from MoReBench-Theory covering both theories, using five language models of varying sizes and capabilities through a neuro-symbolic pipeline based on iterative symbolic refinement. The quality of the resulting reasoning was then evaluated using a multidimensional rubric developed by moral philosophy experts. Results show that the pipeline only improved logical process scores for small-to-medium models that produced relatively short answers when responding without the pipeline, while overall scores declined across all models as the pipeline's output format was not designed to satisfy all rubric dimensions comprehensively. The pipeline also can not systematically mitigate the inter-theory performance disparity.

![Pipeline Architecture](images/pipeline.png)

## Neuro-Symbolic Pipeline

The pipeline implements an iterative refinement loop for moral reasoning:

1. **Extraction** — Extracts agents, actions, patients, and arguments for both action options from the dilemma text.
2. **Semantic Prompting** — Generates explanatory facts and a hypothesis based on the moral theory and extraction results.
3. **Autoformalization** — Translates the explanatory chain into Prolog rules.
4. **Knowledge Base Construction** — Builds a knowledge base with weak unification rules.
5. **Symbolic Solver** — Runs the Prolog solver to produce proof chains with confidence scores.
6. **Abductive Inference** — If no valid proof is found, generates missing facts to bridge gaps in reasoning.
7. **Deductive Inference** — Re-derives the hypothesis from the updated explanatory chain.

Steps 3–7 repeat iteratively (up to `MAX_ITERATIONS` times) until a valid symbolic proof is obtained.

## Setup

1. Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and configure your API keys:
```bash
cp .env.example .env
```
- `API_KEY` is the primary key for the LLM provider used in most pipeline steps.
- `AUTOFORMALIZATION_API_KEY` is a separate key for the autoformalization step.

3. Configure the language model, API provider, and other settings in `config.json`

## Usage

### Running the Neuro-Symbolic Pipeline

```bash
python3 logic_reasoning_theory.py --config config.json
```

This processes all dilemmas in `data/morebench_theory_filtered.csv` and writes results to `generations_theory/pipeline_results.jsonl`.

### Running MoReBench Evaluation

After obtaining pipeline outputs, evaluate them using an LLM judge:

```bash
python3 run_judge_on_pipeline_output.py \
    -i generations_theory/pipeline_results.jsonl \
    -ap vertex-gpt-oss \
    -ak nesy-ai \
    -jm openai/gpt-oss-120b-maas \
    -n 10
```

- `-i` — Input pipeline results file
- `-ap` — API provider (openrouter, groq, vertex-gpt-oss)
- `-ak` — API key (or GCP project ID for vertex)
- `-jm` — Judge model name
- `-n` — Number of parallel requests

Results are saved to `model_resp_judgements/`.

### Calculating MoReBench Scores

For valid-only scores (using pipeline validity filter):

```bash
python3 calculate_morebench_theory.py \
    -i model_resp_judgements/model_resp_pipeline_results_gemma-3-4b-it.jsonl \
    -f human \
    -es 1547 \
    -p generations_theory/pipeline_results_gemma-3-4b-it.jsonl
```

- `-i` — Judgement data file
- `-f` — Output format (`human` or `latex`)
- `-es` — Expected number of judgement entries
- `-p` — Pipeline results file for filtering valid tasks only

To calculate scores on all data (without valid-only filtering), omit the `-p` argument.

## Experimental Results

This repository contains experiment outputs for five language models:

| Model | Pipeline Result | Without Pipeline |
|-------|----------------|------------------|
| Llama 3.1 8B Instant | `pipeline_results_llama-3.1-8b-instant.jsonl` | `without_pipeline_llama-3.1-8b-instant.jsonl` |
| Gemma 3 4B IT | `pipeline_results_gemma-3-4b-it.jsonl` | `without_pipeline_gemma-3-4b-it.jsonl` |
| Llama 4 Scout 17B 16E | `pipeline_results_llama-4-scout-17b-16e-instruct.jsonl` | `without_pipeline_llama-4-scout-17b-16e-instruct.jsonl` |
| Llama 3.3 70B Versatile | `pipeline_results_llama-3.3-70b-versatile.jsonl` | `without_pipeline_llama-3.3-70b-versatile.jsonl` |
| GPT-OSS 120B | `pipeline_results_gpt-oss-120b.jsonl` | `without_pipeline_gpt-oss-120b.jsonl` |

Corresponding evaluation results (judge responses) are in `model_resp_judgements/` and execution logs in `pipeline_logs/`.

The *without pipeline* baseline experiments were generated using a modified version of [MoReBench](https://github.com/fahmi-ramadhan/morebench) that adds support for Groq and Vertex AI API providers and external dataset files.

## Acknowledgements

This work builds upon the following open-source projects:

- **[Logic-Explainer](https://github.com/neuro-symbolic-ai/explanation_based_ethical_reasoning)** — The neuro-symbolic framework for ethical reasoning that forms the foundation of the pipeline implemented in this repository.
- **[MoReBench](https://github.com/morebench/morebench)** — The Moral Reasoning Benchmark that provides the evaluation framework, rubric dimensions, and moral dilemma scenarios used in this study.
