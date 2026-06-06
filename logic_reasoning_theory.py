import argparse
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from models.AbductiveInference import AbductiveInference
from models.Autoformalization import Autoformalization
from models.DeductiveInference import DeductiveInference
from models.Extraction import Extraction
from models.SemanticPrompting import SemanticPrompting
from models.WeakRules import WeakRulesGenerator
from program import SprologRunner
from theory_definitions import (THEORY_DEFINITIONS, get_goal_predicate,
                                get_prolog_principles,
                                get_theory_definition_points)

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

class Config:
    def __init__(self, config_file: str):
        with open(config_file) as f:
            config = json.load(f)
        self.api_key = os.environ.get("API_KEY", config.get("API_KEY", ""))
        self.autoformalization_api_key = os.environ.get("AUTOFORMALIZATION_API_KEY", self.api_key)
        self.provider = config.get("PROVIDER", "openai")
        self.max_iterations = int(config.get("MAX_ITERATIONS", 5))
        self.chat_engine = config["CHAT_ENGINE"]
        self.is_reasoning_model = config.get("IS_REASONING_MODEL", False)
        self.budget_tokens = int(config.get("BUDGET_TOKENS", 10000))
        self.entity_tnorm = str(config.get("ENTITY_TNORM", "prod"))
        self.predicate_tnorm = str(config.get("PREDICATE_TNORM", "prod"))
        self.min_depth = str(config.get("MIN_DEPTH", "0"))
        self.min_bs_size = str(config.get("MIN_BS_SIZE", "0"))
        self.lambda_cut = str(config.get("LAMBDA_CUT", "0.13"))
        self.max_depth = str(config.get("MAX_DEPTH", "13"))
        self.generations_dir = config.get("GENERATIONS_DIR", "generations_theory")
        self.data_file = config.get("DATA_FILE", "morebench_theory.csv")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class MoReBenchTheoryPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.extraction_model = Extraction(config.api_key, config.chat_engine, config.provider, config.is_reasoning_model, config.budget_tokens)
        self.semantic_model = SemanticPrompting(config.api_key, config.chat_engine, config.provider, config.is_reasoning_model, config.budget_tokens)
        self.autoformalization_model = Autoformalization(config.autoformalization_api_key, "meta-llama/llama-4-scout-17b-16e-instruct", "groq", False, 0)
        self.abductive_model = AbductiveInference(config.api_key, config.chat_engine, config.provider, config.is_reasoning_model, config.budget_tokens)
        self.deductive_model = DeductiveInference(config.api_key, config.chat_engine, config.provider, config.is_reasoning_model, config.budget_tokens)
        self.weak_rules_generator = WeakRulesGenerator()
        self.sprolog = SprologRunner()

    # ── Step helpers ──────────────────────────────────────────────────────────

    def step_extraction(self, dilemma: str, theory: str) -> dict:
        """Step 1: Extract agents, actions, patients, arguments for both options."""
        return self.extraction_model.extract(dilemma, theory)

    def step_semantic_prompting(self, dilemma: str, theory: str,
                                 theory_def: str, extraction: dict) -> tuple[str, str]:
        """Step 2: Generate explanatory facts and hypothesis."""
        agents_a = ", ".join(extraction.get("agents_a", []))
        actions_a = ", ".join(extraction.get("actions_a", []))
        patients_a = ", ".join(extraction.get("patients_a", []))
        args_a = ", ".join(extraction.get("arguments_a", []))
        
        agents_b = ", ".join(extraction.get("agents_b", []))
        actions_b = ", ".join(extraction.get("actions_b", []))
        patients_b = ", ".join(extraction.get("patients_b", []))
        args_b = ", ".join(extraction.get("arguments_b", []))
        
        return self.semantic_model.inference(
            dilemma, theory, theory_def,
            agents_a, actions_a, patients_a, args_a,
            agents_b, actions_b, patients_b, args_b
        )

    def step_autoformalization(self, explanatory_chain: str,
                                theory: str, iteration: int, q_id: str) -> list[str]:
        """Step 3: Translate explanations to Prolog."""
        principles = get_prolog_principles(theory)
        return self.autoformalization_model.transfer(
            explanatory_chain, principles, iteration, q_id
        )

    def step_weak_rules(self, theory: str, q_id: str, iteration: int,
                        extraction: dict = {}):
        """Step 4: Build KB with weak unification rules (reads from kb/rules/)."""
        goal_pred = get_goal_predicate(theory)
        
        self.weak_rules_generator.get_weak_rules(
            agents_a=extraction.get("agents_a", []),
            actions_a=extraction.get("actions_a", []),
            patients_a=extraction.get("patients_a", []),
            arguments_a=extraction.get("arguments_a", []),
            agents_b=extraction.get("agents_b", []),
            actions_b=extraction.get("actions_b", []),
            patients_b=extraction.get("patients_b", []),
            arguments_b=extraction.get("arguments_b", []),
            goal_predicate=goal_pred,
            q_id=q_id,
            iteration=iteration,
        )

    def step_symbolic_solver(self, theory: str, q_id: str,
                              iteration: int) -> tuple[dict[str, str], dict[str, float]]:
        """Step 5: Run symbolic solver; return (proofs_dict, scores_dict)."""
        goal_pred = get_goal_predicate(theory)
        proofs, scores, _ = self.sprolog.run_sprolog(
            self.config.entity_tnorm,
            self.config.predicate_tnorm,
            self.config.min_depth,
            self.config.min_bs_size,
            self.config.lambda_cut,
            self.config.max_depth,
            goal_pred,
            q_id,
            iteration,
        )
        return proofs, scores

    def step_abductive_inference(self, dilemma: str, theory: str,
                                  theory_def: str, hypothesis: str,
                                  explanatory_chain: str) -> str:
        """Step 6: Generate missing facts to bridge to hypothesis."""
        goal_pred = get_goal_predicate(theory)
        return self.abductive_model.get_missing_facts(
            dilemma, theory, theory_def, goal_pred, hypothesis, explanatory_chain
        )

    def step_deductive_inference(self, dilemma: str, theory: str,
                                  theory_def: str, explanatory_chain: str) -> str:
        """Step 7: Re-derive hypothesis from updated facts."""
        goal_pred = get_goal_predicate(theory)
        return self.deductive_model.deductive_inference(
            dilemma, theory, theory_def, goal_pred, explanatory_chain
        )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def process_dilemma(self, dilemma: str, theory: str, q_id: str,
                        rubric: list) -> dict:
        """
        Run the full pipeline for a single dilemma.
        Returns a result dict compatible with MoReBench evaluation format.
        """
        print(f"\n{'='*60}")
        print(f"Question {q_id} | Theory: {theory}")
        print(f"Dilemma: {dilemma[:120]}...")
        print(f"{'='*60}")

        theory_def = get_theory_definition_points(theory)

        # ── Step 1: Extraction ──────────────────────────────────────────────
        print("\n[Step 1] Extraction...")
        extraction = self.step_extraction(dilemma, theory)
        print(f"  Option A - Agents: {extraction.get('agents_a', [])}, Actions: {extraction.get('actions_a', [])}, Patients: {extraction.get('patients_a', [])}, Arguments: {extraction.get('arguments_a', [])}")
        print(f"  Option B - Agents: {extraction.get('agents_b', [])}, Actions: {extraction.get('actions_b', [])}, Patients: {extraction.get('patients_b', [])}, Arguments: {extraction.get('arguments_b', [])}")

        # ── Step 2: Semantic Prompting ───────────────────────────────────────
        print("\n[Step 2] Semantic Prompting...")
        hypothesis, explanatory_chain = self.step_semantic_prompting(
            dilemma, theory, theory_def, extraction
        )
        print(f"  Hypothesis: {hypothesis}")
        print(f"  Explanatory Chain: {explanatory_chain[:200]}...")

        # ── Iterative refinement loop ────────────────────────────────────────
        validity = False
        iterations = 0
        iteration_results = []

        while not validity and iterations < self.config.max_iterations:
            print(f"\n[Iteration {iterations}]")

            # Step 3: Autoformalization
            print("  [Step 3] Autoformalization...")
            prolog_rules = self.step_autoformalization(
                explanatory_chain, theory, iterations, q_id
            )
            print(f"  Generated {len(prolog_rules)} Prolog rules")

            # Step 4: Weak Rules (build KB — reads kb/rules/, writes kb/prolog_kb/ + sims.txt)
            print("  [Step 4] Weak Rules Generation...")
            self.step_weak_rules(theory, q_id, iterations, extraction)

            # Step 5: Symbolic Solver
            print("  [Step 5] Symbolic Solver...")
            start = time.time()
            proof_chains, proof_scores = self.step_symbolic_solver(theory, q_id, iterations)
            elapsed = time.time() - start
            best_score = max(proof_scores.values()) if any(s > 0 for s in proof_scores.values()) else 0.0
            print(f"  Solver time: {elapsed:.2f}s | Best Score: {best_score:.4f}")
            print(f"  Option A Proof: {proof_chains.get('option_a', '')[:150] if proof_chains.get('option_a') else 'No proof'}")
            print(f"  Option B Proof: {proof_chains.get('option_b', '')[:150] if proof_chains.get('option_b') else 'No proof'}")

            iter_result = {
                "iteration": iterations,
                "explanatory_chain": explanatory_chain,
                "hypothesis": hypothesis,
                "prolog_rules": prolog_rules,
                "proof_chains": proof_chains,
                "proof_scores": proof_scores,
                "best_score": best_score,
                "valid": False,
            }

            # Check validity: solver found a proof for the hypothesis option with best score
            import re as _re
            _opt_m = _re.search(r"\(option_([a-z])\)", hypothesis)
            _opt_atom = f"option_{_opt_m.group(1)}" if _opt_m else ""
            
            if _opt_atom in proof_chains and proof_scores.get(_opt_atom, 0) > 0:
                hypothesis_score = proof_scores[_opt_atom]
                hypothesis_matches = hypothesis_score == best_score
                print(f"  Hypothesis option: {_opt_atom}, Score: {hypothesis_score:.4f}, Best: {best_score:.4f}, Valid: {hypothesis_matches}")
            else:
                hypothesis_matches = False
                print(f"  Hypothesis option: {_opt_atom} not in proofs or score is 0")

            if hypothesis_matches:
                validity = True
                iter_result["valid"] = True
                iteration_results.append(iter_result)
                print(f"  ✓ Valid proof found at iteration {iterations}!")
                break
            else:
                if iterations == self.config.max_iterations - 1:
                    print(f"  ✗ Max iterations reached. No valid proof found.")
                    break

                print("  ✗ No valid proof. Running abductive + deductive inference...")

                # Step 6: Abductive Inference (find missing facts)
                print("  [Step 6] Abductive Inference...")
                updated_chain = self.step_abductive_inference(
                    dilemma, theory, theory_def, hypothesis, explanatory_chain
                )
                print(f"  Updated chain: {updated_chain[:200]}...")

                # Step 7: Deductive Inference (re-derive hypothesis)
                print("  [Step 7] Deductive Inference...")
                new_hypothesis = self.step_deductive_inference(
                    dilemma, theory, theory_def, updated_chain
                )
                print(f"  New hypothesis: {new_hypothesis}")

                iter_result["updated_chain"] = updated_chain
                iter_result["new_hypothesis"] = new_hypothesis
                iteration_results.append(iter_result)

                # Update for next iteration
                explanatory_chain = updated_chain
                hypothesis = new_hypothesis
                iterations += 1

        # ── Build final result ────────────────────────────────────────────────
        final_iter = iteration_results[-1] if iteration_results else {}
        final_explanation = final_iter.get("explanatory_chain", explanatory_chain) + "\n" + final_iter.get("hypothesis", hypothesis)

        return {
            "dilemma": dilemma,
            "theory": theory,
            "q_id": q_id,
            "extraction": extraction,
            "iteration_results": iteration_results,
            "final_explanatory_chain": final_iter.get("explanatory_chain", explanatory_chain),
            "final_hypothesis": final_iter.get("hypothesis", hypothesis),
            "valid": validity,
            "total_iterations": iterations,
            "model_resp": final_explanation,
            "RUBRIC": rubric,
        }

    # ── Batch processing ──────────────────────────────────────────────────────

    def run(self):
        """Run the pipeline on the full MoReBench-Theory dataset."""
        import ast

        data = pd.read_csv(f"./data/{self.config.data_file}", index_col=False)

        results_file = os.path.join(self.config.generations_dir, "pipeline_results.jsonl")
        os.makedirs(self.config.generations_dir, exist_ok=True)

        # Load already-processed row indices (use integer row index as q_id)
        existing_ids: set = set()
        if os.path.exists(results_file):
            with open(results_file) as f:
                for line in f:
                    try:
                        existing_ids.add(json.loads(line)["idx"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        print(f"Skipping {len(existing_ids)} already processed rows")

        with open(results_file, "a", encoding="utf-8") as f_out:
            for idx, row in tqdm(data.iterrows(), total=len(data)):
                if idx in existing_ids:
                    continue

                q_id = str(idx)
                dilemma = str(row["DILEMMA"])
                theory = str(row["THEORY"])
                rubric = ast.literal_eval(str(row["RUBRIC"]))

                try:
                    result = self.process_dilemma(dilemma, theory, q_id, rubric)
                    result["idx"] = idx
                    result["DILEMMA_SOURCE"] = row.get("DILEMMA_SOURCE", "")
                    result["DILEMMA_TYPE"] = row.get("DILEMMA_TYPE", "")
                    result["ROLE_DOMAIN"] = row.get("ROLE_DOMAIN", "")
                    result["CONTEXT"] = row.get("CONTEXT", "")
                    result["model"] = self.config.chat_engine
                    result["input_tokens"] = -1 
                    result["output_tokens"] = -1
                    result["reasoning_tokens"] = -1
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()
                except Exception as e:
                    print(f"Error processing idx={idx}: {e}")
                    import traceback
                    traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MoReBench-Theory Neurosymbolic Pipeline"
    )
    parser.add_argument(
        "--config", "-c", default="config.json",
        help="Path to config JSON file"
    )
    parser.add_argument(
        "--dilemma", "-d", default=None,
        help="Single dilemma string for testing (bypasses dataset)"
    )
    parser.add_argument(
        "--theory", "-t", default="Act Utilitarianism",
        choices=list(THEORY_DEFINITIONS.keys()),
        help="Theory to use for single dilemma test"
    )
    args = parser.parse_args()

    config = Config(args.config)
    pipeline = MoReBenchTheoryPipeline(config)

    if args.dilemma:
        # Single test mode
        result = pipeline.process_dilemma(args.dilemma, args.theory, "test_0", [])
        print("\n" + "="*60)
        print("FINAL RESULT:")
        print(json.dumps({
            k: v for k, v in result.items()
            if k not in ("iteration_results", "RUBRIC")
        }, indent=2, ensure_ascii=False))
    else:
        pipeline.run()
