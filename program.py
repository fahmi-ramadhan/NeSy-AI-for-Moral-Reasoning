import os
import shlex
import subprocess


class SprologRunner:
    def __init__(self):
        self.BASE_PATH = os.path.dirname(os.path.abspath(__file__))

    def get_goal(self, goal_predicate: str) -> str:
        goals = [
            f"{goal_predicate}(option_a).",
            f"{goal_predicate}(option_b).",
        ]
        return "|".join(goals)

    def query(self, goal: str, entity_tnorm: str, predicate_tnorm: str,
              min_depth: str, min_bs_size: str, lambda_cut: str,
              max_depth: str, q_id: str, iteration: int) -> list:

        lambda_cut_str = "|".join([str(lambda_cut)] * len(goal.split("|")))
        kb_path = os.path.join(
            self.BASE_PATH, "kb", "prolog_kb", f"question_{q_id}", f"{iteration}it.txt"
        )
        sims_path = os.path.join(self.BASE_PATH, "kb", "sims.txt")

        os.makedirs(os.path.dirname(sims_path), exist_ok=True)
        if not os.path.exists(sims_path):
            open(sims_path, "w").close()

        spyrolog_bin = os.path.join(self.BASE_PATH, "spyrolog")

        cmd = shlex.split(
            f"{spyrolog_bin} {kb_path} {sims_path} {goal} "
            f"{max_depth} {lambda_cut_str} "
            f"{entity_tnorm}|{predicate_tnorm} {min_bs_size}"
        )

        try:
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120
            )
        except (subprocess.TimeoutExpired, ValueError):
            return []

        if result.returncode != 0:
            print(f"  [spyrolog crash] rc={result.returncode} stderr={result.stderr[:200]}")
            print(f"  [spyrolog crash] kb={kb_path}")
            # Print KB to diagnose parse error
            try:
                with open(kb_path) as _f:
                    _lines = _f.readlines()
                print(f"  [spyrolog crash] KB ({len(_lines)} lines):")
                for _i, _l in enumerate(_lines):
                    print(f"    {_i+1}: {repr(_l.rstrip())}")
            except Exception:
                pass

        results = []
        try:
            for r in result.stdout.split(b"\n"):
                if not r:
                    continue
                split = r.split(b" ")
                if len(split) < 3:
                    results.append([float(split[0]), int(split[1]), "", ""])
                else:
                    results.append([
                        float(split[0]),
                        int(split[1]),
                        b" ".join(split[3:]).decode(),
                        split[2].decode(),
                    ])
            return results
        except (ValueError, IndexError):
            return []

    def run_sprolog(self, entity_tnorm: str, predicate_tnorm: str,
                    min_depth: str, min_bs_size: str, lambda_cut: str,
                    max_depth: str, goal_predicate: str,
                    q_id: str, iteration: int) -> tuple[dict[str, str], dict[str, float], list]:
        """
        Returns (proofs_dict, scores_dict, raw_results).
        proofs_dict: {'option_a': best_proof_chain_for_a, 'option_b': best_proof_chain_for_b}
        scores_dict: {'option_a': best_score_a, 'option_b': best_score_b}
        """
        goal = self.get_goal(goal_predicate)
        raw = self.query(
            goal, entity_tnorm, predicate_tnorm,
            min_depth, min_bs_size, lambda_cut, max_depth, q_id, iteration
        )

        proofs: dict[str, str] = {"option_a": "", "option_b": ""}
        scores: dict[str, float] = {"option_a": 0.0, "option_b": 0.0}
        option_rules: dict[str, dict[str, float]] = {"option_a": {}, "option_b": {}}

        if raw:
            try:
                for score, depth, rule, unification in raw:
                    if not rule:
                        continue
                    if "option_a" in rule:
                        option_rules["option_a"][rule] = score
                    elif "option_b" in rule:
                        option_rules["option_b"][rule] = score

                for opt in ["option_a", "option_b"]:
                    if option_rules[opt]:
                        best_rule = max(option_rules[opt], key=option_rules[opt].get) # type: ignore
                        scores[opt] = option_rules[opt][best_rule]
                        proofs[opt] = best_rule

                best_score = max(scores.values()) if any(s > 0 for s in scores.values()) else 0.0
                print(f"  [spyrolog] option_a_score={scores['option_a']:.4f}, option_b_score={scores['option_b']:.4f}, best={best_score:.4f}")

                return proofs, scores, raw
            except (ValueError, TypeError):
                pass

        print(f"  [spyrolog] no proof found for {goal_predicate}")
        return proofs, scores, raw
