import os
import re

from models.utils import OpenAIModel

SYSTEM_PROMPT = """You are an expert in converting natural language explanations into Prolog rules.
Output ONLY a numbered list of Prolog rules. No prose, no explanation, no markdown."""

USER_TEMPLATE = """Translate the explanatory chain below into Prolog rules.

CRITICAL RULES FOR VARIABLES AND ARGUMENTS:
- Use variable (X) for GENERAL rules that apply to ANY action/option.
- Use option_a or option_b for FACTS that specifically describe ONE option's consequence.

Examples:
- "Option B generates revenue to fund future R&D" -> generates_revenue(option_b). = 1.0
- "Option A saves the current patients on the waiting list" -> saves_current_patients(option_a). = 1.0
- "Saving lives maximizes overall well-being" -> maximizes_overall_wellbeing(X) :- saves_lives(X). = 1.0
- "Reduces mortality by 25%" -> reduces_mortality_by_25_percent(option_b). = 1.0 (NOT reduces_mortality(option_b, 0.25))

FORMAT (follow exactly):
  head_predicate(X) :- body_predicate(X). = 1.0     [general rule - uses X]
  leaf_predicate(option_atom). = 1.0                [specific fact - uses option_a/option_b]

REQUIREMENTS:
1. All predicate names and arguments: lowercase, underscores only (no hyphens, no colons, no spaces, no percent symbols, no quotes, no dots in predicate names, no dollar signs, and no other special characters).
2. Predicate names and arguments MUST start with a letter (a-z), NOT with numbers, symbols, or dollar signs. For monetary values, include the amount in the predicate name after a word (e.g., generates_8_5_million instead of $8.5_million).
3. If there is a predicate name or argument with dot (e.g. 2.3_something), replace the dot with an underscore (e.g. 2_3_something).
4. Use (X) for rules that are general. Use option_a, option_b, etc. for facts that describe specific option consequences. The options are labeled with letters: option_a, option_b.
5. When the explanatory chain mentions "Option A" or "Option B" (or "option a", "option b"), the corresponding Prolog fact MUST use the matching option_a or option_b atom.

EXPLANATORY CHAIN TO FORMALIZE:
{explanatory_chain}

COMPLETE EXAMPLE (for some explanatory chain, not necessarily related to the current dilemma):
1. justified_virtue_ethics(X) :- virtuous_person_would_do(X). = 1.0
2. virtuous_person_would_do(X) :- balances_justice_and_compassion(X). = 1.0
3. balances_justice_and_compassion(X) :- proportionate_punishment(X), avoids_cruelty(X). = 1.0
4. proportionate_punishment(X) :- crime_severity_matches_sentence(X). = 1.0
5. avoids_cruelty(X) :- considers_defendants_wellbeing(X). = 1.0
6. crime_severity_matches_sentence(option_b). = 1.0
7. considers_defendants_wellbeing(option_b). = 1.0
8. better_outcome_for_all(option_b). = 1.0

Now translate the explanatory chain above. Extract predicates and relationships as stated — do not add or assume any connections beyond what's explicitly in the chain.

Answer (numbered list only):"""


class Autoformalization:
    def __init__(self, api_key: str, engine: str, provider: str = "openai",
                 is_reasoning_model: bool = False, budget_tokens: int = 10000):
        self.model = OpenAIModel(api_key, engine, provider, is_reasoning_model, budget_tokens)

    def transfer(self, explanatory_chain: str, theory_principles: list,
                 iteration: int, q_id: str) -> list:
        """
        Translates explanatory_chain to Prolog rules.
        Saves the translated explanatory_chain and theory_principles to kb/rules/question_{q_id}/{iteration}it.txt
        Returns list of Prolog rule strings.
        """

        user_prompt = USER_TEMPLATE.format(
            explanatory_chain=explanatory_chain,
        )

        VALID_RULE = re.compile(
            r'^[a-z][a-z0-9_]*\('
            r'[^)]+\)'
            r'(\s*:-\s*.+?)?'
            r'\.\s*=\s*[\d.]+\s*$'
        )

        def is_valid_rule(line: str) -> bool:
            if not VALID_RULE.match(line):
                return False
            # Reject facts with multiple arguments (e.g., pred(option_a, 0.25))
            if ':-' not in line:
                args_part = line.split('(')[1].split(')')[0]
                if ',' in args_part:
                    return False
            # Reject predicates starting with $ or numbers
            head_match = re.match(r'^([a-z][a-z0-9_]*)', line)
            if not head_match:
                return False
            head_pred = head_match.group(1)
            if head_pred[0].isdigit() or head_pred.startswith('$'):
                return False
            # Reject predicates with dots in the name
            if '.' in head_pred:
                return False
            return True

        result_list = []
        for attempt in range(10):
            inference_result = self.model.chat(SYSTEM_PROMPT, user_prompt) or ""

            if not inference_result.strip():
                print(f"  [Autoformalization attempt {attempt}] empty response, retrying...")
                continue

            lines = inference_result.split("\n")
            parsed = []
            for line in lines:
                # Strip numbering like "1. " or "1) "
                line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
                # Strip markdown backticks
                line = line.strip("`")
                if is_valid_rule(line):
                    parsed.append(line)
                elif line:
                    print(f"  [Autoformalization skip] {line[:80]}")

            print(f"  [Autoformalization attempt {attempt}] {len(parsed)} rules")

            if parsed:
                result_list = parsed
                break
            elif parsed:
                continue

        # add theory principles to the transferred rules
        for principle in theory_principles:
            result_list.insert(0, principle)

        directory = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "kb", "rules", f"question_{q_id}"
        )
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, f"{iteration}it.txt"), "w") as f:
            f.write("\n".join(result_list))

        return result_list
