import re

from models.utils import OpenAIModel

SYSTEM_PROMPT = """You are an expert in ethical reasoning and moral philosophy.
Your task is to reason about moral dilemmas using a specific ethical theory.
Answer only in plain text. No markdown, no bold, no asterisks."""

USER_TEMPLATE = """You are reasoning about a moral dilemma using {theory}.

{theory_definition}

Moral Dilemma:
{dilemma}

Extracted Semantic Roles:
Option A:
- Agents: {agents_a}
- Actions: {actions_a}
- Patients: {patients_a}
- Arguments: {args_a}

Option B:
- Agents: {agents_b}
- Actions: {actions_b}
- Patients: {patients_b}
- Arguments: {args_b}

Task:
1. Generate a numbered list of explanatory facts (f1, f2, ...) that support a justified conclusion under {theory}.
   CRITICAL RULES for facts:
   - Each fact must be a single complete sentence ending with a period.
   - Do NOT use abbreviations like "e.g." or "i.e." - write them out as "for example" or "that is".
   - Do NOT use parentheses containing examples - move examples into the main sentence.
   - Keep each fact to one plain sentence with no nested clauses.
   - **VERY IMPORTANT**: When describing consequences or effects that apply to a SPECIFIC option, you MUST explicitly state "Option A" or "Option B" in the fact. For example:
     - "Option B generates revenue to fund future R&D."
     - "Option A saves the current patients on the waiting list."
     - NOT "Generating revenue funds future R&D" (missing the option reference)
   - Facts that don't specifically describe an option can be general principles.
2. State which option is justified.

Format your answer EXACTLY as shown (no extra text before or after):
EXPLANATORY FACTS:
f1: <one plain sentence fact - MUST include "Option A" or "Option B" if it describes a specific option's consequence>
f2: <one plain sentence fact>
fn: <one plain sentence fact>
JUSTIFIED: option_a
(or option_b — use the lowercase letter of the justified option)"""


class SemanticPrompting:
    def __init__(self, api_key: str, engine: str, provider: str = "openai",
                 is_reasoning_model: bool = False, budget_tokens: int = 10000):
        self.model = OpenAIModel(api_key, engine, provider, is_reasoning_model, budget_tokens)

    def inference(self, dilemma: str, theory: str, theory_definition: str,
                  agents_a: str, actions_a: str, patients_a: str, args_a: str,
                  agents_b: str, actions_b: str, patients_b: str, args_b: str) -> tuple[str, str]:
        """
        Returns (hypothesis_prolog, explanatory_chain) as strings.
        hypothesis_prolog e.g. 'justified_contractarian(option_b)'
        explanatory_chain is newline-joined facts (NOT dot-split, to avoid e.g. corruption).
        """
        from theory_definitions import get_goal_predicate
        goal_predicate = get_goal_predicate(theory)

        user_prompt = USER_TEMPLATE.format(
            theory=theory,
            theory_definition=theory_definition,
            dilemma=dilemma,
            agents_a=agents_a,
            actions_a=actions_a,
            patients_a=patients_a,
            args_a=args_a,
            agents_b=agents_b,
            actions_b=actions_b,
            patients_b=patients_b,
            args_b=args_b,
        )

        for _ in range(10):
            result = self.model.chat(SYSTEM_PROMPT, user_prompt) or ""
            hypothesis, facts = self._parse(result, goal_predicate)
            if hypothesis and facts:
                return hypothesis, facts

        return "", ""

    def _parse(self, text: str, goal_predicate: str) -> tuple[str, str]:
        VALID_OPTIONS = set("abcdefghij")
        facts = []
        justified_letter = ""

        lines = text.splitlines()
        in_facts = False
        for line in lines:
            line = line.strip()
            if line.startswith("EXPLANATORY FACTS:"):
                in_facts = True
                continue
            if line.startswith("JUSTIFIED:"):
                in_facts = False
                # Extract e.g. "option_b" or just "b"
                m = re.search(r"option[_\s]?([a-j])", line, re.IGNORECASE)
                if m:
                    justified_letter = m.group(1).lower()
                continue
            if in_facts and re.match(r"^f\d+:", line):
                fact = re.sub(r"^f\d+:\s*", "", line).strip()
                if fact:
                    facts.append(fact)

        if not justified_letter and facts:
            # fallback: scan full text
            m = re.search(r"JUSTIFIED:\s*option[_\s]?([a-j])", text, re.IGNORECASE)
            if m:
                justified_letter = m.group(1).lower()

        if justified_letter not in VALID_OPTIONS:
            return "", ""

        hypothesis = f"{goal_predicate}(option_{justified_letter})"
        # Join facts with " | " as separator — safe against period-splitting corruption
        explanatory_chain = " | ".join(facts)
        return hypothesis, explanatory_chain
