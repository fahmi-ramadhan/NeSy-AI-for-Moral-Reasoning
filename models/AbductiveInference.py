import re

from models.utils import OpenAIModel

SYSTEM_PROMPT = """You are an expert in ethical reasoning and abductive inference.
Your task is to identify missing logical steps (facts) that, when added to existing reasoning,
would allow the conclusion (hypothesis) to be entailed under the given ethical theory.
Answer only with a numbered list of facts. No explanation, no prose outside the list."""

USER_TEMPLATE = """Theory: {theory}

{theory_definition}

Moral Dilemma:
{dilemma}

Goal to prove: {goal_predicate}({option_atom})

Existing explanatory facts (pipe-separated):
{explanatory_chain}

Hypothesis (conclusion to reach): {hypothesis}

Task: Provide the complete updated list of facts (existing + any new missing ones).
The facts must form a logical chain that supports proving {goal_predicate}({option_atom}).

CRITICAL RULES for writing facts:
- Each fact must be a single complete sentence ending with a period.
- Do NOT use "e.g." or "i.e." — write out "for example" or "that is" instead.
- Do NOT use parentheses containing examples.
- Keep each fact to one plain sentence.

Format:
1. <fact one>
2. <fact two>
...
n. <fact n>"""


class AbductiveInference:
    def __init__(self, api_key: str, engine: str, provider: str = "openai",
                 is_reasoning_model: bool = False, budget_tokens: int = 10000):
        self.model = OpenAIModel(api_key, engine, provider, is_reasoning_model, budget_tokens)

    def get_missing_facts(self, dilemma: str, theory: str, theory_definition: str,
                          goal_predicate: str, hypothesis: str,
                          explanatory_chain: str) -> str:
        """
        Returns updated explanatory chain (pipe-separated facts) as a string.
        """
        m = re.search(r"\((option_[a-z])\)", hypothesis)
        option_atom = m.group(1) if m else "option_b"

        user_prompt = USER_TEMPLATE.format(
            theory=theory,
            theory_definition=theory_definition,
            dilemma=dilemma,
            goal_predicate=goal_predicate,
            option_atom=option_atom,
            explanatory_chain=explanatory_chain,
            hypothesis=hypothesis,
        )

        for _ in range(10):
            result: str = self.model.chat(SYSTEM_PROMPT, user_prompt) or ""
            facts = self._parse(result)
            if facts:
                return " | ".join(facts)

        return explanatory_chain  # fallback

    def _parse(self, text: str) -> list[str]:
        facts = []
        for line in text.splitlines():
            line = line.strip()
            if re.match(r"^\d+\.", line):
                fact = re.sub(r"^\d+\.\s*", "", line).strip()
                if fact:
                    facts.append(fact)
        return facts
