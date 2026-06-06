import re

from models.utils import OpenAIModel

SYSTEM_PROMPT = """You are an expert in ethical reasoning and moral philosophy.
Your task is to derive a moral conclusion from given facts under a specific ethical theory.
Answer in plain text only. No markdown, no bold, no asterisks."""

USER_TEMPLATE = """Theory: {theory}

{theory_definition}

Moral Dilemma:
{dilemma}

Explanatory facts:
{explanatory_chain}

Goal predicate to prove: {goal_predicate}(option_X)

Question: Based on the facts above and the {theory} framework, which option is morally justified?

IMPORTANT INSTRUCTIONS:
- The options are labeled with letters: A, B, C, etc.
- You MUST end your answer with exactly this format on its own line:
  JUSTIFIED: option_a
  or
  JUSTIFIED: option_b
  (use the lowercase letter matching the justified option)
- Answer in plain text only."""


class DeductiveInference:
    def __init__(self, api_key: str, engine: str, provider: str = "openai",
                 is_reasoning_model: bool = False, budget_tokens: int = 10000):
        self.model = OpenAIModel(api_key, engine, provider, is_reasoning_model, budget_tokens)

    def deductive_inference(self, dilemma: str, theory: str, theory_definition: str,
                            goal_predicate: str, explanatory_chain: str) -> str:
        """
        Returns hypothesis string e.g. 'justified_act_utilitarian(option_b)'
        """
        user_prompt = USER_TEMPLATE.format(
            theory=theory,
            theory_definition=theory_definition,
            dilemma=dilemma,
            goal_predicate=goal_predicate,
            explanatory_chain=explanatory_chain,
        )

        for _ in range(10):
            result: str = self.model.chat(SYSTEM_PROMPT, user_prompt) or ""
            hypothesis = self._parse(result, goal_predicate)
            if hypothesis:
                return hypothesis

        return f"{goal_predicate}(option_unknown)"

    def _parse(self, text: str, goal_predicate: str) -> str:
        """
        Priority 1: explicit JUSTIFIED: option_X tag
        Priority 2: "Option A/B" pattern in text
        Priority 3: any option_X mention
        Explicitly rejects option_t and other non-letter artifacts.
        """
        VALID_OPTIONS = set("abcdefghij")  # only real option letters

        # Priority 1: explicit structured tag
        match = re.search(r"JUSTIFIED:\s*option_([a-z])", text, re.IGNORECASE)
        if match:
            letter = match.group(1).lower()
            if letter in VALID_OPTIONS:
                return f"{goal_predicate}(option_{letter})"

        # Priority 2: "The justified action is Option B" / "Option B is justified"
        match = re.search(
            r"(?:justified\s+(?:action|option)\s+is\s+(?:Option\s+)?|"
            r"(?:Option\s+))([A-J])\b",
            text, re.IGNORECASE
        )
        if match:
            letter = match.group(1).lower()
            if letter in VALID_OPTIONS:
                return f"{goal_predicate}(option_{letter})"

        # Priority 3: any standalone "option_X" or "option X" with a valid letter
        for m in re.finditer(r"\boption[_\s]([a-j])\b", text, re.IGNORECASE):
            letter = m.group(1).lower()
            if letter in VALID_OPTIONS:
                return f"{goal_predicate}(option_{letter})"

        return ""
