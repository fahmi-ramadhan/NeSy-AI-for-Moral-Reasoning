from models.utils import OpenAIModel

EXTRACTION_SYSTEM_PROMPT = """You are an expert in semantic role labeling and moral reasoning.
Your task is to extract structured semantic information from moral dilemmas.
Answer only with the requested structured format."""

EXTRACTION_USER_TEMPLATE = """Extract semantic role information from the moral dilemma below.
Moral Dilemma:
{dilemma}

Task: Extract the semantic roles for EACH OPTION separately. For each option, identify:
1. AGENTS: Who performs the action (comma-separated)
2. ACTIONS: What action is performed (comma-separated) 
3. PATIENTS: Who/what receives the action (comma-separated)
4. ARGUMENTS: Additional semantic information like manner, location, time, etc. (comma-separated)

IMPORTANT: Extract for BOTH Option A and Option B. Use SRL (Semantic Role Labeling) format.

Format your answer exactly as:
AGENTS_A: <comma-separated list>
ACTIONS_A: <comma-separated list>
PATIENTS_A: <comma-separated list>
ARGUMENTS_A: <comma-separated list>
---
AGENTS_B: <comma-separated list>
ACTIONS_B: <comma-separated list>
PATIENTS_B: <comma-separated list>
ARGUMENTS_B: <comma-separated list>"""


class Extraction:
    def __init__(self, api_key: str, engine: str, provider: str = "openai",
                 is_reasoning_model: bool = False, budget_tokens: int = 10000):
        self.model = OpenAIModel(api_key, engine, provider, is_reasoning_model, budget_tokens)

    def extract(self, dilemma: str, theory: str) -> dict:
        """Extract agents, actions, patients, arguments for both options."""
        user_prompt = EXTRACTION_USER_TEMPLATE.format(
            theory=theory,
            dilemma=dilemma
        )
        
        result = ""
        for _ in range(5):
            result = self.model.chat(EXTRACTION_SYSTEM_PROMPT, user_prompt) or ""
            parsed = self._parse(result)
            if parsed["agents_a"] or parsed["agents_b"]:
                return parsed
        
        return {
            "agents_a": [], "actions_a": [], "patients_a": [], "arguments_a": [],
            "agents_b": [], "actions_b": [], "patients_b": [], "arguments_b": [],
            "raw": result
        }

    def _parse(self, text: str) -> dict:
        agents_a, actions_a, patients_a, arguments_a = [], [], [], []
        agents_b, actions_b, patients_b, arguments_b = [], [], [], []
        
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("---"):
                continue
            elif line.startswith("AGENTS_A:"):
                agents_a = [s.strip() for s in line.replace("AGENTS_A:", "").split(",") if s.strip()]
            elif line.startswith("ACTIONS_A:"):
                actions_a = [s.strip() for s in line.replace("ACTIONS_A:", "").split(",") if s.strip()]
            elif line.startswith("PATIENTS_A:"):
                patients_a = [s.strip() for s in line.replace("PATIENTS_A:", "").split(",") if s.strip()]
            elif line.startswith("ARGUMENTS_A:"):
                arguments_a = [s.strip() for s in line.replace("ARGUMENTS_A:", "").split(",") if s.strip()]
            elif line.startswith("AGENTS_B:"):
                agents_b = [s.strip() for s in line.replace("AGENTS_B:", "").split(",") if s.strip()]
            elif line.startswith("ACTIONS_B:"):
                actions_b = [s.strip() for s in line.replace("ACTIONS_B:", "").split(",") if s.strip()]
            elif line.startswith("PATIENTS_B:"):
                patients_b = [s.strip() for s in line.replace("PATIENTS_B:", "").split(",") if s.strip()]
            elif line.startswith("ARGUMENTS_B:"):
                arguments_b = [s.strip() for s in line.replace("ARGUMENTS_B:", "").split(",") if s.strip()]

        return {
            "agents_a": agents_a,
            "actions_a": actions_a,
            "patients_a": patients_a,
            "arguments_a": arguments_a,
            "agents_b": agents_b,
            "actions_b": actions_b,
            "patients_b": patients_b,
            "arguments_b": arguments_b,
            "raw": text
        }
