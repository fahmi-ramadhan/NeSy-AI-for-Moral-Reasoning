import tenacity

from utils import setup_client


class OpenAIModel:
    def __init__(self, api_key: str, engine: str, provider: str = "openai",
                 is_reasoning_model: bool = False, budget_tokens: int = 10000):
        self.api_key = api_key
        self.engine = engine
        self.provider = provider.lower()
        self.is_reasoning_model = is_reasoning_model
        self.budget_tokens = budget_tokens
        self.client = setup_client(self.provider, self.api_key)

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=2, min=15, max=120),
        retry=tenacity.retry_if_exception_type(Exception),
        stop=tenacity.stop_after_attempt(5)
    )
    def completion_with_backoff(self, **kwargs):
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as e:
            print(e)
            raise e

    def chat(self, system_prompt: str, user_prompt: str,
             temperature: float = 0, max_tokens: int = 1500) -> str:
        try:
            max_completion_tokens = self.budget_tokens + max_tokens if self.is_reasoning_model else max_tokens
            if self.engine == "google/gemma-3-4b-it:free":
                messages = [
                    {"role": "user", "content": system_prompt + "\n" + user_prompt}
                ]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            if self.provider == "vertex-gpt-oss":
                response = self.completion_with_backoff(
                    model=self.engine,
                    temperature=1,
                    top_p=0.95,
                    max_tokens=max_completion_tokens,
                    reasoning_effort="high",
                    messages=messages
                )
            else:
                response = self.completion_with_backoff(
                    model=self.engine,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                    messages=messages
                )
            result = ""
            for choice in response.choices:
                result += choice.message.content
            return result
        except Exception as e:
            print(f"Error in chat: {e}")
            return ""
