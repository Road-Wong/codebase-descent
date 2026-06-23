import os
import time
from openai import OpenAI
from typing import List, Dict, Any, Optional


class LLMKernel:
    """
    LLM API wrapper for code generation.
    Implements the gradient descent operator in code space.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "mimo-v2.5",
        timeout: float = 120.0,
        max_retries: int = 2,
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def generate_code(
        self,
        task_description: str,
        current_code: Optional[str] = None,
        error_feedback: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        raw: bool = False,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate or improve code based on task and feedback.

        Args:
            raw: If True, return the full LLM response without stripping
                 code blocks. Use this when the expected output contains
                 protocol markers (SEARCH/REPLACE) rather than pure code.
            system_prompt: Override the default system prompt.
        """
        messages = self._build_messages(task_description, current_code, error_feedback, history, system_prompt=system_prompt)
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    stream=False,
                    timeout=self.timeout,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                response = completion.choices[0].message.content
                if raw:
                    return response
                return self._extract_code(response)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(2 * (attempt + 1))  # backoff: 2s, 4s
                    continue
        raise RuntimeError(f"LLM API call failed after {self.max_retries+1} attempts: {str(last_error)}")

    def _build_messages(
        self,
        task_description: str,
        current_code: Optional[str],
        error_feedback: Optional[str],
        history: Optional[List[Dict[str, str]]],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        if system_prompt is None:
            system_prompt = "You are an expert programmer. Generate clean, correct Python code based on the task description and feedback. Only output the code, no explanations."
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        task_msg = f"Task: {task_description}\n\n"
        if current_code:
            task_msg += f"Current code:\n```python\n{current_code}\n```\n\n"
        if error_feedback:
            task_msg += f"Test failures and errors:\n{error_feedback}\n\n"
            task_msg += "Please fix the code to pass all tests. Output only the corrected code."
        else:
            task_msg += "Please implement the code to solve this task."
        messages.append({"role": "user", "content": task_msg})
        if history:
            for item in history[-5:]:
                if 'code' in item:
                    messages.append({"role": "assistant", "content": f"```python\n{item['code']}\n```"})
                if 'feedback' in item:
                    messages.append({"role": "user", "content": f"Feedback: {item['feedback']}"})
        return messages

    def _extract_code(self, response: str) -> str:
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        return response.strip()

    def generate_with_thinking(
        self,
        task_description: str,
        current_code: Optional[str] = None,
        error_feedback: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Dict[str, str]:
        """Generate code with chain-of-thought reasoning."""
        messages = [
            {"role": "system", "content": "You are an expert programmer. First analyze the problem, then generate the code."},
            {"role": "user", "content": f"Task: {task_description}\n\n" + (f"Current code:\n```python\n{current_code}\n```\n\n" if current_code else "") + (f"Errors:\n{error_feedback}\n\n" if error_feedback else "") + "First explain your approach, then provide the code."},
        ]
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    stream=False,
                    timeout=self.timeout,
                    extra_body={"thinking": {"type": "enabled"}},
                )
                response = completion.choices[0].message.content
                parts = response.split("```python")
                thinking = parts[0].strip() if len(parts) > 1 else ""
                code = self._extract_code(response)
                return {"thinking": thinking, "code": code}
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(2 * (attempt + 1))
                    continue
        raise RuntimeError(f"LLM API call failed after {self.max_retries+1} attempts: {str(last_error)}")
