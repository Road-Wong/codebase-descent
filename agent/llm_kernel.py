import os
from openai import OpenAI
from typing import List, Dict, Any, Optional


class LLMKernel:
    """
    LLM API wrapper for code generation.
    Implements the gradient descent operator in code space.
    """
    
    def __init__(self, api_key: str, base_url: str, model: str = "mimo-v2.5"):
        """
        Initialize LLM client.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for API endpoint
            model: Model name to use
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        
    def generate_code(
        self,
        task_description: str,
        current_code: Optional[str] = None,
        error_feedback: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        Generate or improve code based on task and feedback.
        This is the core update operator: c_{t+1} = A(c_t, feedback, history)
        
        Args:
            task_description: The coding task specification
            current_code: Current code state c_t
            error_feedback: Error messages from tests (symbolic gradient)
            history: Previous iterations (for momentum)
            temperature: Sampling temperature (learning rate analog)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated code string
        """
        messages = self._build_messages(
            task_description,
            current_code,
            error_feedback,
            history
        )
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                stream=False,
                extra_body={
                    "thinking": {"type": "disabled"}
                }
            )
            
            response = completion.choices[0].message.content
            code = self._extract_code(response)
            return code
            
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {str(e)}")
    
    def _build_messages(
        self,
        task_description: str,
        current_code: Optional[str],
        error_feedback: Optional[str],
        history: Optional[List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """Build message list for LLM prompt."""
        messages = [
            {
                "role": "system",
                "content": "You are an expert programmer. Generate clean, correct Python code based on the task description and feedback. Only output the code, no explanations."
            }
        ]
        
        # Add task description
        task_msg = f"Task: {task_description}\n\n"
        
        # Add current code if exists
        if current_code:
            task_msg += f"Current code:\n```python\n{current_code}\n```\n\n"
        
        # Add error feedback (symbolic gradient)
        if error_feedback:
            task_msg += f"Test failures and errors:\n{error_feedback}\n\n"
            task_msg += "Please fix the code to pass all tests. Output only the corrected code."
        else:
            task_msg += "Please implement the code to solve this task."
        
        messages.append({
            "role": "user",
            "content": task_msg
        })
        
        # Add history for context (momentum)
        if history:
            for item in history[-5:]:  # Keep last 5 iterations
                if 'code' in item:
                    messages.append({
                        "role": "assistant",
                        "content": f"```python\n{item['code']}\n```"
                    })
                if 'feedback' in item:
                    messages.append({
                        "role": "user",
                        "content": f"Feedback: {item['feedback']}"
                    })
        
        return messages
    
    def _extract_code(self, response: str) -> str:
        """Extract code from LLM response."""
        # Try to extract from code blocks
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
        
        # Return full response if no code blocks found
        return response.strip()
    
    def generate_with_thinking(
        self,
        task_description: str,
        current_code: Optional[str] = None,
        error_feedback: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Dict[str, str]:
        """
        Generate code with chain-of-thought reasoning.
        Implements gradient preconditioning.
        
        Returns:
            Dictionary with 'thinking' and 'code' keys
        """
        messages = [
            {
                "role": "system",
                "content": "You are an expert programmer. First analyze the problem, then generate the code."
            },
            {
                "role": "user",
                "content": f"Task: {task_description}\n\n"
                          f"Current code:\n```python\n{current_code}\n```\n\n" if current_code else "" +
                          f"Errors:\n{error_feedback}\n\n" if error_feedback else "" +
                          "First explain your approach, then provide the code."
            }
        ]
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                stream=False,
                extra_body={
                    "thinking": {"type": "enabled"}
                }
            )
            
            response = completion.choices[0].message.content
            
            # Split thinking and code
            parts = response.split("```python")
            thinking = parts[0].strip() if len(parts) > 1 else ""
            code = self._extract_code(response)
            
            return {
                'thinking': thinking,
                'code': code
            }
            
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {str(e)}")
