# app/llm_client.py

import os
from groq import Groq

# Client will read GROQ_API_KEY from env 
client = Groq()  

MODEL_NAME = "llama-3.1-8b-instant"  # fast + cheap 


def _chat_once(system_prompt: str, user_prompt: str) -> str:
    """Single non-streaming chat completion, returns text content only."""
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    return completion.choices[0].message.content or ""
    

def generate_answer(prompt: str) -> str:
    system = (
    "You are a research assistant. Use ONLY the provided context (paper titles and excerpts). "
    "Mention specific paper titles when relevant and do not hallucinate new papers."
)

    return _chat_once(system_prompt=system, user_prompt=prompt)


def critique_question(prompt: str) -> str:
    system = (
        "You are an expert research mentor. "
        "Critique the question and suggest how to make it more precise. "
        "Be honest but polite, in 2–4 bullet points."
    )
    return _chat_once(system_prompt=system, user_prompt=prompt)
