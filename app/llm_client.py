from groq import AsyncGroq  

client = AsyncGroq()
MODEL_NAME = "llama-3.1-8b-instant"

<<<<<<< HEAD
# Client will read GROQ_API_KEY from env 
client = Groq()  

MODEL_NAME = "llama-3.1-8b-instant"  # fast + cheap 


def _chat_once(system_prompt: str, user_prompt: str) -> str:
    """Single non-streaming chat completion, returns text content only."""
    completion = client.chat.completions.create(
=======
async def _chat_once(system_prompt: str, user_prompt: str) -> str:
    completion = await client.chat.completions.create(
>>>>>>> 7017d2e (Refactor RAG pipeline and update README documentation)
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    return completion.choices[0].message.content or ""




async def generate_answer(prompt: str) -> str:
    system = (
    "You are a research assistant specialized in machine learning and NLP.\n"
    "Answer the question primarily by synthesizing the provided context.\n"
    "Do NOT give generic textbook definitions unless explicitly asked.\n"
    "When relevant, compare or contrast the retrieved papers.\n"
    "Attribute claims using paper titles when they appear in the context.\n"
    "If the context is insufficient, say so explicitly.\n"
    "Do NOT invent papers or methods."
)
    return await _chat_once(system_prompt=system, user_prompt=prompt)


