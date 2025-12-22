from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os

from backend.rag import RAGEngine

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    
    allow_credentials=True,
    allow_methods=["*"],     
    allow_headers=["*"],
)

rag = RAGEngine()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class QueryRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(request: QueryRequest):
    retrieved = rag.retrieve(request.question, k=5)

    context = "\n\n".join([c["text"] for c in retrieved])
    sources = list({c["source"] for c in retrieved})

    system_prompt = (
        "You are a retrieval-augmented assistant. "
        "Answer the user's question using ONLY the provided context. "
        "Format the answer in clear Markdown. "
        "If the answer is not present, say 'I don't know.'"
    )

    user_prompt = f"""
Context:
{context}

Question:
{request.question}
"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=512
    )

    return {
        "answer": completion.choices[0].message.content.strip(),
        "sources": sources,
        "chunks_used": len(retrieved)
    }
