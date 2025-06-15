import os
import base64
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
from google.generativeai import types
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
# ---- Gemini Setup ----

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def describe_image(image_base64: str):
    try:
        image_bytes = base64.b64decode(image_base64)
        image = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        response = gemini_model.generate_content(
            ["Describe this image.", image]
        )
        return response.text
    except Exception as e:
        return f"[Image context could not be extracted: {e}]"

# ---- OpenAI Setup ----

client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://aipipe.org/openai/v1")
embed_model = "text-embedding-3-small"
llm_model = "gpt-4o-mini"  # or "gpt-3.5-turbo"

# ---- Load Embeddings/Chunks ----
data = np.load("chunks_embeddings.npz", allow_pickle=True)
texts = data["texts"]
sources = data["sources"]
chunk_ids = data["chunk_ids"]
types = data["types"]
embeddings = data["embeddings"]

# ---- FastAPI Setup ----
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domain(s) in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64-encoded image

class LinkOut(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkOut]

def embed_query(text: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=embed_model,
        input=text,
        encoding_format="float"
    )
    return np.array(resp.data[0].embedding)

def search(query_emb: np.ndarray, k=5):
    sims = embeddings @ query_emb / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb))
    topk_idx = sims.argsort()[-k:][::-1]
    return topk_idx, sims

def make_rag_prompt(question, selected_chunks):
    context = "\n\n".join(selected_chunks)
    prompt = (
        "You are a helpful teaching assistant for the IIT Madras Online Data Science course. "
        "A student has asked the following question. Use only the information from the provided context SAbelow to answer. "
        "\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )
    return prompt

@app.post("/api/", response_model=QueryResponse)
async def query_api(req: QueryRequest):
    # 1. Extract image context if provided
    question = req.question
    if req.image:
        image_context = describe_image(req.image)
        question = f"{req.question}\n\n[Image context: {image_context}]"
    # 2. Embed
    q_emb = embed_query(question)
    # 3. Retrieve top chunks
    topk_idx, sims = search(q_emb, k=5)
    selected_chunks = []
    links = []
    used_sources = set()
    for idx in topk_idx:
        content = texts[idx]
        src = sources[idx]
        if src not in used_sources:
            links.append({"url": src, "text": content[:120] + ("..." if len(content) > 120 else "")})
            used_sources.add(src)
        selected_chunks.append(content)
    # 4. RAG LLM synthesis
    prompt = make_rag_prompt(req.question, selected_chunks)
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "You are a helpful teaching assistant for the IIT Madras Online Data Science course. Answer as concisely as possible, only using the context provided."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=600,
    )
    answer = response.choices[0].message.content.strip()
    return {"answer": answer, "links": links}


if __name__ == "__main__":
    uvicorn.run("app:app",reload=True)