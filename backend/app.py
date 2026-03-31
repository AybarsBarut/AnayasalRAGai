from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

# Import our RAG system
from backend.rag import AnayasaRAG

app = FastAPI(title="Anayasa AI")

# CORS setup for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the RAG system
# This might take some time to start if it needs to download sentence-transformers the first time
rag_system = None

def get_rag_system():
    global rag_system
    if rag_system is None:
        try:
            rag_system = AnayasaRAG(model_name="llama3")
        except Exception as e:
            print(f"Error initializing RAG system: {e}")
            raise e
    return rag_system

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

@app.on_event("startup")
async def startup_event():
    # Pre-load the RAG system when the server starts
    get_rag_system()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        rag = get_rag_system()
        answer = rag.interact(request.query)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount the frontend directory to serve static files
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    print(f"Warning: Frontend directory not found at {frontend_dir}")
