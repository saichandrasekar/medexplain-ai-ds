from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.routers import predict, documents, rag, agent
from backend.config import ENV

from backend.services.embedding_service import _load_model
from backend.routers.predict import _load_artifacts

from backend.services.rag_service import  _init_chromadb

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup

    print("Loading - XGBoost Model")
    _load_artifacts()
    print("Loaded - XGBoost Model")
    
    _load_model()
    print("BGE model loaded and ready")

    print("Loading - Vector Store")
    _init_chromadb()
    print("Loaded - Vector Store")

    yield  # App runs here
    
    # Shutdown (add cleanup here if needed)
    print("Shutting down")

app = FastAPI(title="MedExplain Service", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(predict.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(rag.router, prefix="/api/v1")
app.include_router(agent.router, prefix="/api/v1")

@app.get('/')
def root():
    return {
        "service": app.title,
        "version": app.version,
        "env": ENV,
    }

@app.get("/health")
def health():
    return {"status": "ok"}
