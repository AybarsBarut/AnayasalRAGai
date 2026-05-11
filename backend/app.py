from contextlib import asynccontextmanager
import logging
import os
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import get_settings
from backend.exceptions import ConstitutionalError, error_payload
from backend.logging_config import configure_logging
from backend.schemas import ChatRequest, ChatResponse, HealthResponse
from backend.security import SecurityMiddleware, sanitize_query, verify_api_key

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

LEGAL_DISCLAIMER = (
    "\n\n---\n"
    "**Yasal Uyarı:** Bu yanıt yasal tavsiye değildir. "
    "Resmî ve bağlayıcı bilgi için güncel mevzuat ve yetkili profesyonel kaynaklar kontrol edilmelidir."
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if settings.eager_load_rag:
        get_rag_system()
    else:
        logger.info("rag_eager_load_skipped")
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Türkiye Cumhuriyeti Anayasası verisi üzerinde çalışan RAG tabanlı soru-cevap API'si. "
        "Swagger/OpenAPI dokümantasyonu `/docs` ve `/openapi.json` üzerinden erişilebilir."
    ),
    lifespan=lifespan,
)

app.add_middleware(SecurityMiddleware, settings=settings)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allowed_origins),
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the RAG system
# This might take some time to start if it needs to download sentence-transformers the first time
rag_system = None


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid4()))
    request.state.request_id = request_id
    started_at = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request_failed method=%s path=%s request_id=%s",
            request.method,
            request.url.path,
            request_id,
        )
        raise

    duration_ms = (time.perf_counter() - started_at) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_completed method=%s path=%s status=%s duration_ms=%.2f request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        request_id,
    )
    return response


@app.exception_handler(ConstitutionalError)
async def constitutional_exception_handler(request: Request, exc: ConstitutionalError):
    request_id = getattr(request.state, "request_id", None)
    logger.warning(
        "application_error code=%s path=%s request_id=%s details=%s",
        exc.code,
        request.url.path,
        request_id,
        exc.details,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(
            code=exc.code,
            message=exc.message,
            details=exc.details,
            request_id=request_id,
        ),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", None)
    logger.warning(
        "request_validation_error path=%s request_id=%s errors=%s",
        request.url.path,
        request_id,
        exc.errors(),
    )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error_payload(
            code="request_validation_error",
            message="İstek gövdesi doğrulanamadı.",
            details=exc.errors(),
            request_id=request_id,
        ),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(
            code="http_error",
            message=str(exc.detail),
            request_id=request_id,
        ),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", None)
    logger.exception(
        "unhandled_error path=%s request_id=%s",
        request.url.path,
        request_id,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_payload(
            code="internal_server_error",
            message="Beklenmeyen bir sunucu hatası oluştu.",
            request_id=request_id,
        ),
    )


def get_rag_system():
    global rag_system
    if rag_system is None:
        try:
            from backend.rag import AnayasaRAG

            rag_system = AnayasaRAG(model_name="llama3")
        except Exception as e:
            logger.exception("rag_initialization_failed")
            raise ConstitutionalError("RAG sistemi başlatılamadı.") from e
    return rag_system


def append_legal_disclaimer(answer: str) -> str:
    normalized = answer.lower()
    if "yasal tavsiye değildir" in normalized or "yasal tavsiye degildir" in normalized:
        return answer
    return f"{answer}{LEGAL_DISCLAIMER}"


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        rag_loaded=rag_system is not None,
    )


@app.get("/api/health/models", tags=["Health"])
async def model_health():
    rag_loaded = rag_system is not None
    model_status = "ok" if rag_loaded else "not_loaded"
    return {
        "status": "ok",
        "models": {
            "llm": {"name": "llama3", "status": model_status},
            "embedding": {"name": "BAAI/bge-m3", "status": model_status},
        },
    }


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(chat_request: ChatRequest, request: Request):
    """
    Kullanıcı sorusunu temizler, güvenlik denetimlerinden geçirir ve Anayasa RAG motoruyla yanıtlar.
    """
    await verify_api_key(request, settings)
    query = sanitize_query(chat_request.query, settings)
    rag = get_rag_system()
    answer = rag.interact(query)

    if not answer or not answer.strip():
        raise ConstitutionalError("RAG sistemi boş yanıt döndürdü.")

    request_id = getattr(request.state, "request_id", str(uuid4()))
    logger.info("chat_completed request_id=%s query_length=%s", request_id, len(query))
    return ChatResponse(answer=append_legal_disclaimer(answer), request_id=request_id)

# Mount the frontend directory to serve static files
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    logger.warning("frontend_directory_not_found path=%s", frontend_dir)
