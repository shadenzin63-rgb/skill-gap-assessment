import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .config import settings
from .llm import generate_report
from .models import SubmitRequest
from .questions import QUESTIONS
from .scoring import compute_scores

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")
limiter = Limiter(key_func=get_remote_address)
MAX_BODY_BYTES = 10_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.mongo = AsyncIOMotorClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)
    app.state.db = app.state.mongo[settings.mongodb_db]
    yield
    app.state.mongo.close()


app = FastAPI(title="Skill Gap Assessment", lifespan=lifespan, docs_url=None, redoc_url=None)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
        return JSONResponse({"detail": "Request too large"}, status_code=413)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/questions")
async def questions():
    return [{"id": q["id"], "text": q["text"]} for q in QUESTIONS]  # competency stays server-side


@app.post("/api/submit")
@limiter.limit("10/minute")
async def submit(request: Request, body: SubmitRequest):
    db = request.app.state.db
    now = datetime.now(timezone.utc)
    answers = [a.model_dump() for a in body.answers]

    try:
        sub = await db.submissions.insert_one(
            {"user_name": body.user_name, "answers": answers, "created_at": now}
        )
    except Exception:
        log.exception("DB insert failed")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

    scores, overall = compute_scores(body.answers)
    report = await generate_report(scores, overall)

    try:
        await db.reports.insert_one(
            {
                "submission_id": sub.inserted_id,
                "scores": scores,
                "report": report.model_dump(),
                "created_at": now,
            }
        )
    except Exception:
        log.exception("DB report insert failed")  # user still gets the report

    log.info("submission %s processed (generated_by=%s)", sub.inserted_id, report.generated_by)
    return {"submission_id": str(sub.inserted_id), "scores": scores, "report": report.model_dump()}
from pathlib import Path
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parent.parent


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")