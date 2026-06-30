import asyncio
import time
from contextlib import asynccontextmanager
from functools import partial

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from src.inference import InferenceEngine
from src.model import VALID_AA, MAX_SEQUENCE_LENGTH, MODEL_REGISTRY
from src.reference import build_reference_embeddings
from src.search import ProteinSearchIndex
from src.pipeline import aggregate_go_terms
from src.jobs import create_job, get_job, run_embed_job, run_annotate_job

# ---------------------------------------------------------------------------
# Shared state — loaded once at startup, reused across all requests.
# ---------------------------------------------------------------------------

engine: InferenceEngine | None = None
search_index: ProteinSearchIndex | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, search_index
    engine = InferenceEngine(model_size="small")
    embeddings, proteins = build_reference_embeddings(engine)
    search_index = ProteinSearchIndex(embeddings, proteins)
    yield
    engine = None
    search_index = None


app = FastAPI(
    title="ESM2 Inference Service",
    description="Protein sequence embedding and nearest-neighbor search via Meta's ESM2.",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class EmbedRequest(BaseModel):
    sequence: str
    layer: int = -1

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        v = v.upper().strip()
        if not v:
            raise ValueError("sequence must not be empty")
        if len(v) > MAX_SEQUENCE_LENGTH:
            raise ValueError(f"sequence length {len(v)} exceeds maximum {MAX_SEQUENCE_LENGTH}")
        invalid = set(v) - VALID_AA
        if invalid:
            raise ValueError(f"invalid amino acid characters: {sorted(invalid)}")
        return v


class EmbedResponse(BaseModel):
    embedding: list[float]
    dim: int
    model: str = "small"


class BatchEmbedRequest(BaseModel):
    sequences: list[str]
    layer: int = -1

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("sequences list must not be empty")
        if len(v) > 64:
            raise ValueError("batch size must not exceed 64 sequences")
        return [s.upper().strip() for s in v]


class BatchEmbedResponse(BaseModel):
    embeddings: list[list[float]]
    count: int
    dim: int
    model: str = "small"


class SearchRequest(BaseModel):
    sequence: str
    k: int = 5

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        v = v.upper().strip()
        if not v:
            raise ValueError("sequence must not be empty")
        if len(v) > MAX_SEQUENCE_LENGTH:
            raise ValueError(f"sequence length {len(v)} exceeds maximum {MAX_SEQUENCE_LENGTH}")
        invalid = set(v) - VALID_AA
        if invalid:
            raise ValueError(f"invalid amino acid characters: {sorted(invalid)}")
        return v

    @field_validator("k")
    @classmethod
    def validate_k(cls, v: int) -> int:
        if not 1 <= v <= 20:
            raise ValueError("k must be between 1 and 20")
        return v


class SearchResponse(BaseModel):
    query_length: int
    results: list[dict]


class AnnotateRequest(BaseModel):
    sequence: str
    k: int = 5

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        v = v.upper().strip()
        if not v:
            raise ValueError("sequence must not be empty")
        if len(v) > MAX_SEQUENCE_LENGTH:
            raise ValueError(f"sequence length {len(v)} exceeds maximum {MAX_SEQUENCE_LENGTH}")
        invalid = set(v) - VALID_AA
        if invalid:
            raise ValueError(f"invalid amino acid characters: {sorted(invalid)}")
        return v

    @field_validator("k")
    @classmethod
    def validate_k(cls, v: int) -> int:
        if not 1 <= v <= 20:
            raise ValueError("k must be between 1 and 20")
        return v


class AnnotateResponse(BaseModel):
    query_length: int
    predicted_go_terms: list[dict]
    nearest_neighbors: list[dict]
    inference_time_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": engine is not None,
        "index_ready": search_index is not None,
        "active_model": engine.model.model_size if engine else None,
    }


@app.get("/models")
async def list_models():
    active = engine.model.model_size if engine else None
    return [
        {
            "id": model_id,
            "active": model_id == active,
            **config,
        }
        for model_id, config in MODEL_REGISTRY.items()
    ]


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    loop = asyncio.get_running_loop()
    try:
        embeddings = await loop.run_in_executor(
            None, partial(engine.embed, [request.sequence], request.layer)
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    embedding = embeddings[0]
    return EmbedResponse(embedding=embedding, dim=len(embedding))


@app.post("/embed/batch", response_model=BatchEmbedResponse)
async def embed_batch(request: BatchEmbedRequest):
    loop = asyncio.get_running_loop()
    try:
        embeddings = await loop.run_in_executor(
            None, partial(engine.embed, request.sequences, request.layer)
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return BatchEmbedResponse(
        embeddings=embeddings,
        count=len(embeddings),
        dim=len(embeddings[0]),
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    loop = asyncio.get_running_loop()
    try:
        embeddings = await loop.run_in_executor(
            None, partial(engine.embed, [request.sequence])
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    results = search_index.query(embeddings[0], k=request.k)
    return SearchResponse(query_length=len(request.sequence), results=results)


@app.post("/pipeline/annotate", response_model=AnnotateResponse)
async def annotate(request: AnnotateRequest):
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()

    try:
        embeddings = await loop.run_in_executor(
            None, partial(engine.embed, [request.sequence])
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    neighbors = search_index.query(embeddings[0], k=request.k)
    predicted_go_terms = aggregate_go_terms(neighbors)

    inference_time_ms = round((time.perf_counter() - t0) * 1000, 2)

    return AnnotateResponse(
        query_length=len(request.sequence),
        predicted_go_terms=predicted_go_terms,
        nearest_neighbors=neighbors,
        inference_time_ms=inference_time_ms,
    )


# ---------------------------------------------------------------------------
# Async job endpoints
# ---------------------------------------------------------------------------

@app.post("/jobs/embed", status_code=202)
async def submit_embed_job(request: BatchEmbedRequest, background_tasks: BackgroundTasks):
    job = create_job("embed")
    background_tasks.add_task(
        run_embed_job, job.job_id, request.sequences, request.layer, engine
    )
    return {"job_id": job.job_id, "status": job.status}


@app.post("/jobs/annotate", status_code=202)
async def submit_annotate_job(request: AnnotateRequest, background_tasks: BackgroundTasks):
    job = create_job("annotate")
    background_tasks.add_task(
        run_annotate_job, job.job_id, request.sequence, request.k, engine, search_index
    )
    return {"job_id": job.job_id, "status": job.status}


@app.get("/jobs/{job_id}")
async def poll_job(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job '{job_id}' not found")

    response: dict = {
        "job_id": job.job_id,
        "type": job.type,
        "status": job.status,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
        "duration_ms": job.duration_ms(),
    }
    if job.status == "complete":
        response["result"] = job.result
    elif job.status == "failed":
        response["error"] = job.error

    return response
