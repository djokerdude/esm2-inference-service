import asyncio
import time
import uuid
from dataclasses import dataclass, field
from functools import partial

from src.pipeline import aggregate_go_terms

# In-memory job store. Keyed by job_id.
# Production replacement: Redis with TTL so completed jobs expire automatically.
_store: dict[str, "Job"] = {}


@dataclass
class Job:
    job_id: str
    type: str                        # "embed" | "annotate"
    status: str                      # "queued" | "running" | "complete" | "failed"
    created_at: float
    completed_at: float | None = None
    result: dict | None = None
    error: str | None = None

    def duration_ms(self) -> float | None:
        if self.completed_at is None:
            return None
        return round((self.completed_at - self.created_at) * 1000, 2)


def create_job(job_type: str) -> "Job":
    job = Job(
        job_id=str(uuid.uuid4()),
        type=job_type,
        status="queued",
        created_at=time.time(),
    )
    _store[job.job_id] = job
    return job


def get_job(job_id: str) -> "Job | None":
    return _store.get(job_id)


# ---------------------------------------------------------------------------
# Background task functions
# These run after the HTTP response is sent. Each updates the job store
# in-place so polling clients see live status changes.
# ---------------------------------------------------------------------------

async def run_embed_job(job_id: str, sequences: list[str], layer: int, engine) -> None:
    job = _store[job_id]
    job.status = "running"
    loop = asyncio.get_running_loop()
    try:
        embeddings = await loop.run_in_executor(
            None, partial(engine.embed, sequences, layer)
        )
        job.result = {
            "embeddings": embeddings,
            "count": len(embeddings),
            "dim": len(embeddings[0]),
        }
        job.status = "complete"
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
    finally:
        job.completed_at = time.time()


async def run_annotate_job(job_id: str, sequence: str, k: int, engine, search_index) -> None:
    job = _store[job_id]
    job.status = "running"
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    try:
        embeddings = await loop.run_in_executor(
            None, partial(engine.embed, [sequence])
        )
        neighbors = search_index.query(embeddings[0], k=k)
        job.result = {
            "query_length": len(sequence),
            "predicted_go_terms": aggregate_go_terms(neighbors),
            "nearest_neighbors": neighbors,
            "inference_time_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        job.status = "complete"
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
    finally:
        job.completed_at = time.time()
