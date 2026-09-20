"""API HTTP mínima para o Duo RAG.

Esta camada desacopla clientes web da implementação local do RAG. O endpoint
de upload usa processamento em background para que a requisição não fique
bloqueada durante OCR, embeddings e catalogação. Em produção, o estado dos
jobs deve ser movido para PostgreSQL/uma fila persistente; o dicionário em
memória é apenas o adaptador inicial para o piloto.
"""
from __future__ import annotations

import logging
import secrets
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src import config
from src.ingestion.chunk_and_index import index_processed_document
from src.ingestion.extract import extract_and_save
from src.ingestion.metadata_extractor import extract_and_save_metadata
from src.rag.pipeline import ask

logger = logging.getLogger(__name__)

app = FastAPI(title="Duo RAG API", version="0.1.0")
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="duo-ingest")
_jobs: dict[str, dict[str, str]] = {}
_jobs_lock = threading.Lock()


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    used_catalog: bool


def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Protege a API quando DUO_RAG_API_KEY estiver configurada."""
    expected = config.API_KEY
    if expected and (
        not x_api_key or not secrets.compare_digest(x_api_key, expected)
    ):
        raise HTTPException(status_code=401, detail="API key inválida")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(_require_api_key)])
def query(request: QueryRequest) -> QueryResponse:
    result = ask(request.question)
    return QueryResponse(
        answer=result.answer,
        used_catalog=result.used_catalog,
        sources=[
            {
                "source_file": source.source_file,
                "page_number": source.page_number,
                "used_ocr": source.used_ocr,
            }
            for source in result.sources
        ],
    )


def _set_job(job_id: str, **values: str) -> None:
    with _jobs_lock:
        _jobs.setdefault(job_id, {}).update(values)


def _process_upload(job_id: str, destination: Path) -> None:
    try:
        _set_job(job_id, status="extracting")
        processed_path = extract_and_save(destination, force=True)
        _set_job(job_id, status="indexing")
        index_processed_document(processed_path)
        _set_job(job_id, status="cataloging")
        extract_and_save_metadata(processed_path, force=True)
        _set_job(job_id, status="ready")
    except Exception:
        logger.exception("Falha no processamento do job %s", job_id)
        _set_job(job_id, status="failed")


@app.post("/documents", status_code=202, dependencies=[Depends(_require_api_key)])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict[str, str]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Envie um arquivo PDF")

    safe_name = Path(file.filename).name
    destination = config.RAW_DIR / safe_name
    destination.write_bytes(await file.read())
    job_id = str(uuid.uuid4())
    _set_job(job_id, status="queued", filename=safe_name)
    background_tasks.add_task(_executor.submit, _process_upload, job_id, destination)
    return {"job_id": job_id, "status": "queued", "filename": safe_name}


@app.get("/jobs/{job_id}", dependencies=[Depends(_require_api_key)])
def job_status(job_id: str) -> dict[str, str]:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    return {"job_id": job_id, **job}
