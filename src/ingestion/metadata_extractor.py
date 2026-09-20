"""Extração de metadados estruturados por documento usando o LLM local.

Para cada documento processado (``data/processed/<nome>.json``), pede ao
LLM (via Ollama) para extrair campos estruturados — cidade, endereço,
proprietários, valor, data, matrícula, tipo de documento — e salva o
resultado em ``data/catalog/<nome>.json``.

Esse catálogo é usado pelo pipeline de RAG para responder perguntas de
agregação/contagem (ex.: "quantos imóveis existem em Rio das Ostras?"),
que a busca semântica por chunk não resolve bem, pois só enxerga alguns
trechos de cada vez em vez do conjunto completo de documentos.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import ollama

from src import config

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """Você é um assistente que extrai dados estruturados de documentos jurídicos/cartorários (escrituras, certidões, instituições de condomínio) de uma firma de arquitetura.

Leia o texto abaixo (extraído de um documento; pode conter erros de OCR) e responda APENAS com um JSON válido, sem nenhum texto adicional, com exatamente estas chaves:

{
  "tipo_documento": "string curta (ex: Escritura de Compra e Venda, Certidão, Instituição de Condomínio)",
  "cidade": "cidade onde fica o imóvel, ou null se não encontrado",
  "endereco": "endereço/descrição do imóvel (rua, lote, quadra, loteamento), ou null",
  "proprietarios": ["lista de nomes de proprietários/compradores mencionados"],
  "valor": "valor da transação em texto (ex: R$ 150.000,00), ou null",
  "data": "data do documento/escritura (ex: 26/12/2018), ou null",
  "matricula": "número de matrícula do imóvel, ou null"
}

Se alguma informação não estiver clara no texto, use null (ou lista vazia para proprietarios). Não invente dados que não estejam no texto.

TEXTO DO DOCUMENTO:
"""


@dataclass
class DocumentMetadata:
    source_file: str
    tipo_documento: str | None
    cidade: str | None
    endereco: str | None
    proprietarios: list[str]
    valor: str | None
    data: str | None
    matricula: str | None
    file_hash: str
    extracted_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def _catalog_path_for(stem: str) -> Path:
    return config.CATALOG_DIR / f"{stem}.json"


def is_already_extracted(stem: str, file_hash: str) -> bool:
    """Verifica se o catálogo desse documento já existe e está atualizado."""
    path = _catalog_path_for(stem)
    if not path.exists():
        return False
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if existing.get("file_hash") != file_hash:
        return False
    # Não preserve um resultado vazio causado por uma resposta interrompida
    # do modelo; ele deve ser tentado novamente na próxima ingestão.
    fields = (
        existing.get("tipo_documento"),
        existing.get("cidade"),
        existing.get("endereco"),
        existing.get("valor"),
        existing.get("data"),
        existing.get("matricula"),
    )
    return any(fields) or bool(existing.get("proprietarios"))


def _full_text(doc: dict) -> str:
    text = "\n\n".join(page["text"] for page in doc["pages"])
    return text[: config.METADATA_EXTRACT_CHAR_LIMIT]


def _parse_llm_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        # Remove blocos de código markdown (```json ... ```), caso o
        # modelo ignore a instrução de responder só com JSON puro.
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


def extract_metadata_for_document(doc: dict) -> DocumentMetadata:
    """Chama o LLM local para extrair os campos estruturados de um documento."""
    source_file = doc["source_file"]
    text = _full_text(doc)

    response = ollama.chat(
        model=config.OLLAMA_LLM_MODEL,
        messages=[{"role": "user", "content": _EXTRACTION_PROMPT + text}],
        format="json",
    )
    raw_content = response["message"]["content"]

    try:
        parsed = _parse_llm_json(raw_content)
    except (json.JSONDecodeError, IndexError):
        logger.warning(
            "Não foi possível interpretar o JSON de metadados de %s; "
            "usando valores vazios",
            source_file,
        )
        parsed = {}

    proprietarios = parsed.get("proprietarios") or []
    if isinstance(proprietarios, str):
        proprietarios = [proprietarios]

    return DocumentMetadata(
        source_file=source_file,
        tipo_documento=parsed.get("tipo_documento"),
        cidade=parsed.get("cidade"),
        endereco=parsed.get("endereco"),
        proprietarios=proprietarios,
        valor=parsed.get("valor"),
        data=parsed.get("data"),
        matricula=parsed.get("matricula"),
        file_hash=doc.get("file_hash", ""),
        extracted_at=datetime.now(timezone.utc).isoformat(),
    )


def extract_and_save_metadata(json_path: Path, force: bool = False) -> Path:
    """Extrai e salva os metadados estruturados de um documento processado.

    Pula a extração (mantendo o catálogo existente) se o documento não
    mudou desde a última extração, a menos que ``force=True``.
    """
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    stem = json_path.stem
    out_path = _catalog_path_for(stem)

    if not force and is_already_extracted(stem, doc.get("file_hash", "")):
        logger.info(
            "Pulando metadados de %s (já extraídos e sem alterações)",
            doc["source_file"],
        )
        return out_path

    logger.info("Extraindo metadados estruturados de %s", doc["source_file"])
    metadata = extract_metadata_for_document(doc)
    out_path.write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


def extract_all_metadata(force: bool = False) -> int:
    """Extrai metadados estruturados de todos os documentos processados."""
    count = 0
    for json_path in sorted(config.PROCESSED_DIR.glob("*.json")):
        try:
            extract_and_save_metadata(json_path, force=force)
        except (ollama.ResponseError, ConnectionError, TimeoutError, OSError) as exc:
            # Um erro do Ollama em um documento não deve impedir a ingestão
            # dos demais. O arquivo pode ser reprocessado na próxima execução.
            logger.error(
                "Falha ao extrair metadados de %s: %s",
                json_path.name,
                exc,
            )
            continue
        count += 1
    return count
