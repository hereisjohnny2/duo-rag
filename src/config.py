"""Configuração central do Duo RAG.

Centraliza caminhos, parâmetros de chunking e nomes de modelos, para que o
resto do código (ingestão, RAG, UI, empacotamento) não tenha valores
espalhados/hardcoded.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _base_dir() -> Path:
    """Retorna o diretório base da aplicação.

    Quando empacotado com PyInstaller, os dados devem ficar ao lado do
    executável (não dentro do bundle temporário do PyInstaller), então
    usamos o diretório do executável nesse caso.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = _base_dir()

# --- Caminhos de dados ---
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma"
# Catálogo de metadados estruturados por documento (cidade, endereço,
# proprietários, valor, data, matrícula etc.), extraídos via LLM. Usado para
# responder perguntas de agregação/contagem que a busca semântica por chunk
# não resolve bem (ex.: "quantos imóveis existem em Rio das Ostras?").
CATALOG_DIR = DATA_DIR / "catalog"

for _dir in (RAW_DIR, PROCESSED_DIR, CHROMA_DIR, CATALOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --- Chroma ---
CHROMA_COLLECTION_NAME = "duo_documentos"

# --- OCR ---
OCR_LANGUAGE = os.environ.get("DUO_RAG_OCR_LANG", "por")


def _autodetect_tesseract_cmd() -> str | None:
    """Tenta localizar o executável do Tesseract em instalações comuns no
    Windows quando ele não está corretamente resolvível via PATH (causa
    comum do erro "PermissionError: [WinError 5] Access is denied" quando o
    Python tenta invocar apenas "tesseract" via subprocess).
    """
    candidates = [
        os.environ.get("LOCALAPPDATA", "") + r"\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


# Caminho customizado do binário do Tesseract. Pode ser definido explicitamente
# via env var DUO_RAG_TESSERACT_CMD; caso contrário, tentamos autodetectar em
# instalações comuns do Windows antes de recorrer ao PATH.
TESSERACT_CMD = os.environ.get("DUO_RAG_TESSERACT_CMD") or _autodetect_tesseract_cmd()
# Limite mínimo de caracteres "úteis" por página para considerar que ela já
# tem texto nativo extraível (abaixo disso, tratamos como scan e usamos OCR).
MIN_NATIVE_TEXT_CHARS = 20

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Ollama (usado quando LLM_PROVIDER/EMBEDDING_PROVIDER = "ollama") ---
OLLAMA_HOST = os.environ.get("DUO_RAG_OLLAMA_HOST", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.environ.get("DUO_RAG_LLM_MODEL", "llama3.1:8b")
OLLAMA_EMBEDDING_MODEL = os.environ.get(
    "DUO_RAG_EMBEDDING_MODEL", "nomic-embed-text"
)

# --- Provedor de LLM/embeddings ---
# "ollama": 100% local, exige Ollama rodando (recomendado apenas quando a
#   máquina tem RAM/CPU suficiente para o modelo, ex. 4GB+ para llama3.1:8b).
# "deepseek": usa a API do DeepSeek (compatível com a API da OpenAI) para o
#   LLM. Recomendado em instâncias de nuvem pequenas/baratas, onde não há
#   recursos para rodar um LLM localmente.
LLM_PROVIDER = os.environ.get("DUO_RAG_LLM_PROVIDER", "ollama").lower()
DEEPSEEK_API_KEY = os.environ.get("DUO_RAG_DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get(
    "DUO_RAG_DEEPSEEK_BASE_URL", "https://api.deepseek.com"
)
DEEPSEEK_MODEL = os.environ.get("DUO_RAG_DEEPSEEK_MODEL", "deepseek-chat")

# "ollama": embeddings via Ollama (modelo nomic-embed-text).
# "local": modelo de embeddings leve rodando via sentence-transformers, sem
#   depender de GPU/serviço externo. Usado como padrão quando o provedor de
#   LLM é uma API externa, pois o DeepSeek não oferece endpoint de
#   embeddings; manter os embeddings locais também evita expor o texto dos
#   documentos a uma segunda API.
EMBEDDING_PROVIDER = os.environ.get(
    "DUO_RAG_EMBEDDING_PROVIDER", "local" if LLM_PROVIDER == "deepseek" else "ollama"
).lower()
# Modelo multilíngue pequeno (~470MB, roda bem em CPU/pouca RAM) — necessário
# porque os documentos da Duo estão em português.
LOCAL_EMBEDDING_MODEL = os.environ.get(
    "DUO_RAG_LOCAL_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

API_KEY = os.environ.get("DUO_RAG_API_KEY", "")

# --- Retrieval ---
RETRIEVAL_TOP_K = 8

# --- Extração de metadados estruturados ---
# Quantidade máxima de caracteres do texto do documento enviados ao LLM para
# extração de metadados (cidade, endereço, proprietários etc.). As
# informações relevantes de escrituras costumam aparecer nas primeiras
# páginas, então um limite evita prompts muito longos/lentos.
METADATA_EXTRACT_CHAR_LIMIT = 8000
