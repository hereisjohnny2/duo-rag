"""Gera o executável Windows standalone (PyInstaller) do Duo RAG.

Empacota o launcher (que sobe o Streamlit e abre o navegador) junto com
todos os assets do Streamlit/Chroma/Ollama necessários. O resultado fica em
``dist/DuoRAG/`` (modo --onedir, mais confiável que --onefile para apps
Streamlit).

Uso:
    python scripts\\build_exe.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / "dist" / "DuoRAG"

README_USUARIO = """\
Duo RAG - Assistente de documentos (uso local)
================================================

Como usar
---------
1. Instale o Ollama (https://ollama.com/download) e, com ele aberto, rode no
   terminal (uma única vez):
       ollama pull llama3.1
       ollama pull nomic-embed-text

2. (Opcional, só se for indexar PDFs escaneados) Instale o Tesseract OCR:
       https://github.com/UB-Mannheim/tesseract/wiki
   e marque o pacote de idioma "Portuguese" durante a instalação.

3. Dê duplo clique em "DuoRAG.exe". O programa vai abrir sozinho no seu
   navegador (endereço http://localhost:8501).

4. Na aba "Adicionar documentos", envie os PDFs das escrituras. Na aba
   "Consultar", pergunte o que quiser sobre os documentos enviados.

Onde ficam os dados
--------------------
Os documentos e o índice de busca ficam salvos na pasta "data" ao lado do
DuoRAG.exe. Faça backup dessa pasta periodicamente. Não é necessário
conexão com a internet para usar o programa (o Ollama e o Tesseract também
rodam localmente na sua máquina).
"""


def run_pyinstaller() -> None:
    launcher = ROOT / "src" / "app" / "launcher.py"
    streamlit_app = ROOT / "src" / "app" / "streamlit_app.py"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--name",
        "DuoRAG",
        "--onedir",
        "--console",  # mantém uma janela de console visível com logs/erros
        "--add-data",
        f"{streamlit_app};.",
        "--collect-all",
        "streamlit",
        "--collect-all",
        "altair",
        "--collect-all",
        "pydeck",
        "--collect-all",
        "chromadb",
        "--collect-all",
        "ollama",
        "--collect-all",
        "pytesseract",
        "--collect-all",
        "pymupdf",
        "--copy-metadata",
        "streamlit",
        "--hidden-import",
        "streamlit.web.cli",
        "--hidden-import",
        "src",
        "--hidden-import",
        "src.config",
        "--hidden-import",
        "src.ingestion",
        "--hidden-import",
        "src.ingestion.extract",
        "--hidden-import",
        "src.ingestion.ocr",
        "--hidden-import",
        "src.ingestion.chunk_and_index",
        "--hidden-import",
        "src.rag",
        "--hidden-import",
        "src.rag.retriever",
        "--hidden-import",
        "src.rag.generator",
        "--hidden-import",
        "src.rag.pipeline",
        "--hidden-import",
        "src.app",
        "--hidden-import",
        "src.app.streamlit_app",
        "--paths",
        str(ROOT),
        str(launcher),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def write_user_readme() -> None:
    (DIST_DIR / "README-USUARIO.txt").write_text(README_USUARIO, encoding="utf-8")


def main() -> None:
    run_pyinstaller()
    write_user_readme()
    print(f"\nBuild concluído. Pasta portátil gerada em: {DIST_DIR}")


if __name__ == "__main__":
    main()
