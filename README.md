# Duo RAG

Sistema de RAG (Retrieval-Augmented Generation) local para a firma de
arquitetura **Duo**, alimentado pelos documentos internos da firma
(inicialmente escrituras de imóveis, em PDF — nativos ou digitalizados).

Tudo roda **localmente** (LLM e embeddings via [Ollama](https://ollama.com)),
sem enviar documentos sensíveis para serviços externos.

## Arquitetura

```
data/raw/         PDFs originais (não versionados, sensíveis)
data/processed/   Texto extraído por documento (JSON, com metadados)
data/catalog/     Metadados estruturados por documento (cidade, endereço,
                  proprietários, valores, datas e matrícula)
data/chroma/      Vector store persistido (ChromaDB)

src/config.py         Configuração central (caminhos, modelos, chunking)
src/ingestion/        Extração de texto (nativo + OCR) e indexação no Chroma
src/rag/               Retrieval + geração de resposta (Ollama)
src/app/                Interface Streamlit (chat + upload de documentos)

scripts/ingest_all.py  Processa todos os PDFs pendentes em data/raw
```

## Pré-requisitos

1. **Python 3.10+**
2. **[Ollama](https://ollama.com/download)** instalado e rodando, com os
   modelos baixados:
   ```
   ollama pull llama3.1
   ollama pull nomic-embed-text
   ```
3. **Tesseract OCR** instalado (para páginas escaneadas), com o pacote de
   idioma português (`por`):
   - Windows: instalador em https://github.com/UB-Mannheim/tesseract/wiki
   - O app tenta detectar automaticamente o Tesseract em locais comuns
     (`%LOCALAPPDATA%\Tesseract-OCR`, `C:\Program Files\Tesseract-OCR`).
     Se mesmo assim ocorrer o erro `PermissionError: [WinError 5] Access is
     denied`, isso indica que o Windows não resolveu corretamente o
     `tesseract` via PATH. Defina a variável de ambiente
     `DUO_RAG_TESSERACT_CMD` com o caminho completo do executável, ex.:
     ```powershell
     $env:DUO_RAG_TESSERACT_CMD = "C:\Users\<seu_usuario>\AppData\Local\Tesseract-OCR\tesseract.exe"
     ```

## Instalação (desenvolvimento)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Uso

1. Coloque os PDFs em `data/raw/`.
2. Rode a extração + indexação de todos os documentos pendentes. Além do
   texto/OCR e dos embeddings, o comando extrai um catálogo estruturado de
   cada documento usando o Ollama:
   ```powershell
   python scripts\ingest_all.py
   ```
3. Suba a interface:
   ```powershell
   streamlit run src\app\streamlit_app.py
   ```
4. Pela interface você também pode adicionar novos documentos (upload),
   que são processados, catalogados e indexados automaticamente.

Perguntas de contagem ou agregação, como “quantos imóveis existem em Rio das
Ostras?”, usam o catálogo completo em vez de apenas os chunks mais similares.
Para contagens por cidade, a aplicação informa separadamente documentos
encontrados e endereços distintos, pois o mesmo imóvel pode ter mais de um
documento.

## Executável para usuários leigos (Windows)

Além da execução via Python, o app pode ser empacotado como um `.exe`
standalone (PyInstaller), para que usuários sem Python instalado consigam
rodar com duplo clique. Veja `scripts/build_exe.py` e
`README-USUARIO.txt` (gerado junto ao build) para instruções.

> Observação: mesmo empacotado em `.exe`, o **Ollama** e o **Tesseract OCR**
> continuam sendo pré-requisitos externos instalados no Windows do usuário
> — são serviços/binários de sistema, não bibliotecas Python, e por isso
> não são embutidos no executável.

## Roadmap para produção online (futuro)

- Trocar o Ollama local por um serviço de inferência hospedado (ou um
  servidor Ollama dedicado), mantendo a mesma interface `src/rag`.
- Mover o ChromaDB para um servidor dedicado (modo cliente-servidor) em
  vez do modo embutido em arquivo.
- Containerizar com Docker (ingestão + API + UI).
- Trocar/complementar a UI Streamlit por uma API (FastAPI) + frontend web,
  adicionando autenticação e controle de acesso por usuário.

## Primeiro passo para nuvem

Foi adicionada uma API FastAPI em `src/api/main.py` com:

- `GET /health` para health checks;
- `POST /query` para perguntas;
- `POST /documents` para upload assíncrono de PDFs;
- `GET /jobs/{job_id}` para acompanhar OCR, indexação e catalogação.

Quando `DUO_RAG_API_KEY` estiver definida, os endpoints protegidos exigem o
header `X-API-Key`. Para executar localmente:

```powershell
$env:DUO_RAG_API_KEY = "troque-esta-chave"
uvicorn src.api.main:app --reload
```

Também há um `Dockerfile` e `docker-compose.cloud.yml`. O container é um
adaptador inicial para piloto: os dados ainda ficam no volume Docker e os
jobs ficam em memória. Antes de produção, devem ser migrados para object
storage, PostgreSQL/pgvector e uma fila persistente, conforme o plano de
migração.
