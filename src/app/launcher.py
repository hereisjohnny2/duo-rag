"""Launcher usado tanto em desenvolvimento quanto no executável empacotado
(PyInstaller). Sobe o servidor Streamlit programaticamente e abre o
navegador padrão do usuário automaticamente no endereço local, para que
usuários leigos só precisem dar duplo clique no `.exe`.
"""
from __future__ import annotations

import sys
import threading
import time
import webbrowser
from pathlib import Path

PORT = 8501


def _resolve_app_script() -> Path:
    """Localiza o script principal do Streamlit, em dev ou dentro do bundle."""
    if getattr(sys, "frozen", False):
        # PyInstaller extrai os arquivos de dados para sys._MEIPASS em tempo
        # de execução (modo --onefile) ou para a pasta do executável (--onedir).
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        return base / "streamlit_app.py"
    return Path(__file__).resolve().parent / "streamlit_app.py"


def _open_browser_when_ready(url: str) -> None:
    time.sleep(3)
    webbrowser.open(url)


def main() -> None:
    app_script = _resolve_app_script()
    url = f"http://localhost:{PORT}"

    # Quando empacotado com PyInstaller, o Streamlit não consegue detectar
    # que foi "instalado normalmente" (checa se __file__ contém
    # "site-packages"), então assume erroneamente developmentMode=True, o
    # que conflita com --server.port. Forçamos explicitamente via env var.
    import os

    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"

    threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()

    sys.argv = [
        "streamlit",
        "run",
        str(app_script),
        "--server.port",
        str(PORT),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--server.fileWatcherType",
        "none",
    ]

    from streamlit.web import cli as stcli

    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
