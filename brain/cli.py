from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.prompt import Prompt

from brain import __version__
from brain.app_config import AppConfig, load_config
from brain.errors import ConfigError
from brain.indexer import run_index
from brain.ingest.apple_mail import discover_envelope_indices
from brain.ingest.screenshots import tesseract_available
from brain.rag import build_agent, run_chat_turn
from brain.tools.builtin import ToolContext
from brain.vectorstore import get_embeddings

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

# Reduce HF/transformers terminal noise (can override in env).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _load_app_config() -> AppConfig:
    try:
        return load_config()
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


_HTTP_VERBOSE_LOGGERS = frozenset(
    {"httpx", "httpcore", "urllib3", "huggingface_hub", "sentence_transformers", "transformers"},
)


def _setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    # Embedding / HF Hub stacks spam INFO on every run; keep them quiet unless log_level is DEBUG.
    if numeric > logging.DEBUG:
        for name in _HTTP_VERBOSE_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)


@app.command()
def doctor() -> None:
    """Check Ollama, embeddings, Chroma dir; warn if optional OCR/Mail pieces are missing."""
    _setup_logging("INFO")
    cfg = _load_app_config()
    ok = True

    def check_required(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        sym = "[green]OK[/green]" if passed else "[red]FAIL[/red]"
        console.print(f"{sym} [bold]{name}[/bold]: {detail}")
        if not passed:
            ok = False

    def check_optional(name: str, passed: bool, detail: str) -> None:
        sym = "[green]OK[/green]" if passed else "[yellow]WARN[/yellow]"
        console.print(f"{sym} [bold]{name}[/bold]: {detail}")

    # Ollama
    url = cfg.ollama_base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        models = [m.get("name", "") for m in data.get("models", [])]
        check_required("Ollama", True, f"reachable; models: {', '.join(models[:5]) or '(none)'}")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        check_required("Ollama", False, str(e))

    # Embeddings
    try:
        emb = get_embeddings(cfg.embedding_model_name)
        _ = emb.embed_query("ping")
        check_required("Embeddings", True, cfg.embedding_model_name)
    except Exception as e:
        check_required("Embeddings", False, repr(e))

    # Chroma dir
    try:
        cfg.chroma_persist_directory.mkdir(parents=True, exist_ok=True)
        probe = cfg.chroma_persist_directory / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        check_required("Chroma dir", True, str(cfg.chroma_persist_directory))
    except OSError as e:
        check_required("Chroma dir", False, str(e))

    # Optional: screenshot OCR
    tess_ok = tesseract_available()
    tess_detail = (
        "on PATH (screenshot OCR)"
        if tess_ok
        else "not on PATH (optional; needed for screenshot OCR)"
    )
    check_optional("Tesseract", tess_ok, tess_detail)

    # Optional: mail ingest needs Mail.app data and macOS layout
    indices = discover_envelope_indices(cfg.mail_library)
    mail_detail = (
        f"{len(indices)} Envelope Index file(s); first: {indices[0]}"
        if indices
        else f"no Envelope Index under {cfg.mail_library} (optional; needed for mail ingest)"
    )
    check_optional("Apple Mail", bool(indices), mail_detail)

    console.print(f"\n[dim]brain {__version__}[/dim]")
    if not ok:
        raise typer.Exit(code=1)


@app.command()
def index(
    full: bool = typer.Option(False, "--full", help="Rebuild index state and re-embed all files"),
) -> None:
    """Index configured paths into Chroma."""
    cfg = _load_app_config()
    _setup_logging(cfg.log_level)
    console.print("[bold]Indexing…[/bold]")
    counts = run_index(cfg, full=full)
    console.print_json(data=counts)


@app.command()
def chat() -> None:
    """Interactive RAG chat in the terminal."""
    cfg = _load_app_config()
    _setup_logging(cfg.log_level)
    ctx = ToolContext()
    agent_holder: list[Any] = []

    def get_agent() -> Any:
        if not agent_holder:
            agent_holder.append(build_agent(cfg, ctx))
        return agent_holder[0]

    console.print(
        "[dim]Commands: /quit, /sources (last retrieval paths).[/dim]",
    )
    while True:
        try:
            line = Prompt.ask("\n[bold]You[/bold]")
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye.")
            break
        text = line.strip()
        if not text:
            continue
        if text in ("/quit", "/exit", "/q"):
            break
        if text == "/sources":
            if ctx.last_sources:
                for s in ctx.last_sources:
                    console.print(f"- {s}")
            else:
                console.print("(no retrieval yet)")
            continue
        with console.status("[bold green]Thinking…"):
            try:
                reply = run_chat_turn(text, cfg=cfg, get_agent=get_agent)
            except Exception as e:
                console.print(f"[red]{e!r}[/red]")
                continue
        console.print(Markdown(reply))


def main() -> None:
    """Entry point for the `brain` console script."""
    app()


if __name__ == "__main__":
    main()
