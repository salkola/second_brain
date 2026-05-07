from __future__ import annotations

import io
import shutil
from pathlib import Path

from langchain_core.documents import Document
from PIL import Image

from brain.app_config import AppConfig
from brain.chunking import sanitize_doc_metadata, split_text
from brain.ingest.fs import sorted_files_under_roots

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".tiff",
    ".tif",
    ".heic",
}


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def iter_screenshot_files(roots: list[Path]) -> list[Path]:
    return sorted_files_under_roots(roots, _IMAGE_SUFFIXES)


def _ocr_image(path: Path, max_side: int = 2000) -> str:
    import pytesseract

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        buf.seek(0)
        return pytesseract.image_to_string(Image.open(buf)) or ""


def iter_screenshot_documents(cfg: AppConfig, path: Path) -> list[Document]:
    if not tesseract_available():
        return []
    source = str(path.resolve())
    try:
        text = _ocr_image(path)
    except Exception:
        return []
    if not text.strip():
        return []
    chunks = split_text(
        text,
        source=source,
        source_type="screenshot",
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        base_metadata={"source": source, "source_type": "screenshot"},
    )
    return [
        Document(page_content=c.text, metadata=sanitize_doc_metadata(c.metadata)) for c in chunks
    ]
