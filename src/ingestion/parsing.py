from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import pdfplumber
from pypdf import PdfReader


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PageText:
    """Extracted text for a single PDF page."""
    page: int
    content: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PageTable:
    """Extracted table for a single PDF page."""
    page: int
    table_index: int        # 0-based index when multiple tables appear on one page
    headers: list[str]
    rows: list[list[str]]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PDFMetadata:
    """Metadata extracted from the PDF file."""
    file_path: str
    file_size_bytes: int
    total_pages: int
    pdf_version: str | None = None
    creator: str | None = None
    producer: str | None = None
    creation_date: str | None = None
    modification_date: str | None = None
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class PDFParser:
    """
    Parse a PDF file and separately extract text, tables, and metadata.

    Parameters
    ----------
    pdf_path : str | Path
        Path to the PDF file.
    """

    # Mapping from pypdf metadata keys to friendly names
    _META_MAP: dict[str, str] = {
        "/Title":        "title",
        "/Author":       "author",
        "/Subject":      "subject",
        "/Keywords":     "keywords",
        "/Creator":      "creator",
        "/Producer":     "producer",
        "/CreationDate": "creation_date",
        "/ModDate":      "modification_date",
    }

    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = Path(pdf_path).resolve()
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_metadata(self) -> PDFMetadata:
        """
        Extract document metadata from the PDF.

        Returns
        -------
        PDFMetadata
            Dataclass containing all available metadata fields.
        """
        reader = PdfReader(self.pdf_path)
        raw_meta = reader.metadata or {}

        known: dict[str, Any] = {}
        extra: dict[str, Any] = {}

        for key, value in raw_meta.items():
            friendly = self._META_MAP.get(key)
            if friendly:
                known[friendly] = str(value) if value is not None else None
            else:
                extra[key.lstrip("/")] = str(value)

        # pdfinfo-style extras from pdfplumber
        pdf_version = None
        with pdfplumber.open(self.pdf_path) as pdf:
            pdf_version = pdf.doc.pdf_version if hasattr(pdf.doc, "pdf_version") else None

        return PDFMetadata(
            file_path=str(self.pdf_path),
            file_size_bytes=self.pdf_path.stat().st_size,
            total_pages=len(reader.pages),
            pdf_version=str(pdf_version) if pdf_version else None,
            creator=known.get("creator"),
            producer=known.get("producer"),
            creation_date=known.get("creation_date"),
            modification_date=known.get("modification_date"),
            title=known.get("title"),
            author=known.get("author"),
            subject=known.get("subject"),
            keywords=known.get("keywords"),
            extra=extra,
        )

    def extract_text(
        self,
        page_range: tuple[int, int] | None = None,
        strip_whitespace: bool = True,
        skip_empty: bool = True,
    ) -> list[PageText]:
        """
        Extract plain text from every page (or a page range).

        Parameters
        ----------
        page_range : (start, end) | None
            1-based, inclusive page range. None means all pages.
        strip_whitespace : bool
            Collapse excessive whitespace and strip leading/trailing space.
        skip_empty : bool
            Skip pages that yield no text after stripping.

        Returns
        -------
        list[PageText]
            One entry per non-empty page containing the page number and text.
        """
        results: list[PageText] = []

        with pdfplumber.open(self.pdf_path) as pdf:
            pages = self._select_pages(pdf.pages, page_range)
            for page_obj in pages:
                raw = page_obj.extract_text() or ""
                if strip_whitespace:
                    raw = re.sub(r"\n{3,}", "\n\n", raw)   # collapse multiple blank lines
                    raw = raw.strip()
                if skip_empty and not raw:
                    continue
                results.append(PageText(page=page_obj.page_number, content=raw))

        return results

    def extract_tables(
        self,
        page_range: tuple[int, int] | None = None,
        infer_headers: bool = True,
        clean_cells: bool = True,
    ) -> list[PageTable]:
        """
        Extract all tables from every page (or a page range).

        Parameters
        ----------
        page_range : (start, end) | None
            1-based, inclusive page range. None means all pages.
        infer_headers : bool
            Treat the first row of each table as headers.
        clean_cells : bool
            Replace None cells with empty strings and strip whitespace.

        Returns
        -------
        list[PageTable]
            One entry per table found, including page number and table index.
        """
        results: list[PageTable] = []

        with pdfplumber.open(self.pdf_path) as pdf:
            pages = self._select_pages(pdf.pages, page_range)
            for page_obj in pages:
                raw_tables = page_obj.extract_tables() or []
                for t_idx, raw_table in enumerate(raw_tables):
                    if not raw_table:
                        continue

                    cleaned = self._clean_table(raw_table, clean_cells)

                    if infer_headers and len(cleaned) > 0:
                        headers = [str(h).strip() for h in cleaned[0]]
                        rows = cleaned[1:]
                    else:
                        headers = []
                        rows = cleaned

                    results.append(PageTable(
                        page=page_obj.page_number,
                        table_index=t_idx,
                        headers=headers,
                        rows=rows,
                    ))

        return results

    def parse(
        self,
        page_range: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """
        Run all extractions in one call.

        Returns
        -------
        dict with keys:
            "metadata"  -> PDFMetadata.to_dict()
            "text"      -> list of PageText.to_dict()
            "tables"    -> list of PageTable.to_dict()
        """
        metadata = self.extract_metadata()
        text     = self.extract_text(page_range=page_range)
        tables   = self.extract_tables(page_range=page_range)

        return {
            "metadata": metadata.to_dict(),
            "text":     [t.to_dict() for t in text],
            "tables":   [t.to_dict() for t in tables],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_pages(pages, page_range: tuple[int, int] | None):
        """Slice pdfplumber page list by a 1-based, inclusive range."""
        if page_range is None:
            return pages
        start, end = page_range
        # pdfplumber pages are 0-indexed internally
        return pages[start - 1 : end]

    @staticmethod
    def _clean_table(
        raw: list[list[Any]],
        clean_cells: bool,
    ) -> list[list[str]]:
        """Normalise a raw table: replace None, strip whitespace."""
        if not clean_cells:
            return raw
        cleaned = []
        for row in raw:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    # Collapse internal newlines (common in merged cells)
                    cell_str = re.sub(r"\s*\n\s*", " ", str(cell)).strip()
                    cleaned_row.append(cell_str)
            cleaned.append(cleaned_row)
        return cleaned