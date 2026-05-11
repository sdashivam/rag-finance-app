"""
Document chunking and FAISS index management.

Provides section-aware text chunking with metadata preservation
and FAISS vector index construction for semantic retrieval.
"""

import re
import pickle
import numpy as np
import faiss, torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SectionAwareChunker:
    """Chunks text while preserving section context and metadata.

    Splits documents into semantic units with attached metadata
    for downstream retrieval and answer aggregation.

    Attributes:
        splitter: RecursiveCharacterTextSplitter configured for financial documents

    Args:
        config: Optional config dict with chunk_size and chunk_overlap.
                Defaults: chunk_size=1000, chunk_overlap=200
    """
    def __init__(self, config: dict = None):
        chunk_size = 1000
        chunk_overlap = 200
        if config:
            chunk_size = config.get('chunk_size', chunk_size)
            chunk_overlap = config.get('chunk_overlap', chunk_overlap)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk(self, parsed_data: dict) -> list[dict]:
        """
        Takes parsed data and produces chunks with section-aware metadata.
        Each chunk is a dictionary with:
        - "text": the chunked text content
        - "metadata": a dictionary containing:
        - "source": the original file path
        - "page": the page number
        - "section": the section name (if detected)
        - "type": "text" or "table"
        """
        chunks = []
        file_path = parsed_data["metadata"]["file_path"]
        

        # 1. Process Text Sections
        for item in parsed_data["text"]:
            page_num = item.get("page", 0)  # 0-based internal page index
            pdf_page = item.get("pdf_page")  # Visible PDF page number (may be None)
            content = item["content"]

            # Heuristic for section detection: split by lines that look like headers
            # (e.g., "SECTION 1: BALANCE SHEET" or lines in ALL CAPS)
            parts = re.split(r'(\n[A-Z\s]{5,}\n)', content)

            current_section = "General"
            for part in parts:
                if re.match(r'\n[A-Z\s]{5,}\n', part):
                    current_section = part.strip()
                    continue

                if not part.strip():
                    continue

                split_texts = self.splitter.split_text(part)
                for text in split_texts:
                    chunk_meta = {
                        "source": file_path,
                        "page": page_num,  # Internal 0-based index for FAISS/BM25
                        "section": current_section,
                        "type": "text"
                    }
                    # Add visible PDF page number if available
                    if pdf_page is not None:
                        chunk_meta["pdf_page"] = pdf_page

                    chunks.append({
                        "text": text,
                        "metadata": chunk_meta
                    })

        # 2. Process Tables
        for table in parsed_data["tables"]:
            page_num = table.get("page", 0)
            pdf_page = table.get("pdf_page")

            header_str = " | ".join(table["headers"])
            rows_str = "\n".join([" | ".join(row) for row in table["rows"]])
            # Use pdf_page for display if available, otherwise fallback to page
            display_page = pdf_page if pdf_page is not None else page_num
            table_content = f"Table (Page {display_page}):\n{header_str}\n{'-' * len(header_str)}\n{rows_str}"

            chunk_meta = {
                "source": file_path,
                "page": page_num,  # Internal 0-based index
                "section": "Table",
                "type": "table"
            }
            if pdf_page is not None:
                chunk_meta["pdf_page"] = pdf_page

            chunks.append({
                "text": table_content,
                "metadata": chunk_meta
            })

        return chunks

class FAISSIndexManager:
    """Handles embedding generation and FAISS indexing.

    Args:
        config: Optional config dict with model_name and device settings.
                device can be "auto" (cuda if available), "cuda", or "cpu".
    """
    def __init__(self, config: dict = None):
        model_name = "all-MiniLM-L6-v2"
        device = "cpu"

        if config:
            model_name = config.get('embedding_model', config.get('retriever_model', model_name))
            device_setting = config.get('device', 'auto')

            if device_setting == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            elif device_setting in ("cuda", "cpu"):
                device = device_setting
            else:
                device = "cpu"

        self.model = SentenceTransformer(model_name, device=device)
        self.index = None
        self.metadata = []

    def build_index(self, chunks: list[dict]):
        """
        Builds a FAISS index from the provided chunks.
            Each chunk is expected to be a dictionary with "text" and "metadata" keys.
            The method generates embeddings for the text and builds a FAISS index in memory.
            Metadata is stored in a list for later retrieval during querying.
            The index is built using L2 distance (IndexFlatL2) for efficient similarity search.
            The embeddings are converted to float32 before being added to the index, as required by FAISS.
        """
        self.metadata = []
        for chunk in chunks:
            text = str(chunk.get("text", "")).strip()
            if not text:
                continue
            self.metadata.append({
                "text": text,
                "metadata": chunk.get("metadata", {}) or {},
            })

        if not self.metadata:
            raise ValueError("No non-empty chunks were produced; refusing to build an empty FAISS index.")

        texts = [item["text"] for item in self.metadata]
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except TypeError:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, a_min=1e-12, a_max=None)

        if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
            raise ValueError(
                f"Unexpected embedding shape {embeddings.shape}; expected ({len(texts)}, dimension)."
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("Embedding matrix contains NaN or infinite values.")
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype("float32"))
        print("FAISS index built successfully.")

    def save(self, output_path: str, index_filename: str = "index.faiss", metadata_filename: str = "metadata.pkl"):
        """
        Saves the FAISS index and associated metadata to disk.
            The FAISS index is saved as a binary file, while the metadata is serialized using pickle
                and saved as a separate file. The method ensures that the output directory exists before saving.
                The index is saved using the faiss.write_index function, and the metadata is saved using pickle.dump.
                A confirmation message is printed upon successful saving of the index and metadata.
        """
        if self.index is None or not self.metadata:
            raise ValueError("Cannot save FAISS artifacts before building a non-empty index.")

        path = Path(output_path)
        path.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(path / index_filename))
        with open(path / metadata_filename, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Index and metadata saved to {output_path}")

    @staticmethod
    def inspect_saved(
        output_path: str,
        index_filename: str = "index.faiss",
        metadata_filename: str = "metadata.pkl",
    ) -> dict:
        """
        Returns a validation report for persisted FAISS artifacts.
        This is intentionally lightweight so benchmark/main can decide whether
        to reuse an existing index or rebuild it.
        """
        path = Path(output_path)
        index_path = path / index_filename
        metadata_path = path / metadata_filename
        report = {
            "index_path": str(index_path),
            "metadata_path": str(metadata_path),
            "index_exists": index_path.exists(),
            "metadata_exists": metadata_path.exists(),
            "valid": False,
            "errors": [],
            "faiss_ntotal": 0,
            "faiss_dimension": 0,
            "metadata_count": 0,
            "non_empty_text_count": 0,
            "metadata_with_fields_count": 0,
        }

        if not report["index_exists"]:
            report["errors"].append("FAISS index file is missing.")
        if not report["metadata_exists"]:
            report["errors"].append("FAISS metadata file is missing.")
        if report["errors"]:
            return report

        try:
            index = faiss.read_index(str(index_path))
            report["faiss_ntotal"] = int(index.ntotal)
            report["faiss_dimension"] = int(index.d)
        except Exception as exc:
            report["errors"].append(f"Could not read FAISS index: {exc}")
            return report

        try:
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
        except Exception as exc:
            report["errors"].append(f"Could not read metadata pickle: {exc}")
            return report

        if not isinstance(metadata, list):
            report["errors"].append(f"Metadata must be a list, got {type(metadata).__name__}.")
            return report

        report["metadata_count"] = len(metadata)
        report["non_empty_text_count"] = sum(
            1 for item in metadata
            if isinstance(item, dict) and str(item.get("text", "")).strip()
        )
        report["metadata_with_fields_count"] = sum(
            1 for item in metadata
            if isinstance(item, dict) and isinstance(item.get("metadata"), dict) and item["metadata"]
        )

        if report["faiss_ntotal"] != report["metadata_count"]:
            report["errors"].append(
                f"FAISS vectors ({report['faiss_ntotal']}) do not match metadata rows "
                f"({report['metadata_count']})."
            )
        if report["faiss_ntotal"] <= 0:
            report["errors"].append("FAISS index has no vectors.")
        if report["non_empty_text_count"] <= 0:
            report["errors"].append("Metadata contains no non-empty chunk text.")

        report["valid"] = not report["errors"]
        return report

    @staticmethod
    def debug_chunk_metadata(chunks: list[dict], sample_count: int = 5) -> dict:
        """
        Debug utility to inspect chunk metadata and validate page number consistency.

        Parameters
        ----------
        chunks : list[dict]
            List of chunk dictionaries with "text" and "metadata" keys
        sample_count : int
            Number of sample chunks to inspect

        Returns
        -------
        dict
            Debug report with page number statistics and any inconsistencies
        """
        if not chunks:
            return {"error": "No chunks provided"}

        report = {
            "total_chunks": len(chunks),
            "samples": [],
            "page_stats": {"min": None, "max": None, "unique_values": set()},
            "pdf_page_stats": {"min": None, "max": None, "has_values": False},
            "inconsistencies": [],
        }

        internal_pages = set()
        pdf_pages = set()

        for i, chunk in enumerate(chunks[:sample_count]):
            meta = chunk.get("metadata", {})
            page = meta.get("page")
            pdf_page = meta.get("pdf_page")

            report["samples"].append({
                "index": i,
                "page": page,
                "pdf_page": pdf_page,
                "section": meta.get("section"),
                "type": meta.get("type"),
                "text_preview": chunk.get("text", "")[:100],
            })

            if page is not None:
                internal_pages.add(page)
            if pdf_page is not None:
                pdf_pages.add(pdf_page)

        # Calculate stats
        if internal_pages:
            report["page_stats"]["min"] = min(internal_pages)
            report["page_stats"]["max"] = max(internal_pages)
            report["page_stats"]["unique_values"] = len(internal_pages)

        if pdf_pages:
            report["pdf_page_stats"]["min"] = min(pdf_pages)
            report["pdf_page_stats"]["max"] = max(pdf_pages)
            report["pdf_page_stats"]["has_values"] = True

        # Check for inconsistencies
        for i, chunk in enumerate(chunks):
            meta = chunk.get("metadata", {})
            page = meta.get("page")
            pdf_page = meta.get("pdf_page")

            # If pdf_page is set, it should be different from page (unless offset is 0)
            if page is not None and pdf_page is not None and page == pdf_page:
                report["inconsistencies"].append(
                    f"Chunk {i}: page={page} and pdf_page={pdf_page} are identical - "
                    "possible that page_offset was not applied or PDF has no page labels"
                )

        # Convert set to list for JSON serialization
        report["page_stats"]["unique_values"] = list(internal_pages)
        report["inconsistencies"] = report["inconsistencies"][:10]  # Limit to first 10

        return report
