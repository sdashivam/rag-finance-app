import re
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SectionAwareChunker:
    """
    Chunks text while preserving section context and metadata.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
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
            page_num = item["page"]
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
                    chunks.append({
                        "text": text,
                        "metadata": {
                            "source": file_path,
                            "page": page_num,
                            "section": current_section,
                            "type": "text"
                        }
                    })

        # 2. Process Tables
        for table in parsed_data["tables"]:
            header_str = " | ".join(table["headers"])
            rows_str = "\n".join([" | ".join(row) for row in table["rows"]])
            table_content = f"Table (Page {table['page']}):\n{header_str}\n{'-' * len(header_str)}\n{rows_str}"
            
            chunks.append({
                "text": table_content,
                "metadata": {
                    "source": file_path,
                    "page": table["page"],
                    "section": "Table",
                    "type": "table"
                }
            })

        return chunks

class FAISSIndexManager:
    """
    Handles embedding generation and FAISS indexing.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
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
        self.metadata = [{"text": c["text"], "metadata": c["metadata"]} for c in chunks]
        texts = [item["text"] for item in self.metadata]
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
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
        path = Path(output_path)
        path.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(path / index_filename))
        with open(path / metadata_filename, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Index and metadata saved to {output_path}")