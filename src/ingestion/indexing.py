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

    def build_index(self, chunks: list[dict]): # This method builds the index in memory
        self.metadata = [{"text": c["text"], "metadata": c["metadata"]} for c in chunks]
        texts = [item["text"] for item in self.metadata]
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype("float32"))
        print("FAISS index built successfully.")

    def save(self, output_path: str, index_filename: str = "index.faiss"): # This method saves the index to disk
        path = Path(output_path)
        path.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(path / index_filename))
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Index and metadata saved to {output_path}")