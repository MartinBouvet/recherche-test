"""
Base de connaissances construite à partir des Eurocodes fournis par l'utilisateur.
Extrait automatiquement le texte des PDF/DOCX, le segmente, génère des embeddings
et fournit des extraits pertinents pour enrichir les prompts du LLM.
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - dépendance optionnelle
    PdfReader = None

try:
    from docx import Document
except ImportError:  # pragma: no cover - dépendance optionnelle
    Document = None


class KnowledgeBase:
    """
    Indexe les documents Eurocodes et fournit un moteur de recherche sémantique léger.
    """

    def __init__(
        self,
        documents_dir: str,
        embedding_model,
        cache_dir: str = "knowledge_cache",
        chunk_size: int = 1600,
        chunk_overlap: int = 200,
    ) -> None:
        self.documents_dir = Path(documents_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embeddings_path = self.cache_dir / "eurocodes_embeddings.npy"
        self.metadata_path = self.cache_dir / "eurocodes_metadata.pkl"

        self.enabled = self.documents_dir.exists()
        self.metadata: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

        if not self.enabled:
            print(f"⚠️  Dossier Eurocodes introuvable: {self.documents_dir}")
            return

        if PdfReader is None:
            print("⚠️  La bibliothèque pypdf est absente. Installez-la pour indexer les Eurocodes.")
            self.enabled = False
            return

        print(f"Chargement de la base Eurocodes depuis {self.documents_dir}...")
        self._load_or_build_index()

    # ------------------------------------------------------------------ #
    # Construction / chargement de l'index
    # ------------------------------------------------------------------ #
    def _load_or_build_index(self) -> None:
        if self.embeddings_path.exists() and self.metadata_path.exists():
            try:
                self.embeddings = np.load(self.embeddings_path)
                with open(self.metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)
                print(f"✅ {len(self.metadata)} extraits Eurocodes chargés depuis le cache.")
                return
            except Exception as exc:
                print(f"⚠️  Impossible de charger le cache Eurocodes ({exc}). Reconstruction...")

        self._build_index_from_documents()

    def _build_index_from_documents(self) -> None:
        files: List[Path] = []
        files.extend(self.documents_dir.rglob("*.pdf"))
        if Document is not None:
            files.extend(self.documents_dir.rglob("*.docx"))

        if not files:
            print("⚠️  Aucun document Eurocode trouvé.")
            self.enabled = False
            return

        all_texts: List[str] = []
        self.metadata = []

        for file_path in files:
            text = self._extract_text(file_path)
            if not text:
                continue

            chunks = self._chunk_text(text)
            for idx, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                self.metadata.append(
                    {
                        "text": chunk,
                        "source": str(file_path.relative_to(self.documents_dir)),
                        "chunk_index": idx,
                    }
                )
                all_texts.append(chunk)

        if not all_texts:
            print("⚠️  Aucun texte exploitable n'a été extrait des Eurocodes.")
            self.enabled = False
            return

        print(f"Génération des embeddings Eurocodes pour {len(all_texts)} extraits...")
        self.embeddings = self.embedding_model.encode(
            all_texts,
            batch_size=16,
            show_progress_bar=True,
        )
        np.save(self.embeddings_path, self.embeddings)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print("✅ Base Eurocodes indexée et mise en cache.")

    # ------------------------------------------------------------------ #
    # Extraction & chunking
    # ------------------------------------------------------------------ #
    def _extract_text(self, file_path: Path) -> str:
        if file_path.suffix.lower() == ".pdf":
            return self._extract_text_from_pdf(file_path)
        if file_path.suffix.lower() == ".docx" and Document is not None:
            return self._extract_text_from_docx(file_path)
        return ""

    def _extract_text_from_pdf(self, file_path: Path) -> str:
        if PdfReader is None:
            return ""
        try:
            reader = PdfReader(str(file_path))
            pages_text = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                pages_text.append(page_text)
            return "\n".join(pages_text)
        except Exception as exc:
            print(f"⚠️  Impossible d'extraire {file_path.name}: {exc}")
            return ""

    def _extract_text_from_docx(self, file_path: Path) -> str:
        if Document is None:
            return ""

        try:
            doc = Document(str(file_path))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as exc:
            print(f"⚠️  Impossible d'extraire {file_path.name}: {exc}")
            return ""

    def _chunk_text(self, text: str) -> List[str]:
        cleaned = re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()
        if not cleaned:
            return []

        chunks: List[str] = []
        start = 0
        step = max(self.chunk_size - self.chunk_overlap, 200)
        while start < len(cleaned):
            end = min(len(cleaned), start + self.chunk_size)
            chunk = cleaned[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(cleaned):
                break
            start += step
        return chunks

    # ------------------------------------------------------------------ #
    # Recherche & génération de contexte
    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.enabled or self.embeddings is None or not self.metadata:
            return []

        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(
                {
                    "text": self.metadata[idx]["text"],
                    "source": self.metadata[idx]["source"],
                    "score": float(similarities[idx]),
                }
            )
        return results

    def build_context_block(
        self, query: str, top_k: int = 5, max_chars: int = 2500
    ) -> str:
        """
        Retourne un bloc de contexte textuel prêt à être injecté dans un prompt LLM.
        """
        snippets = self.search(query, top_k=top_k)
        if not snippets:
            return ""

        context_lines = [
            "Connaissances Eurocodes (ne cite que ce qui est pertinent) :"
        ]
        current_len = 0
        for snippet in snippets:
            formatted = f"- Source: {snippet['source']}\n  {snippet['text'].strip()}"
            if current_len + len(formatted) > max_chars:
                break
            context_lines.append(formatted)
            current_len += len(formatted)

        return "\n".join(context_lines)


