"""
Vector Store for RAG Pipeline

Provides vector storage and retrieval for semantic similarity search
using FAISS for efficient nearest neighbor search.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from .models import DefectDomain, DefectEmbedding, JiraDefect, TriagePrediction

logger = logging.getLogger(__name__)


class DefectVectorStore:
    """
    Vector store for defect embeddings using FAISS.
    
    Supports efficient similarity search for RAG retrieval
    across large-scale defect datasets.
    """
    
    def __init__(
        self,
        dimension: int = 384,  # all-MiniLM-L6-v2 dimension
        index_type: str = "flat",
    ) -> None:
        """
        Initialize the vector store.
        
        Args:
            dimension: Embedding dimension size.
            index_type: FAISS index type ('flat', 'ivf', 'hnsw').
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = self._create_index()
        self.metadata: dict[int, dict] = {}  # Maps index position to metadata
        self.defect_ids: list[str] = []  # Maps position to defect_id
        self._id_to_position: dict[str, int] = {}
        
        logger.info(f"Vector store initialized with {index_type} index")
    
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on index_type."""
        if self.index_type == "flat":
            # Exact search - best for smaller datasets
            return faiss.IndexFlatIP(self.dimension)  # Inner product (cosine)
        elif self.index_type == "ivf":
            # Approximate search with inverted file
            quantizer = faiss.IndexFlatIP(self.dimension)
            return faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif self.index_type == "hnsw":
            # Hierarchical Navigable Small World graph
            return faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def add_embedding(self, embedding: DefectEmbedding) -> None:
        """
        Add a single embedding to the store.
        
        Args:
            embedding: DefectEmbedding to add.
        """
        if embedding.defect_id in self._id_to_position:
            logger.warning(f"Defect {embedding.defect_id} already exists, skipping")
            return
        
        vector = np.array([embedding.embedding], dtype=np.float32)
        # Normalize for cosine similarity
        faiss.normalize_L2(vector)
        
        position = len(self.defect_ids)
        self.index.add(vector)
        
        self.defect_ids.append(embedding.defect_id)
        self._id_to_position[embedding.defect_id] = position
        self.metadata[position] = {
            "defect_id": embedding.defect_id,
            "domain": embedding.domain.value,
            **embedding.metadata,
        }
    
    def add_embeddings(self, embeddings: list[DefectEmbedding]) -> None:
        """
        Add multiple embeddings efficiently.
        
        Args:
            embeddings: List of embeddings to add.
        """
        new_embeddings = [
            e for e in embeddings 
            if e.defect_id not in self._id_to_position
        ]
        
        if not new_embeddings:
            logger.warning("No new embeddings to add")
            return
        
        vectors = np.array(
            [e.embedding for e in new_embeddings], 
            dtype=np.float32,
        )
        faiss.normalize_L2(vectors)
        
        start_position = len(self.defect_ids)
        self.index.add(vectors)
        
        for i, emb in enumerate(new_embeddings):
            position = start_position + i
            self.defect_ids.append(emb.defect_id)
            self._id_to_position[emb.defect_id] = position
            self.metadata[position] = {
                "defect_id": emb.defect_id,
                "domain": emb.domain.value,
                **emb.metadata,
            }
        
        logger.info(f"Added {len(new_embeddings)} embeddings to store")
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        domain_filter: Optional[DefectDomain] = None,
    ) -> list[dict]:
        """
        Search for similar defects.
        
        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.
            domain_filter: Optional domain to filter results.
            
        Returns:
            List of matching results with metadata and scores.
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
        
        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)
        
        # Search more if filtering
        search_k = top_k * 3 if domain_filter else top_k
        distances, indices = self.index.search(query, min(search_k, self.index.ntotal))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            meta = self.metadata.get(int(idx), {})
            
            # Apply domain filter if specified
            if domain_filter and meta.get("domain") != domain_filter.value:
                continue
            
            results.append({
                "defect_id": self.defect_ids[idx],
                "similarity_score": float(dist),
                "metadata": meta,
            })
            
            if len(results) >= top_k:
                break
        
        return results
    
    def get_domain_statistics(self) -> dict[str, int]:
        """Get count of defects per domain."""
        stats: dict[str, int] = {}
        for meta in self.metadata.values():
            domain = meta.get("domain", "unknown")
            stats[domain] = stats.get(domain, 0) + 1
        return stats
    
    def save(self, path: Path) -> None:
        """Save index and metadata to disk."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save metadata
        with open(path / "metadata.json", "w") as f:
            json.dump({
                "defect_ids": self.defect_ids,
                "metadata": {str(k): v for k, v in self.metadata.items()},
                "dimension": self.dimension,
                "index_type": self.index_type,
            }, f, indent=2)
        
        logger.info(f"Vector store saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "DefectVectorStore":
        """Load index and metadata from disk."""
        with open(path / "metadata.json") as f:
            data = json.load(f)
        
        store = cls(
            dimension=data["dimension"],
            index_type=data["index_type"],
        )
        store.index = faiss.read_index(str(path / "index.faiss"))
        store.defect_ids = data["defect_ids"]
        store.metadata = {int(k): v for k, v in data["metadata"].items()}
        store._id_to_position = {
            did: i for i, did in enumerate(store.defect_ids)
        }
        
        logger.info(f"Loaded vector store with {store.index.ntotal} vectors")
        return store
