"""
PyTorch Embeddings Module

Generates embeddings for defect texts using PyTorch and sentence transformers
for semantic similarity search in the RAG pipeline.
"""

import logging
from functools import lru_cache
from typing import Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .models import DefectDomain, DefectEmbedding, JiraDefect

logger = logging.getLogger(__name__)


class DefectEmbeddingGenerator:
    """
    Generates PyTorch embeddings for JIRA defects.
    
    Uses sentence-transformers for semantic encoding, enabling
    context-aware defect categorization and similarity search.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "auto",
    ) -> None:
        """
        Initialize the embedding generator.
        
        Args:
            model_name: HuggingFace model name for embeddings.
            device: Target device ('cuda', 'cpu', or 'auto').
        """
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.model = self._load_model()
        logger.info(f"Embedding model loaded on {self.device}")
    
    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device
    
    @lru_cache(maxsize=1)
    def _load_model(self) -> SentenceTransformer:
        """Load the sentence transformer model."""
        model = SentenceTransformer(self.model_name, device=self.device)
        return model
    
    def _prepare_defect_text(self, defect: JiraDefect) -> str:
        """
        Prepare defect text for embedding generation.
        
        Combines relevant fields into a single text representation
        for comprehensive semantic encoding.
        """
        parts = [
            f"Summary: {defect.summary}",
            f"Description: {defect.description}",
            f"Component: {defect.component}",
            f"Platform: {defect.platform}",
        ]
        
        if defect.error_log:
            # Truncate long error logs for embedding
            error_snippet = defect.error_log[:500]
            parts.append(f"Error: {error_snippet}")
        
        if defect.steps_to_reproduce:
            parts.append(f"Steps: {defect.steps_to_reproduce[:300]}")
        
        return " | ".join(parts)
    
    def generate_embedding(
        self,
        defect: JiraDefect,
    ) -> DefectEmbedding:
        """
        Generate embedding for a single defect.
        
        Args:
            defect: JIRA defect to embed.
            
        Returns:
            DefectEmbedding with vector representation.
        """
        text = self._prepare_defect_text(defect)
        
        with torch.no_grad():
            embedding = self.model.encode(
                text,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            embedding_list = embedding.cpu().numpy().tolist()
        
        # Preliminary domain classification based on keywords
        domain = self._classify_domain_from_text(text)
        
        return DefectEmbedding(
            defect_id=defect.id,
            embedding=embedding_list,
            domain=domain,
            metadata={
                "text_length": len(text),
                "platform": defect.platform,
                "component": defect.component,
            },
        )
    
    def generate_batch_embeddings(
        self,
        defects: list[JiraDefect],
        batch_size: int = 32,
    ) -> list[DefectEmbedding]:
        """
        Generate embeddings for multiple defects efficiently.
        
        Args:
            defects: List of JIRA defects.
            batch_size: Batch size for processing.
            
        Returns:
            List of DefectEmbedding objects.
        """
        texts = [self._prepare_defect_text(d) for d in defects]
        
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=True,
            )
            embeddings_np = embeddings.cpu().numpy()
        
        results = []
        for defect, emb in zip(defects, embeddings_np):
            domain = self._classify_domain_from_text(
                self._prepare_defect_text(defect)
            )
            results.append(
                DefectEmbedding(
                    defect_id=defect.id,
                    embedding=emb.tolist(),
                    domain=domain,
                    metadata={
                        "platform": defect.platform,
                        "component": defect.component,
                    },
                )
            )
        
        return results
    
    def _classify_domain_from_text(self, text: str) -> DefectDomain:
        """
        Preliminary domain classification based on keywords.
        
        This is refined by the ML classifier in trend analysis.
        """
        text_lower = text.lower()
        
        domain_keywords = {
            DefectDomain.MEMORY: [
                "memory", "allocation", "oom", "heap", "buffer", "leak",
                "hip_memory", "gpu memory", "vram",
            ],
            DefectDomain.INFRASTRUCTURE: [
                "setup", "infra", "network", "node", "cluster", "ssh",
                "connection", "environment", "provisioning",
            ],
            DefectDomain.SYSTEM_CALL: [
                "syscall", "system call", "kernel", "ioctl", "driver call",
                "api error", "undefined symbol",
            ],
            DefectDomain.PERFORMANCE: [
                "slow", "performance", "latency", "throughput", "bottleneck",
                "optimization", "benchmark", "regression",
            ],
            DefectDomain.FIRMWARE: [
                "firmware", "bios", "uefi", "smi", "ras", "boot",
            ],
            DefectDomain.DRIVER: [
                "driver", "amdgpu", "rocm", "hip", "kmd", "umd",
            ],
            DefectDomain.CONFIGURATION: [
                "config", "configuration", "setting", "parameter", "env var",
            ],
        }
        
        for domain, keywords in domain_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return domain
        
        return DefectDomain.UNKNOWN
    
    def compute_similarity(
        self,
        query_embedding: list[float],
        candidate_embeddings: list[list[float]],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        Compute cosine similarity between query and candidates.
        
        Args:
            query_embedding: Query vector.
            candidate_embeddings: List of candidate vectors.
            top_k: Number of top results to return.
            
        Returns:
            List of (index, similarity_score) tuples.
        """
        query = np.array(query_embedding)
        candidates = np.array(candidate_embeddings)
        
        # Normalize vectors
        query_norm = query / np.linalg.norm(query)
        candidates_norm = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # Cosine similarity
        similarities = np.dot(candidates_norm, query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
