# SPDX-License-Identifier: Apache-2.0
"""
Engine abstraction for oMLX inference.

Provides multiple engine implementations:
- BatchedEngine: Continuous batching for multiple concurrent users
- EmbeddingEngine: Batch embedding generation using mlx-embeddings
- RerankerEngine: Document reranking using SequenceClassification models

Also re-exports core engine components for backwards compatibility.
"""

# Re-export from parent engine.py for backwards compatibility
from ..engine_core import AsyncEngineCore, EngineConfig, EngineCore
from .base import BaseEngine, BaseNonStreamingEngine, GenerationOutput
from .batched import BatchedEngine
from .embedding import EmbeddingEngine
from .reranker import RerankerEngine

__all__ = [
    "BaseEngine",
    "BaseNonStreamingEngine",
    "GenerationOutput",
    "BatchedEngine",
    "EmbeddingEngine",
    "RerankerEngine",
    # Core engine components
    "EngineCore",
    "AsyncEngineCore",
    "EngineConfig",
]
