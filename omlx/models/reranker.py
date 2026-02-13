# SPDX-License-Identifier: Apache-2.0
"""
MLX Reranker Model wrapper.

This module provides a wrapper for document reranking using SequenceClassification
models on Apple's MLX framework.

Supports:
- ModernBertForSequenceClassification (via mlx-embeddings)
- XLMRobertaForSequenceClassification (omlx native implementation)
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

from ..model_discovery import SUPPORTED_RERANKER_ARCHITECTURES

logger = logging.getLogger(__name__)


@dataclass
class RerankOutput:
    """Output from rerank operation."""

    scores: list[float]
    """Relevance scores for each document (0 to 1)."""

    indices: list[int]
    """Document indices sorted by score (descending)."""

    total_tokens: int
    """Total number of tokens processed."""


class MLXRerankerModel:
    """
    Wrapper around mlx-embeddings for document reranking.

    This class provides a unified interface for loading and running
    SequenceClassification models for reranking on Apple's MLX framework.

    Supported architectures:
    - ModernBertForSequenceClassification
    - BertForSequenceClassification
    - XLMRobertaForSequenceClassification

    Example:
        >>> model = MLXRerankerModel("BAAI/bge-reranker-v2-m3")
        >>> model.load()
        >>> output = model.rerank("What is ML?", ["ML is...", "Weather is..."])
        >>> print(output.scores)  # [0.95, 0.12]
    """

    def __init__(self, model_name: str):
        """
        Initialize the MLX reranker model.

        Args:
            model_name: HuggingFace model name or local path
        """
        self.model_name = model_name

        self.model = None
        self.processor = None
        self._loaded = False
        self._num_labels: int | None = None

    def _get_architecture(self) -> str | None:
        """Get the model architecture from config.json."""
        config_path = Path(self.model_name) / "config.json"
        if not config_path.exists():
            return None

        try:
            with open(config_path) as f:
                config = json.load(f)
            architectures = config.get("architectures", [])
            return architectures[0] if architectures else None
        except (json.JSONDecodeError, IOError):
            return None

    def _load_xlm_roberta(self) -> Tuple[Any, Any]:
        """Load XLMRoberta model using omlx native implementation."""
        import mlx.core as mx
        from mlx.utils import tree_unflatten
        from safetensors import safe_open
        from transformers import AutoTokenizer

        from .xlm_roberta import Model, ModelArgs

        model_path = Path(self.model_name)

        # Load config
        with open(model_path / "config.json") as f:
            config_dict = json.load(f)

        config = ModelArgs(**{
            k: v for k, v in config_dict.items()
            if k in ModelArgs.__dataclass_fields__
        })

        # Create model
        model = Model(config)

        # Load weights
        weights = {}
        weight_files = list(model_path.glob("*.safetensors"))
        for wf in weight_files:
            with safe_open(wf, framework="mlx") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)

        # Sanitize weights (remove "roberta." prefix, etc.)
        weights = model.sanitize(weights)

        # Load weights into model
        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        return model, tokenizer

    def load(self) -> None:
        """Load the model and processor/tokenizer."""
        if self._loaded:
            return

        # Check architecture before loading
        self._validate_architecture()

        arch = self._get_architecture()
        logger.info(f"Loading reranker model: {self.model_name} (arch={arch})")

        try:
            if arch == "XLMRobertaForSequenceClassification":
                # Use omlx native implementation
                self.model, self.processor = self._load_xlm_roberta()
                self._num_labels = getattr(self.model.config, "num_labels", None)
            else:
                # Use mlx-embeddings for other architectures (ModernBert, etc.)
                from mlx_embeddings import load
                self.model, self.processor = load(self.model_name)

                # Get num_labels from model config
                if hasattr(self.model, "config"):
                    config = self.model.config
                    self._num_labels = getattr(config, "num_labels", None)

            self._loaded = True
            logger.info(
                f"Reranker model loaded successfully: {self.model_name} "
                f"(num_labels={self._num_labels})"
            )

        except ImportError as e:
            raise ImportError(
                "mlx-embeddings or transformers is required for reranking. "
                "Install with: pip install mlx-embeddings transformers"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: list[str],
        max_length: int = 512,
    ) -> RerankOutput:
        """
        Rerank documents by relevance to the query.

        Args:
            query: The search query
            documents: List of documents to rerank
            max_length: Maximum token length for each query-document pair

        Returns:
            RerankOutput with scores, sorted indices, and token count
        """
        if not self._loaded:
            self.load()

        import mlx.core as mx

        if not documents:
            return RerankOutput(scores=[], indices=[], total_tokens=0)

        # Get the underlying tokenizer from TokenizerWrapper (mlx-embeddings only)
        # Don't unwrap transformers tokenizers which also have _tokenizer attribute
        processor = self.processor
        processor_class = type(processor).__name__
        if processor_class == "TokenizerWrapper" and hasattr(processor, "_tokenizer"):
            processor = processor._tokenizer

        # Tokenize query-document pairs
        # SequenceClassification models expect pairs as (query, document)
        pairs = [(query, doc) for doc in documents]

        # Batch encode all pairs
        inputs = processor(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="np",
        )

        # Convert to MLX arrays
        input_ids = mx.array(inputs["input_ids"])
        attention_mask = mx.array(inputs["attention_mask"])

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Extract scores from pooler_output
        # pooler_output shape: (batch_size, num_labels)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            logits = outputs.pooler_output
        else:
            raise ValueError(
                "Model output does not contain pooler_output. "
                "Ensure the model is a SequenceClassification model."
            )

        # Ensure computation is done
        mx.eval(logits)

        # Extract relevance scores
        # For binary classification (num_labels=1), score is already sigmoid applied
        # For multi-class, take the positive class probability
        if logits.shape[-1] == 1:
            # Binary classification: sigmoid already applied by model
            scores = logits.squeeze(-1).tolist()
        else:
            # Multi-class: take last column (typically "relevant" class)
            scores = logits[:, -1].tolist()

        # Sort indices by score (descending)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        sorted_indices = [idx for idx, _ in indexed_scores]

        # Count tokens
        total_tokens = self._count_tokens(query, documents)

        return RerankOutput(
            scores=scores,
            indices=sorted_indices,
            total_tokens=total_tokens,
        )

    def _count_tokens(self, query: str, documents: list[str]) -> int:
        """Count total tokens in query-document pairs."""
        total = 0

        processor = self.processor
        processor_class = type(processor).__name__
        if processor_class == "TokenizerWrapper" and hasattr(processor, "_tokenizer"):
            processor = processor._tokenizer

        def get_token_count(text: str, add_special: bool = True) -> int:
            """Get token count for text, handling different tokenizer types."""
            if hasattr(processor, "encode"):
                tokens = processor.encode(text, add_special_tokens=add_special)
                # Handle different return types
                if isinstance(tokens, list):
                    return len(tokens)
                elif hasattr(tokens, "ids"):
                    # tokenizers.Encoding object
                    return len(tokens.ids)
                else:
                    return len(tokens)
            else:
                # Fallback to word count estimate
                return len(text.split()) + (2 if add_special else 0)

        # Count query tokens once
        query_len = get_token_count(query, add_special=True)

        # Count document tokens
        for doc in documents:
            doc_len = get_token_count(doc, add_special=False)
            # Each pair includes query + doc + special tokens
            total += query_len + doc_len + 3  # [CLS], [SEP], [SEP]

        return total

    @property
    def num_labels(self) -> int | None:
        """Get the number of classification labels."""
        return self._num_labels

    def _validate_architecture(self) -> None:
        """
        Validate that the model architecture is supported.

        Raises:
            ValueError: If the architecture is not supported by mlx-embeddings
        """
        config_path = Path(self.model_name) / "config.json"
        if not config_path.exists():
            # If no config.json, let mlx-embeddings handle validation
            return

        try:
            with open(config_path) as f:
                config = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read config.json: {e}")
            return

        architectures = config.get("architectures", [])
        if not architectures:
            return

        arch = architectures[0]
        if arch not in SUPPORTED_RERANKER_ARCHITECTURES:
            supported_list = ", ".join(sorted(SUPPORTED_RERANKER_ARCHITECTURES))
            raise ValueError(
                f"Unsupported reranker architecture: {arch}. "
                f"Currently supported architectures: {supported_list}. "
                f"This model may be supported in a future version of mlx-embeddings."
            )

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self._loaded:
            return {"loaded": False, "model_name": self.model_name}

        info = {
            "loaded": True,
            "model_name": self.model_name,
            "num_labels": self._num_labels,
        }

        # Try to get model config
        if hasattr(self.model, "config"):
            config = self.model.config
            info.update(
                {
                    "model_type": getattr(config, "model_type", None),
                    "hidden_size": getattr(config, "hidden_size", None),
                    "max_position_embeddings": getattr(
                        config, "max_position_embeddings", None
                    ),
                }
            )

        return info

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"<MLXRerankerModel model={self.model_name} status={status}>"
