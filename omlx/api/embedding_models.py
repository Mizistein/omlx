# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for OpenAI-compatible Embeddings API.

These models define the request and response schemas for:
- /v1/embeddings endpoint
"""

import time
import uuid
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """
    Request for creating embeddings.

    OpenAI-compatible request format for the /v1/embeddings endpoint.
    """

    input: Union[str, List[str]]
    """Input text(s) to embed. Can be a single string or list of strings."""

    model: str
    """ID of the model to use."""

    encoding_format: Literal["float", "base64"] = "float"
    """
    The format to return embeddings in.
    - "float": Returns a list of floats (default)
    - "base64": Returns a base64-encoded string of little-endian floats
    """

    dimensions: Optional[int] = None
    """
    The number of dimensions the output embeddings should have.
    Only supported by some models. If not supported, returns full dimensions.
    """


class EmbeddingData(BaseModel):
    """A single embedding result."""

    object: str = "embedding"
    """The object type, always "embedding"."""

    index: int
    """The index of the embedding in the input list."""

    embedding: Union[List[float], str]
    """
    The embedding vector.
    - List[float] when encoding_format="float"
    - str (base64) when encoding_format="base64"
    """


class EmbeddingUsage(BaseModel):
    """Token usage statistics for embedding request."""

    prompt_tokens: int
    """Number of tokens in the input."""

    total_tokens: int
    """Total number of tokens used (same as prompt_tokens for embeddings)."""


class EmbeddingResponse(BaseModel):
    """
    Response from creating embeddings.

    OpenAI-compatible response format for the /v1/embeddings endpoint.
    """

    object: str = "list"
    """The object type, always "list"."""

    data: List[EmbeddingData]
    """List of embedding objects."""

    model: str
    """The model used for embedding."""

    usage: EmbeddingUsage
    """Usage statistics."""
