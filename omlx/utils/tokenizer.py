# SPDX-License-Identifier: Apache-2.0
"""
Tokenizer utilities for oMLX.

This module provides shared tokenizer configuration and fixes that are used
across multiple modules in the codebase.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def is_harmony_model(model_name: str, config: dict[str, Any] | None = None) -> bool:
    """
    Check if the model uses Harmony format.

    Harmony format is used by gpt-oss models with special tokens like
    <|start|>, <|channel|>, <|message|>, <|end|>, <|return|>, <|call|>.

    Detection priority:
    1. model_type == "gpt_oss" in config.json
    2. Fallback: model_name contains "gpt-oss" or "gptoss" (case-insensitive)

    Args:
        model_name: The model name or path.
        config: Optional model config dict (from config.json).

    Returns:
        True if the model uses Harmony format.
    """
    # Primary detection: config.model_type
    if config is not None:
        model_type = config.get("model_type", "")
        if model_type == "gpt_oss":
            logger.debug(f"Harmony model detected via config.model_type: {model_name}")
            return True

    # Fallback detection: model name pattern
    if model_name:
        name_lower = model_name.lower()
        if "gpt-oss" in name_lower or "gptoss" in name_lower:
            logger.debug(f"Harmony model detected via model name pattern: {model_name}")
            return True

    return False


def is_qwen3_model(model_name: str) -> bool:
    """
    Check if the model is a Qwen3 model.

    Args:
        model_name: The model name or path.

    Returns:
        True if the model is a Qwen3 model.
    """
    model_lower = model_name.lower()
    return "qwen3" in model_lower or "Qwen3" in model_name


def get_tokenizer_config(
    model_name: str,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """
    Get tokenizer configuration with model-specific fixes.

    This function centralizes tokenizer configuration to ensure consistent
    behavior across different modules.

    Args:
        model_name: The model name or path.
        trust_remote_code: Whether to trust remote code.

    Returns:
        Dictionary of tokenizer configuration options.
    """
    config: dict[str, Any] = {"trust_remote_code": trust_remote_code}

    # Apply Qwen3 fix if needed
    if is_qwen3_model(model_name):
        config["eos_token"] = "<|im_end|>"
        logger.debug("Qwen3 detected: setting eos_token to <|im_end|>")

    return config


def apply_qwen3_fix(
    tokenizer_config: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """
    Apply Qwen3 tokenizer fix to an existing config.

    Qwen3 has a known issue where eos_token changed from <|im_end|> to
    <|endoftext|>, but the chat template still uses <|im_end|>. This
    function applies the fix if needed.

    Args:
        tokenizer_config: Existing tokenizer configuration dict.
        model_name: The model name or path.

    Returns:
        Updated tokenizer configuration with Qwen3 fix applied if needed.
    """
    if is_qwen3_model(model_name):
        tokenizer_config["eos_token"] = "<|im_end|>"
        logger.debug("Qwen3 detected: setting eos_token to <|im_end|>")

    return tokenizer_config
