# SPDX-License-Identifier: Apache-2.0
"""HuggingFace model downloader for oMLX admin panel.

Downloads models from HuggingFace Hub using huggingface_hub's snapshot_download
with directory-size-based progress polling.
"""

import asyncio
import enum
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    RepositoryNotFoundError,
)

logger = logging.getLogger(__name__)


class DownloadStatus(str, enum.Enum):
    """Status of a download task."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DownloadTask:
    """Represents a single model download task."""

    task_id: str
    repo_id: str
    status: DownloadStatus = DownloadStatus.PENDING
    progress: float = 0.0
    total_size: int = 0
    downloaded_size: int = 0
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0

    def to_dict(self) -> dict:
        """Serialize task to a JSON-compatible dict."""
        return {
            "task_id": self.task_id,
            "repo_id": self.repo_id,
            "status": self.status.value,
            "progress": round(self.progress, 1),
            "total_size": self.total_size,
            "downloaded_size": self.downloaded_size,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U64": 8, "U32": 4, "U16": 2, "U8": 1,
    "BOOL": 1,
}

# Minimum downloads to be included in recommendations.
_MIN_DOWNLOADS = 100


def _calc_safetensors_disk_size(safetensors: dict) -> int:
    """Calculate actual disk size in bytes from safetensors parameters.

    safetensors.total is the parameter count, not bytes.
    We need to multiply each dtype's parameter count by its byte width.
    """
    params = safetensors.get("parameters", {})
    if not params:
        return 0
    return sum(count * _DTYPE_BYTES.get(dtype, 1) for dtype, count in params.items())


def _format_model_size(size_bytes: int) -> str:
    """Format model size in bytes to a human-readable string."""
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.1f} GB"


class HFDownloader:
    """Manages HuggingFace model downloads with progress tracking.

    Uses huggingface_hub.snapshot_download() for actual downloads and polls
    the target directory size to estimate progress.

    Args:
        model_dir: Directory where downloaded models are stored.
        on_complete: Async callback invoked when a download completes successfully.
    """

    @staticmethod
    async def get_recommended_models(
        max_memory_bytes: int,
        limit: int = 30,
    ) -> dict:
        """Fetch trending and popular mlx-community models that fit in memory.

        Queries HuggingFace Hub for text-generation models from mlx-community,
        filtered by system memory capacity.

        Args:
            max_memory_bytes: Maximum model size in bytes (typically system memory).
            limit: Number of models to fetch per category from HF API.

        Returns:
            Dict with 'trending' and 'popular' lists (up to 10 each).
        """
        api = HfApi()

        async def _fetch(sort: str) -> list[dict]:
            models = await asyncio.to_thread(
                api.list_models,
                author="mlx-community",
                sort=sort,
                limit=limit,
                pipeline_tag="text-generation",
                expand=["safetensors", "downloads", "likes", "trendingScore"],
            )
            results = []
            for m in models:
                if not m.safetensors or not m.safetensors.get("parameters"):
                    continue
                downloads = m.downloads or 0
                if downloads < _MIN_DOWNLOADS:
                    continue
                size = _calc_safetensors_disk_size(m.safetensors)
                if size <= 0 or size > max_memory_bytes:
                    continue
                results.append(
                    {
                        "repo_id": m.id,
                        "name": m.id.split("/")[-1],
                        "downloads": downloads,
                        "likes": m.likes or 0,
                        "trending_score": m.trending_score or 0,
                        "size": size,
                        "size_formatted": _format_model_size(size),
                    }
                )
            return results

        trending, popular = await asyncio.gather(
            _fetch("trendingScore"),
            _fetch("downloads"),
        )

        return {
            "trending": trending[:10],
            "popular": popular[:10],
        }

    def __init__(
        self,
        model_dir: str,
        on_complete: Optional[Callable] = None,
    ):
        self._model_dir = Path(model_dir)
        self._tasks: dict[str, DownloadTask] = {}
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._progress_tasks: dict[str, asyncio.Task] = {}
        self._on_complete = on_complete
        self._cancelled: set[str] = set()

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    def update_model_dir(self, new_dir: str) -> None:
        """Update the model directory path."""
        self._model_dir = Path(new_dir)

    async def start_download(
        self, repo_id: str, hf_token: str = ""
    ) -> DownloadTask:
        """Start downloading a model from HuggingFace.

        Args:
            repo_id: HuggingFace repository ID (e.g., "mlx-community/Llama-3-8B-4bit").
            hf_token: Optional HuggingFace token for gated models.

        Returns:
            The created DownloadTask.

        Raises:
            ValueError: If repo_id format is invalid or download is already queued.
        """
        repo_id = repo_id.strip()
        if "/" not in repo_id or len(repo_id.split("/")) != 2:
            raise ValueError(
                f"Invalid repository ID: '{repo_id}'. "
                "Expected format: 'owner/model' (e.g., 'mlx-community/Llama-3-8B-4bit')"
            )

        # Check for duplicate active downloads
        for task in self._tasks.values():
            if task.repo_id == repo_id and task.status in (
                DownloadStatus.PENDING,
                DownloadStatus.DOWNLOADING,
            ):
                raise ValueError(
                    f"Download for '{repo_id}' is already in progress"
                )

        task_id = str(uuid.uuid4())
        task = DownloadTask(task_id=task_id, repo_id=repo_id)
        self._tasks[task_id] = task

        # Start download in background
        self._active_tasks[task_id] = asyncio.create_task(
            self._run_download(task_id, hf_token)
        )

        logger.info(f"Download queued: {repo_id} (task_id={task_id})")
        return task

    async def cancel_download(self, task_id: str) -> bool:
        """Cancel an active download.

        Args:
            task_id: The task ID to cancel.

        Returns:
            True if the task was found and cancelled.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return False

        if task.status not in (DownloadStatus.PENDING, DownloadStatus.DOWNLOADING):
            return False

        # Mark as cancelled
        self._cancelled.add(task_id)
        task.status = DownloadStatus.CANCELLED

        # Stop progress polling
        progress_task = self._progress_tasks.pop(task_id, None)
        if progress_task and not progress_task.done():
            progress_task.cancel()

        # Cancel the download task
        active_task = self._active_tasks.pop(task_id, None)
        if active_task and not active_task.done():
            active_task.cancel()

        # Clean up partial download
        self._cleanup_partial(task)

        logger.info(f"Download cancelled: {task.repo_id} (task_id={task_id})")
        return True

    def remove_task(self, task_id: str) -> bool:
        """Remove a completed, failed, or cancelled task from the list.

        Args:
            task_id: The task ID to remove.

        Returns:
            True if the task was found and removed.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return False

        if task.status in (DownloadStatus.PENDING, DownloadStatus.DOWNLOADING):
            return False

        del self._tasks[task_id]
        self._cancelled.discard(task_id)
        return True

    def get_tasks(self) -> list[dict]:
        """Return all tasks as serializable dicts, ordered by creation time."""
        return [
            task.to_dict()
            for task in sorted(self._tasks.values(), key=lambda t: t.created_at)
        ]

    async def shutdown(self) -> None:
        """Cancel all active downloads and clean up."""
        # Cancel all progress polling tasks
        for task_id, progress_task in list(self._progress_tasks.items()):
            if not progress_task.done():
                progress_task.cancel()
        self._progress_tasks.clear()

        # Cancel all active download tasks
        for task_id, active_task in list(self._active_tasks.items()):
            if not active_task.done():
                active_task.cancel()
                task = self._tasks.get(task_id)
                if task and task.status == DownloadStatus.DOWNLOADING:
                    task.status = DownloadStatus.CANCELLED
        self._active_tasks.clear()

        logger.info("HF Downloader shut down")

    async def _run_download(self, task_id: str, hf_token: str) -> None:
        """Execute a download task.

        Fetches repo info for total size, then runs snapshot_download in a thread
        while polling the target directory for progress updates.
        """
        task = self._tasks[task_id]
        task.status = DownloadStatus.DOWNLOADING
        task.started_at = time.time()

        # Derive model name from repo_id (last part)
        model_name = task.repo_id.split("/")[-1]
        target_dir = self._model_dir / model_name

        try:
            # Get total repo size for progress estimation
            try:
                api = HfApi()
                model_info = await asyncio.to_thread(
                    api.model_info,
                    task.repo_id,
                    token=hf_token or None,
                    expand=["safetensors"],
                )
                if model_info.safetensors and model_info.safetensors.get("parameters"):
                    # Calculate size from safetensors metadata (most accurate)
                    task.total_size = _calc_safetensors_disk_size(dict(model_info.safetensors))
                elif model_info.siblings:
                    # Fallback to siblings size if safetensors not available
                    task.total_size = sum(
                        s.size for s in model_info.siblings if s.size
                    )
            except Exception as e:
                logger.warning(
                    f"Could not fetch repo info for {task.repo_id}: {e}. "
                    "Progress estimation will be unavailable."
                )

            # Start progress polling
            self._progress_tasks[task_id] = asyncio.create_task(
                self._poll_progress(task_id, target_dir)
            )

            # Run snapshot_download in a thread (blocking call)
            await asyncio.to_thread(
                snapshot_download,
                repo_id=task.repo_id,
                local_dir=str(target_dir),
                token=hf_token or None,
            )

            # Check if cancelled while downloading
            if task_id in self._cancelled:
                return

            # Success
            task.status = DownloadStatus.COMPLETED
            task.progress = 100.0
            task.downloaded_size = task.total_size or self._get_dir_size(target_dir)
            task.completed_at = time.time()

            logger.info(
                f"Download completed: {task.repo_id} -> {target_dir} "
                f"({time.time() - task.started_at:.1f}s)"
            )

            # Trigger model pool refresh
            if self._on_complete:
                try:
                    await self._on_complete()
                except Exception as e:
                    logger.error(f"Error in download completion callback: {e}")

        except asyncio.CancelledError:
            if task.status != DownloadStatus.CANCELLED:
                task.status = DownloadStatus.CANCELLED
            self._cleanup_partial(task)
        except RepositoryNotFoundError:
            task.status = DownloadStatus.FAILED
            task.error = f"Repository not found: {task.repo_id}"
            logger.error(f"Repository not found: {task.repo_id}")
        except GatedRepoError:
            task.status = DownloadStatus.FAILED
            task.error = (
                f"Repository '{task.repo_id}' is gated. "
                "Please provide a valid HF token with access."
            )
            logger.error(f"Gated repo access denied: {task.repo_id}")
        except Exception as e:
            if task_id not in self._cancelled:
                task.status = DownloadStatus.FAILED
                task.error = str(e)
                logger.error(f"Download failed for {task.repo_id}: {e}")
        finally:
            # Stop progress polling
            progress_task = self._progress_tasks.pop(task_id, None)
            if progress_task and not progress_task.done():
                progress_task.cancel()

            # Remove from active tasks
            self._active_tasks.pop(task_id, None)

    async def _poll_progress(self, task_id: str, target_dir: Path) -> None:
        """Poll the target directory size to estimate download progress."""
        task = self._tasks.get(task_id)
        if task is None:
            return

        try:
            while task.status == DownloadStatus.DOWNLOADING:
                await asyncio.sleep(2)

                if task.status != DownloadStatus.DOWNLOADING:
                    break

                current_size = self._get_dir_size(target_dir)
                task.downloaded_size = current_size

                if task.total_size > 0:
                    # Cap at 99% until snapshot_download confirms completion
                    task.progress = min(
                        (current_size / task.total_size) * 100, 99.0
                    )
        except asyncio.CancelledError:
            pass

    @staticmethod
    def _get_dir_size(path: Path) -> int:
        """Calculate total size of all files in a directory."""
        if not path.exists():
            return 0
        total = 0
        try:
            for f in path.rglob("*"):
                if f.is_file():
                    try:
                        total += f.stat().st_size
                    except OSError:
                        pass
        except OSError:
            pass
        return total

    def _cleanup_partial(self, task: DownloadTask) -> None:
        """Remove partially downloaded model directory."""
        model_name = task.repo_id.split("/")[-1]
        target_dir = self._model_dir / model_name
        if target_dir.exists():
            try:
                shutil.rmtree(target_dir)
                logger.info(f"Cleaned up partial download: {target_dir}")
            except Exception as e:
                logger.error(f"Failed to clean up {target_dir}: {e}")
