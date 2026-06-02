from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Final

DEFAULT_BEST_CHECKPOINT_NAME: Final[str] = "checkpoint-best"
_CHECKPOINT_PREFIX: Final[str] = "checkpoint-"


def update_best_checkpoint(
    *,
    run_dir: str | Path,
    checkpoint_dir: str | Path,
    best_checkpoint_name: str = DEFAULT_BEST_CHECKPOINT_NAME,
) -> Path:
    """
    Materialize `run_dir/<best_checkpoint_name>` from `checkpoint_dir`.

    This prefers a hardlink copy (fast and storage efficient), and falls back to a
    full copy when hardlinks are not supported.
    """
    if best_checkpoint_name in {"", ".", ".."} or Path(best_checkpoint_name).name != best_checkpoint_name:
        raise ValueError(f"best_checkpoint_name must be a single directory name, got {best_checkpoint_name!r}")

    run_dir_path = Path(run_dir)
    checkpoint_dir_path = Path(checkpoint_dir)
    if not checkpoint_dir_path.is_dir():
        raise NotADirectoryError(f"checkpoint_dir must be a directory, got {checkpoint_dir_path!s}")

    best_path = run_dir_path / best_checkpoint_name

    if best_path.is_symlink() or best_path.is_file():
        best_path.unlink()
    elif best_path.is_dir():
        shutil.rmtree(best_path)
    elif os.path.lexists(best_path):
        raise RuntimeError(f"Refusing to overwrite unsupported path type at {best_path!s}")

    try:
        shutil.copytree(checkpoint_dir_path, best_path, copy_function=os.link)
        return best_path
    except OSError:
        shutil.copytree(checkpoint_dir_path, best_path)
        return best_path


def prune_old_checkpoints(*, run_dir: str | Path, keep_last: int) -> list[Path]:
    """
    Delete old `checkpoint-<step>/` directories under `run_dir`.

    Args:
        run_dir: Directory containing training checkpoints.
        keep_last: Number of newest `checkpoint-<step>/` directories to keep.

    Returns:
        List of deleted checkpoint directories.
    """
    if keep_last < 0:
        raise ValueError(f"keep_last must be >= 0, got {keep_last}")

    run_dir_path = Path(run_dir)
    if not run_dir_path.is_dir():
        raise NotADirectoryError(f"run_dir must be a directory, got {run_dir_path!s}")

    checkpoints: list[tuple[int, Path]] = []
    for child in run_dir_path.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith(_CHECKPOINT_PREFIX):
            continue
        _, _, suffix = name.partition("-")
        if not suffix.isdigit():
            continue
        checkpoints.append((int(suffix), child))

    if keep_last == 0:
        to_delete = checkpoints
    else:
        checkpoints.sort(key=lambda item: item[0])
        to_delete = checkpoints[:-keep_last]

    deleted: list[Path] = []
    for _, path in to_delete:
        shutil.rmtree(path)
        deleted.append(path)

    return deleted
