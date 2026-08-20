"""Make the pip-installed NVIDIA CUDA DLLs visible to CTranslate2 on Windows.

CTranslate2 (the engine under faster-whisper) links against cuBLAS and cuDNN at
load time. On Linux the wheels' RPATH handles this. On Windows nothing does, so
importing WhisperModel with device="cuda" fails with a bare
"Library cublas64_12.dll is not found" unless the DLL directories are registered
first. Call `register()` before constructing a WhisperModel.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_registered = False


def nvidia_dll_dirs() -> list[Path]:
    """Directories inside the installed nvidia-*-cu12 wheels that hold DLLs."""
    dirs: list[Path] = []
    try:
        import nvidia  # noqa: PLC0415
    except ImportError:
        return dirs

    for root in map(Path, nvidia.__path__):
        # Windows wheels use bin/, Linux wheels use lib/.
        dirs.extend(p for p in root.glob("*/bin") if p.is_dir())
        dirs.extend(p for p in root.glob("*/lib") if p.is_dir())
    return dirs


def register() -> list[Path]:
    """Register NVIDIA DLL directories. Idempotent. Returns what was added."""
    global _registered
    dirs = nvidia_dll_dirs()
    if _registered or not dirs:
        return dirs

    for d in dirs:
        if sys.platform == "win32":
            os.add_dll_directory(str(d))
        os.environ["PATH"] = f"{d}{os.pathsep}{os.environ.get('PATH', '')}"

    _registered = True
    return dirs
