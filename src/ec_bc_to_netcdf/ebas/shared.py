"""Shared helpers used by EBAS conversion scripts.
"""

from __future__ import annotations

from typing import Iterable, List
import os
import numpy as np

__all__ = ["build_id_array_from_unix_seconds", "list_nas_files"]


def build_id_array_from_unix_seconds(timestamps: Iterable[int]) -> np.ndarray:
    """Build EBAS id array from UNIX timestamps."""
    id_values: List[int] = []
    for t_sec in timestamps:
        try:
            s_time = str(int(t_sec))
            if len(s_time) > 1:
                id_val = int("2" + s_time[1:])
            elif len(s_time) == 1:
                id_val = int("2" + s_time)
            else:
                id_val = 20
        except Exception:
            id_val = int(t_sec)
        id_values.append(id_val)
    return np.array(id_values, dtype=np.int64)


def list_nas_files(source_dir: str, sort: bool = False, case_insensitive_ext: bool = False) -> List[str]:
    """Return absolute paths to .nas files in directory, optionally sorted."""
    source_dir_abs = os.path.abspath(source_dir)
    if not os.path.isdir(source_dir_abs):
        return []
    try:
        names = os.listdir(source_dir_abs)
    except Exception:
        return []
    if sort:
        names = sorted(names)
    if case_insensitive_ext:
        return [os.path.join(source_dir_abs, f) for f in names if f.lower().endswith('.nas')]
    return [os.path.join(source_dir_abs, f) for f in names if f.endswith('.nas')]


