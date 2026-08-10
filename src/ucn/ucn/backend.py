"""
Array backend that runs the network on the GPU with CuPy when available and
falls back to NumPy on the CPU otherwise.

The switch is controlled by the environment variable UCN_GPU.  Set it to 1 to
request the GPU.  When CuPy or its CUDA runtime cannot be loaded the code
silently stays on NumPy, so the same source runs in both places.

On Windows the CUDA runtime libraries ship as separate pip wheels under
site-packages/nvidia/*/bin.  Those directories are not on the default DLL
search path, so I register them before importing CuPy.  Without this step
CuPy finds nvrtc but fails to load cublas.
"""
import os
import numpy as _np

_WANT_GPU = os.environ.get("UCN_GPU", "0") == "1"


def _register_cuda_dll_dirs() -> None:
    """Add pip-installed CUDA runtime bin folders to the Windows DLL path."""
    import glob
    import site
    roots = set()
    try:
        roots.update(site.getsitepackages())
    except Exception:
        pass
    try:
        roots.add(site.getusersitepackages())
    except Exception:
        pass
    for sp in roots:
        pattern = os.path.join(sp, "nvidia", "*", "bin")
        for bindir in glob.glob(pattern):
            if os.path.isdir(bindir):
                try:
                    os.add_dll_directory(bindir)
                except Exception:
                    pass


xp = _np
ON_GPU = False

if _WANT_GPU:
    try:
        if os.name == "nt":
            _register_cuda_dll_dirs()
        import cupy as _cp
        # Force context initialization so a broken runtime fails here, not deep
        # inside training where it would be harder to diagnose.
        _cp.zeros(1) + 1
        xp = _cp
        ON_GPU = True
        print("[backend] GPU enabled: running on CuPy.", flush=True)
    except Exception as e:  # pragma: no cover - depends on host
        print(f"[backend] GPU requested but CuPy failed ({e}). "
              f"Falling back to NumPy CPU.", flush=True)
        xp = _np
        ON_GPU = False


def to_device(a):
    """Move a host array onto the compute device (no-op on CPU)."""
    if a is None:
        return None
    return xp.asarray(a) if ON_GPU else a


def to_cpu(a):
    """Bring a device array back to a NumPy host array (no-op on CPU)."""
    if a is None:
        return None
    if ON_GPU:
        return xp.asnumpy(a)
    return a


def new_rng(seed: int):
    """Return a random generator on the active backend."""
    return xp.random.default_rng(seed)
