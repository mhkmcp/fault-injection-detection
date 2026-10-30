"""Accelerator choice and timing helpers: CUDA (NVIDIA), MPS (Apple silicon) or CPU."""

import os
import platform
import subprocess

import torch

ACCELERATORS = ("cuda", "mps")  # preference order for `auto`


def available(device: str) -> bool:
    kind = torch.device(device).type
    if kind == "cuda":
        return torch.cuda.is_available()
    if kind == "mps":
        return torch.backends.mps.is_available()
    return kind == "cpu"


def default() -> str:
    return next((d for d in ACCELERATORS if available(d)), "cpu")


def resolve(device: str) -> str:
    """`auto` → the best available device; anything else must exist on this machine."""
    if device == "auto":
        return default()
    if not available(device):
        hint = (" (CPU-only torch build: run `uv sync` so the CUDA wheel from pyproject.toml is "
                "installed)" if device.startswith("cuda") else "")
        raise SystemExit(f"device {device!r} is not available here{hint}; use --device auto")
    return device


def accelerators(devices: list[str]) -> list[str]:
    """`devices` minus the ones this machine lacks; CPU always stays."""
    return [d for d in devices if available(d)]


def synchronize(device: str) -> None:
    kind = torch.device(device).type
    if kind == "cuda":
        torch.cuda.synchronize(device)
    elif kind == "mps":
        torch.mps.synchronize()


def allocated_mb(device: str) -> float | None:
    """Tensor memory held by the accelerator's allocator; None on CPU."""
    kind = torch.device(device).type
    if kind == "cuda":
        return torch.cuda.memory_allocated(device) / 2**20
    if kind == "mps":
        return torch.mps.current_allocated_memory() / 2**20
    return None


def empty_cache(device: str) -> None:
    kind = torch.device(device).type
    if kind == "cuda":
        torch.cuda.empty_cache()
    elif kind == "mps":
        torch.mps.empty_cache()


def _sysctl(key: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True,
                              check=False).stdout.strip()
    except FileNotFoundError:
        return ""


def _memory_gb() -> float:
    system = platform.system()
    if system == "Darwin":
        return int(_sysctl("hw.memsize") or 0) / 2**30
    if system == "Windows":
        import ctypes

        class Status(ctypes.Structure):  # MEMORYSTATUSEX
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                        ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong),
                        ("avail_virtual", ctypes.c_ulonglong),
                        ("avail_ext", ctypes.c_ulonglong)]

        s = Status(length=ctypes.sizeof(Status))
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return s.total / 2**30 if ok else 0.0
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return 0.0


def machine() -> dict:
    """Host description for sidecars, on macOS, Windows and Linux."""
    system = platform.system()
    if system == "Darwin":
        chip, os_name = _sysctl("machdep.cpu.brand_string"), f"macOS {platform.mac_ver()[0]}"
    else:
        chip = platform.processor() or platform.machine()
        os_name = f"{system} {platform.release()} ({platform.version()})"
    gpu = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gpu = {"name": p.name, "memory_gb": round(p.total_memory / 2**30, 1),
               "cuda": torch.version.cuda}
    return {"chip": chip, "memory_gb": round(_memory_gb(), 1), "cpu_count": os.cpu_count() or 0,
            "os": os_name, "gpu": gpu}


if __name__ == "__main__":  # python -m fidnn.devices: what `--device auto` picks on this machine
    print(f"torch {torch.__version__}, CUDA build {torch.version.cuda}, "
          f"CUDA available {torch.cuda.is_available()}")
    print(f"--device auto -> {default()}")
    print(machine())
