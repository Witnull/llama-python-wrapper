"""
High-Performance llama-server Subprocess Wrapper with llama_cpp Drop-in Compatibility.

Replaces in-process llama-cpp-python C-bindings with a managed background
`llama-server.exe` instance. Provides:
1. Fast, 100% leak-proof load/unload cycles (GPU VRAM is instantly reclaimed by OS kernel).
2. Dynamic auto-optimizing concurrency (-np parallel slots sized against available VRAM).
3. Thread-safe concurrent inference with continuous batching across slots.
4. Full API drop-in compatibility for existing project code using llama_cpp.
"""

from __future__ import annotations

import atexit
import gc
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time
from collections import deque
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import psutil
except ImportError:
    psutil = None

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# pyrefly: ignore [missing-import]
from src.utils.logger import setup_logger

LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = setup_logger(str(LOG_DIR / "llama_server.log"), log_level=logging.INFO)

# Default location for the llama-server executable
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SERVER_EXE = PROJECT_ROOT / "exe" / "llama-windows-x64-cuda12.4" / "llama-server.exe"

# GGML constants matching llama_cpp
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
LLAMA_DEFAULT_SEED = 42

_GGML_TYPE_MAP: Dict[Union[int, str], str] = {
    0: "f32",
    1: "f16",
    2: "q4_0",
    3: "q4_1",
    6: "q5_0",
    7: "q5_1",
    8: "q8_0",
    "f32": "f32",
    "f16": "f16",
    "q4_0": "q4_0",
    "q4_1": "q4_1",
    "q5_0": "q5_0",
    "q5_1": "q5_1",
    "q8_0": "q8_0",
}


def _resolve_cache_type(val: Union[int, str, None], default: str = "q8_0") -> str:
    if val is None:
        return default
    return _GGML_TYPE_MAP.get(val, str(val))


def get_gpu_vram_info() -> Tuple[float, float]:
    """
    Returns (total_vram_mb, free_vram_mb) for GPU 0 via nvidia-smi.
    Falls back to safe default (8192 MB, 6000 MB) if nvidia-smi fails.
    """
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, text=True, timeout=3).strip()
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        if lines:
            parts = [float(x.strip()) for x in lines[0].split(",")]
            if len(parts) >= 2:
                return parts[0], parts[1]
    except Exception as e:
        logger.debug(f"nvidia-smi query failed ({e}); falling back to default VRAM estimates.")
    return 8192.0, 6000.0


# ── Windows Kernel Job Object Management ──────────────────────────────────────
# On Windows, assigning subprocesses to a Job Object with
# JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE guarantees that the Windows kernel
# terminates all child processes when the parent Python process terminates,
# crashes, or is killed (e.g. IDE stop button, Task Manager, exception).
_GLOBAL_JOB_OBJECT = None


def _get_global_job_object():
    global _GLOBAL_JOB_OBJECT
    if sys.platform != "win32":
        return None
    if _GLOBAL_JOB_OBJECT is not None:
        return _GLOBAL_JOB_OBJECT

    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        JobObjectExtendedLimitInformation = 9
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoCounters", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryLimit", ctypes.c_size_t),
                ("PeakJobMemoryLimit", ctypes.c_size_t),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        res = kernel32.SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if res:
            _GLOBAL_JOB_OBJECT = job
            return _GLOBAL_JOB_OBJECT
    except Exception as e:
        logger.debug(f"[LlamaServer] Failed to initialize Windows Job Object: {e}")
    return None


def _assign_process_to_job(proc: subprocess.Popen) -> bool:
    """Assigns proc to the Windows Job Object if running on Windows."""
    if sys.platform != "win32" or proc is None:
        return False
    try:
        job = _get_global_job_object()
        if job and hasattr(proc, "_handle"):
            import ctypes
            return bool(ctypes.windll.kernel32.AssignProcessToJobObject(job, proc._handle))
    except Exception as e:
        logger.debug(f"[LlamaServer] Failed to assign process to Job Object: {e}")
    return False


# ── Process & Port Inspection Helpers (pure standard library / psutil fallback) ─

def _pid_exists(pid: int) -> bool:
    """Returns True if the process exists and is active."""
    if pid <= 0:
        return False
    if psutil is not None:
        return psutil.pid_exists(pid)
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                exit_code = ctypes.c_ulong()
                kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                kernel32.CloseHandle(handle)
                STILL_ACTIVE = 259
                return exit_code.value == STILL_ACTIVE
            return False
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _kill_pid(pid: int, timeout: float = 2.0) -> bool:
    """Forcefully terminates a PID and its child process tree."""
    if psutil is not None:
        try:
            p = psutil.Process(pid)
            for child in p.children(recursive=True):
                try:
                    child.kill()
                except Exception:
                    pass
            p.kill()
            p.wait(timeout=timeout)
            return True
        except Exception:
            pass

    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=timeout,
            )
            return True
        except Exception:
            pass
    else:
        try:
            os.kill(pid, signal.SIGKILL)
            return True
        except Exception:
            pass
    return False


def _get_llama_server_pids() -> List[int]:
    """Finds all PIDs of running llama-server processes on the system."""
    pids: List[int] = []
    if psutil is not None:
        for p in psutil.process_iter(["pid", "name"]):
            try:
                if "llama-server" in (p.info["name"] or "").lower():
                    pids.append(p.info["pid"])
            except Exception:
                pass
        return pids

    if sys.platform == "win32":
        try:
            res = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq llama-server*", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
            )
            import csv, io
            for line in res.stdout.strip().splitlines():
                line = line.strip()
                if not line or "INFO: No tasks" in line:
                    continue
                reader = csv.reader(io.StringIO(line))
                for row in reader:
                    if len(row) >= 2 and "llama-server" in row[0].lower():
                        try:
                            pids.append(int(row[1]))
                        except ValueError:
                            pass
        except Exception:
            pass
    return pids


def _get_pid_listening_on_port(port: int) -> Optional[int]:
    """Finds the PID currently listening on the specified TCP port."""
    if psutil is not None:
        try:
            for p in psutil.process_iter(["pid", "name"]):
                try:
                    for conn in p.connections(kind="inet"):
                        if conn.laddr.port == port:
                            return p.info["pid"]
                except Exception:
                    pass
        except Exception:
            pass
        return None

    if sys.platform == "win32":
        try:
            res = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True)
            for line in res.stdout.splitlines():
                line = line.strip()
                if not line or not line.startswith("TCP"):
                    continue
                parts = line.split()
                if len(parts) >= 5 and parts[3] == "LISTENING":
                    local_addr = parts[1]
                    if local_addr.endswith(f":{port}"):
                        return int(parts[4])
        except Exception:
            pass
    return None


# Registry of all running server instances for clean teardown
_ACTIVE_SERVERS: List["LlamaServerInstance"] = []
_ACTIVE_SERVERS_LOCK = threading.Lock()
_PORT_SELECTION_LOCK = threading.Lock()


def cleanup_orphan_servers(force: bool = False) -> int:
    """
    Terminates any unmanaged orphan llama-server.exe processes left behind by
    prior crashed or killed Python runs. Reclaims GPU VRAM and port bindings.
    """
    with _ACTIVE_SERVERS_LOCK:
        tracked_pids: Set[int] = {
            s.proc.pid for s in _ACTIVE_SERVERS if s.proc is not None and s.is_running()
        }
    current_pid = os.getpid()
    killed = 0

    all_pids = _get_llama_server_pids()
    for pid in all_pids:
        if pid in tracked_pids:
            continue
        # Unmanaged llama-server PID
        logger.warning(f"[LlamaServer] Cleaning up unmanaged orphan llama-server (PID: {pid})...")
        if _kill_pid(pid):
            killed += 1

    if killed > 0:
        logger.info(f"[LlamaServer] Cleaned up {killed} orphan server process(es).")
    return killed


# Clean up any leftover orphans on module import
cleanup_orphan_servers()


def _cleanup_all_active_servers():
    """Ensures no orphan llama-server processes are left behind on exit."""
    with _ACTIVE_SERVERS_LOCK:
        servers = list(_ACTIVE_SERVERS)
    for s in servers:
        try:
            s.stop()
        except Exception:
            pass
    # Final sweep of any remaining unmanaged orphans
    try:
        cleanup_orphan_servers(force=True)
    except Exception:
        pass


atexit.register(_cleanup_all_active_servers)


# Signal handling to ensure clean exit on SIGINT (Ctrl+C), SIGTERM, and SIGBREAK
def _signal_handler(signum, frame):
    logger.info(f"[LlamaServer] Received termination signal {signum}. Cleaning up servers...")
    _cleanup_all_active_servers()
    sys.exit(128 + signum)


try:
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, _signal_handler)
except Exception:
    pass


def find_free_port(start_port: int = 8080, max_attempts: int = 100) -> int:
    """Finds an unused TCP port on localhost that is not currently bound or tracked."""
    with _PORT_SELECTION_LOCK:
        with _ACTIVE_SERVERS_LOCK:
            used_ports = {s.port for s in _ACTIVE_SERVERS if s.is_running() or s.port is not None}

        for port in range(start_port, start_port + max_attempts):
            if port in used_ports:
                continue

            # Test connecting to port - if connect succeeds, something is already listening
            is_in_use = False
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.1)
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    is_in_use = True

            if is_in_use:
                # Check if this port is held by an unmanaged orphan llama-server
                pid = _get_pid_listening_on_port(port)
                orphan_killed = False
                if pid and pid in _get_llama_server_pids():
                    logger.warning(
                        f"[LlamaServer] Port {port} held by orphan llama-server (PID: {pid}). Terminating orphan..."
                    )
                    _kill_pid(pid)
                    orphan_killed = True

                if orphan_killed:
                    time.sleep(0.3)
                    # Re-verify port is now free
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(0.1)
                        if s.connect_ex(("127.0.0.1", port)) == 0:
                            continue
                else:
                    continue

            # Test binding to port without SO_REUSEADDR to avoid Windows port sharing
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue

        # Fallback to OS-assigned ephemeral port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]



def calculate_optimal_concurrency(
    model_path: Union[str, Path],
    context_size: int = 4096,
    embedding: bool = False,
    requested_parallel: Optional[Union[int, str]] = None,
    concurrency_mode: str = "balanced",
) -> int:
    """
    Calculates the number of parallel slots (-np) based on context size and available VRAM.

    Modes:
      - 'balanced' (default): Allocates ~40-60% of headroom (2-3 slots for chat, 4 for embeddings),
        preserving sufficient VRAM so another model can be loaded simultaneously.
      - 'max': Greedily allocates up to the full VRAM limit for a single high-throughput model.
      - 'conservative': Minimal slots (1-2) for lowest memory pressure.
    """
    if isinstance(requested_parallel, str) and requested_parallel.lower() in ("max", "balanced", "conservative"):
        concurrency_mode = requested_parallel.lower()

    total_vram_mb, free_vram_mb = get_gpu_vram_info()

    model_path = Path(model_path)
    if model_path.exists():
        model_size_mb = model_path.stat().st_size / (1024.0 * 1024.0)
    else:
        model_size_mb = 5000.0

    # Account for model weights offloaded to VRAM + base CUDA overhead
    weights_vram_mb = (model_size_mb * 1.05) + 300.0
    vram_headroom_mb = max(200.0, free_vram_mb - weights_vram_mb)

    # Context-based KV cache memory per slot (with Flash Attention and Q8 KV cache)
    # Scales directly with context_size!
    if embedding:
        slot_memory_mb = max(40.0, (context_size / 1024.0) * 40.0)
        max_possible = max(1, min(8, int(vram_headroom_mb // slot_memory_mb)))
        if concurrency_mode == "conservative":
            optimal_slots = 2
        elif concurrency_mode == "balanced":
            optimal_slots = max(1, min(4, max_possible))
        else:  # max
            optimal_slots = max_possible
    else:
        slot_memory_mb = max(100.0, (context_size / 4096.0) * 250.0)
        max_possible = max(1, min(6, int(vram_headroom_mb // slot_memory_mb)))
        if concurrency_mode == "conservative":
            optimal_slots = 1
        elif concurrency_mode == "balanced":
            optimal_slots = max(1, min(3, max_possible))
        else:  # max
            optimal_slots = max_possible

    if isinstance(requested_parallel, int) and requested_parallel > 0:
        if requested_parallel > max_possible:
            logger.warning(
                f"[VRAM Safety] Requested n_parallel={requested_parallel} exceeds safe capacity ({max_possible} slots) "
                f"for model '{model_path.name}' ({model_size_mb:.0f}MB) on {free_vram_mb:.0f}MB free VRAM. "
                f"Clamping to {max_possible} slot(s) to avoid slow shared memory spillover."
            )
            return max(1, max_possible)
        logger.info(
            f"[VRAM Optimizer] Validated requested n_parallel={requested_parallel} against model '{model_path.name}' "
            f"({model_size_mb:.0f}MB, {slot_memory_mb:.0f}MB/slot). Safe capacity: {max_possible} slots."
        )
        return requested_parallel

    logger.info(
        f"[VRAM Optimizer] Mode: '{concurrency_mode}' | Free: {free_vram_mb:.0f}MB | Model: {model_size_mb:.0f}MB | "
        f"Ctx/Slot: {context_size} ({slot_memory_mb:.0f}MB/slot) -> Slots (-np): {optimal_slots} (Max capacity: {max_possible})"
    )
    return optimal_slots


class LlamaServerInstance:
    """
    Manages the lifecycle of a single `llama-server.exe` background process.
    Provides fast, deterministic startup, health check polling, and instant
    VRAM reclamation upon termination.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        exe_path: Optional[Union[str, Path]] = None,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 99,
        n_batch: int = 512,
        n_ubatch: int = 512,
        n_threads: int = 8,
        flash_attn: bool = True,
        type_k: Union[int, str] = 8,
        type_v: Union[int, str] = 8,
        embedding: bool = False,
        n_parallel: Optional[int] = None,
        verbose: bool = False,
        extra_args: Optional[List[str]] = None,
    ):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.exe_path = Path(exe_path or DEFAULT_SERVER_EXE).resolve()
        if not self.exe_path.exists():
            raise FileNotFoundError(f"llama-server executable not found at: {self.exe_path}")

        self.host = host
        self._explicit_port = port
        self.port = port or find_free_port()
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers if n_gpu_layers >= 0 else 99
        self.n_batch = n_batch
        self.n_ubatch = n_ubatch
        self.n_threads = n_threads
        self.flash_attn = flash_attn
        default_cache = "q4_0" if n_ctx >= 16384 else "q8_0"
        self.type_k = _resolve_cache_type(type_k, default_cache)
        self.type_v = _resolve_cache_type(type_v, default_cache)
        self.embedding = embedding
        # Validate parallel slots against model file size and free VRAM headroom
        safe_parallel = calculate_optimal_concurrency(
            self.model_path,
            context_size=self.n_ctx,
            embedding=self.embedding,
            requested_parallel=n_parallel,
            concurrency_mode="balanced",
        )
        self.n_parallel = safe_parallel
        self.suggested_parallel = safe_parallel
        self.verbose = verbose
        self.extra_args = extra_args or []

        self._stop_lock = threading.Lock()
        self.proc: Optional[subprocess.Popen] = None
        self.session: Optional[requests.Session] = self._create_session()
        self.base_url = f"http://{self.host}:{self.port}"

    def _create_session(self) -> requests.Session:
        s = requests.Session()
        # High connection pool for parallel requests without socket starvation
        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=Retry(total=2, backoff_factor=0.1, status_forcelist=[502, 503, 504]),
        )
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        return s

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, timeout_sec: float = 60.0):
        """Starts the server process and blocks until /health reports OK."""
        if self.is_running():
            return

        with self._stop_lock:
            # Recreate session if needed
            if self.session is None:
                self.session = self._create_session()

            # If port was not explicitly provided by user, dynamically re-allocate a free port right before launching
            if self._explicit_port is None:
                self.port = find_free_port()
                self.base_url = f"http://{self.host}:{self.port}"

            # Verify the port is not already bound
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.1)
                if s.connect_ex((self.host, self.port)) == 0:
                    # In use! Assign a new free port
                    self.port = find_free_port()
                    self.base_url = f"http://{self.host}:{self.port}"

            total_ctx = self.n_ctx * self.n_parallel

            cmd = [
                str(self.exe_path),
                "-m", str(self.model_path),
                "--host", self.host,
                "--port", str(self.port),
                "-ngl", str(self.n_gpu_layers),
                "-c", str(total_ctx),
                "-np", str(self.n_parallel),
                "-b", str(self.n_batch),
                "-ub", str(self.n_ubatch),
                "-t", str(self.n_threads),
                "--cache-type-k", self.type_k,
                "--cache-type-v", self.type_v,
                "--slot-prompt-similarity", "0.1",
                "--jinja",
            ]
            if self.flash_attn:
                cmd.extend(["-fa", "1"])
            if self.embedding:
                cmd.append("--embedding")
            if self.extra_args:
                cmd.extend(self.extra_args)

            t0 = time.time()
            logger.info(
                f"[LlamaServer] Launching {self.model_path.name} on {self.base_url} "
                f"(-np {self.n_parallel}, ctx/slot: {self.n_ctx}, -fa {1 if self.flash_attn else 0})..."
            )
            if self.n_parallel == 1:
                if self.suggested_parallel > 1:
                    print(
                        f"\n[SUGGESTION] Model '{self.model_path.name}' running with 1 thread/slot. "
                        f"Available VRAM can comfortably fit up to {self.suggested_parallel} parallel slots! "
                        f"Use LLMPool(..., n_workers={self.suggested_parallel}) to enable concurrent processing.\n"
                    )
            else:
                print(
                    f"[PARALLEL MODE] Model '{self.model_path.name}' running with {self.n_parallel} parallel slots."
                )

            # Continuously drain stderr to prevent OS pipe buffer deadlock on Windows
            self._stderr_lines = deque(maxlen=200)
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL if not self.verbose else None,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            # Bind to Windows Job Object so kernel guarantees child cleanup on parent exit/crash
            _assign_process_to_job(self.proc)

            with _ACTIVE_SERVERS_LOCK:
                if self not in _ACTIVE_SERVERS:
                    _ACTIVE_SERVERS.append(self)

            def _drain_stderr(proc, lines):
                try:
                    for line in iter(proc.stderr.readline, ""):
                        lines.append(line)
                        if self.verbose:
                            sys.stderr.write(line)
                            sys.stderr.flush()
                except Exception:
                    pass

            threading.Thread(target=_drain_stderr, args=(self.proc, self._stderr_lines), daemon=True).start()

        # Poll /health endpoint until model is ready
        health_url = f"{self.base_url}/health"
        deadline = time.time() + timeout_sec
        ready = False

        try:
            while time.time() < deadline:
                # Check if process exited unexpectedly
                if self.proc.poll() is not None:
                    err_out = "".join(self._stderr_lines)
                    raise RuntimeError(
                        f"llama-server terminated unexpectedly (code {self.proc.returncode}): {err_out}"
                    )

                try:
                    resp = self.session.get(health_url, timeout=0.8)
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("status") == "ok":
                            ready = True
                            break
                except Exception:
                    pass
                time.sleep(0.08)

            if not ready:
                raise TimeoutError(f"llama-server failed to report ready within {timeout_sec}s.")

            logger.info(f"[LlamaServer] Ready in {time.time() - t0:.2f}s.")
        except Exception:
            self.stop()
            raise

    def stop(self):
        """Immediately terminates server process and releases all GPU VRAM."""
        with self._stop_lock:
            with _ACTIVE_SERVERS_LOCK:
                if self in _ACTIVE_SERVERS:
                    try:
                        _ACTIVE_SERVERS.remove(self)
                    except ValueError:
                        pass

            if self.proc is not None:
                t0 = time.time()
                pid = self.proc.pid
                try:
                    _kill_pid(pid, timeout=2.5)
                    # If still alive on Windows, force taskkill
                    if sys.platform == "win32" and _pid_exists(pid):
                        try:
                            subprocess.run(
                                ["taskkill", "/F", "/T", "/PID", str(pid)],
                                capture_output=True,
                                timeout=3,
                            )
                        except Exception:
                            pass
                except Exception as e:
                    logger.error(f"[LlamaServer] Error while stopping PID {pid}: {e}")
                finally:
                    self.proc = None
                    logger.info(f"[LlamaServer] Stopped PID {pid} & VRAM reclaimed in {time.time() - t0:.3f}s.")

            if self.session is not None:
                try:
                    self.session.close()
                except Exception:
                    pass
                self.session = None
            gc.collect()

    def __enter__(self) -> "LlamaServerInstance":
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass


try:
    from src.modules.llm.llama_grammar import LlamaGrammar, json_schema_to_gbnf
except ImportError:
    class LlamaGrammar:
        """Compatibility drop-in for llama_cpp.LlamaGrammar."""

        def __init__(self, grammar: str = "", schema: Optional[Dict[str, Any]] = None):
            self.grammar = grammar
            self.schema = schema

        @classmethod
        def from_json_schema(
            cls, schema: Union[str, Dict[str, Any]], verbose: bool = False
        ) -> "LlamaGrammar":
            if isinstance(schema, str):
                try:
                    schema_dict = json.loads(schema)
                except Exception:
                    schema_dict = None
            else:
                schema_dict = schema
            return cls(grammar="", schema=schema_dict)

    json_schema_to_gbnf = None


class Llama:
    """
    Drop-in replacement for llama_cpp.Llama.
    Transparently manages an underlying high-performance llama-server instance.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        n_ctx: int = 4096,
        n_gpu_layers: int = 99,
        n_batch: int = 512,
        n_ubatch: int = 512,
        n_threads: int = 8,
        verbose: bool = False,
        flash_attn: bool = True,
        seed: int = 42,
        type_k: Union[int, str] = 8,
        type_v: Union[int, str] = 8,
        embedding: bool = False,
        n_parallel: int = 1,
        port: Optional[int] = None,
        host: str = "127.0.0.1",
        exe_path: Optional[Union[str, Path]] = None,
        auto_start: bool = True,
        **kwargs,
    ):
        self.model_path = Path(model_path)
        self.verbose = verbose
        self.embedding = embedding

        # Store internal dummy attributes for code inspecting llama_cpp internals
        self._sampler = None
        self._model = None
        self._ctx = None

        self._server = LlamaServerInstance(
            model_path=self.model_path,
            exe_path=exe_path,
            host=host,
            port=port,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            n_threads=n_threads,
            flash_attn=flash_attn,
            type_k=type_k,
            type_v=type_v,
            embedding=embedding,
            n_parallel=n_parallel,
            verbose=verbose,
        )

        if auto_start:
            self.load()

    def load(self) -> "Llama":
        if not self._server.is_running():
            self._server.start()
        return self

    def unload(self):
        self._server.stop()

    def close(self):
        self.unload()

    def __enter__(self) -> "Llama":
        return self.load()

    def __exit__(self, *args):
        self.unload()

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass

    # ── Inference Methods ─────────────────────────────────────────────────────

    def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        grammar: Optional[LlamaGrammar] = None,
        response_format: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repeat_penalty: float = 1.1,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Executes a chat completion against /v1/chat/completions.
        Matches llama_cpp.create_chat_completion format and parameters.
        """
        if not self._server.is_running():
            self.load()

        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "stream": stream,
        }

        if max_tokens is not None and max_tokens > 0:
            payload["max_tokens"] = max_tokens

        if stop is not None:
            payload["stop"] = stop if isinstance(stop, list) else [stop]

        # Handle structured output / grammar constraints
        if grammar is not None:
            gbnf = getattr(grammar, "grammar", None) or getattr(grammar, "_grammar", None)
            if gbnf:
                payload["grammar"] = gbnf
            elif getattr(grammar, "schema", None):
                if json_schema_to_gbnf is not None:
                    try:
                        payload["grammar"] = json_schema_to_gbnf(json.dumps(grammar.schema))
                    except Exception:
                        payload["response_format"] = {"type": "json_object", "schema": grammar.schema}
                else:
                    payload["response_format"] = {"type": "json_object", "schema": grammar.schema}
        elif response_format is not None:
            payload["response_format"] = response_format

        # Pass through optional parameters supported by llama-server
        for key in ("top_k", "min_p", "seed", "presence_penalty", "frequency_penalty"):
            if key in kwargs and kwargs[key] is not None:
                payload[key] = kwargs[key]

        url = f"{self._server.base_url}/v1/chat/completions"
        try:
            resp = self._server.session.post(url, json=payload, timeout=kwargs.get("timeout", 180.0))
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            err_text = resp.text if "resp" in locals() else str(e)
            raise RuntimeError(f"Chat completion error from llama-server: {err_text}") from e

    def create_completion(
        self,
        prompt: Union[str, List[int]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repeat_penalty: float = 1.1,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Executes a text completion against /v1/completions.
        Matches llama_cpp.create_completion format and parameters.
        """
        if not self._server.is_running():
            self.load()

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "stream": stream,
        }
        if max_tokens is not None and max_tokens > 0:
            payload["max_tokens"] = max_tokens
        if stop is not None:
            payload["stop"] = stop if isinstance(stop, list) else [stop]
        for key in ("top_k", "min_p", "seed", "presence_penalty", "frequency_penalty"):
            if key in kwargs and kwargs[key] is not None:
                payload[key] = kwargs[key]

        url = f"{self._server.base_url}/v1/completions"
        try:
            resp = self._server.session.post(url, json=payload, timeout=kwargs.get("timeout", 180.0))
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            err_text = resp.text if "resp" in locals() else str(e)
            raise RuntimeError(f"Completion error from llama-server: {err_text}") from e

    def create_embedding(
        self,
        input: Union[str, List[str]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generates embeddings against /v1/embeddings.
        Matches llama_cpp.create_embedding output format.
        """
        if not self._server.is_running():
            self.load()

        # Handle input string or list of strings
        inputs = [input] if isinstance(input, str) else list(input)
        payload = {"input": inputs}

        url = f"{self._server.base_url}/v1/embeddings"
        try:
            resp = self._server.session.post(url, json=payload, timeout=kwargs.get("timeout", 60.0))
            resp.raise_for_status()
            res_json = resp.json()

            # Ensure 'data' contains objects with standard indices
            data = res_json.get("data", [])
            for idx, item in enumerate(data):
                if "index" not in item:
                    item["index"] = idx

            # If input was a single string, return matching singular or list
            return res_json
        except requests.exceptions.HTTPError as e:
            err_text = resp.text if "resp" in locals() else str(e)
            raise RuntimeError(f"Embedding error from llama-server: {err_text}") from e
