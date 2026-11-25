"""Child processes that end with Localive, even when its window is closed or it is killed.

Windows: every child joins a Job Object with "kill on job close"; the job handle is
owned by Localive, so the system closes it (and ends the children) however Localive
ends. Linux: the child asks the kernel for SIGTERM when its parent dies.
Without this, closing the console left the llama-servers running, and the next start
reused them with their old settings.
"""

from __future__ import annotations

import logging
import subprocess
import sys

log = logging.getLogger(__name__)
_job = None


def _windows_job():
    global _job
    if _job is None:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]  # fmt: skip

        class Basic(ctypes.Structure):
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

        class Io(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]  # fmt: skip

        class Extended(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", Basic),
                ("IoInfo", Io),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        info = Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        extended_limit_information = 9
        if not job or not kernel32.SetInformationJobObject(
            job, extended_limit_information, ctypes.byref(info), ctypes.sizeof(info)
        ):
            raise OSError(ctypes.get_last_error(), "CreateJobObject")
        _job = (kernel32, job)
    return _job


def bind(process: subprocess.Popen) -> None:
    """Make ``process`` end with Localive (Windows; Linux uses ``preexec``)."""
    if sys.platform != "win32":
        return
    try:
        kernel32, job = _windows_job()
        if not kernel32.AssignProcessToJobObject(job, int(process._handle)):  # noqa: SLF001
            raise OSError("AssignProcessToJobObject")
    except (OSError, AttributeError) as exc:  # never block startup for this
        log.warning("Le processus %s ne sera pas arrêté avec Localive : %s", process.pid, exc)


def preexec():
    """Linux: SIGTERM to the child when Localive dies (None elsewhere)."""
    if not sys.platform.startswith("linux"):
        return None

    def die_with_parent() -> None:
        import ctypes
        import signal

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG

    return die_with_parent
