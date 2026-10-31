"""Windows memory gate and termination restricted to the helper's owned process."""

import ctypes
import os
import subprocess


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)]
    _fields_ += [(name, ctypes.c_ulonglong) for name in
                ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual",
                 "avail_virtual", "avail_extended")]


def resources():
    if os.name != "nt":
        raise RuntimeError("Only the local Windows queue may use this gate")
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise RuntimeError("Cannot read current physical memory")
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                     text=True, timeout=10)
    values = output.strip().splitlines()
    if len(values) != 1:
        raise RuntimeError("This local queue expects exactly one GPU")
    return {"available_ram_gib": status.avail_phys / 2**30, "free_gpu_mib": float(values[0])}


def stop_owned(process):
    if process.poll() is None:
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=20)
        process.wait(timeout=20)
