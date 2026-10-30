# SPDX-License-Identifier: Apache-2.0
"""FABRIC (MNNVL) memory after cuCheckpointProcess restore, driver API only.

    python3 fabric_after_restore.py create
        A child creates a FABRIC allocation, its parent checkpoints and restores it, then the child creates and
        exports another one. Expected: 0 0. Driver 610.43.02: 0 800 (CUDA_ERROR_NOT_PERMITTED).

    python3 fabric_after_restore.py export DIR       # on node A, DIR shared with node B
    python3 fabric_after_restore.py import DIR       # on node B, same NVLink domain
        `export` writes the fabric handle of a fresh allocation (filled with 7) to DIR. `import` uses FABRIC once,
        is checkpointed and restored by its parent, then imports the handle. Expected: 0 [7, 7, 7, 7]. Driver
        610.43.02: 800 (CUDA_ERROR_NOT_PERMITTED); with only the IMEX subscription restored, 1
        (CUDA_ERROR_INVALID_VALUE).

Run with LD_PRELOAD=libfabric_restore_shim.so to see the workaround restore the expected results.
"""

import ctypes as C
import multiprocessing as mp
import os
import sys
import time

FABRIC = 8  # CU_MEM_HANDLE_TYPE_FABRIC


class Loc(C.Structure):
    _fields_ = [("type", C.c_int), ("id", C.c_int)]


class Flags(C.Structure):
    _fields_ = [("compressionType", C.c_ubyte), ("gpuDirectRDMACapable", C.c_ubyte), ("usage", C.c_ushort),
                ("reserved", C.c_ubyte * 4)]


class Prop(C.Structure):
    _fields_ = [("type", C.c_int), ("requestedHandleTypes", C.c_int), ("location", Loc),
                ("win32HandleMetaData", C.c_void_p), ("allocFlags", Flags)]


class Access(C.Structure):
    _fields_ = [("location", Loc), ("flags", C.c_int)]


def cuda():
    cu = C.CDLL("libcuda.so.1")
    assert cu.cuInit(0) == 0
    dev, ctx = C.c_int(), C.c_void_p()
    cu.cuDeviceGet(C.byref(dev), 0)
    cu.cuDevicePrimaryCtxRetain(C.byref(ctx), dev)
    cu.cuCtxSetCurrent(ctx)
    prop = Prop(type=1, requestedHandleTypes=FABRIC, location=Loc(1, 0))  # pinned, device 0
    gran = C.c_size_t()
    cu.cuMemGetAllocationGranularity(C.byref(gran), C.byref(prop), 0)
    return cu, prop, gran.value


def create(cu, prop, size, fill=None):
    """Returns (rc of cuMemCreate, rc of cuMemExportToShareableHandle, handle bytes)."""
    h = C.c_ulonglong()
    rc = cu.cuMemCreate(C.byref(h), C.c_size_t(size), C.byref(prop), C.c_ulonglong(0))
    if rc:
        return rc, None, None
    if fill is not None:
        va = C.c_ulonglong()
        cu.cuMemAddressReserve(C.byref(va), C.c_size_t(size), C.c_size_t(0), C.c_ulonglong(0), C.c_ulonglong(0))
        cu.cuMemMap(va, C.c_size_t(size), C.c_size_t(0), h, C.c_ulonglong(0))
        cu.cuMemSetAccess(va, C.c_size_t(size), C.byref(Access(Loc(1, 0), 3)), C.c_size_t(1))
        cu.cuMemsetD8_v2(va, C.c_ubyte(fill), C.c_size_t(size))
        cu.cuCtxSynchronize()
    handle = (C.c_ubyte * 64)()
    return rc, cu.cuMemExportToShareableHandle(handle, h, FABRIC, C.c_ulonglong(0)), bytes(handle)


def checkpoint_restore(pid):
    cu = C.CDLL("libcuda.so.1")
    cu.cuInit(0)
    return [(op, getattr(cu, "cuCheckpointProcess" + op)(pid, None)) for op in ("Lock", "Checkpoint", "Restore", "Unlock")]


def run_checkpointed(target, *args):
    ready, go = mp.Event(), mp.Event()
    p = mp.Process(target=target, args=(ready, go, *args))
    p.start()
    ready.wait()
    time.sleep(1)
    print("checkpoint/restore:", checkpoint_restore(p.pid), flush=True)
    go.set()
    p.join(120)
    if p.is_alive():
        print("child hung after restore", flush=True)
        p.kill()


def create_child(ready, go):
    cu, prop, size = cuda()
    print("before checkpoint: create, export =", create(cu, prop, size)[:2], flush=True)
    ready.set()
    go.wait()
    print("after restore:     create, export =", create(cu, prop, size)[:2], flush=True)


def import_child(ready, go, d):
    cu, prop, size = cuda()
    create(cu, prop, size)  # use FABRIC once before the checkpoint, like NCCL with MNNVL
    ready.set()
    go.wait()
    while not os.path.exists(f"{d}/handle"):
        time.sleep(0.5)
    handle = (C.c_ubyte * 64).from_buffer_copy(open(f"{d}/handle", "rb").read())
    h = C.c_ulonglong()
    rc = cu.cuMemImportFromShareableHandle(C.byref(h), handle, FABRIC)
    value = None
    if rc == 0:
        va = C.c_ulonglong()
        cu.cuMemAddressReserve(C.byref(va), C.c_size_t(size), C.c_size_t(0), C.c_ulonglong(0), C.c_ulonglong(0))
        cu.cuMemMap(va, C.c_size_t(size), C.c_size_t(0), h, C.c_ulonglong(0))
        cu.cuMemSetAccess(va, C.c_size_t(size), C.byref(Access(Loc(1, 0), 3)), C.c_size_t(1))
        out = (C.c_ubyte * 4)()
        cu.cuMemcpyDtoH_v2(out, va, C.c_size_t(4))
        value = list(out)
    print("after restore:     import =", rc, value, flush=True)
    open(f"{d}/done", "w").close()


def export(d):
    cu, prop, size = cuda()
    rc, rc_export, handle = create(cu, prop, size, fill=7)
    assert rc == 0 and rc_export == 0, (rc, rc_export)
    os.makedirs(d, exist_ok=True)
    open(f"{d}/handle.tmp", "wb").write(handle)
    os.replace(f"{d}/handle.tmp", f"{d}/handle")
    print("exported; waiting for the importer", flush=True)
    while not os.path.exists(f"{d}/done"):
        time.sleep(0.5)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    mode = sys.argv[1]
    if mode == "create":
        run_checkpointed(create_child)
    elif mode == "export":
        export(sys.argv[2])
    elif mode == "import":
        run_checkpointed(import_child, sys.argv[2])
    else:
        sys.exit(__doc__)
