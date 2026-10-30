# cuda-ckpt-fabric-restore

An `LD_PRELOAD` workaround for multi-node NVLink (MNNVL) FABRIC memory after a CUDA process checkpoint restore
(`cuCheckpointProcessRestore`) on NVIDIA driver 610.43.02, plus a driver-API-only reproducer.

## The problem

Once a process is restored, it can no longer use FABRIC memory:

- If the process used FABRIC memory before the checkpoint, creating a FABRIC allocation and importing a FABRIC
  handle both return `CUDA_ERROR_NOT_PERMITTED` (800).
- Even when that is fixed, or if the process never used FABRIC memory before, importing returns
  `CUDA_ERROR_INVALID_VALUE` (1).

NCCL with MNNVL re-creates and re-imports its buffers in `ncclCommResume`, so suspend → checkpoint → restore → resume
fails on MNNVL even with the NCCL fixes in [NVIDIA/nccl#2530](https://github.com/NVIDIA/nccl/pull/2530).

## Root cause

We compared libcuda's RM ioctls in a fresh process and in a restored one (strace plus an `LD_PRELOAD` ioctl
logger). The restore recreates the process's RM client under a new handle and drops two pieces of state that
libcuda sets up only once:

1. **IMEX channel subscription.**
   - Before its first FABRIC use, libcuda opens `/dev/nvidia-caps-imex-channels/channelN` and sends
     `NV0000_CTRL_CMD_CLIENT_SUBSCRIBE_TO_IMEX_CHANNEL` (0xd08) on its root client.
   - After restore, the channel fd is gone and libcuda does not subscribe the new client.
   - The FABRIC allocation (`NV_MEMORY_FABRIC`) still succeeds, but the export that completes it
     (`NV_MEMORY_EXPORT`) and every import (`NV_MEMORY_FABRIC_IMPORT_V2`) are refused with
     `NV_ERR_INSUFFICIENT_PERMISSIONS`.
2. **OS event for fabric imports.**
   - libcuda allocates an OS event at init (`/dev/nvidiaN`, `NV_ESC_REGISTER_FD`, `NV_ESC_ALLOC_OS_EVENT`) and
     passes it as `NV00F9_ALLOCATION_PARAMETERS.pOsEvent`.
   - After restore, it passes -1. RM fails the import with `NV_ERR_OBJECT_NOT_FOUND`.
   - If the import is made to succeed with a valid event, libcuda then waits on fd -1 forever.

Separately, FABRIC allocations that went through the checkpoint can no longer be imported from another node
(`CUDA_ERROR_INVALID_HANDLE`). Allocations created after the restore can. NCCL re-creates every released buffer on
resume, so it is not affected by this.

## The workaround

`fabric_restore_shim.c` interposes `ioctl`, `poll` and `ppoll`. It records the channel libcuda subscribed to and
how libcuda allocated its OS event. It then retries only the FABRIC calls that RM refused:

- **Refused with `NV_ERR_INSUFFICIENT_PERMISSIONS` on a client that is not subscribed:** subscribe that client,
  then retry.
- **Import still refused while `pOsEvent` is -1:** allocate an equivalent OS event on the client, retry with it,
  and redirect libcuda's wait on fd -1 to that event.

Calls that RM accepts pass through unchanged. That includes the allocations the driver itself re-creates during
the restore. An earlier version subscribed proactively and disturbed the restore.

```bash
make
LD_PRELOAD=$PWD/libfabric_restore_shim.so <your program>   # every process that is checkpointed
FABRIC_RESTORE_SHIM_VERBOSE=1                              # log each retry to stderr
```

This relies on RM ioctl layouts and libcuda behaviour that are not public interfaces. It is tested only on driver
610.43.02, aarch64, GB300 NVL72 with IMEX channel 0. Remove it once the driver restores this state itself.

## Reproducer

`repro/fabric_after_restore.py` uses only the driver API (ctypes). It does not need NCCL, PyTorch or a container.

```bash
python3 repro/fabric_after_restore.py create                 # one node
python3 repro/fabric_after_restore.py export /shared/dir     # node A
python3 repro/fabric_after_restore.py import /shared/dir     # node B, same NVLink domain
```

Results on GB300, driver 610.43.02:

| | `create`: create, export after restore | `import`: import after restore |
|---|---|---|
| stock driver | 800, – | 800 |
| only the IMEX subscription restored | 0, 0 | 1 |
| with `libfabric_restore_shim.so` | **0, 0** | **0, `[7, 7, 7, 7]`** |

The "only the IMEX subscription restored" row comes from an earlier experiment that re-sent the subscription from
Python, without the OS-event fix.

## Validation with NCCL and vLLM

The setup is 2 × 4 GB300 in one NVL72 rack, with `NCCL_MNNVL_ENABLE=1` and NCCL 2.30.7 plus NVIDIA/nccl#2530.

- **NCCL collectives.** We ran `all_gather`, `all_gatherv`, `all_reduce` and `reduce_scatterv` from 256 to 4M
  elements, both eagerly and replayed from a CUDA graph, through suspend → checkpoint → restore → resume.
  - All results were correct.
  - Each GPU held 3 MiB while checkpointed.
  - Each rank re-imported its 64 peer buffers.
- **vLLM end to end.** vLLM served DeepSeek-V4.1-Flash with DP8/EP8 and `--enable-nccl-comm-suspend`, using the
  CUDA checkpoint executor for sleep.
  - gsm8k accuracy stayed at 0.90–0.91 before sleep and after each of 3 wakes.
  - Each GPU held ≤129 MiB while asleep.

## License

Apache-2.0
