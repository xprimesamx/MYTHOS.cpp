# `distributed.cpp` — Distributed Training

**Path:** `src/distributed.cpp`

Distributed training support: data parallelism across multiple devices/nodes.

## Functions

| Function | Description |
|----------|-------------|
| `init_distributed()` | Initialize distributed environment |
| `all_reduce(tensor, op)` | All-reduce across workers |
| `broadcast(tensor, root)` | Broadcast from root worker |
| `barrier()` | Synchronize all workers |
| `get_world_size()` | Number of workers |
| `get_rank()` | Current worker rank |

## Data Parallel Training

```
Worker 0:        data_0 → forward → loss → grad → all_reduce(grad)
Worker 1:        data_1 → forward → loss → grad → all_reduce(grad)
Worker 2:        data_2 → forward → loss → grad → all_reduce(grad)
                                                    ↓
                                             All workers get avg(grad)
                                                    ↓
                                             optimizer.step()
```

## Communication Backend

- Single-node: shared memory (fast)
- Multi-node: MPI (planned)
- Gradient averaging via all-reduce
