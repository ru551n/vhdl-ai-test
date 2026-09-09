"""HIR scheduling (doc/tosa_compiler_plan.md §11 row 06, §13 M7).

`schedule` turns a `stage='mapped'` `HirModule` into `stage='scheduled'`:
a deterministic topological order over `HirOp.deps` unioned with the
*implied* dependencies of the dataflow (an op reading a buffer depends on
the op that writes it -- or, for a buffer view, on whoever writes the
bytes it shares, see `HirModule.storage_dependencies`). Ties break on the original op order so the same
input always yields the same schedule. The implied deps are folded back
into each op's `deps` so the dependency structure is explicit in dumps,
not left implicit in the buffer graph.
"""

from __future__ import annotations

import heapq

from cnnc.errors import CompilerError
from cnnc.hir.ir import HirModule
from cnnc.hir.verify import verify_hir


class ScheduleError(CompilerError):
    """Raised when `schedule` cannot find a total order (dependency cycle)."""


def _implied_deps(module: HirModule) -> dict[str, set[str]]:
    writer_of: dict[str, str] = {}
    for op in module.ops:
        for buffer_id in op.writes:
            writer_of[buffer_id] = op.id
    implied: dict[str, set[str]] = {op.id: set() for op in module.ops}
    for op in module.ops:
        for buffer_id in op.reads:
            # Through aliases too: a buffer view has no writer of its own,
            # so the op that fills it is the one writing the storage they
            # share (`HirModule.storage_dependencies`).
            for source in module.storage_dependencies(buffer_id):
                writer = writer_of.get(source)
                if writer is not None and writer != op.id:
                    implied[op.id].add(writer)
    return implied


def schedule(module: HirModule) -> HirModule:
    implied = _implied_deps(module)
    all_deps = {op.id: set(op.deps) | implied[op.id] for op in module.ops}

    order_index = {op.id: i for i, op in enumerate(module.ops)}
    dependents: dict[str, set[str]] = {op.id: set() for op in module.ops}
    indegree: dict[str, int] = {}
    for op_id, deps in all_deps.items():
        indegree[op_id] = len(deps)
        for dep in deps:
            dependents[dep].add(op_id)

    # Min-heap keyed by original op order: among all currently-ready ops
    # (indegree 0), the earliest one in the input order is scheduled next.
    # This is the "stable" tie-break the plan asks for.
    heap = [(order_index[op_id], op_id) for op_id, d in indegree.items() if d == 0]
    heapq.heapify(heap)

    seq_of: dict[str, int] = {}
    seq = 0
    while heap:
        _, op_id = heapq.heappop(heap)
        seq_of[op_id] = seq
        seq += 1
        for dependent in dependents[op_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                heapq.heappush(heap, (order_index[dependent], dependent))

    if len(seq_of) != len(module.ops):
        stuck = sorted(op_id for op_id in order_index if op_id not in seq_of)
        raise ScheduleError(f"dependency cycle among ops: {stuck}", stage="schedule")

    new_ops = [op.replace(seq=seq_of[op.id], deps=tuple(sorted(all_deps[op.id]))) for op in module.ops]
    new_ops.sort(key=lambda op: op.seq)

    scheduled = module.with_ops(new_ops).replace(stage="scheduled")
    verify_hir(scheduled)
    return scheduled
