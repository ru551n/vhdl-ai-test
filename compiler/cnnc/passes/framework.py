"""Pass framework (doc/tosa_compiler_plan.md §11): `PassContext`, the
`Pass` protocol, and `run_pipeline`.

Passes are pure: given a `Graph` they return a `Graph`, never mutating
their input (the GIR dataclasses are frozen, so this is the natural
style). A pass with nothing to do may return the *same* `Graph` object
(cheap, and `graph2 is graph` is then a valid "no-op" check); it must
never mutate `graph` in place.

Numbering: `run_pipeline`'s dumps start at `first_index` (default `2`),
matching doc/tosa_compiler_plan.md §11's `NN_<stage>` scheme where `00`/`01`
are reserved for the parsed-MLIR and imported-GIR dumps written by the
driver (M8), before any pass in this package runs.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Protocol, Sequence

from cnnc.gir.ir import Graph
from cnnc.gir.printer import print_gir, to_json
from cnnc.gir.verify import verify
from cnnc.target.contract import Target


@dataclasses.dataclass
class PassContext:
    """Shared, mutable state threaded through one `run_pipeline` call.

    `notes` accumulates human-readable manifest notes (e.g. rounding-mode
    substitutions performed by `legalize_rescale`); `dump_dir`, when set,
    makes `run_pipeline` write a `NN_<pass.name>.txt`/`.json` dump after
    every pass.
    """

    target: Target | None = None
    notes: list[str] = dataclasses.field(default_factory=list)
    dump_dir: Path | None = None


class Pass(Protocol):
    name: str

    def run(self, graph: Graph, ctx: PassContext) -> Graph: ...


def run_pipeline(graph: Graph, passes: Sequence[Pass], ctx: PassContext, first_index: int = 2) -> Graph:
    """Run `passes` in order, verifying and (optionally) dumping after
    each one. Returns the final `Graph`."""
    g = graph
    for idx, p in enumerate(passes, start=first_index):
        g = p.run(g, ctx)
        verify(g)
        if ctx.dump_dir is not None:
            ctx.dump_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{idx:02d}_{p.name}"
            (ctx.dump_dir / f"{stem}.txt").write_text(print_gir(g))
            (ctx.dump_dir / f"{stem}.json").write_text(json.dumps(to_json(g), indent=2, sort_keys=True))
    return g
