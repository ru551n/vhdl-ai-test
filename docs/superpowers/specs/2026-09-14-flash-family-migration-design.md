# Flash family migration: vhdl-ai-test → awesome-vunit-vcs

- **Date:** 2026-09-14
- **Status:** draft, awaiting review
- **Source:** `vhdl-ai-test/modules/flash_model` (on `main`)
- **Target:** `github.com/ru551n/awesome-vunit-vcs`, library `awesome_vunit_vcs`

## Goal

Move the QSPI NOR flash verification component into awesome-vunit-vcs as a *flash* family that
follows that repository's architecture, then remove it from vhdl-ai-test. Behaviour stays the same
apart from the error-model change described below.

## Decisions already made

| Topic | Decision |
|---|---|
| Scope | **Move.** The flash VC lives only in awesome-vunit-vcs; vhdl-ai-test drops `modules/flash_model`. |
| Approach | **Conform** to awesome-vunit-vcs: per-VC Python session with a backend object, reports routed to the VC checker. |
| Top level | Entity **`flash`**. |
| Configuration | **All configuration, including actor/logger/checker, lives in the `flash : flash_t` generic**, built by `new_flash(...)`. The entity has no other generics. |
| QSPI master | **Inside the flash family** (`vhdl/flash/`). |
| Delivery | PR from a **separate clone**; merged by this session when every CI job is green. |

## Constraints

- Other agents commit directly to awesome-vunit-vcs `main`. Never touch that checkout, its branches,
  index or working tree; work in a separate clone. Re-read `origin/main` before every push.
- **No changes** to `src/awesome_vunit_vcs/vhdl/common/**`, `src/awesome_vunit_vcs/common/**`,
  `tests/vhdl/run.py`, `pyproject.toml` or the CI workflows.
- Shared files this work does touch — `docs/index.rst`, `docs/python_api.rst`, `docs/roadmap.md` — are
  announced to the active awesome-vunit-vcs session before they are edited.
- CI gates that must pass: ruff lint and format, strict mypy, pytest (Python matrix), packaging check,
  HDL tests on GHDL and NVC, Sphinx docs with `-W`. Strict mypy covers `src/awesome_vunit_vcs` only
  (`[tool.mypy] files`); the tests must pass ruff but are not type-checked.
- The docs build does not install the package: `docs/conf.py` puts `src` on `sys.path`, and
  `docs/requirements.txt` provides only Sphinx extensions and numpy. Every `awesome_vunit_vcs.flash`
  module must therefore import with the standard library and numpy alone — no VUnit, no bridge — or
  its `automodule` entry fails the `-W` build.

## Target layout

| vhdl-ai-test (today) | awesome-vunit-vcs |
|---|---|
| `sim/flash_model.vhd` | `src/awesome_vunit_vcs/vhdl/flash/flash.vhd` (entity `flash`) |
| `sim/flash_model_pkg.vhd` + body | `vhdl/flash/flash_pkg.vhd` + body (`flash_t`, `new_flash`, procedures) |
| `sim/flash_model_protocol_checker.vhd` | `vhdl/flash/flash_protocol_checker.vhd` |
| `sim/qspi_pkg.vhd` | `vhdl/flash/qspi_pkg.vhd` |
| `sim/qspi_master_pkg.vhd`, `qspi_master.vhd` | `vhdl/flash/qspi_master_pkg.vhd`, `qspi_master.vhd` |
| `sim/qspi_flash_cmd_pkg.vhd` | `vhdl/flash/qspi_flash_cmd_pkg.vhd` |
| — | `vhdl/flash/flash_context.vhd` |
| `python/flash_model/{array,commands,device,directive,images,mode,protection,sfdp,timing}.py` | `src/awesome_vunit_vcs/flash/` (same modules) |
| `python/flash_model/profiles.py` | `src/awesome_vunit_vcs/flash/config.py` (`FlashConfig`: defaults and validation, no part table) |
| `python/flash_model_bridge.py`, `python/flash_model/registry.py` | **removed**; replaced by `flash/vunit_backend.py` (`FlashBackend`) |
| `python/tests/*` | `tests/python/test_flash_*.py` |
| `test/tb_flash_model.vhd`, `test/tb_qspi_master.vhd` | `tests/vhdl/tb_flash.vhd`, `tests/vhdl/tb_qspi_master.vhd` |
| `doc/*.md` | `docs/flash.rst` |

Every migrated file gets the repository's MPL-2.0 header.

## Configuration: `flash_t`

The handle is a record of private fields; `new_flash` builds it, with defaults equal to today's
`generic_16mib` profile. VHDL is the single source of truth; the Python backend receives the
configuration when it is created.

| Group | Fields (defaults) |
|---|---|
| Identity | `std_cfg` from `create_std_cfg(id, provider, vc_name, unexpected_msg_type_policy)`: actor, logger, checker, id |
| Geometry | `size_bytes` (16 MiB), `page_bytes` (256), `sector_bytes` (4 KiB), `block32_bytes` (32 KiB, 0 = none), `block_bytes` (64 KiB) |
| Addressing | `addr_bytes` power-up mode (3), `addr_modes` advertised in SFDP (`both`, `three_only`, `four_only`; default `both`) |
| Identification | `jedec_id` (`16#EF4018#`), `electronic_id` (-1 = derived from the capacity byte) |
| Status registers | `sr1_default` (`16#00#`), `sr2_default` (`16#02#`, QE set), `sr3_default` (`16#00#`) |
| Busy times | `t_pp` 700 us, `t_se` 45 ms, `t_be32` 120 ms, `t_be64` 150 ms, `t_ce` 20 s, `t_w` 10 ms, `t_rst` 30 us, `t_res1` 3 us, `t_res2` 1.8 us |
| Timing | `timing_enabled` initial state (true); still switchable at run time with `flash_set_timing_enable` |
| Pin checks | `t_sck_min` 7.519 ns, `t_sck_high_min` 3 ns, `t_sck_low_min` 3 ns, `t_slch` 5 ns, `t_chsh` 5 ns, `t_shsl` 30 ns, `t_dvch` 2 ns, `t_chdx` 3 ns (0 = not checked) |
| Output delays | `t_clqv` 6 ns, `t_shqz` 6 ns |
| Checking | `protocol_checks` (true) |

The named Python parts (`w25q128jv`, `w25q32jv`, `mt25ql256`, `generic_32mib_4b`) are not carried
forward: they differ only in size, JEDEC ID and power-up addressing, which are `new_flash` arguments.

Pin-timing limits and output delays are used only by VHDL, so they are no longer sent to Python and
`get_timing_limits` is removed. Busy-time overrides at run time (`flash_set_timing`) remain.

## VHDL ↔ Python boundary

All bridge access uses `vcs_python_pkg`, except array arguments, which use `call(...)` directly as
`gmii_monitor` already does for `expect_payload`.

- **Creation:** `session := new_vc_session(get_id(flash))`, then
  `create_backend(session, "awesome_vunit_vcs.flash.vunit_backend", "FlashBackend", <arguments>)`.
  The arguments are built from the handle with `py_str`/`py_bool` and integer literals; busy times are
  passed as `(hi, lo)` femtosecond pairs. Each instance has its own session and `vc`, so the instance
  registry and instance ids disappear.
- **Handshake:** `backend_integer(session, "layout_version()")` checked against `c_layout_version`.
- **Time:** simulation times cross as femtosecond halves, `t = hi * 2**30 + lo`, the convention of
  `vcs_python_pkg` and `common/vunit_bridge.py` (`join_time`/`split_time`). The split is computed in
  `flash_pkg` with the same `2**30 fs` constant. No `real` values cross the boundary.
- **Per transaction:** `backend_integer(session, "cs_assert(hi, lo)")` → packed directive;
  `backend_integer(session, "xfer(b)")`, or `"xfer(b, hi, lo)"` when the previous directive was
  volatile → packed directive; `backend_integer_array(session, "cs_deassert(bits, hi, lo)")` →
  `[hi, lo]` busy duration, deallocated after use. The packed directive layout (30 bits) and
  `LAYOUT_VERSION` are unchanged.
- **Control plane:** `backend_exec` for scalar calls (`reset()`, `preload_fill(...)`,
  `load_image(...)`, `set_timing_enable(...)`, `set_timing(name, hi, lo)`, `set_protection(...)`,
  `check_content_fill(...)`); `backend_integer_array` for `read_back(...)` and
  `written_regions()`; `backend_integer` for `get_stat(...)`;
  `call("vc.preload", arg(data), arg(addr), session => session)` and the same for `check_content`.
- **Performance:** `call` is itself an `eval` of a generated call string, so the per-byte path costs
  the same as today.

String arguments must not contain `"` or `\`: file names, stat names and timing names passed here do
not.

## Error model

- Content and request checks in the model (`check_content`, `check_content_fill`, invalid control
  requests) queue a `Report(Severity.ERROR)` instead of raising. After each control-plane call the VC
  runs `log_reports(session, logger, checker)`, so they become `check_failed` on the VC checker:
  counted and mockable.
- Internal errors raise, and the bridge reports them on the VC's logger through the session identity.
- The pin-level protocol checker is unchanged and reports on the VC checker.
- **Behaviour change:** a content mismatch no longer stops the simulation with a Python traceback; it
  is an ordinary check failure. A new HDL test mocks the checker and asserts the mismatch is reported.

## Python package

- `awesome_vunit_vcs.flash`: the device model modules, simulator independent.
- `config.py` replaces `profiles.py`: a `FlashConfig` dataclass with defaults and the existing
  validation (powers of two, divisibility, `addr_bytes` 3 or 4); `electronic_id` and SFDP use it.
- `timing.py` keeps the busy table and deadline; the pin limits are removed.
- `vunit_backend.py`: `FlashBackend(config...)` with the methods listed above plus `take_reports()`.
- Passes ruff (lint and format) and strict mypy. The known work: the `Command` table in
  `commands.py` (57 errors from one `**dict` spread), untyped dicts in `profiles.py`, two in `sfdp.py`,
  one in `device.py`.

## Tests

- **Python:** the existing suite moves to `tests/python/test_flash_*.py` with imports from
  `awesome_vunit_vcs.flash`. Tests of the removed pieces (`profiles` table, `get_timing_limits`, the
  registry) are replaced by tests of `FlashConfig` validation and of `FlashBackend`, including the
  report queue.
- **HDL:** `tests/vhdl/tb_flash.vhd` (today's 26 cases) and `tests/vhdl/tb_qspi_master.vhd` (12
  cases). The two-SCK-period runs move inside `tb_qspi_master` through `set_sck_period`, so
  `tests/vhdl/run.py` is unchanged. New cases:
  - a content mismatch reported as a mocked check failure;
  - a non-default `flash_t` (32 MiB, 4-byte power-up, another JEDEC ID, a changed pin limit) proven
    over the bus: JEDEC ID, addressing, and the protocol checker firing at the changed limit.

## Documentation

- New `docs/flash.rst`: instantiating `flash`, `new_flash` configuration, the three ways to
  initialize, checking, timing, statistics, the QSPI master.
- `docs/index.rst`: `flash` in the user guide toctree, and one intro sentence noting flash as a
  family beyond Ethernet.
- `docs/python_api.rst`: a Flash section of `automodule` entries.
- `docs/roadmap.md`: one line recording the flash family.

## Delivery

1. Clone to `~/git/awesome-vunit-vcs-flash`; branch `feat/flash-family` from `origin/main`; venv with
   `tests/packaging/unreleased-requirements.txt` and `pip install -e ".[dev]"`.
2. Message the active awesome-vunit-vcs session about the shared docs files before editing them.
3. Implement, run every CI gate locally (HDL on NVC), push, open the PR.
4. Before merging: rebase onto the latest `origin/main`, let CI re-run, and merge only when every job
   is green, as a **rebase merge**. The repository has no merge commits and no earlier PRs (changes
   land directly on `main`), and `main` is unprotected, so nothing server-side enforces CI: the
   all-green gate is held by this session.
5. Then, in vhdl-ai-test, a separate PR removes `modules/flash_model`, the `flash-model` CI job and
   the mentions in `.github/workflows/ci.yml` and `requirements.txt`; merged when its CI is green.
6. Follow-ups outside both repositories: `vhdl-skills/shared/Vunit.md` §7 cites the flash registry as
   the example of state that must survive `exec_file`; replace it with a reference to per-VC sessions
   in awesome-vunit-vcs. Update the matching memory note.

## Out of scope

- New flash features or commands.
- Changes to awesome-vunit-vcs common infrastructure, CI, packaging or `tests/vhdl/run.py`.
- Publishing to PyPI.

## Risks

- **Concurrent edits** to the shared docs files by the active session: announce first, rebase late.
- **Unreleased pins** (VUnit package-hooks branch, vunit-python-bridge) may move under the PR; CI on
  the rebased branch is the gate.
- **Behaviour change** in content-check reporting; covered by the new mocked test and documented in
  `docs/flash.rst`.
