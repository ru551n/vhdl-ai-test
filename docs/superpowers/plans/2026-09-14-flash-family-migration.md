# Flash family migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move vhdl-ai-test's `modules/flash_model` QSPI NOR flash VC into awesome-vunit-vcs as the `flash` family (entity `flash`, all configuration in one `flash_t` generic), merge it via a PR, then delete it from vhdl-ai-test.

**Architecture:** The simulator-independent device model moves to `awesome_vunit_vcs.flash` with a `FlashConfig` dataclass instead of named profiles, time in integer femtoseconds, and a `FlashBackend` class in `flash/vunit_backend.py` that VHDL creates through `vcs_python_pkg`. VHDL owns pins and time; every configuration value lives in `flash_t`. Errors become reports (`common/reports.py`), never exceptions escaping into the bridge.

**Tech Stack:** VHDL-2008 (GHDL + NVC), VUnit 5.0.0.dev12 fork with package hooks, vunit-python-bridge, Python 3.10+ (ruff 0.16.7, mypy strict, pytest, numpy), Sphinx.

**Spec:** `docs/superpowers/specs/2026-09-14-flash-family-migration-design.md` (vhdl-ai-test). Source of the port: `git show origin/main:modules/flash_model/...` in `/home/sebbe/git/vhdl-ai-test`.

## Global Constraints

- Work only in `/home/sebbe/git/awesome-vunit-vcs-flash` (branch `feat/flash-family`) and worktrees made from it. NEVER touch `/home/sebbe/git/awesome-vunit-vcs` (another session's checkout).
- Do not modify: `src/awesome_vunit_vcs/vhdl/common/**`, `src/awesome_vunit_vcs/common/**`, `src/awesome_vunit_vcs/ethernet/**`, `tests/vhdl/run.py`, `pyproject.toml`, `.github/**`, `vunit_pkg.toml`.
- Shared docs touched only: `docs/index.rst`, `docs/python_api.rst`, `docs/roadmap.md` (flash as its own "Other VC families" section, not in the Ethernet table).
- Follow `CONTRIBUTING.md` exactly: MPL-2.0 header on every file; no `g_`/`c_`/`v_`/`p_` prefixes except `p_` record fields; architectures `a`/`tb`; `main` process; `_inst` labels; no column alignment; `std_ulogic` ports; VHDL-2008.
- Python: ruff (line 120, E,F,W,I,B,UP,SIM,RUF), `ruff format`, mypy strict over `src/awesome_vunit_vcs`, py310 compatible, imports only stdlib + numpy (the docs build does not install the package).
- Venv: `/home/sebbe/git/awesome-vunit-vcs-flash/.venv/bin/`. In a worktree, prefix Python commands with `PYTHONPATH=$PWD/src` so the worktree's package shadows the editable install.
- Gates (all must pass): `ruff check .`, `ruff format --check .`, `mypy`, `pytest`, `VUNIT_SIMULATOR=nvc python tests/vhdl/run.py -p 4`, `VUNIT_SIMULATOR=ghdl python tests/vhdl/run.py -p 4`, `sphinx-build -W --keep-going -b html docs <tmp>`.
- Backend methods never raise: any exception becomes `Report(Severity.FAILURE, ...)`; model check failures (content mismatch) become `Report(Severity.ERROR, ...)`.
- No reals cross the bridge. Times cross as `(hi, lo)` with `fs = hi * 2**30 + lo` (`common.vunit_bridge.join_time/split_time`).
- Directive packing (30 bits, `LAYOUT_VERSION = 1`) is unchanged.
- Commits end with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01846z9LGpF2PiS5xXA9qXUX`.

---

## File map

| New file (awesome-vunit-vcs) | From (vhdl-ai-test `modules/flash_model/`) | Task |
|---|---|---|
| `src/awesome_vunit_vcs/flash/__init__.py` | `python/flash_model/__init__.py` | 1 |
| `flash/{array,commands,directive,images,mode,protection,sfdp}.py` | same names | 1 |
| `flash/timing.py` | `timing.py`, femtoseconds, no limits | 1 |
| `flash/config.py` | `profiles.py` → `FlashConfig` | 1 |
| `flash/device.py` | `device.py`, femtoseconds, `FlashConfig` | 1 |
| `flash/vunit_backend.py` | replaces `flash_model_bridge.py` + `registry.py` | 2 |
| `tests/python/flash_harness.py`, `tests/python/test_flash_*.py` | `python/tests/*` | 1, 2 |
| `vhdl/flash/qspi_pkg.vhd`, `qspi_master_pkg.vhd`, `qspi_master.vhd`, `qspi_flash_cmd_pkg.vhd` | `sim/` same names | 3 |
| `vhdl/flash/flash_context.vhd` | new | 3 (qspi), 4 (adds flash_pkg) |
| `tests/vhdl/tb_qspi_master.vhd` | `test/tb_qspi_master.vhd` | 3 |
| `vhdl/flash/flash_pkg.vhd` (declaration + body in one file) | `sim/flash_model_pkg.vhd` + `-body.vhd` | 4 |
| `vhdl/flash/flash.vhd` | `sim/flash_model.vhd` | 4 |
| `vhdl/flash/flash_protocol_checker.vhd` | `sim/flash_model_protocol_checker.vhd` | 4 |
| `tests/vhdl/tb_flash.vhd` | `test/tb_flash_model.vhd` | 4 |
| `docs/flash.rst`; edits to `docs/index.rst`, `docs/python_api.rst`, `docs/roadmap.md` | `doc/*.md` (content source) | 5 |

Execution order: Tasks 1→2 (agent A, worktree `~/git/awesome-vunit-vcs-flash-python`, branch `flash-python`) run in parallel with Task 3 (agent B, the clone itself). Then merge A into `feat/flash-family` (rebase, linear). Then Task 4 (agent C, clone) in parallel with Task 5 (agent D, worktree `~/git/awesome-vunit-vcs-flash-docs`, branch `flash-docs`). Then Task 6 (coordinator). Task 7–8 after merge.

---

### Task 1: Python core in `awesome_vunit_vcs.flash`

**Files:**
- Create: `src/awesome_vunit_vcs/flash/{__init__,array,commands,config,device,directive,images,mode,protection,sfdp,timing}.py`
- Create: `tests/python/flash_harness.py`, `tests/python/test_flash_{array,commands,config,device,directive,images,protection,sfdp,timing}.py`

**Interfaces:**
- Produces (exact):
  ```python
  # flash/config.py
  KIB = 1024; MIB = 1024 * KIB
  class AddrModes(IntEnum): BOTH = 0; THREE_ONLY = 3; FOUR_ONLY = 4
  BUSY_KEYS: tuple[str, ...] = ("tPP","tSE","tBE32","tBE64","tCE","tW","tRST","tRES1","tRES2")
  DEFAULT_BUSY_FS: Mapping[str, int]  # tPP 700 us, tSE 45 ms, tBE32 120 ms, tBE64 150 ms, tCE 20 s, tW 10 ms, tRST 30 us, tRES1 3 us, tRES2 1.8 us, all in fs
  @dataclass(frozen=True)
  class FlashConfig:
      size_bytes: int = 16 * MIB; page_bytes: int = 256; sector_bytes: int = 4 * KIB
      block32_bytes: int = 32 * KIB  # 0 = no 32 KiB block erase
      block_bytes: int = 64 * KIB; addr_bytes: int = 3; addr_modes: AddrModes = AddrModes.BOTH
      jedec_id: int = 0xEF4018; electronic_id: int | None = None  # None = derived
      sr1_default: int = 0x00; sr2_default: int = 0x02; sr3_default: int = 0x00
      busy_fs: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_BUSY_FS))
      timing_enabled: bool = True
      def __post_init__(self) -> None: ...  # raises ValueError (old profiles._validate + busy keys exactly BUSY_KEYS, non-negative + addr_bytes consistent with addr_modes + status bytes 0..255 + jedec_id 24-bit)
      def jedec_id_bytes(self) -> bytes: ...
      def device_id(self) -> int: ...  # old profiles.electronic_id
  # flash/timing.py
  class Timing:
      def __init__(self, busy_fs: Mapping[str, int], *, enabled: bool = True) -> None
      enabled: bool
      def set_busy(self, name: str, duration_fs: int) -> None   # KeyError unknown, ValueError negative
      def set_enable(self, enable: bool) -> None
      def busy_fs(self, name: str | None) -> int
      def start_busy(self, now_fs: int, name: str | None) -> int  # returns duration fs
      def is_busy(self, now_fs: int) -> bool
      def deadline_fs(self) -> int
      def clear_busy(self) -> None
  # flash/device.py
  class ContentMismatch(Exception): ...   # raised by check_content / check_content_fill only
  class FlashDevice:
      def __init__(self, config: FlashConfig) -> None
      def reset_state(self) -> None
      def cs_assert(self, now_fs: int) -> int
      def xfer(self, byte_in: int, now_fs: int | None = None) -> int
      def cs_deassert(self, trailing_bits: int, now_fs: int) -> int   # busy duration fs, 0 if none
      def get_stat(self, name: str) -> int   # "busy_deadline_ps" becomes "busy_deadline_fs"
      def preload(self, addr: int, data: bytes) -> None
      def preload_fill(self, addr: int, num_bytes: int, value: int) -> None
      def load_image(self, path: str, fmt: str | None = None, base: int = 0) -> int
      def read_back(self, addr: int, num_bytes: int) -> bytes
      def check_content(self, addr: int, expected: bytes) -> None
      def check_content_fill(self, addr: int, num_bytes: int, value: int) -> None
      def written_regions(self) -> list[tuple[int, int]]
      def set_protection(self, addr: int, num_bytes: int, locked: bool) -> None
      def set_timing(self, name: str, duration_fs: int) -> None
      def set_timing_enable(self, enable: bool) -> None
  # flash/sfdp.py
  def build(config: FlashConfig) -> bytes; def basic_parameter_table(config: FlashConfig) -> list[int]; def read(image: bytes, addr: int, length: int) -> bytes
  ```
  Removed: `profiles.py`, `registry.py`, `LIMIT_KEYS`, `Timing.limits_ps`, `FlashDevice.timing_limits_ps`, all float seconds.

- [ ] **Step 1: Worktree.** `cd /home/sebbe/git/awesome-vunit-vcs-flash && git worktree add -b flash-python ../awesome-vunit-vcs-flash-python feat/flash-family`. All following commands run in `/home/sebbe/git/awesome-vunit-vcs-flash-python` with `V=/home/sebbe/git/awesome-vunit-vcs-flash/.venv/bin` and `PYTHONPATH=$PWD/src`.
- [ ] **Step 2: Copy sources verbatim first.** For each module in the file map: `git -C /home/sebbe/git/vhdl-ai-test show origin/main:modules/flash_model/python/flash_model/<m>.py > src/awesome_vunit_vcs/flash/<m>.py`; same for tests into `tests/python/test_flash_<m>.py` and `harness.py` → `tests/python/flash_harness.py`. Rewrite imports: `flash_model.` → `awesome_vunit_vcs.flash.`, `from flash_model import X` → `from awesome_vunit_vcs.flash import X`, `from .harness`/`tests.harness` → `from flash_harness import ...` (check how `tests/python/test_*.py` import `helpers` and do the same). Add the MPL-2.0 header (copy the 3 `#` lines from `src/awesome_vunit_vcs/common/reports.py`'s sibling files, e.g. `ethernet/vunit_backend.py`) to each file. Commit "flash: import the model sources unchanged" only after Step 3 passes; do not commit a red tree.
- [ ] **Step 3: Tests first for the new shapes.** Edit tests before code:
  - `test_flash_config.py` (replaces test_profiles): defaults equal the table above; `FlashConfig(size_bytes=3*MIB)` raises ValueError; `FlashConfig(page_bytes=300)` raises; `FlashConfig(addr_bytes=5)` raises; `FlashConfig(busy_fs={"tPP": 1})` raises (missing keys); negative busy raises; `FlashConfig(addr_bytes=3, addr_modes=AddrModes.FOUR_ONLY)` raises; `device_id()` for 0xEF4018 is 0x17 and honours `electronic_id=0x42`; `jedec_id_bytes()` is `b"\xef\x40\x18"`; `block32_bytes=0` is accepted.
  - `test_flash_timing.py`: every seconds value → fs int (e.g. `700e-6` → `700_000_000_000`); remove limits tests; `Timing(DEFAULT_BUSY_FS)`.
  - `test_flash_device.py`, `flash_harness.py`: `FlashDevice(profiles.build(...))` → `FlashDevice(FlashConfig(...))`; named profiles map to configs: `generic_16mib` → `FlashConfig()`, `w25q32jv` → `FlashConfig(size_bytes=4*MIB, jedec_id=0xEF4016)`, `mt25ql256` → `FlashConfig(size_bytes=32*MIB, jedec_id=0x20BA19)`, `generic_32mib_4b` → `FlashConfig(size_bytes=32*MIB, jedec_id=0xEF4019, addr_bytes=4)`; all `now_s` floats → `now_fs` ints (`Host` in harness keeps an int fs clock); `check_content` mismatch expects `ContentMismatch`.
  - `test_flash_sfdp.py`: profile dicts → `FlashConfig`; `addr_modes_only` → `addr_modes`.
  Run `PYTHONPATH=$PWD/src $V/pytest -q tests/python -k flash`. Expected: FAIL (imports of config / fs signatures).
- [ ] **Step 4: Implement.** Write `config.py` per Interfaces (move `_validate` logic into `__post_init__`; `busy_fs` is copied into a plain dict in `__post_init__` via `object.__setattr__` so the frozen instance does not alias the caller's mapping). Convert `timing.py` and `device.py` to integer fs (`self.now_fs: int = 0`; `wip(now_fs)`; `cs_deassert` returns `self.timing.start_busy(...)`). `device.py` reads `config.*` attributes instead of dict keys; `sfdp.py` likewise (`config.block32_bytes or None`, `config.addr_modes`). Delete `profiles.py`. `__init__.py` exports `FlashConfig`, `FlashDevice`, `ContentMismatch`, `AddrModes`.
- [ ] **Step 5: Lint/type clean.** `$V/ruff format src/awesome_vunit_vcs/flash tests/python`; `$V/ruff check --fix .` then fix the rest by hand (B905: add `strict=True` to `zip`; SIM108; B007). `$V/mypy` must print `Success`. For `commands.py`'s `Command(**_READ_COMMON)` spreads (57 mypy errors): replace each `_X_COMMON` dict with a factory, e.g. `def _read(opcode: int, name: str, **overrides: Any) -> Command: return replace(_READ_BASE, opcode=opcode, name=name, **overrides)` where `_READ_BASE = Command(0, "", Op.READ, ...)` and `dataclasses.replace` is used; keep the table data identical and `test_flash_commands.py` unchanged apart from imports.
- [ ] **Step 6: Gates.** `PYTHONPATH=$PWD/src $V/pytest -q` → all pass (122 existing + flash); `$V/ruff check .`, `$V/ruff format --check .`, `$V/mypy` clean.
- [ ] **Step 7: Commit** (`git add src/awesome_vunit_vcs/flash tests/python; git commit`) with message "Add the flash family's device model" and a body explaining FlashConfig, femtoseconds and the dropped profiles.

### Task 2: `FlashBackend`

**Files:**
- Create: `src/awesome_vunit_vcs/flash/vunit_backend.py`
- Create: `tests/python/test_flash_backend.py` (replaces `test_bridge.py`)

**Interfaces:**
- Consumes: Task 1 (`FlashConfig`, `FlashDevice`, `ContentMismatch`, `BUSY_KEYS`, `directive.LAYOUT_VERSION`, `directive.ignore_rest`), `common.reports.{ReportQueue, Severity, encode_reports}`, `common.vunit_bridge.{join_time, split_time}`.
- Produces (exact; VHDL Task 4 calls these strings):
  ```python
  class FlashBackend:
      def __init__(self, name: str, *, size_bytes: int, page_bytes: int, sector_bytes: int,
                   block32_bytes: int, block_bytes: int, addr_bytes: int, addr_modes: int,
                   jedec_id: int, electronic_id: int, sr1_default: int, sr2_default: int,
                   sr3_default: int, busy: Mapping[str, tuple[int, int]], timing_enabled: bool) -> None
          # electronic_id < 0 → None. An invalid config does not raise: it queues a FAILURE
          # report and builds FlashDevice(FlashConfig()) so later calls stay harmless.
      def layout_version(self) -> int
      def num_reports(self) -> int
      def take_reports(self) -> str                       # encode_reports(self.reports.take())
      def cs_assert(self, hi: int, lo: int) -> int         # directive; on exception FAILURE + ignore_rest()
      def xfer(self, byte_in: int, hi: int = -1, lo: int = 0) -> int  # hi < 0 → no time
      def cs_deassert(self, trailing_bits: int, hi: int, lo: int) -> npt.NDArray[np.int32]  # [busy_hi, busy_lo, num_reports]
      def reset(self) -> int                               # all control methods return num_reports()
      def preload(self, data: Any, addr: int) -> int       # data: sequence of ints 0..255 (numpy array from the bridge)
      def preload_fill(self, addr: int, num_bytes: int, value: int) -> int
      def load_image(self, path: str, fmt: str, base: int) -> int   # fmt "auto" → None
      def read_back(self, addr: int, num_bytes: int) -> npt.NDArray[np.int32]   # empty on failure
      def check_content(self, expected: Any, addr: int) -> int      # ContentMismatch → ERROR
      def check_content_fill(self, addr: int, num_bytes: int, value: int) -> int
      def written_regions(self) -> npt.NDArray[np.int32]  # flat [addr, len, ...]
      def set_timing_enable(self, enable: bool) -> int
      def set_timing(self, name: str, hi: int, lo: int) -> int
      def set_protection(self, addr: int, num_bytes: int, locked: bool) -> int
      def get_stat(self, name: str) -> int                 # unknown → FAILURE, returns 0
  ```
  Report text always starts with the VC `name` (as `MonitorBackend` does). FAILURE text: `f"{name}: {method} raised {type(exc).__name__}: {exc}"`.

- [ ] **Step 1: Failing tests** in `test_flash_backend.py` (same worktree/env as Task 1):
  - `layout_version() == LAYOUT_VERSION`.
  - A read of the JEDEC ID through `cs_assert/xfer/cs_deassert` with `(hi, lo) = split_time(t)` returns bytes EF 40 18 (port `test_bridge.py`'s bus-level tests; keep the directive-following harness style).
  - `cs_deassert` after a sector erase with timing enabled returns `[*split_time(45_000_000_000_000), 0]`.
  - Busy times: `busy={"tSE": split_time(1_000)}` plus the others → erase returns duration 1000 fs.
  - `check_content([0x00], 0)` on an erased device returns 1; `take_reports()` decodes (`decode_reports`) to one `Severity.ERROR` whose message contains `"0x00000000"` and starts with the name.
  - `xfer(0x03)` with no `cs_assert` first, or `preload([300], 0)`, or `get_stat("nope")`, or `set_timing("tXX", 0, 1)` → returns without raising, one `Severity.FAILURE` report.
  - An invalid config (`size_bytes=3`) → constructor does not raise, `num_reports() == 1` FAILURE.
  - `read_back` returns `np.int32` dtype; `written_regions()` after a page program is `[page_addr, 256]`.
  - Two backends are independent (preload one, read the other → 0xFF).
  Run `PYTHONPATH=$PWD/src $V/pytest -q tests/python/test_flash_backend.py` → FAIL (module missing).
- [ ] **Step 2: Implement** `vunit_backend.py`. One private helper `_guard(self, method: str, fn: Callable[[], T], fallback: T) -> T` catching `ContentMismatch` → ERROR and `Exception` → FAILURE; every public method goes through it. `_bytes(values)` validates 0..255 with `ValueError` (port from the old bridge). Module docstring: the VHDL call strings above, the error model, and why there is no registry (one session per VC).
- [ ] **Step 3:** tests pass; ruff/format/mypy clean over the whole repo; full `pytest` green.
- [ ] **Step 4: Commit** "Add the flash backend the VHDL component creates".

### Task 3: QSPI master side (VHDL only)

**Files:**
- Create: `src/awesome_vunit_vcs/vhdl/flash/{qspi_pkg,qspi_master_pkg,qspi_master,qspi_flash_cmd_pkg,flash_context}.vhd`
- Create: `tests/vhdl/tb_qspi_master.vhd`

**Interfaces:**
- Produces: packages `qspi_pkg`, `qspi_master_pkg`, `qspi_flash_cmd_pkg` in library `awesome_vunit_vcs` with the same public subprograms and types as the originals (names unchanged; only prefix removal on generics/constants/variables). Entity `qspi_master` with `generic (qspi_master : qspi_master_t)` and ports `m2s : out qspi_m2s_t := qspi_m2s_init; s2m : in qspi_s2m_t` (renamed from `qspi_m2s`/`qspi_s2m`). If a simulator rejects a generic named like its entity, use `handle` and note it in the commit body. `flash_context`:
  ```vhdl
  context flash_context is
    library vunit_lib;
    context vunit_lib.vunit_context;
    context vunit_lib.com_context;
    use vunit_lib.sync_pkg.all;
    library awesome_vunit_vcs;
    use awesome_vunit_vcs.qspi_pkg.all;
    use awesome_vunit_vcs.qspi_master_pkg.all;
    use awesome_vunit_vcs.qspi_flash_cmd_pkg.all;
  end context;
  ```
  (Mirror `vhdl/ethernet/ethernet_context.vhd` for exact shape.) Task 4 adds `use awesome_vunit_vcs.flash_pkg.all;`.

- [ ] **Step 1:** Copy the four sources and `tb_qspi_master.vhd` from `git -C /home/sebbe/git/vhdl-ai-test show origin/main:modules/flash_model/{sim,test}/...`. Replace `library flash_model; use flash_model.X.all;` with `use work.X.all;` in sources; testbench uses `library awesome_vunit_vcs; context awesome_vunit_vcs.flash_context;`. Add MPL headers (copy from `vhdl/ethernet/ethernet_pkg.vhd`), keeping each file's prose header below it.
- [ ] **Step 2:** Apply house style: rename every `g_x`→`x`, `c_x`→`x`, `v_x`→`x` (resolve any collision a rename creates by picking a descriptive name, never by re-adding a prefix); un-align column-aligned declarations/maps; comments stay. `vc_name` in `create_std_cfg` for the master: `"qspi_master"`; provider: reuse the family's provider string — look at `ethernet_pkg.vhd` for how `ethernet_provider` is defined and define `flash_provider : string := "awesome_vunit_vcs"`-style equivalently in `qspi_master_pkg` (Task 4's `flash_pkg` uses the same constant by `use work.qspi_master_pkg.all` or its own).
- [ ] **Step 3: Two SCK periods without `run.py` configs.** `tests/vhdl/run.py` must not change, so the generic `g_sck_period_ns` goes away: every test case whose expectations derive from the period runs inside `for period_idx in periods'range loop set_sck_period(net, master, periods(period_idx)); ... end loop;` with `constant periods : time_vector := (20 ns, 33 ns);`. Keep all 12 `run("test_...")` names.
- [ ] **Step 4: Run** from the clone root: `VUNIT_SIMULATOR=nvc .venv/bin/python tests/vhdl/run.py -p 4 'lib.tb_qspi_master.*'` then the same with `ghdl`. Expected: `pass 12 of 12` on both. Grep the output for `pass 12 of 12`, not just the exit code.
- [ ] **Step 5: Commit** "Add the QSPI master VC to the flash family".

### Task 4: `flash` entity, `flash_pkg`, protocol checker, `tb_flash`

Precondition: coordinator has rebased `flash-python` (Tasks 1–2) onto `feat/flash-family` in the clone.

**Files:**
- Create: `src/awesome_vunit_vcs/vhdl/flash/{flash_pkg,flash,flash_protocol_checker}.vhd`
- Modify: `src/awesome_vunit_vcs/vhdl/flash/flash_context.vhd` (add `use awesome_vunit_vcs.flash_pkg.all;`)
- Create: `tests/vhdl/tb_flash.vhd`

**Interfaces:**
- Consumes: Task 2 call strings; Task 3 packages; `vcs_python_pkg` (`new_vc_session`, `create_backend`, `backend_integer`, `backend_integer_array`, `log_reports`, `py_str`, `py_bool`); `python_bridge.python_context` only for `python_session_t` and the array form `call("vc.preload", arg(data), arg(address), session => session)` (precedent: `vhdl/ethernet/gmii_monitor.vhd`).
- Produces:
  ```vhdl
  type flash_addr_modes_t is (both, three_only, four_only);
  type flash_t is record
    p_std_cfg : std_cfg_t;
    p_size_bytes, p_page_bytes, p_sector_bytes, p_block_bytes : positive;
    p_block32_bytes : natural;                 -- 0 = none
    p_addr_bytes : positive range 3 to 4;
    p_addr_modes : flash_addr_modes_t;
    p_jedec_id : natural;
    p_electronic_id : integer;                 -- -1 = derived from jedec_id
    p_sr1_default, p_sr2_default, p_sr3_default : natural range 0 to 255;
    p_t_pp, p_t_se, p_t_be32, p_t_be64, p_t_ce, p_t_w, p_t_rst, p_t_res1, p_t_res2 : delay_length;
    p_timing_enabled : boolean;
    p_t_sck_min, p_t_sck_high_min, p_t_sck_low_min, p_t_slch, p_t_chsh, p_t_shsl, p_t_dvch, p_t_chdx : delay_length;  -- 0 = not checked
    p_t_clqv, p_t_shqz : delay_length;
    p_protocol_checks : boolean;
  end record;
  impure function new_flash(
    id : id_t := null_id;
    size_bytes : positive := 16 * 1024 * 1024; page_bytes : positive := 256; sector_bytes : positive := 4096;
    block32_bytes : natural := 32768; block_bytes : positive := 65536;
    addr_bytes : positive range 3 to 4 := 3; addr_modes : flash_addr_modes_t := both;
    jedec_id : natural := 16#EF4018#; electronic_id : integer := -1;
    sr1_default : natural range 0 to 255 := 16#00#; sr2_default : natural range 0 to 255 := 16#02#; sr3_default : natural range 0 to 255 := 16#00#;
    t_pp : delay_length := 700 us; t_se : delay_length := 45 ms; t_be32 : delay_length := 120 ms; t_be64 : delay_length := 150 ms;
    t_ce : delay_length := 20 sec; t_w : delay_length := 10 ms; t_rst : delay_length := 30 us; t_res1 : delay_length := 3 us; t_res2 : delay_length := 1800 ns;
    timing_enabled : boolean := true;
    t_sck_min : delay_length := 7519 ps; t_sck_high_min : delay_length := 3 ns; t_sck_low_min : delay_length := 3 ns;
    t_slch : delay_length := 5 ns; t_chsh : delay_length := 5 ns; t_shsl : delay_length := 30 ns;
    t_dvch : delay_length := 2 ns; t_chdx : delay_length := 3 ns;
    t_clqv : delay_length := 6 ns; t_shqz : delay_length := 6 ns;
    protocol_checks : boolean := true;
    unexpected_msg_type_policy : unexpected_msg_type_policy_t := fail
  ) return flash_t;
  impure function get_id(flash : flash_t) return id_t;   -- also get_logger, get_checker, as_sync
  constant flash_layout_version : natural := 1;          -- + the c_dir_* constants without the c_ prefix (dir_action_shift, ...)
  function decode_directive(packed : integer) return flash_directive_t;
  -- procedures: exactly the old flash_model_pkg set with flash_model_t → flash_t:
  -- flash_preload (integer_array_t and std_ulogic_vector), flash_preload_fill, flash_load_image,
  -- flash_read_back (+ await_flash_read_back_reply, blocking form), flash_check_content,
  -- flash_check_content_fill, flash_get_written_regions, flash_set_timing_enable,
  -- flash_set_timing(net, flash, name : string; duration : delay_length), flash_set_protection,
  -- flash_wait_until_ready, flash_reset, flash_get_stat
  ```
  Entity: `entity flash is generic (flash : flash_t); port (m2s : in qspi_m2s_t; s2m : out qspi_s2m_t := qspi_s2m_init); end entity;` (same fallback rule as Task 3 if the generic name is rejected). Removed: `profile`, `p_state`/instance id, `g_python_bridge_path`, `g_protocol_checks`, `to_seconds`, `now_seconds`, `c_tl_*`. Checker entity: `flash_protocol_checker` with `generic (flash : flash_t); port (m2s : in qspi_m2s_t)`; limits come from the handle, `0 ns` disables that one check, `p_protocol_checks = false` disables all.

- [ ] **Step 1: Port `tb_flash.vhd` first** (tests define the API): copy `tb_flash_model.vhd`, rename entity/instances, use `flash_context`, `new_flash(...)`, drop `profile =>`, drop `g_protocol_checks` generic maps in favour of `new_flash(protocol_checks => false)`, master ports `m2s`/`s2m`. Convert the two mock-based negative tests to the repo style: `disable_stop(get_logger(checked_flash), error)` in setup; after the traffic `check_equal(get_log_count(get_logger(checked_flash), error), 1, ...)` (use the actual count the traffic produces, then prove it with a comment) and `reset_log_count(get_logger(checked_flash), error)`; the unchecked VC asserts count 0. Keep all 26 test names. Add:
  - `test_content_mismatch_is_a_check_failure`: `disable_stop(get_logger(flash_a), error)`; `flash_check_content(net, flash_a, 16#000100#, bytes_of((0 => 16#00#)))` on erased flash → `get_log_count(..., error) = 1`; reset the count.
  - `test_non_default_configuration`: a fifth pair (`custom_master`, `custom_flash := new_flash(size_bytes => 32 * 1024 * 1024, addr_bytes => 4, jedec_id => 16#20BA19#, t_shsl => 60 ns, timing_enabled => false)`), proving over the bus: read ID = 20 BA 19; preload at `16#1800000#` (above 16 MiB) and read back with a 4-byte-address read (no EN4B needed: powers up 4-byte); and, with the master's default 50 ns CS deselect, two back-to-back commands give a tSHSL error count of 1 on `custom_flash` (disable_stop/count/reset).
  - `test_metavalue_on_io_is_reported`: a sixth bus driven directly by the testbench (no master): signal record `raw_m2s`; bit-bang CS low, 8 SCK cycles with `io(0) <= 'X'` on one of them, CS high → error count 1 on `raw_flash`, reset count.
- [ ] **Step 2: Implement `flash_pkg.vhd`** (declaration + body in one file, like `ethernet_pkg.vhd` is organized; check whether ethernet splits files and match it). Port procedures and message types from `flash_model_pkg` / `-body`; add a body-private `function to_python_time(value : time) return string` returning `"(" & hi & ", " & lo & ")"` with `hi = value / (2**30 * 1 fs)`, `lo = (value - hi * 2**30 * 1 fs) / 1 fs` (mirrors `vcs_python_pkg.time_split`; `2**30 * 1 fs` as a local constant `time_split`).
- [ ] **Step 3: Implement `flash.vhd`**. `main` process:
  1. `session := new_vc_session(get_id(flash))`; `create_backend(session, "awesome_vunit_vcs.flash.vunit_backend", "FlashBackend", py_str(get_name(get_id(flash))) & ", size_bytes=" & ... & ", busy={'tPP': " & to_python_time(flash.p_t_pp) & ", ...}, timing_enabled=" & py_bool(...))`; `addr_modes` sent as 0/3/4; then `log_reports` if `backend_integer(session, "num_reports()") > 0`.
  2. `check_equal(checker, backend_integer(session, "layout_version()"), flash_layout_version, ...)`.
  3. Message loop: each control message maps to its call string from Task 2; after every call returning a count, `if count > 0 then log_reports(session, logger, checker); end if;`. After `read_back`/`written_regions` (arrays), check `num_reports()`. Arrays passed with `call("vc.preload", arg(data), arg(address), session => session)` and `call("vc.check_content", arg(data), arg(address), session => session)` returning integer counts. Pop message fields into variables before building the call (argument evaluation order is undefined).
  The `session` must be shared with the `pins` process: declare it as a constant in the architecture is impossible (impure), so create it in `main`, pass it to `pins` via a signal of `python_session_t` is not allowed either if the type is an access/protected — check `python_session_t`'s definition in the python_bridge package: if it is a plain record/integer, create it in a constant `session : python_session_t := new_vc_session(get_id(flash))` at architecture level and `create_backend` once in `main` before raising `initialized`; `pins` waits for `initialized`.
  4. `pins` process: same shift engine; `cs_assert("cs_assert(" & hi_lo(now) & ")")`, `xfer(b)` or `xfer(b, hi, lo)` when volatile, `cs_deassert` via `backend_integer_array`, read `[busy_hi, busy_lo, num_reports]`, `deallocate`, busy duration `busy_hi * time_split + busy_lo * 1 fs`, `log_reports` when `num_reports > 0`. Output delays from `flash.p_t_clqv/p_t_shqz` (no signals needed). Metavalue check in the receive loop: before sampling, if `is_x` on any of the sampled lanes (`io(0)` for x1, `io(1 downto 0)` for x2, all four for x4; mirror qspi_pkg's lane mapping) → `check_failed(checker, "Metavalue on io at " & to_string(now))`, then sample with `to_01`.
  5. `busy_timer` process and `busy_started/busy_finished` scheme unchanged.
  6. `flash_protocol_checker_inst : entity work.flash_protocol_checker generic map (flash => flash) port map (m2s => m2s);`
- [ ] **Step 4: Port the protocol checker** from the ps integer_vector to the handle's `delay_length` fields; `0 ns` = skip. Message prefix `"flash protocol: "`; keep the "CS high time between commands X is shorter than the Y minimum" wording.
- [ ] **Step 5: Run** `VUNIT_SIMULATOR=nvc .venv/bin/python tests/vhdl/run.py -p 4` and `ghdl`. Expected: all tests of tb_gmii, tb_xgmii (whatever is on main), tb_qspi_master (12) and tb_flash (29) pass on both; grep `pass N of N` and that `tb_flash` shows 29.
- [ ] **Step 6: Mutation proof** (do not commit): set `protocol_checks => true` on the unchecked VC → `test_protocol_checks_can_be_switched_off` must fail; restore. Make `FlashBackend.check_content` return 0 without comparing → `test_content_mismatch_is_a_check_failure` fails; restore. Record both in the report.
- [ ] **Step 7: Commit** "Add the flash VC and its protocol checker".

### Task 5: Documentation

Precondition: Tasks 1–2 merged. Worktree `git worktree add -b flash-docs ../awesome-vunit-vcs-flash-docs feat/flash-family`.

**Files:**
- Create: `docs/flash.rst`
- Modify: `docs/index.rst` (toctree: `flash` after `gmii`; one intro sentence that the flash family provides a QSPI NOR flash model and QSPI master), `docs/python_api.rst` (new `Flash` section after the Ethernet PHY section with `.. automodule::` for `awesome_vunit_vcs.flash.config`, `.device`, `.array`, `.commands`, `.directive`, `.timing`, `.protection`, `.mode`, `.sfdp`, `.images`, `.vunit_backend`), `docs/roadmap.md` (append `## Other VC families` with a short paragraph: QSPI NOR flash (x1/x2/x4, 3/4-byte addressing, SFDP, protection, busy timing, pin-level checks) is available; parallel NOR, SPI NAND, eMMC not planned).

- [ ] **Step 1:** Write `flash.rst` in the tone and structure of `docs/gmii.rst`, content from `git show origin/main:modules/flash_model/doc/flash_model.md` and `flash_model_ffi_contract.md`, rewritten for this repo: Interface (port table `m2s`/`s2m` fields from `qspi_pkg`), Creating the component (`new_flash` with an options table grouped geometry / identity / status defaults / busy times / pin checks / output delays), Initializing content (the three tiers: `flash_preload`, `flash_preload_fill`, `flash_load_image` with formats .hex/.srec/.s19/.bin/.json), Inspecting content (`flash_read_back`, `flash_check_content[_fill]`, `flash_get_written_regions`), Timing (busy times, `flash_set_timing` names tPP..tRES2, `flash_set_timing_enable`, `flash_wait_until_ready`), Protocol checks (list, `0 ns` disables one, `protocol_checks => false` all), Errors (checker vs logger, the negative-test pattern with `disable_stop`), Supported commands (table from `commands.py` COMMAND_TABLE: opcode, name, notes), The QSPI master and command layer (`qspi_transfer`, `qspi_flash_*` procedures), Using the model from Python (`FlashDevice(FlashConfig(...))` snippet), Performance note (one bridge call per byte, fine to ~64 KiB of bus traffic per test; image-sized content goes through preload/check procedures).
- [ ] **Step 2:** Build: `$V/pip install -q -r docs/requirements.txt` (into the clone venv if missing) and `PYTHONPATH=$PWD/src $V/sphinx-build -W --keep-going -b html docs /tmp/claude-1000/flash-docs-build`. Fix every warning in the docstrings of `src/awesome_vunit_vcs/flash/*.py` (typical: RST inline literal needs double backticks where a single backtick is followed by a letter, indentation of bullet continuation lines, `*` in text). Docstring-only edits in `flash/`.
- [ ] **Step 3:** `$V/ruff check .`, `$V/ruff format --check .` (ruff formats Markdown code blocks in roadmap.md), `$V/mypy` still clean.
- [ ] **Step 4: Commit** "Document the flash family".

### Task 6: Integrate, PR, merge (coordinator)

- [ ] Rebase `flash-docs` onto `feat/flash-family` (after Task 4), resolve any flash.rst ↔ flash_pkg API drift by editing the docs to the real API.
- [ ] Fresh gates in the clone: ruff, format, mypy, pytest, HDL on NVC and GHDL (`pass N of N`), Sphinx -W, `pytest tests/packaging` (builds the wheel: confirms `vhdl/flash/*.vhd` and `flash/*.py` ship).
- [ ] Review the whole diff against the spec and CONTRIBUTING (prefixes, alignment, headers, `--@` markers).
- [ ] `git fetch && git rebase origin/main`; rerun HDL + pytest if main moved; `git push -u origin feat/flash-family`; `gh pr create` with a body: summary, layout, deviations from the old flash_model (profiles → FlashConfig/flash_t, fs time, reports instead of exceptions, content mismatch now a check failure, no generic sck period config), test counts, what reviewers should check.
- [ ] SendMessage `awesome-vunit-vcs [6f78ab]`: PR URL.
- [ ] Watch CI with `gh run list --branch feat/flash-family` / `gh run view <id> --json jobs`; verify HDL jobs' logs show `pass N of N` with tb_flash included (strip ANSI).
- [ ] SendMessage the coordinator "about to merge"; wait briefly for objections; rebase onto latest main if needed (re-run CI); `gh pr merge <n> --rebase --delete-branch`.

### Task 7: Remove flash_model from vhdl-ai-test

- [ ] Branch `refactor/remove-flash-model` from origin/main (carry the spec + plan commits from `refactor/migrate-flash-model`).
- [ ] `git rm -r modules/flash_model`; remove the `flash-model` job from `.github/workflows/ci.yml`; remove flash_model mentions in `requirements.txt` comments and any docs (`git grep -n flash_model`), keeping `vu.add_python()` in run.py only if another testbench uses the bridge (`git grep -n python_context`), otherwise keep it anyway (harmless) and say so in the PR.
- [ ] Local NVC run of the remaining suite; push; PR pointing to the awesome-vunit-vcs PR; merge on green (merge commit, as the repo's earlier PRs).

### Task 8: Follow-ups

- [ ] vhdl-skills `shared/Vunit.md` §7: replace the flash_model registry example with the awesome-vunit-vcs pattern (one Python session per VC via `new_session(id)`, backend object `vc`), commit + push.
- [ ] Update memory note `vunit-python-execute-resets-module-globals` to drop the stale registry path.
- [ ] Report to the user: links, what changed, what they must do (nothing mandatory; optional: review the merged PR, consumers of flash_model switch to `vu.add_package("awesome-vunit-vcs")`).
