# flash_model — using the verification component

A QSPI NOR flash device for VUnit testbenches. Instantiate it, wire it to your
controller, and drive it from VHDL. The device model itself is Python, but a
testbench never says so.

See `flash_model_req.md` for what it guarantees and `flash_model_proposal.md`
for why it is built the way it is.

## Instantiating

```vhdl
constant c_flash : flash_model_t := new_flash_model(profile => "generic_16mib");
...
flash_model_inst : entity flash_model.flash_model
  generic map (
    g_flash => c_flash
  )
  port map (
    sck => flash_sck,
    cs_n => flash_cs_n,
    io => flash_io
  );
```

`new_flash_model` takes geometry overrides (`size_bytes`, `page_bytes`,
`sector_bytes`, `block_bytes`, `addr_bytes`, `jedec_id`) for a part that is not
worth adding to `python/flash_model/profiles.py`.

Several instances may coexist; each gets its own model, its own logger and its
own checker.

## Initializing the array

Everything you do not initialize reads `0xFF`, like an erased part — you never
need to fill the device first. Storage is sparse, so only what you write costs
memory.

There are three ways in, chosen by how much data there is. They differ in how
much crosses the language boundary, which is what makes a 16 MiB image cheap.

**Scattered literals** — the bytes cross. Good to a few KiB.

```vhdl
-- A vector literal, most significant byte first:
flash_preload(net, c_flash, 16#001000#, std_ulogic_vector'(x"DE_AD_BE_EF"));

-- Or computed data, as an integer_array_t of bytes:
variable v_data : integer_array_t := new_1d(length => 256, bit_width => 8, is_signed => false);
...
for i in 0 to 255 loop
  set(v_data, i, i);
end loop;
flash_preload(net, c_flash, 16#020000#, v_data);
-- v_data is still valid here: the VC was handed a copy, not your array.
```

**A large uniform region** — only a length crosses, and nothing is materialized,
so this is O(1) however big it is.

```vhdl
flash_preload_fill(net, c_flash, 16#100000#, 1024 * 1024, 16#00#);
```

**An image file** — only the file name crosses; Python opens the file itself.
Intel HEX and S-record are natively sparse formats and stay sparse.

```vhdl
flash_load_image(net, c_flash, tb_path(runner_cfg) & "images/boot.hex");
flash_load_image(net, c_flash, tb_path(runner_cfg) & "fw.bin",
                 format => "bin", base_address => 16#400000#);
```

Accepted formats: `.hex` (Intel HEX), `.srec` / `.s19` (Motorola S-record),
`.bin` (raw, placed at `base_address`) and `.json`. `format => "auto"`, the
default, picks by extension.

Initialization deliberately **bypasses the write-enable latch and block
protection** — it is test setup, not a device operation. That is what lets you
preload a region and then lock it, to prove the DUT cannot program it.

## A typical test-case setup

```vhdl
-- Required. One Python interpreter serves the whole simulation and namespaces
-- are not reset between test cases, so without this the previous test's array,
-- status bits and mode state leak into this one.
flash_reset(net, c_flash);

flash_set_timing_enable(net, c_flash, false);  -- this test does not care about busy time
flash_preload(net, c_flash, 0, std_ulogic_vector'(x"01_02_03_04"));
flash_set_protection(net, c_flash, 16#000000#, 16#001000#, locked => true);
```

## Checking afterwards

```vhdl
-- Compared in Python, so a mismatch reports the address and both values with a
-- traceback -- rather than a VHDL loop reporting only its first bad index.
flash_check_content(net, c_flash, 16#001000#, v_expected);

-- Proves an erase without building a 4 KiB expected array:
flash_check_content_fill(net, c_flash, 16#002000#, 4096, 16#FF#);

-- Read data back into VHDL when you want to inspect it yourself:
flash_read_back(net, c_flash, 16#001000#, 16, v_data);

-- The sparse map of everything the DUT touched, as flat [addr, len] pairs --
-- the natural scoreboard for "did it write only where it was supposed to".
flash_get_written_regions(net, c_flash, v_regions);
```

`flash_read_back` and `flash_get_written_regions` hand you ownership of the
returned `integer_array_t`; `deallocate` it when you are done.

## Timing

```vhdl
flash_set_timing_enable(net, c_flash, true);          -- the global switch
flash_set_timing(net, c_flash, "tSE", 45 ms);         -- override one duration
flash_wait_until_ready(net, c_flash);                 -- block until not busy
```

Durations are named the way the part's datasheet names them: `tPP` (page
program), `tSE` (sector erase), `tBE32` / `tBE64` (block erase), `tCE` (chip
erase), `tW` (status write), `tRST`, `tRES1` / `tRES2` (reset and deep
power-down). Passing an unknown name raises in Python and lists the valid
ones, so a typo tells you what it should have been.

With timing enabled, a program or erase holds write-in-progress for its
duration; every command except a status read is ignored while it runs, and
status polling keeps working. With it disabled, all durations are zero and the
same test runs in no simulated time.

Pin-level timing (SCK period, CS setup/hold, CS deselect) is checked against the
profile and reported through the component's own checker, so a negative test can
`mock`/`unmock` its logger. Clock-to-output and output-disable are applied as
output delays, not checked.

## Statistics

```vhdl
flash_get_stat(net, c_flash, "program_count", v_count);
flash_get_stat(net, c_flash, "erase_count", v_count);
flash_get_stat(net, c_flash, "ignored_command_count", v_count);
```

`ignored_command_count` is the one to reach for when a controller is
misbehaving and you do not yet know why: it counts every command the device
silently dropped. Real parts drop commands without complaint, so a controller
bug can otherwise look like a device that simply did nothing.

Once you know commands *are* being dropped, the granular counters say why:

| stat | meaning |
|---|---|
| `wel_reject_count` | a program or erase arrived without a write enable |
| `wip_reject_count` | a command arrived while the device was busy |
| `protect_reject_count` | a program or erase targeted a locked region |
| `qe_reject_count` | a quad command arrived with quad-enable clear |
| `dpd_reject_count` | a command arrived during deep power-down |
| `unknown_opcode_count` | the opcode is not in the command table |
| `abort_count` | a command ended on a non-multiple of 8 clocks |

Plus the ordinary activity counters: `cmd_count`, `program_count`,
`erase_count`, `chip_erase_count`, `wrsr_count`, `reset_count`,
`bytes_read`, `bytes_programmed`, `bytes_erased`, and `materialized_pages`
— the last being a direct measure of how much the sparse array has actually
allocated.

Current device state is readable the same way: `wip`, `wel`, `qpi`,
`addr_bytes`, `continuous_read`, `timing_enabled`, `sr1`/`sr2`/`sr3`.
