library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
context vunit_lib.vunit_context;
context vunit_lib.com_context;
use vunit_lib.integer_array_pkg.all;
use vunit_lib.sync_pkg.all;
use vunit_lib.vc_pkg.all;

-- Control surface for the QSPI NOR flash verification component
-- (flash_model.vhd). Everything a testbench needs in order to drive the model
-- lives here; a testbench never mentions Python, and never touches the flash
-- pins directly.
--
-- The device model itself is Python (modules/flash_model/python/) and the VC
-- calls into it over VUnit's VHDL-to-Python bridge. This package therefore
-- carries two kinds of declaration:
--
--   * the ordinary VUnit VC apparatus -- a handle record built on
--     vc_pkg.create_std_cfg, one message type per command, and one procedure
--     per command that sends it to the VC's actor;
--   * the VHDL half of the FFI contract -- the packed-directive layout and the
--     timing-limit index order, which MUST agree byte for byte with
--     doc/flash_model_ffi_contract.md and python/flash_model_bridge.py.
--
-- The second kind is guarded at run time: the VC asserts the Python side's
-- layout_version() against c_layout_version when it creates its model
-- instance, so a field-order drift fails at time 0 rather than as an
-- inexplicable wrong byte later in a test.
--
-- See doc/flash_model_req.md and doc/flash_model_proposal.md.
package flash_model_pkg is

  ------------------------------------------------------------------------------
  -- FFI contract: keep in lockstep with doc/flash_model_ffi_contract.md
  ------------------------------------------------------------------------------

  -- Bumped whenever any packed layout or index order below changes.
  constant c_layout_version : natural := 1;

  -- What the VC should do next. Every field describes the SAME, next action --
  -- one tense throughout, so there is no off-by-one between "lanes for the next
  -- phase" and "byte to drive now".
  type flash_action_t is (receive, transmit, ignore_rest);

  subtype lane_count_t is positive range 1 to 4;  -- 1, 2 or 4; 3 is never valid

  type flash_directive_t is record
    action : flash_action_t;
    -- Lane width for THIS action.
    lanes : lane_count_t;
    -- SCK cycles with the IOs high-impedance BEFORE this action. A prefix, not
    -- a phase of its own -- which is what lets 0x6B (address x1, 8 dummy
    -- cycles, data x4) be a single directive, and what covers 0x77, where the
    -- dummy cycles precede a *receive* rather than a transmit.
    pre_dummy_cycles : natural;
    -- Valid when action = transmit.
    byte_out : natural range 0 to 255;
    -- The model's state can change under this byte (a status register, say), so
    -- the VC must pass a fresh `now` on the next xfer call. Array reads leave
    -- this clear and so pay nothing on the hot path.
    is_volatile : boolean;
    -- Reserved for the chunked-prefetch extension; always 1 today.
    num_bytes : positive;
  end record;

  -- Bit layout of the integer the Python side returns from cs_assert and xfer.
  -- A packed scalar rather than an integer_array_t on purpose: python_pkg's
  -- result_array allocates a new integer_array_t on every array-returning call,
  -- so an array-per-byte hot path would allocate -- and leak -- one array per
  -- byte shifted. A scalar allocates nothing.
  constant c_dir_action_shift : natural := 0;
  constant c_dir_action_width : natural := 2;
  constant c_dir_lanes_shift : natural := 2;
  constant c_dir_lanes_width : natural := 3;
  constant c_dir_dummy_shift : natural := 5;
  constant c_dir_dummy_width : natural := 6;
  constant c_dir_byte_out_shift : natural := 11;
  constant c_dir_byte_out_width : natural := 8;
  constant c_dir_flags_shift : natural := 19;
  constant c_dir_flags_width : natural := 4;
  constant c_dir_num_bytes_shift : natural := 23;
  constant c_dir_num_bytes_width : natural := 10;

  constant c_dir_flag_volatile : natural := 0;  -- bit index within the flags field

  -- Unpack what Python returned. Call sites read a record; no bit shifting ever
  -- appears in the VC itself.
  function decode_directive(packed : integer) return flash_directive_t;

  -- Index order of the get_timing_limits() array, in picoseconds. A value of 0
  -- means "not specified, do not check".
  --
  -- Indices 0..7 are CHECKS on DUT-driven pins. Indices 8..9 are DELAYS on
  -- VC-driven pins, scheduled with `after` -- asserting on those would make the
  -- VC fail on its own output.
  constant c_tl_sck_min : natural := 0;
  constant c_tl_sck_high_min : natural := 1;
  constant c_tl_sck_low_min : natural := 2;
  constant c_tl_slch : natural := 3;
  constant c_tl_chsh : natural := 4;
  constant c_tl_shsl : natural := 5;
  constant c_tl_dvch : natural := 6;
  constant c_tl_chdx : natural := 7;
  constant c_tl_clqv : natural := 8;
  constant c_tl_shqz : natural := 9;
  constant c_tl_count : natural := 10;

  constant c_tl_first_delay : natural := c_tl_clqv;  -- 0 .. c_tl_first_delay-1 are checks

  ------------------------------------------------------------------------------
  -- Handle
  ------------------------------------------------------------------------------

  type flash_model_t is record
    -- All private. Use the accessors and procedures below.
    p_std_cfg : std_cfg_t;
    p_size_bytes : positive;
    p_page_bytes : positive;
    p_sector_bytes : positive;
    p_block_bytes : positive;
    p_addr_bytes : positive range 3 to 4;
    p_jedec_id : natural;
    p_profile : string_ptr_t;
  end record;

  -- A pending read-back, redeemed with await_flash_read_back_reply.
  alias flash_reference_t is msg_t;

  -- `profile` names an entry in python/flash_model/profiles.py; the remaining
  -- arguments override that profile's geometry when non-zero, so a test can ask
  -- for an odd-sized part without adding a profile.
  impure function new_flash_model(
    profile : string := "generic_16mib";
    size_bytes : positive := 16 * 1024 * 1024;
    page_bytes : positive := 256;
    sector_bytes : positive := 4096;
    block_bytes : positive := 65536;
    addr_bytes : positive range 3 to 4 := 3;
    jedec_id : natural := 16#EF4018#;
    id : id_t := null_id;
    unexpected_msg_type_policy : unexpected_msg_type_policy_t := fail
  ) return flash_model_t;

  impure function as_sync(flash : flash_model_t) return sync_handle_t;
  impure function get_logger(flash : flash_model_t) return logger_t;
  impure function get_checker(flash : flash_model_t) return checker_t;

  ------------------------------------------------------------------------------
  -- Initialization: tiered by scale, so the bytes crossing the FFI stay bounded
  -- no matter how large the image is.
  ------------------------------------------------------------------------------

  -- Scattered literals. The bytes cross the FFI, so keep this to a few KiB.
  -- Sparse because nothing between preloaded regions is ever allocated.
  procedure flash_preload(
    signal net : inout network_t;
    flash : flash_model_t;
    address : natural;
    data : integer_array_t
  );

  procedure flash_preload(
    signal net : inout network_t;
    flash : flash_model_t;
    address : natural;
    data : std_ulogic_vector
  );

  -- O(1) in num_bytes: Python stores a run, and never materializes the bytes.
  procedure flash_preload_fill(
    signal net : inout network_t;
    flash : flash_model_t;
    address : natural;
    num_bytes : positive;
    value : natural range 0 to 255 := 16#FF#
  );

  -- Only the file name crosses the FFI; Python opens the file itself. Intel HEX
  -- and S-record are natively sparse formats, so a scattered image loads as
  -- scattered regions with no VHDL involvement at all.
  procedure flash_load_image(
    signal net : inout network_t;
    flash : flash_model_t;
    file_name : string;
    format : string := "auto";
    base_address : natural := 0
  );

  ------------------------------------------------------------------------------
  -- Read-back and checking
  ------------------------------------------------------------------------------

  -- Non-blocking first, per the house convention: issue the request, redeem it
  -- only when the value is actually needed.
  procedure flash_read_back(
    signal net : inout network_t;
    flash : flash_model_t;
    address : natural;
    num_bytes : positive;
    variable reference : inout flash_reference_t
  );

  procedure await_flash_read_back_reply(
    signal net : inout network_t;
    variable reference : inout flash_reference_t;
    variable data : out integer_array_t
  );

  -- Blocking convenience wrapper.
  procedure flash_read_back(
    signal net : inout network_t;
    flash : flash_model_t;
    address : natural;
    num_bytes : positive;
    variable data : out integer_array_t
  );

  -- The comparison happens in Python, so a mismatch raises with the address and
  -- both values and arrives as a VUnit FAILURE carrying a traceback -- strictly
  -- more useful than a VHDL loop reporting its first bad index.
  procedure flash_check_content(
    signal net : inout network_t;
    flash : flash_model_t;
    address : natural;
    expected : integer_array_t
  );

  procedure flash_check_content_fill(
    signal net : inout network_t;
    flash : flash_model_t;
    address : natural;
    num_bytes : positive;
    value : natural range 0 to 255
  );

  -- Which regions the DUT actually touched, as a flat [addr, len, addr, len...].
  -- The sparse map is the natural scoreboard for "did it write only where it
  -- was supposed to".
  procedure flash_get_written_regions(
    signal net : inout network_t;
    flash : flash_model_t;
    variable regions : out integer_array_t
  );

  ------------------------------------------------------------------------------
  -- Configuration
  ------------------------------------------------------------------------------

  -- The global timing off switch: collapses every busy duration to zero.
  procedure flash_set_timing_enable(
    signal net : inout network_t;
    flash : flash_model_t;
    enable : boolean
  );

  -- Override one busy duration, e.g. ("sector_erase", 45 ms).
  procedure flash_set_timing(
    signal net : inout network_t;
    flash : flash_model_t;
    name : string;
    duration : delay_length
  );

  -- Lock or unlock a region. A program or erase touching a locked region is
  -- silently ignored, exactly as a real part behaves.
  procedure flash_set_protection(
    signal net : inout network_t;
    flash : flash_model_t;
    address : natural;
    num_bytes : positive;
    locked : boolean
  );

  -- Blocks until the model is no longer write-in-progress.
  procedure flash_wait_until_ready(
    signal net : inout network_t;
    flash : flash_model_t;
    timeout : delay_length := 1 sec
  );

  -- Clear all model state. REQUIRED in test_case_setup: one Python interpreter
  -- serves the whole simulation and namespaces are not reset between test
  -- cases, so state would otherwise leak from one test into the next.
  procedure flash_reset(
    signal net : inout network_t;
    flash : flash_model_t
  );

  -- Named counters kept by the model, e.g. "program_count", "erase_count",
  -- "ignored_command_count".
  procedure flash_get_stat(
    signal net : inout network_t;
    flash : flash_model_t;
    name : string;
    variable value : out integer
  );

  ------------------------------------------------------------------------------
  -- Message types (public so the VC can dispatch on them)
  ------------------------------------------------------------------------------

  constant flash_preload_msg : msg_type_t := new_msg_type("flash preload");
  constant flash_preload_fill_msg : msg_type_t := new_msg_type("flash preload fill");
  constant flash_load_image_msg : msg_type_t := new_msg_type("flash load image");
  constant flash_read_back_msg : msg_type_t := new_msg_type("flash read back");
  constant flash_read_back_reply_msg : msg_type_t := new_msg_type("flash read back reply");
  constant flash_check_content_msg : msg_type_t := new_msg_type("flash check content");
  constant flash_check_content_fill_msg : msg_type_t := new_msg_type("flash check content fill");
  constant flash_written_regions_msg : msg_type_t := new_msg_type("flash written regions");
  constant flash_written_regions_reply_msg : msg_type_t := new_msg_type("flash written regions reply");
  constant flash_set_timing_enable_msg : msg_type_t := new_msg_type("flash set timing enable");
  constant flash_set_timing_msg : msg_type_t := new_msg_type("flash set timing");
  constant flash_set_protection_msg : msg_type_t := new_msg_type("flash set protection");
  constant flash_wait_until_ready_msg : msg_type_t := new_msg_type("flash wait until ready");
  constant flash_reset_msg : msg_type_t := new_msg_type("flash reset");
  constant flash_get_stat_msg : msg_type_t := new_msg_type("flash get stat");
  constant flash_get_stat_reply_msg : msg_type_t := new_msg_type("flash get stat reply");

end package;
