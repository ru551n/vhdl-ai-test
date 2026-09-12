-- Body of flash_model_pkg. Split into its own file following the convention
-- VUnit's own verification components use for packages with a substantial body
-- (bus_master_pkg.vhd / bus_master_pkg-body.vhd, memory_pkg, sync_pkg).
package body flash_model_pkg is

  ------------------------------------------------------------------------------
  -- FFI contract
  ------------------------------------------------------------------------------

  -- Extract `width` bits starting at `shift` from a non-negative integer.
  -- Written with division rather than shift_right so it works on a plain
  -- integer without a detour through unsigned.
  function extract_field(packed : integer; shift : natural; width : natural) return natural is
  begin
    return (packed / (2 ** shift)) mod (2 ** width);
  end function;

  function decode_directive(packed : integer) return flash_directive_t is
    constant c_action : natural := extract_field(packed, c_dir_action_shift, c_dir_action_width);
    constant c_lanes : natural := extract_field(packed, c_dir_lanes_shift, c_dir_lanes_width);
    constant c_flags : natural := extract_field(packed, c_dir_flags_shift, c_dir_flags_width);
    constant c_num_bytes : natural := extract_field(packed, c_dir_num_bytes_shift, c_dir_num_bytes_width);
    variable v_result : flash_directive_t;
  begin
    assert packed >= 0
      report "flash_model_pkg.decode_directive: negative packed directive " & to_string(packed)
      severity failure;

    assert c_action <= flash_action_t'pos(flash_action_t'high)
      report "flash_model_pkg.decode_directive: action code " & to_string(c_action)
        & " is not a flash_action_t -- FFI layout drift?"
      severity failure;

    assert c_lanes = 1 or c_lanes = 2 or c_lanes = 4
      report "flash_model_pkg.decode_directive: lane count " & to_string(c_lanes)
        & " is not 1, 2 or 4"
      severity failure;

    assert c_num_bytes >= 1
      report "flash_model_pkg.decode_directive: num_bytes must be at least 1"
      severity failure;

    v_result := (
      action => flash_action_t'val(c_action),
      lanes => c_lanes,
      pre_dummy_cycles => extract_field(packed, c_dir_dummy_shift, c_dir_dummy_width),
      byte_out => extract_field(packed, c_dir_byte_out_shift, c_dir_byte_out_width),
      is_volatile => (c_flags / (2 ** c_dir_flag_volatile)) mod 2 = 1,
      num_bytes => c_num_bytes
    );
    return v_result;
  end function;

  ------------------------------------------------------------------------------
  -- Handle
  ------------------------------------------------------------------------------

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
  ) return flash_model_t is
  begin
    return (
      p_std_cfg => create_std_cfg(
        id => id,
        provider => "flash_model",
        vc_name => "flash_model",
        unexpected_msg_type_policy => unexpected_msg_type_policy
      ),
      p_size_bytes => size_bytes,
      p_page_bytes => page_bytes,
      p_sector_bytes => sector_bytes,
      p_block_bytes => block_bytes,
      p_addr_bytes => addr_bytes,
      p_jedec_id => jedec_id,
      p_profile => new_string_ptr(profile)
    );
  end function;

  impure function as_sync(flash : flash_model_t) return sync_handle_t is
  begin
    return get_actor(flash.p_std_cfg);
  end function;

  impure function get_logger(flash : flash_model_t) return logger_t is
  begin
    return get_logger(flash.p_std_cfg);
  end function;

  impure function get_checker(flash : flash_model_t) return checker_t is
  begin
    return get_checker(flash.p_std_cfg);
  end function;

  ------------------------------------------------------------------------------
  -- Initialization
  ------------------------------------------------------------------------------

  procedure flash_preload(
    signal net : inout network_t;
    flash : flash_model_t;
    address : natural;
    data : integer_array_t
  ) is
    variable v_msg : msg_t := new_msg(flash_preload_msg);
    -- push_integer_array_t_ref transfers ownership: it nulls the handle it is
    -- given. Push a copy so the caller's array survives the call and can be
    -- reused -- a test that preloads and then checks against the same array
    -- would otherwise fail in a thoroughly confusing way. The VC deallocates
    -- the copy once it has forwarded the bytes to Python.
    variable v_owned : integer_array_t := copy(data);
  begin
    push(v_msg, address);
    push_integer_array_t_ref(v_msg, v_owned);
    send(net, get_actor(flash.p_std_cfg), v_msg);
  end procedure;

  procedure flash_preload(
    signal net : inout network_t;
    flash : flash_model_t;
    address : natural;
    data : std_ulogic_vector
  ) is
    -- Byte-oriented, most significant byte first, matching how a flash image is
    -- written down and how every address in this model is byte-addressed.
    constant c_num_bytes : natural := data'length / 8;
    variable v_data : integer_array_t := new_1d(
      length => c_num_bytes, bit_width => 8, is_signed => false
    );
    constant c_data : std_ulogic_vector(data'length - 1 downto 0) := data;
  begin
    assert data'length mod 8 = 0
      report "flash_model_pkg.flash_preload: vector length " & to_string(data'length)
        & " is not a whole number of bytes"
      severity failure;

    for byte_idx in 0 to c_num_bytes - 1 loop
      set(
        v_data, byte_idx,
        to_integer(unsigned(c_data(c_data'high - 8 * byte_idx downto c_data'high - 8 * byte_idx - 7)))
      );
    end loop;

    flash_preload(net, flash, address, v_data);
    -- The overload above pushed a copy, so this temporary is ours to release.
    deallocate(v_data);
  end procedure;

  procedure flash_preload_fill(
    signal net : inout network_t;
    flash : flash_model_t;
    address : natural;
    num_bytes : positive;
    value : natural range 0 to 255 := 16#FF#
  ) is
    variable v_msg : msg_t := new_msg(flash_preload_fill_msg);
  begin
    push(v_msg, address);
    push(v_msg, num_bytes);
    push(v_msg, value);
    send(net, get_actor(flash.p_std_cfg), v_msg);
  end procedure;

  procedure flash_load_image(
    signal net : inout network_t;
    flash : flash_model_t;
    file_name : string;
    format : string := "auto";
    base_address : natural := 0
  ) is
    variable v_msg : msg_t := new_msg(flash_load_image_msg);
  begin
    push_string(v_msg, file_name);
    push_string(v_msg, format);
    push(v_msg, base_address);
    send(net, get_actor(flash.p_std_cfg), v_msg);
  end procedure;

  ------------------------------------------------------------------------------
  -- Read-back and checking
  ------------------------------------------------------------------------------

  procedure flash_read_back(
    signal net : inout network_t;
    flash : flash_model_t;
    address : natural;
    num_bytes : positive;
    variable reference : inout flash_reference_t
  ) is
  begin
    reference := new_msg(flash_read_back_msg);
    push(reference, address);
    push(reference, num_bytes);
    send(net, get_actor(flash.p_std_cfg), reference);
  end procedure;

  procedure await_flash_read_back_reply(
    signal net : inout network_t;
    variable reference : inout flash_reference_t;
    variable data : out integer_array_t
  ) is
    variable v_reply : msg_t;
  begin
    receive_reply(net, reference, v_reply);
    data := pop_integer_array_t_ref(v_reply);
    delete(reference);
    delete(v_reply);
  end procedure;

  procedure flash_read_back(
    signal net : inout network_t;
    flash : flash_model_t;
    address : natural;
    num_bytes : positive;
    variable data : out integer_array_t
  ) is
    variable v_reference : flash_reference_t;
  begin
    flash_read_back(net, flash, address, num_bytes, v_reference);
    await_flash_read_back_reply(net, v_reference, data);
  end procedure;

  procedure flash_check_content(
    signal net : inout network_t;
    flash : flash_model_t;
    address : natural;
    expected : integer_array_t
  ) is
    variable v_msg : msg_t := new_msg(flash_check_content_msg);
    -- A copy, for the same ownership reason as flash_preload above. `expected`
    -- is very often a literal the test reuses across several checks.
    variable v_owned : integer_array_t := copy(expected);
  begin
    push(v_msg, address);
    push_integer_array_t_ref(v_msg, v_owned);
    send(net, get_actor(flash.p_std_cfg), v_msg);
  end procedure;

  procedure flash_check_content_fill(
    signal net : inout network_t;
    flash : flash_model_t;
    address : natural;
    num_bytes : positive;
    value : natural range 0 to 255
  ) is
    variable v_msg : msg_t := new_msg(flash_check_content_fill_msg);
  begin
    push(v_msg, address);
    push(v_msg, num_bytes);
    push(v_msg, value);
    send(net, get_actor(flash.p_std_cfg), v_msg);
  end procedure;

  procedure flash_get_written_regions(
    signal net : inout network_t;
    flash : flash_model_t;
    variable regions : out integer_array_t
  ) is
    variable v_request : msg_t := new_msg(flash_written_regions_msg);
    variable v_reply : msg_t;
  begin
    request(net, get_actor(flash.p_std_cfg), v_request, v_reply);
    regions := pop_integer_array_t_ref(v_reply);
    delete(v_reply);
  end procedure;

  ------------------------------------------------------------------------------
  -- Configuration
  ------------------------------------------------------------------------------

  procedure flash_set_timing_enable(
    signal net : inout network_t;
    flash : flash_model_t;
    enable : boolean
  ) is
    variable v_msg : msg_t := new_msg(flash_set_timing_enable_msg);
  begin
    push(v_msg, enable);
    send(net, get_actor(flash.p_std_cfg), v_msg);
  end procedure;

  procedure flash_set_timing(
    signal net : inout network_t;
    flash : flash_model_t;
    name : string;
    duration : delay_length
  ) is
    variable v_msg : msg_t := new_msg(flash_set_timing_msg);
  begin
    push_string(v_msg, name);
    push_time(v_msg, duration);
    send(net, get_actor(flash.p_std_cfg), v_msg);
  end procedure;

  procedure flash_set_protection(
    signal net : inout network_t;
    flash : flash_model_t;
    address : natural;
    num_bytes : positive;
    locked : boolean
  ) is
    variable v_msg : msg_t := new_msg(flash_set_protection_msg);
  begin
    push(v_msg, address);
    push(v_msg, num_bytes);
    push(v_msg, locked);
    send(net, get_actor(flash.p_std_cfg), v_msg);
  end procedure;

  procedure flash_wait_until_ready(
    signal net : inout network_t;
    flash : flash_model_t;
    timeout : delay_length := 1 sec
  ) is
    variable v_request : msg_t := new_msg(flash_wait_until_ready_msg);
    variable v_reply : msg_t;
  begin
    request(net, get_actor(flash.p_std_cfg), v_request, v_reply, timeout => timeout);
    delete(v_reply);
  end procedure;

  procedure flash_reset(
    signal net : inout network_t;
    flash : flash_model_t
  ) is
    variable v_request : msg_t := new_msg(flash_reset_msg);
    variable v_reply : msg_t;
  begin
    -- Blocking: a test case's setup must not race the first stimulus against a
    -- model that has not been cleared yet.
    request(net, get_actor(flash.p_std_cfg), v_request, v_reply);
    delete(v_reply);
  end procedure;

  procedure flash_get_stat(
    signal net : inout network_t;
    flash : flash_model_t;
    name : string;
    variable value : out integer
  ) is
    variable v_request : msg_t := new_msg(flash_get_stat_msg);
    variable v_reply : msg_t;
  begin
    push_string(v_request, name);
    request(net, get_actor(flash.p_std_cfg), v_request, v_reply);
    value := pop_integer(v_reply);
    delete(v_reply);
  end procedure;

end package body;
