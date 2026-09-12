library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
context vunit_lib.vunit_context;
context vunit_lib.com_context;
use vunit_lib.integer_array_pkg.all;
use vunit_lib.sync_pkg.all;
use vunit_lib.vc_pkg.all;

library flash_model;
use flash_model.qspi_pkg.all;

-- Control surface for the protocol-generic QSPI master verification component
-- (qspi_master.vhd).
--
-- This package knows nothing about flash. It offers exactly one primitive --
-- a QSPI transaction -- described in the vocabulary of the bus itself:
--
--   CS low
--     -> command bytes   at 'cmd_lanes'    (the opcode, typically)
--     -> address bytes   at 'addr_lanes'   (address, mode byte, ...)
--     -> write bytes     at 'wr_lanes'     (a program payload, typically)
--     -> 'dummy_cycles'  SCK cycles with every master IO tri-stated
--     -> 'num_read_bytes' read bytes at 'read_lanes'
--   CS high
--
-- Three separate write phases rather than one is what lets a caller express
-- every shape doc/flash_model_ffi_contract.md describes with one transaction:
-- a quad-IO read sends its opcode at x1 and its address at x4; a quad page
-- program sends opcode and address at x1 and its payload at x4. Any phase may
-- be empty. The caller composes the bytes; the VC owns CS framing, SCK
-- generation, lane placement and tri-stating.
--
-- Calls follow the house non-blocking-first shape (shared/Vunit.md section 13):
-- 'qspi_transfer' with a 'reference' queues the transaction and returns at
-- once, and 'await_qspi_transfer_reply' redeems the read data when the caller
-- actually needs it. Blocking convenience overloads that do both in one call
-- are provided for directed sequences. Every transaction is acknowledged with
-- a reply whether or not it reads anything, so a redeemed reference is also
-- the point at which the transaction is known to have completed on the bus.
--
-- The handle is built on vc_pkg.create_std_cfg, so the VC gets an id, a
-- logger, a checker and an unexpected-message-type policy like every other
-- full-featured VUnit VC. 'as_sync' makes sync_pkg.wait_until_idle work.
--
-- SPI mode 0 only (CPOL = 0, CPHA = 0): SCK idles low, the master changes its
-- outputs on the falling edge and both ends sample on the rising edge. That is
-- the mode every QSPI NOR flash supports; mode 3 differs only in the idle
-- level and is not needed here.
package qspi_master_pkg is

  ------------------------------------------------------------------------------
  -- Handle
  ------------------------------------------------------------------------------

  type qspi_master_t is record
    -- Private. Use the accessors below.
    p_std_cfg : std_cfg_t;
    p_sck_period : delay_length;
    -- Minimum CS-high time between two transactions. A real device specifies
    -- this as tSHSL and ignores a command that arrives too soon after the
    -- previous one, so a master that deselects for less than tSHSL is a bug
    -- even though nothing on the bus looks wrong. It is NOT derived from the
    -- SCK period: tSHSL is a property of the device, not of the bus speed, and
    -- tying the two makes a fast bus silently violate a slow part.
    p_cs_deselect_time : delay_length;
  end record;

  constant c_qspi_default_sck_period : delay_length := 20 ns;

  -- Comfortably above the tSHSL of the JEDEC baseline profile (30 ns). Chosen
  -- as a default that does not violate a typical part rather than as the
  -- fastest legal value; a test that wants to probe the limit sets it down.
  constant c_qspi_default_cs_deselect_time : delay_length := 50 ns;

  impure function new_qspi_master(
    sck_period : delay_length := c_qspi_default_sck_period;
    cs_deselect_time : delay_length := c_qspi_default_cs_deselect_time;
    id : id_t := null_id;
    unexpected_msg_type_policy : unexpected_msg_type_policy_t := fail
  ) return qspi_master_t;

  -- The configured minimum CS-high time between transactions.
  impure function cs_deselect_time(qspi_master : qspi_master_t) return delay_length;

  impure function get_id(qspi_master : qspi_master_t) return id_t;
  impure function get_actor(qspi_master : qspi_master_t) return actor_t;
  impure function get_logger(qspi_master : qspi_master_t) return logger_t;
  impure function get_checker(qspi_master : qspi_master_t) return checker_t;
  impure function as_sync(qspi_master : qspi_master_t) return sync_handle_t;

  -- The SCK period the VC starts out with. The value actually in force can be
  -- changed at run time with 'set_sck_period' below and is then owned by the
  -- VC process, so this accessor reports the initial value only.
  function sck_period(qspi_master : qspi_master_t) return delay_length;

  ------------------------------------------------------------------------------
  -- Transactions
  ------------------------------------------------------------------------------

  -- Handle to an issued, not yet redeemed transaction.
  alias qspi_transfer_reference_t is msg_t;

  -- Non-blocking: queue one transaction and return immediately.
  --
  -- 'cmd', 'addr' and 'wr_data' are byte arrays (one byte per element,
  -- 0 .. 255); any of them may be 'null_integer_array' for an empty phase. The
  -- arrays are read during this call only, so the caller keeps ownership and
  -- may deallocate or reuse them as soon as it returns.
  procedure qspi_transfer(
    signal net : inout network_t;
    constant qspi_master : in qspi_master_t;
    constant cmd : in integer_array_t;
    variable reference : inout qspi_transfer_reference_t;
    constant cmd_lanes : in lane_count_t := 1;
    constant addr : in integer_array_t := null_integer_array;
    constant addr_lanes : in lane_count_t := 1;
    constant wr_data : in integer_array_t := null_integer_array;
    constant wr_lanes : in lane_count_t := 1;
    constant dummy_cycles : in natural := 0;
    constant num_read_bytes : in natural := 0;
    constant read_lanes : in lane_count_t := 1
  );

  -- Blocking: redeem a reference. 'data' is replaced by a freshly allocated
  -- byte array holding the 'num_read_bytes' bytes read, which the caller then
  -- owns and must deallocate. Any array previously held by 'data' is
  -- deallocated first.
  procedure await_qspi_transfer_reply(
    signal net : inout network_t;
    variable reference : inout qspi_transfer_reference_t;
    variable data : inout integer_array_t
  );

  -- Blocking: redeem a reference, discarding any read data.
  procedure await_qspi_transfer_reply(
    signal net : inout network_t;
    variable reference : inout qspi_transfer_reference_t
  );

  -- Blocking convenience: issue and redeem in one call.
  procedure qspi_transfer(
    signal net : inout network_t;
    constant qspi_master : in qspi_master_t;
    constant cmd : in integer_array_t;
    variable data : inout integer_array_t;
    constant cmd_lanes : in lane_count_t := 1;
    constant addr : in integer_array_t := null_integer_array;
    constant addr_lanes : in lane_count_t := 1;
    constant wr_data : in integer_array_t := null_integer_array;
    constant wr_lanes : in lane_count_t := 1;
    constant dummy_cycles : in natural := 0;
    constant num_read_bytes : in natural := 0;
    constant read_lanes : in lane_count_t := 1
  );

  -- Blocking convenience for a write-only transaction.
  procedure qspi_transfer(
    signal net : inout network_t;
    constant qspi_master : in qspi_master_t;
    constant cmd : in integer_array_t;
    constant cmd_lanes : in lane_count_t := 1;
    constant addr : in integer_array_t := null_integer_array;
    constant addr_lanes : in lane_count_t := 1;
    constant wr_data : in integer_array_t := null_integer_array;
    constant wr_lanes : in lane_count_t := 1;
    constant dummy_cycles : in natural := 0
  );

  ------------------------------------------------------------------------------
  -- Run-time configuration
  ------------------------------------------------------------------------------

  -- Blocking, acknowledged: change the SCK period used by every subsequent
  -- transaction. Queued transactions issued before this call still run at the
  -- old period, since the VC handles its messages in order.
  procedure set_sck_period(
    signal net : inout network_t;
    constant qspi_master : in qspi_master_t;
    constant period : in delay_length
  );

  ------------------------------------------------------------------------------
  -- Message types, for the VC implementation
  ------------------------------------------------------------------------------

  constant qspi_transfer_msg : msg_type_t := new_msg_type("qspi transfer");
  constant qspi_transfer_reply_msg : msg_type_t := new_msg_type("qspi transfer reply");
  constant qspi_master_set_sck_period_msg : msg_type_t := new_msg_type("qspi master set sck period");

end package;

package body qspi_master_pkg is

  impure function new_qspi_master(
    sck_period : delay_length := c_qspi_default_sck_period;
    cs_deselect_time : delay_length := c_qspi_default_cs_deselect_time;
    id : id_t := null_id;
    unexpected_msg_type_policy : unexpected_msg_type_policy_t := fail
  ) return qspi_master_t is
  begin
    return (
      p_std_cfg => create_std_cfg(
        id => id,
        provider => "flash_model",
        vc_name => "qspi_master",
        unexpected_msg_type_policy => unexpected_msg_type_policy
      ),
      p_sck_period => sck_period,
      p_cs_deselect_time => cs_deselect_time
    );
  end function;

  impure function cs_deselect_time(qspi_master : qspi_master_t) return delay_length is
  begin
    return qspi_master.p_cs_deselect_time;
  end function;

  impure function get_id(qspi_master : qspi_master_t) return id_t is
  begin
    return get_id(qspi_master.p_std_cfg);
  end function;

  impure function get_actor(qspi_master : qspi_master_t) return actor_t is
  begin
    return get_actor(qspi_master.p_std_cfg);
  end function;

  impure function get_logger(qspi_master : qspi_master_t) return logger_t is
  begin
    return get_logger(qspi_master.p_std_cfg);
  end function;

  impure function get_checker(qspi_master : qspi_master_t) return checker_t is
  begin
    return get_checker(qspi_master.p_std_cfg);
  end function;

  impure function as_sync(qspi_master : qspi_master_t) return sync_handle_t is
  begin
    return get_actor(qspi_master.p_std_cfg);
  end function;

  function sck_period(qspi_master : qspi_master_t) return delay_length is
  begin
    return qspi_master.p_sck_period;
  end function;

  -- Bytes go into the message one integer at a time rather than by reference:
  -- pushing an 'integer_array_t' would hand ownership of the caller's array to
  -- the VC, and a caller that composes a command from a constant array would
  -- then lose it on the first call.
  procedure push_byte_phase(
    constant msg : in msg_t;
    constant bytes : in integer_array_t;
    constant lanes : in lane_count_t
  ) is
    variable v_length : natural := 0;
  begin
    if not is_null(bytes) then
      v_length := length(bytes);
    end if;

    push_integer(msg, v_length);
    push_integer(msg, lanes);
    for index in 0 to v_length - 1 loop
      push_integer(msg, get(bytes, index));
    end loop;
  end procedure;

  procedure qspi_transfer(
    signal net : inout network_t;
    constant qspi_master : in qspi_master_t;
    constant cmd : in integer_array_t;
    variable reference : inout qspi_transfer_reference_t;
    constant cmd_lanes : in lane_count_t := 1;
    constant addr : in integer_array_t := null_integer_array;
    constant addr_lanes : in lane_count_t := 1;
    constant wr_data : in integer_array_t := null_integer_array;
    constant wr_lanes : in lane_count_t := 1;
    constant dummy_cycles : in natural := 0;
    constant num_read_bytes : in natural := 0;
    constant read_lanes : in lane_count_t := 1
  ) is
    alias request_msg : msg_t is reference;
  begin
    request_msg := new_msg(qspi_transfer_msg);

    push_byte_phase(request_msg, cmd, cmd_lanes);
    push_byte_phase(request_msg, addr, addr_lanes);
    push_byte_phase(request_msg, wr_data, wr_lanes);
    push_integer(request_msg, dummy_cycles);
    push_integer(request_msg, num_read_bytes);
    push_integer(request_msg, read_lanes);

    send(net, get_actor(qspi_master), request_msg);
  end procedure;

  procedure await_qspi_transfer_reply(
    signal net : inout network_t;
    variable reference : inout qspi_transfer_reference_t;
    variable data : inout integer_array_t
  ) is
    alias request_msg : msg_t is reference;
    variable reply_msg : msg_t;
  begin
    receive_reply(net, request_msg, reply_msg);

    if not is_null(data) then
      deallocate(data);
    end if;
    data := pop_ref(reply_msg);

    delete(request_msg);
    delete(reply_msg);
  end procedure;

  procedure await_qspi_transfer_reply(
    signal net : inout network_t;
    variable reference : inout qspi_transfer_reference_t
  ) is
    variable v_data : integer_array_t := null_integer_array;
  begin
    await_qspi_transfer_reply(net, reference, v_data);
    deallocate(v_data);
  end procedure;

  procedure qspi_transfer(
    signal net : inout network_t;
    constant qspi_master : in qspi_master_t;
    constant cmd : in integer_array_t;
    variable data : inout integer_array_t;
    constant cmd_lanes : in lane_count_t := 1;
    constant addr : in integer_array_t := null_integer_array;
    constant addr_lanes : in lane_count_t := 1;
    constant wr_data : in integer_array_t := null_integer_array;
    constant wr_lanes : in lane_count_t := 1;
    constant dummy_cycles : in natural := 0;
    constant num_read_bytes : in natural := 0;
    constant read_lanes : in lane_count_t := 1
  ) is
    variable v_reference : qspi_transfer_reference_t;
  begin
    qspi_transfer(
      net => net,
      qspi_master => qspi_master,
      cmd => cmd,
      reference => v_reference,
      cmd_lanes => cmd_lanes,
      addr => addr,
      addr_lanes => addr_lanes,
      wr_data => wr_data,
      wr_lanes => wr_lanes,
      dummy_cycles => dummy_cycles,
      num_read_bytes => num_read_bytes,
      read_lanes => read_lanes
    );
    await_qspi_transfer_reply(net, v_reference, data);
  end procedure;

  procedure qspi_transfer(
    signal net : inout network_t;
    constant qspi_master : in qspi_master_t;
    constant cmd : in integer_array_t;
    constant cmd_lanes : in lane_count_t := 1;
    constant addr : in integer_array_t := null_integer_array;
    constant addr_lanes : in lane_count_t := 1;
    constant wr_data : in integer_array_t := null_integer_array;
    constant wr_lanes : in lane_count_t := 1;
    constant dummy_cycles : in natural := 0
  ) is
    variable v_reference : qspi_transfer_reference_t;
  begin
    qspi_transfer(
      net => net,
      qspi_master => qspi_master,
      cmd => cmd,
      reference => v_reference,
      cmd_lanes => cmd_lanes,
      addr => addr,
      addr_lanes => addr_lanes,
      wr_data => wr_data,
      wr_lanes => wr_lanes,
      dummy_cycles => dummy_cycles
    );
    await_qspi_transfer_reply(net, v_reference);
  end procedure;

  procedure set_sck_period(
    signal net : inout network_t;
    constant qspi_master : in qspi_master_t;
    constant period : in delay_length
  ) is
    variable request_msg : msg_t;
    variable ack : boolean;
  begin
    request_msg := new_msg(qspi_master_set_sck_period_msg);
    push_time(request_msg, period);
    request(net, get_actor(qspi_master), request_msg, ack);
    assert ack report "Failed on set_sck_period command" severity failure;
  end procedure;

end package body;
