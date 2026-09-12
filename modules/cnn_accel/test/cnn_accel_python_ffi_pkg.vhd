library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
context vunit_lib.python_context;
use vunit_lib.integer_array_pkg.all;
use vunit_lib.memory_pkg.all;
use vunit_lib.check_pkg.all;

-- Reusable helpers for a testbench that writes/reads DDR through VUnit's
-- Python FFI (python_pkg's 'call') instead of CSV files, so that writing
-- the byte-at-a-time 'write_word'/'read_word' loop is a one-line call at
-- every call site rather than duplicated per testbench.
--
-- 'to_bytes' serves the other direction: python_pkg's 'arg'/'kwarg' have
-- no unsigned/std_ulogic overload (deliberately -- they would make a
-- string literal argument ambiguous), and a 32-bit counter does not fit
-- VHDL's signed 'integer' anyway, so a wide value crosses to Python as a
-- little-endian list of byte values that the receiving function
-- reassembles with int.from_bytes(bytes(value), "little").
--
-- Both directions work in whole bytes over a byte-addressed 'memory_t',
-- matching the convention 'accel_v2.memimage.MemoryImage' and every
-- 'python_bridge' module already use on the Python side.
package cnn_accel_python_ffi_pkg is

  -- Call 'function_name()' (no arguments) in the current Python
  -- session, expecting an unsigned byte array of exactly 'num_bytes'
  -- elements back, and write it into 'memory' starting at 'base_addr'.
  -- A no-op when 'num_bytes' is 0 -- a case with nothing to write there
  -- need not special-case the call site.
  procedure ffi_write_bytes(
    memory : memory_t;
    function_name : string;
    base_addr : natural;
    num_bytes : natural
  );

  -- The inverse: read 'num_bytes' bytes out of 'memory' starting at
  -- 'base_addr' into a VUnit 'integer_array_t' of unsigned bytes, ready
  -- to hand to a 'call' as its argument.
  impure function ffi_export_bytes(
    memory : memory_t;
    base_addr : natural;
    num_bytes : natural
  ) return integer_array_t;

  -- Like 'ffi_write_bytes', but calls 'function_name(index)' instead of
  -- 'function_name()' -- for writing one of several regions a case
  -- reports (e.g. 'get_program_regions'/'get_program_data' in
  -- top_level_bridge.py), where one Python function alone cannot name
  -- which region's bytes to return.
  procedure ffi_write_indexed_bytes(
    memory : memory_t;
    function_name : string;
    index : natural;
    base_addr : natural;
    num_bytes : natural
  );

  -- 'value' as a little-endian list of ((value'length + 7) / 8) byte
  -- values (0..255), ready for python_pkg's 'arg'/'kwarg' -- which have
  -- no unsigned overload, and could not carry a full 32-bit counter in a
  -- VHDL 'integer' even if they did. The Python side reassembles it with
  -- int.from_bytes(bytes(value), "little").
  function to_bytes(value : unsigned) return integer_vector;

  -- The same, for a single status bit: 4 bytes holding 0 or 1, so a
  -- std_ulogic flag can be concatenated straight into a counter vector
  -- alongside the real 32-bit counters instead of being special-cased at
  -- every call site ('arg'/'kwarg' have no std_ulogic overload either).
  function to_bytes(value : std_ulogic) return integer_vector;

end package;

package body cnn_accel_python_ffi_pkg is

  procedure ffi_write_bytes(
    memory : memory_t;
    function_name : string;
    base_addr : natural;
    num_bytes : natural
  ) is
    variable data : integer_array_t;
  begin
    if num_bytes = 0 then
      return;
    end if;

    data := call(function_name);
    check_equal(
      length(data), num_bytes,
      "ffi_write_bytes: call(""" & function_name & """) returned "
      & to_string(length(data)) & " bytes, expected " & to_string(num_bytes)
    );

    for i in 0 to num_bytes - 1 loop
      write_word(
        memory => memory,
        address => base_addr + i,
        word => std_logic_vector(to_unsigned(get(data, i), 8))
      );
    end loop;
  end procedure;

  impure function ffi_export_bytes(
    memory : memory_t;
    base_addr : natural;
    num_bytes : natural
  ) return integer_array_t is
    variable data : integer_array_t;
  begin
    data := new_1d(length => num_bytes, bit_width => 8, is_signed => false);
    for i in 0 to num_bytes - 1 loop
      set(
        data, i,
        to_integer(
          u_unsigned(read_word(memory => memory, address => base_addr + i, bytes_per_word => 1))
        )
      );
    end loop;
    return data;
  end function;

  procedure ffi_write_indexed_bytes(
    memory : memory_t;
    function_name : string;
    index : natural;
    base_addr : natural;
    num_bytes : natural
  ) is
    variable data : integer_array_t;
  begin
    if num_bytes = 0 then
      return;
    end if;

    data := call(function_name, arg(index));
    check_equal(
      length(data), num_bytes,
      "ffi_write_indexed_bytes: call(""" & function_name & """, " & to_string(index)
      & ") returned " & to_string(length(data)) & " bytes, expected " & to_string(num_bytes)
    );

    for i in 0 to num_bytes - 1 loop
      write_word(
        memory => memory,
        address => base_addr + i,
        word => std_logic_vector(to_unsigned(get(data, i), 8))
      );
    end loop;
  end procedure;

  function to_bytes(value : unsigned) return integer_vector is
    constant num_bytes : positive := (value'length + 7) / 8;
    alias normalized : unsigned(value'length - 1 downto 0) is value;
    variable result : integer_vector(0 to num_bytes - 1) := (others => 0);
  begin
    -- Per set bit rather than per byte: it needs no guard for the bits of
    -- the last byte that a non-multiple-of-8 'value' does not have.
    for idx in 0 to normalized'length - 1 loop
      if normalized(idx) = '1' then
        result(idx / 8) := result(idx / 8) + 2 ** (idx mod 8);
      end if;
    end loop;
    return result;
  end function;

  function to_bytes(value : std_ulogic) return integer_vector is
  begin
    if value = '1' then
      return integer_vector'(1, 0, 0, 0);
    end if;
    return integer_vector'(0, 0, 0, 0);
  end function;

end package body;
