library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
use vunit_lib.python_pkg.all;
use vunit_lib.integer_array_pkg.all;
use vunit_lib.memory_pkg.all;
use vunit_lib.check_pkg.all;

-- Reusable helpers for a testbench that seeds/exports DDR through
-- VUnit's Python FFI ('python_call') instead of CSV files, so that
-- writing the byte-at-a-time 'write_word'/'read_word' loop is a one-line
-- call at every call site rather than duplicated per testbench.
--
-- Both directions work in whole bytes over a byte-addressed 'memory_t',
-- matching the convention 'accel_v2.memimage.MemoryImage' and every
-- 'python_bridge' module already use on the Python side.
package cnn_accel_python_ffi_pkg is

  -- Call 'function_name()' (no arguments) in the current Python
  -- session, expecting an unsigned byte array of exactly 'num_bytes'
  -- elements back, and write it into 'memory' starting at 'base_addr'.
  -- A no-op when 'num_bytes' is 0 -- a case with nothing to seed there
  -- need not special-case the call site.
  procedure ffi_seed_bytes(
    memory : memory_t;
    function_name : string;
    base_addr : natural;
    num_bytes : natural
  );

  -- The inverse: read 'num_bytes' bytes out of 'memory' starting at
  -- 'base_addr' into a VUnit 'integer_array_t' of unsigned bytes, ready
  -- to hand to a 'python_call' as its argument.
  impure function ffi_export_bytes(
    memory : memory_t;
    base_addr : natural;
    num_bytes : natural
  ) return integer_array_t;

  -- Like 'ffi_seed_bytes', but calls 'function_name(index)' instead of
  -- 'function_name()' -- for seeding one of several regions a case
  -- reports (e.g. 'get_program_regions'/'get_program_data' in
  -- top_level_bridge.py), where one Python function alone cannot name
  -- which region's bytes to return.
  procedure ffi_seed_indexed_bytes(
    memory : memory_t;
    function_name : string;
    index : natural;
    base_addr : natural;
    num_bytes : natural
  );

end package;

package body cnn_accel_python_ffi_pkg is

  procedure ffi_seed_bytes(
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

    data := python_call(function_name);
    check_equal(
      length(data), num_bytes,
      "ffi_seed_bytes: python_call(""" & function_name & """) returned "
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

  procedure ffi_seed_indexed_bytes(
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

    data := python_call(function_name, arg => index);
    check_equal(
      length(data), num_bytes,
      "ffi_seed_indexed_bytes: python_call(""" & function_name & """, " & to_string(index)
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

end package body;
