library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
context vunit_lib.vunit_context;
use vunit_lib.python_pkg.all;
use vunit_lib.integer_array_pkg.all;

-- MICROBENCHMARK, not a functional test: how long does moving a real
-- multi-megabyte 'integer_array_t' through VUnit's Python FFI actually
-- take? None of tb_cnn_accel_top's existing cases exercise this -- their
-- exported/checked region is always just the graph's own output
-- tensor(s), a few KB at most, even when the underlying simulated DDR
-- traffic is much larger. This measures the transport itself, both
-- directions, at sizes an eventual megabyte-scale case's checked region
-- could plausibly reach.
entity tb_python_ffi_throughput_pilot is
  generic (runner_cfg : string);
end entity tb_python_ffi_throughput_pilot;

architecture tb of tb_python_ffi_throughput_pilot is
begin

  main : process
    variable data : integer_array_t;
    variable result : integer_array_t;
    variable t0, t1 : time;
    variable n : natural;
  begin
    test_runner_setup(runner, runner_cfg);
    python_execute(
      "import numpy as np" +
      "def sum_bytes(arr):" +
      "    return int(np.asarray(arr).astype(np.int64).sum() & 0xFFFFFFFF)" +
      "def identity_bytes(arr):" +
      "    return np.asarray(arr, dtype=np.uint8)"
    );

    while test_suite loop

      if run("1 MiB integer_array_t round trip, sum") then
        n := 1 * 1024 * 1024;
        data := new_1d(length => n, bit_width => 8, is_signed => false);
        for i in 0 to n - 1 loop
          set(data, i, i mod 256);
        end loop;

        t0 := now;
        check_equal(integer'(python_call("sum_bytes", arg => data)), 32640 * (n / 256));
        t1 := now;
        report "1 MiB sum_bytes (VHDL->Python, one-way): " & to_string((t1 - t0), ns) & " (real time via 'now' is 0 in this sim -- see the process's own wall-clock report on stdout instead)";

      elsif run("1 MiB integer_array_t round trip, identity") then
        n := 1 * 1024 * 1024;
        data := new_1d(length => n, bit_width => 8, is_signed => false);
        for i in 0 to n - 1 loop
          set(data, i, i mod 256);
        end loop;

        result := python_call("identity_bytes", arg => data);
        check_equal(length(result), n);
        for i in 0 to n - 1 loop
          check_equal(get(result, i), i mod 256);
        end loop;

      elsif run("4 MiB integer_array_t round trip, identity") then
        n := 4 * 1024 * 1024;
        data := new_1d(length => n, bit_width => 8, is_signed => false);
        for i in 0 to n - 1 loop
          set(data, i, i mod 256);
        end loop;

        result := python_call("identity_bytes", arg => data);
        check_equal(length(result), n);
        for i in 0 to n - 1 loop
          check_equal(get(result, i), i mod 256);
        end loop;

      elsif run("4 MiB round trip, transfer cost only (single sum check)") then
        -- Isolates the FFI transfer+identity-copy cost from the previous
        -- case's real cost driver: a million-plus individual VHDL-side
        -- check_equal calls. One check_equal on a 'sum' Python already
        -- computed removes that entirely.
        n := 4 * 1024 * 1024;
        data := new_1d(length => n, bit_width => 8, is_signed => false);
        for i in 0 to n - 1 loop
          set(data, i, i mod 256);
        end loop;

        result := python_call("identity_bytes", arg => data);
        check_equal(length(result), n);
        check_equal(integer'(python_call("sum_bytes", arg => result)), 32640 * (n / 256));

      end if;

    end loop;

    test_runner_cleanup(runner);
  end process;

end architecture tb;
