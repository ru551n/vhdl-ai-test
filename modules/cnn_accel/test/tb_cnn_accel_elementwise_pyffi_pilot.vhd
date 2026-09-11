library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vunit_lib;
context vunit_lib.vunit_context;
use vunit_lib.python_pkg.all;
use vunit_lib.integer_array_pkg.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;
use cnn_accel.cnn_accel_v2_pkg.all;

-- PILOT: DEPTH_TO_SPACE checked live against 'cnn_accel_model.depth_to_space'
-- through VUnit's VHDL-to-Python bridge (`add_vhdl_builtins(python=True)`,
-- fork ru551n/vunit branch feature/python-ffi), instead of the project's
-- usual "generate a reference file in Python, load it in VHDL, compare"
-- flow (see tb_cnn_accel_top.vhd's header). One `python_call` per test case
-- gets the expected bytes from the SAME golden-model function every other
-- opcode's file-based reference already treats as the single source of
-- truth -- the file-generation and reload step is simply gone.
--
-- Not wired through cnn_accel_top/tensor_mem: this instantiates
-- cnn_accel_elementwise directly with small hand-rolled req/stream
-- responders, since no per-entity testbench for it existed yet (every
-- opcode it implements was previously verified only through tb_cnn_accel_top).
entity tb_cnn_accel_elementwise_pyffi_pilot is
  generic (runner_cfg : string);
end entity tb_cnn_accel_elementwise_pyffi_pilot;

architecture tb of tb_cnn_accel_elementwise_pyffi_pilot is

  constant c_clk_period : time := 10 ns;
  constant c_bytes_per_beat : positive := 8; -- T = ACTIVATION_PLANE_CHANNELS

  signal clk : std_ulogic := '0';
  signal reset : std_ulogic := '1';

  signal start : std_ulogic := '0';
  signal opcode : std_ulogic_vector(7 downto 0) := (others => '0');
  signal src0_addr, src1_addr, dst_addr, lut_addr, xfer_bytes : unsigned(31 downto 0) := (others => '0');
  signal in_width, in_height, in_channels, out_channels : unsigned(15 downto 0) := (others => '0');
  signal dts_factor : unsigned(7 downto 0) := (others => '0');
  signal requant_scale : signed(31 downto 0) := (others => '0');
  signal requant_shift : unsigned(7 downto 0) := (others => '0');

  signal done : std_ulogic;
  signal error : std_ulogic;
  signal error_code : err_code_t;

  signal src0_req_m2s, src1_req_m2s, dst_req_m2s, lut_req_m2s : dma_req_m2s_t;
  signal src0_req_s2m, src1_req_s2m, dst_req_s2m, lut_req_s2m : dma_req_s2m_t := (ready => '1');

  signal s_src0_stream_m2s, s_src1_stream_m2s, s_lut_stream_m2s : axi_stream_m2s_t := axi_stream_m2s_init;
  signal s_src0_stream_s2m, s_src1_stream_s2m, s_lut_stream_s2m : axi_stream_s2m_t;
  signal m_dst_stream_m2s : axi_stream_m2s_t;
  signal m_dst_stream_s2m : axi_stream_s2m_t := (ready => '1');

  -- Small flat byte "memories" for this pilot's own request/stream
  -- responders -- plane-major HWC bytes, unpacked (one array element per
  -- logical byte, matching cnn_accel_model's own unpacked representation).
  type byte_array_t is array (natural range <>) of std_ulogic_vector(7 downto 0);
  signal src0_mem : byte_array_t(0 to 1023) := (others => (others => '0'));
  signal dst_mem : byte_array_t(0 to 1023) := (others => (others => '0'));
  signal dst_mem_valid : std_ulogic := '0'; -- pulses once per captured beat, for the checker to sync on if ever needed

begin

  clk <= not clk after c_clk_period / 2;

  reset_gen : process
  begin
    reset <= '1';
    wait for 5 * c_clk_period;
    wait until rising_edge(clk);
    reset <= '0';
    wait;
  end process;

  dut : entity cnn_accel.cnn_accel_elementwise
    generic map (
      g_axi_data_width => 64
    )
    port map (
      clk => clk,
      reset => reset,

      start => start,
      opcode => opcode,
      src0_addr => src0_addr,
      src1_addr => src1_addr,
      dst_addr => dst_addr,
      lut_addr => lut_addr,
      xfer_bytes => xfer_bytes,
      in_width => in_width,
      in_height => in_height,
      in_channels => in_channels,
      out_channels => out_channels,
      dts_factor => dts_factor,
      requant_scale => requant_scale,
      requant_shift => requant_shift,

      done => done,
      error => error,
      error_code => error_code,

      src0_req_m2s => src0_req_m2s,
      src0_req_s2m => src0_req_s2m,
      s_src0_stream_m2s => s_src0_stream_m2s,
      s_src0_stream_s2m => s_src0_stream_s2m,

      src1_req_m2s => src1_req_m2s,
      src1_req_s2m => src1_req_s2m,
      s_src1_stream_m2s => s_src1_stream_m2s,
      s_src1_stream_s2m => s_src1_stream_s2m,

      dst_req_m2s => dst_req_m2s,
      dst_req_s2m => dst_req_s2m,
      m_dst_stream_m2s => m_dst_stream_m2s,
      m_dst_stream_s2m => m_dst_stream_s2m,

      lut_req_m2s => lut_req_m2s,
      lut_req_s2m => lut_req_s2m,
      s_lut_stream_m2s => s_lut_stream_m2s,
      s_lut_stream_s2m => s_lut_stream_s2m
    );

  -- src0 read responder: on every request, stream 'length' bytes back
  -- from 'src0_mem' starting at 'addr', one c_bytes_per_beat-wide beat
  -- per handshake (DEPTH_TO_SPACE issues single-beat requests, but this
  -- loop does not assume that -- it drains whatever length is asked for).
  src0_responder : process
    variable addr, len : natural;
  begin
    loop
      wait until rising_edge(clk) and src0_req_m2s.valid = '1' and src0_req_s2m.ready = '1';
      addr := to_integer(src0_req_m2s.req.addr);
      len := to_integer(src0_req_m2s.req.length);
      while len > 0 loop
        s_src0_stream_m2s.valid <= '1';
        for b in 0 to c_bytes_per_beat - 1 loop
          s_src0_stream_m2s.data(8 * b + 7 downto 8 * b) <= src0_mem(addr + b);
        end loop;
        s_src0_stream_m2s.last <= '1' when len = c_bytes_per_beat else '0';
        wait until rising_edge(clk) and s_src0_stream_s2m.ready = '1';
        s_src0_stream_m2s.valid <= '0';
        addr := addr + c_bytes_per_beat;
        len := len - c_bytes_per_beat;
      end loop;
    end loop;
  end process;

  -- dst write responder: accept every request immediately (ready = '1'
  -- always, set at declaration), then capture each incoming beat into
  -- 'dst_mem' at the address latched from that beat's own request.
  dst_responder : process
    variable addr, len : natural;
  begin
    loop
      wait until rising_edge(clk) and dst_req_m2s.valid = '1' and dst_req_s2m.ready = '1';
      addr := to_integer(dst_req_m2s.req.addr);
      len := to_integer(dst_req_m2s.req.length);
      while len > 0 loop
        wait until rising_edge(clk) and m_dst_stream_m2s.valid = '1' and m_dst_stream_s2m.ready = '1';
        for b in 0 to c_bytes_per_beat - 1 loop
          dst_mem(addr + b) <= m_dst_stream_m2s.data(8 * b + 7 downto 8 * b);
        end loop;
        dst_mem_valid <= '1';
        wait until rising_edge(clk);
        dst_mem_valid <= '0';
        addr := addr + c_bytes_per_beat;
        len := len - c_bytes_per_beat;
      end loop;
    end loop;
  end process;

  main : process
    -- Runs one DEPTH_TO_SPACE command through the DUT and returns after
    -- 'done'. Geometry is loaded into 'src0_mem' by the caller first.
    procedure run_depth_to_space(
      p_in_width, p_in_height, p_in_channels, p_out_channels : natural;
      p_factor : natural
    ) is
    begin
      in_width <= to_unsigned(p_in_width, 16);
      in_height <= to_unsigned(p_in_height, 16);
      in_channels <= to_unsigned(p_in_channels, 16);
      out_channels <= to_unsigned(p_out_channels, 16);
      dts_factor <= to_unsigned(p_factor, 8);
      src0_addr <= to_unsigned(0, 32);
      dst_addr <= to_unsigned(512, 32);
      opcode <= c_opcode_depth_to_space;

      wait until rising_edge(clk);
      start <= '1';
      wait until rising_edge(clk);
      start <= '0';

      wait until rising_edge(clk) and done = '1' for 10 us;
      check_true(done = '1', "DUT never asserted done (watchdog timeout)");
    end procedure;

    variable expected : integer_array_t;
    variable pixels : integer_array_t;
    variable in_w, in_h, in_c, out_c, factor : natural;
    variable out_bytes : natural;
    variable rtl_byte, exp_byte : integer;
  begin
    test_runner_setup(runner, runner_cfg);
    python_execute(file_name => tb_path(runner_cfg) & "python_bridge/depth_to_space_bridge.py");
    wait until reset = '0';

    while test_suite loop

      if run("depth_to_space matches the golden model, live") then
        in_w := 2;
        in_h := 2;
        out_c := 8;   -- one whole channel tile (T = 8)
        factor := 2;
        in_c := factor * factor * out_c; -- 32

        -- Fill src0_mem with a simple, non-trivial pattern -- RAW packed
        -- S6 channel-tiled bytes (2*2*32 = 128 of them), the same domain
        -- 'depth_to_space_check' now unpacks/repacks internally -- and
        -- hand the identical bytes to Python as an integer_array_t, so
        -- both sides start from the same data with no separate
        -- reference-file step.
        pixels := new_1d(length => in_w * in_h * in_c, bit_width => 8, is_signed => true);
        for i in 0 to in_w * in_h * in_c - 1 loop
          src0_mem(i) <= std_ulogic_vector(to_signed((i mod 251) - 125, 8));
          set(pixels, i, (i mod 251) - 125);
        end loop;
        wait until rising_edge(clk);

        expected := python_call(
          "depth_to_space_check",
          arg => pixels,
          kwargs => kw("in_width", in_w) & kw("in_height", in_h) & kw("in_channels", in_c) &
                    kw("out_channels", out_c) & kw("dts_factor", factor)
        );

        run_depth_to_space(in_w, in_h, in_c, out_c, factor);

        out_bytes := in_w * in_h * out_c * factor * factor;
        check_equal(length(expected), out_bytes, "golden model output length");
        for i in 0 to out_bytes - 1 loop
          rtl_byte := to_integer(signed(dst_mem(512 + i)));
          exp_byte := get(expected, i);
          check_equal(rtl_byte, exp_byte, "byte " & to_string(i));
        end loop;
        check_equal(error, '0');

      elsif run("depth_to_space matches the golden model, live, two output tiles") then
        -- A second, independent geometry: asymmetric width/height and
        -- out_channels = 16 (two whole channel tiles, n_tiles_out = 2),
        -- proving the previous case was not a coincidence of one small,
        -- one-tile shape.
        in_w := 3;
        in_h := 2;
        out_c := 16;
        factor := 2;
        in_c := factor * factor * out_c; -- 64

        pixels := new_1d(length => in_w * in_h * in_c, bit_width => 8, is_signed => true);
        for i in 0 to in_w * in_h * in_c - 1 loop
          src0_mem(i) <= std_ulogic_vector(to_signed(((i * 7 + 3) mod 251) - 125, 8));
          set(pixels, i, ((i * 7 + 3) mod 251) - 125);
        end loop;
        wait until rising_edge(clk);

        expected := python_call(
          "depth_to_space_check",
          arg => pixels,
          kwargs => kw("in_width", in_w) & kw("in_height", in_h) & kw("in_channels", in_c) &
                    kw("out_channels", out_c) & kw("dts_factor", factor)
        );

        run_depth_to_space(in_w, in_h, in_c, out_c, factor);

        out_bytes := in_w * in_h * out_c * factor * factor;
        check_equal(length(expected), out_bytes, "golden model output length");
        for i in 0 to out_bytes - 1 loop
          rtl_byte := to_integer(signed(dst_mem(512 + i)));
          exp_byte := get(expected, i);
          check_equal(rtl_byte, exp_byte, "byte " & to_string(i));
        end loop;
        check_equal(error, '0');

      end if;

    end loop;

    test_runner_cleanup(runner);
  end process;

end architecture tb;
