library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library cnn_accel;
use cnn_accel.cnn_accel_regs_pkg.all;

library axi;
use axi.axi_pkg.all;

library axi_lite;
use axi_lite.axi_lite_pkg.all;

-- Pinnable build wrapper around 'cnn_accel_top', used ONLY by the top-level
-- Vivado build project ('cnn_accel_top_build' in 'module_cnn_accel.py'). It is
-- not part of the accelerator: no simulation, no product build and no other
-- synthesis entity instantiates it.
--
-- Why it exists. 'cnn_accel_top' is a full AXI4 master plus an AXI4-Lite slave,
-- which is ~840 port bits. A real Vivado implementation run (place, route,
-- 'write_bitstream') requires every top-level port to be a placed I/O buffer,
-- and this part (xc7a200tfbg484) has 285 I/O -- so 'cnn_accel_top' can never be
-- the top of a full build. tsfpga's out-of-context mode ('-no_iobuf') exists
-- only for 'VivadoNetlistProject', which is synthesis-only by construction
-- ('VivadoProject.build()': 'synth_only = synth_only or self.is_netlist_build'),
-- so it cannot deliver post-place-and-route numbers either. This wrapper is the
-- remaining option: five real pins, and the accelerator's whole bus boundary
-- terminated in registers inside the FPGA.
--
-- What it does, and what that does to the numbers:
--
--  * Every 'cnn_accel_top' input bit is driven by a bit of 'stim', a free-
--    running LFSR-seeded shift register. So each input path *starts* at a real
--    flip-flop, exactly as it would when driven by a neighbouring block, and
--    the data is non-constant, so nothing inside the accelerator can be
--    constant-propagated away.
--  * Every 'cnn_accel_top' output bit is captured in 'out_q' and then reduced,
--    over two further register stages, to the single 'result' pin. So each
--    output path *ends* at a real flip-flop, and no output can be trimmed.
--  * The harness itself is ~900 flip-flops and a few hundred LUTs, and it is a
--    sibling of the 'cnn_accel_top' instance in the hierarchy -- so the
--    hierarchical utilization report attributes it separately and the
--    accelerator's own row is uncontaminated.
--
-- What it deliberately does NOT model: the stimulus is not AXI-legal traffic,
-- so this build says nothing about function. It is a resource and static-timing
-- vehicle only; correctness lives in 'tb_cnn_accel_top' and the golden model.
entity cnn_accel_top_build is
  generic (
    -- The one scaling knob, see 'cnn_accel_constants.py'. Only the default
    -- value is buildable at the top level: 'cnn_accel_top' asserts
    -- 'g_pe_rows = cnn_accel_constant_activation_plane_channels', so the
    -- 16-row point the conv datapath's netlist builds use does not
    -- elaborate here. See 'module_cnn_accel.py' for the full note.
    g_pe_rows : positive := cnn_accel_constant_pe_rows
  );
  port (
    clk : in std_ulogic;
    -- Cold, synchronous active-high reset. Registered once here before it
    -- reaches the accelerator, see 'reset_int' below.
    reset : in std_ulogic;
    --# {{}}
    -- Serial seed for the stimulus shift register that drives every
    -- accelerator input.
    stimulus : in std_ulogic;
    -- Registered exclusive-or reduction of every accelerator output.
    result : out std_ulogic := '0';
    -- The accelerator's interrupt, registered.
    irq : out std_ulogic := '0'
  );
end entity cnn_accel_top_build;

architecture a of cnn_accel_top_build is

  ------------------------------------------------------------------------
  -- Stimulus: one shift register, wide enough that every accelerator input
  -- bit gets its own flip-flop. The taps make it self-sustaining so the
  -- design keeps toggling even with 'stimulus' tied low, and make the
  -- register unusable as an SRL (every stage is tapped or fed back).
  ------------------------------------------------------------------------
  constant c_stim_width : positive := 512;
  signal stim : std_ulogic_vector(c_stim_width - 1 downto 0) := (0 => '1', others => '0');

  ------------------------------------------------------------------------
  -- The accelerator's bus boundary.
  ------------------------------------------------------------------------
  signal s_axi_lite_m2s : axi_lite_m2s_t := axi_lite_m2s_init;
  signal s_axi_lite_s2m : axi_lite_s2m_t := axi_lite_s2m_init;
  signal m_axi_m2s : axi_m2s_t := axi_m2s_init;
  signal m_axi_s2m : axi_s2m_t := axi_s2m_init;
  signal irq_int : std_ulogic := '0';

  -- The accelerator's reset is driven from a flip-flop here rather than
  -- straight from the pin, so that the (very high fan-out) reset distribution
  -- inside the accelerator is a timed register-to-register path like it would
  -- be in a real integration, instead of an untimed input path.
  signal reset_int : std_ulogic := '1';

  ------------------------------------------------------------------------
  -- Output collection. 'to_slv' below is the single definition of which
  -- output bits exist, so 'c_out_width' cannot drift from what is actually
  -- collected: it is that function's own return length.
  ------------------------------------------------------------------------
  function to_slv(
    axi_lite_s2m : axi_lite_s2m_t; axi_m2s : axi_m2s_t; irq : std_ulogic
  ) return std_ulogic_vector is
  begin
    return (
      axi_lite_s2m.read.ar.ready
      & axi_lite_s2m.read.r.valid
      & axi_lite_s2m.read.r.data
      & axi_lite_s2m.read.r.resp
      & axi_lite_s2m.write.aw.ready
      & axi_lite_s2m.write.w.ready
      & axi_lite_s2m.write.b.valid
      & axi_lite_s2m.write.b.resp
      & axi_m2s.read.ar.valid
      & std_ulogic_vector(axi_m2s.read.ar.id)
      & std_ulogic_vector(axi_m2s.read.ar.addr)
      & std_ulogic_vector(axi_m2s.read.ar.len)
      & std_ulogic_vector(axi_m2s.read.ar.size)
      & axi_m2s.read.ar.burst
      & axi_m2s.read.r.ready
      & axi_m2s.write.aw.valid
      & std_ulogic_vector(axi_m2s.write.aw.id)
      & std_ulogic_vector(axi_m2s.write.aw.addr)
      & std_ulogic_vector(axi_m2s.write.aw.len)
      & std_ulogic_vector(axi_m2s.write.aw.size)
      & axi_m2s.write.aw.burst
      & axi_m2s.write.w.valid
      & axi_m2s.write.w.data
      & axi_m2s.write.w.strb
      & axi_m2s.write.w.last
      & std_ulogic_vector(axi_m2s.write.w.id)
      & axi_m2s.write.b.ready
      & irq
    );
  end function;

  -- Elaboration-time probe of the collection function above, purely to learn its
  -- length; the initial-value records make it a constant expression.
  constant c_out_init : std_ulogic_vector := to_slv(axi_lite_s2m_init, axi_m2s_init, '0');
  constant c_out_width : positive := c_out_init'length;

  -- Reduce in chunks so the exclusive-or tree is two shallow register-to-
  -- register stages rather than one deep cone that could itself become the
  -- critical path and mask the accelerator's own worst path.
  constant c_chunk_width : positive := 64;
  constant c_num_chunks : positive := (c_out_width + c_chunk_width - 1) / c_chunk_width;
  constant c_padded_width : positive := c_num_chunks * c_chunk_width;

  signal out_q : std_ulogic_vector(c_padded_width - 1 downto 0) := (others => '0');
  signal chunk_q : std_ulogic_vector(c_num_chunks - 1 downto 0) := (others => '0');

  ------------------------------------------------------------------------
  -- Pad-facing register chains.
  --
  -- No accelerator port -- and no harness signal either -- reaches a pin
  -- directly: 'result' and 'irq' each pass through three further plain
  -- flip-flops after the reduction/'irq_int', and the LAST one of each
  -- chain is packed into the output buffer's own IOB flip-flop (the
  -- 'IOB TRUE' property in 'tcl/cnn_accel_top_build_pinning.xdc'). A
  -- register-to-pad hop is then IOB-flop clock-to-out plus the OBUF, with
  -- no fabric routing at all, which is what takes the two pad paths off
  -- the design's worst-path list where they used to sit at -3.2 ns.
  --
  -- 'shreg_extract' is "no" on every stage of both chains, and it has to
  -- be. Two or more flip-flops in a row on the same clock, with no reset,
  -- no clock enable and no logic between them, is exactly the pattern
  -- Vivado's 'shreg_extract' collapses into an SRL16/SRL32 -- and an SRL
  -- has neither the fixed per-stage placement this chain exists to
  -- provide nor the ability to be packed into an IOB at all, so the
  -- collapse would silently undo the whole fix. 'shreg_extract' (not
  -- 'srl_style') is the right control here: UG901 gives it precedence, and
  -- it prevents the inference rather than steering it. Verified after the
  -- build by checking that the design contains zero SRL primitives.
  --
  -- Three stages, not one, so that the placer has two freely-placeable
  -- hops between the (widely spread) reduction logic and the fixed IOB
  -- site in bank 14. They cost latency only, and this build has no
  -- latency contract -- it is a static-timing and resource vehicle.
  ------------------------------------------------------------------------
  signal result_p1_q : std_ulogic := '0';
  signal result_p2_q : std_ulogic := '0';
  signal result_pad_q : std_ulogic := '0';
  signal irq_p1_q : std_ulogic := '0';
  signal irq_p2_q : std_ulogic := '0';
  signal irq_pad_q : std_ulogic := '0';

  attribute shreg_extract : string;
  attribute shreg_extract of result_p1_q : signal is "no";
  attribute shreg_extract of result_p2_q : signal is "no";
  attribute shreg_extract of result_pad_q : signal is "no";
  attribute shreg_extract of irq_p1_q : signal is "no";
  attribute shreg_extract of irq_p2_q : signal is "no";
  attribute shreg_extract of irq_pad_q : signal is "no";

  function xor_reduce(value : std_ulogic_vector) return std_ulogic is
    variable result : std_ulogic := '0';
  begin
    for idx in value'range loop
      result := result xor value(idx);
    end loop;
    return result;
  end function;

begin

  ------------------------------------------------------------------------
  stim_block : process
  begin
    wait until rising_edge(clk);

    stim <= stim(c_stim_width - 2 downto 0)
      & (stimulus xor stim(c_stim_width - 1) xor stim(c_stim_width - 61) xor stim(0));

    reset_int <= reset;
  end process;

  -- Pure rewiring of shift-register outputs onto the accelerator's inputs;
  -- the ranges are disjoint so every input bit has its own flip-flop.
  m_axi_s2m.read.ar.ready <= stim(0);
  m_axi_s2m.read.r.valid <= stim(1);
  m_axi_s2m.read.r.last <= stim(2);
  m_axi_s2m.read.r.resp <= stim(4 downto 3);
  m_axi_s2m.read.r.id <= u_unsigned(stim(28 downto 5));
  m_axi_s2m.read.r.data <= stim(156 downto 29);
  m_axi_s2m.write.aw.ready <= stim(157);
  m_axi_s2m.write.w.ready <= stim(158);
  m_axi_s2m.write.b.valid <= stim(159);
  m_axi_s2m.write.b.resp <= stim(161 downto 160);
  m_axi_s2m.write.b.id <= u_unsigned(stim(185 downto 162));

  s_axi_lite_m2s.read.ar.valid <= stim(186);
  s_axi_lite_m2s.read.ar.addr <= u_unsigned(stim(250 downto 187));
  s_axi_lite_m2s.read.r.ready <= stim(251);
  s_axi_lite_m2s.write.aw.valid <= stim(252);
  s_axi_lite_m2s.write.aw.addr <= u_unsigned(stim(316 downto 253));
  s_axi_lite_m2s.write.w.valid <= stim(317);
  s_axi_lite_m2s.write.w.data <= stim(381 downto 318);
  s_axi_lite_m2s.write.w.strb <= stim(389 downto 382);
  s_axi_lite_m2s.write.b.ready <= stim(390);

  ------------------------------------------------------------------------
  result_block : process
    variable out_v : std_ulogic_vector(c_padded_width - 1 downto 0) := (others => '0');
  begin
    wait until rising_edge(clk);

    out_v := (others => '0');
    out_v(c_out_width - 1 downto 0) := to_slv(
      axi_lite_s2m => s_axi_lite_s2m, axi_m2s => m_axi_m2s, irq => irq_int
    );
    out_q <= out_v;

    for chunk_idx in 0 to c_num_chunks - 1 loop
      chunk_q(chunk_idx) <= xor_reduce(
        out_q((chunk_idx + 1) * c_chunk_width - 1 downto chunk_idx * c_chunk_width)
      );
    end loop;

    -- Pad chains: reduction/'irq_int' -> p1 -> p2 -> pad flop -> OBUF.
    -- See the declaration comment for why there are three of them and why
    -- every one carries 'shreg_extract = "no"'.
    result_p1_q <= xor_reduce(chunk_q);
    result_p2_q <= result_p1_q;
    result_pad_q <= result_p2_q;

    irq_p1_q <= irq_int;
    irq_p2_q <= irq_p1_q;
    irq_pad_q <= irq_p2_q;
  end process;

  -- The only two pad drivers in the design, each a plain flip-flop that
  -- drives nothing but its own OBUF -- the precondition for IOB packing.
  result <= result_pad_q;
  irq <= irq_pad_q;

  ------------------------------------------------------------------------
  cnn_accel_top_inst : entity cnn_accel.cnn_accel_top
    generic map (
      g_pe_rows => g_pe_rows
    )
    port map (
      clk => clk,
      reset => reset_int,
      --
      s_axi_lite_m2s => s_axi_lite_m2s,
      s_axi_lite_s2m => s_axi_lite_s2m,
      --
      m_axi_m2s => m_axi_m2s,
      m_axi_s2m => m_axi_s2m,
      --
      irq => irq_int
    );

end architecture a;
