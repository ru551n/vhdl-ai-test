library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library axi;
use axi.axi_pkg.all;

-- DDR port concentrator (doc/cnn_accel_top_v2_arch.md section 2's
-- 'cnn_accel_axi_mux' block: "axi.axi_simple_read_crossbar 3:1 + write
-- passthru"). Every requester inside the accelerator that needs external
-- memory -- descriptor fetch, activation/LOAD reads, weight/bias/scale/LUT
-- reads, and the STORE write -- lands here and leaves on the IP's single
-- AXI4 master port.
--
-- Reuse over glue: the read side is hdl-modules' own
-- 'axi.axi_simple_read_crossbar', not a hand-written arbiter. That entity
-- port-locks the whole burst (AR..RLAST) to the granted input before
-- looking at another one, which is exactly what this IP needs and could
-- not get for free otherwise: every read requester here
-- ('cnn_accel_axi_read_dma') hardwires 'ARID'/'RID' to 0 (see that
-- entity's own header comment), so R beats from two different requesters
-- must never be allowed to interleave on the shared port -- there would be
-- no ID to tell them apart. Burst-granularity port locking is the property
-- that makes a single shared ID legal, so the crossbar is load-bearing for
-- correctness here, not merely convenient. Same-ID ordering
-- (shared/Axi4.md) is therefore satisfied by construction: at most one
-- burst is ever in flight on the shared port.
--
-- The only hand-written logic in this entity is (a) the degenerate
-- 1-input bypass generate below and (b) the two byte-counting strobes.
--
-- Degenerate 1-input bypass: with a single requester on a direction, the
-- crossbar's own idle -> wait_for_a_done -> wait_for_data_done ->
-- (wait_for_b_done) state machine would serialize that requester against
-- itself, costing several dead cycles per burst and -- on the write side,
-- where 'cnn_accel_ofmap_dma' issues one AW per 64-bit packet -- would
-- dominate the transfer time. A direct record connection is both free and
-- exactly as protocol-correct (there is nothing to arbitrate), so the
-- 1-input case bypasses the crossbar entirely. This is a pure structural
-- choice: no signal is modified on the way through, so handshake
-- stability, the 4 KiB boundary rule and burst framing all remain
-- whatever the requester itself produced (each 'cnn_accel_axi_read_dma' /
-- 'cnn_accel_ofmap_dma' instance already guarantees them; nothing here
-- re-splits or re-frames a burst).
--
-- Byte counting (spec section 8, 'DDR_RD_BYTES'/'DDR_WR_BYTES'): counted
-- here rather than in 'cnn_accel_cmd_proc' because this is the one place
-- that sees *all* external traffic -- including the descriptor fetches
-- 'cmd_proc' never issues itself ("**all** AXI read bytes (incl.
-- descriptors and weights)"). Only the per-cycle increments are produced
-- here; the accumulators live in 'cmd_proc' with every other counter, so
-- that one module owns when counters clear (at 'START') and this one stays
-- free of run-control state. Reads count a full beat per accepted 'R'
-- handshake (every requester issues whole-beat, 8-byte-aligned bursts);
-- writes count the actual asserted 'WSTRB' lanes, which is the exact byte
-- count even if a future requester ever issues a partial beat.
entity cnn_accel_axi_mux is
  generic (
    -- Data width of the shared AXI4 port; sizes the read byte increment
    -- and bounds the 'WSTRB' popcount.
    g_axi_data_width : positive := 64;
    -- Number of read requesters. Default 3 = descriptor fetch +
    -- activation/LOAD read DMA + weight/LUT read DMA (section 2).
    g_num_read_inputs : positive := 3;
    -- Number of write requesters. Default 1 = the STORE/spill
    -- 'cnn_accel_ofmap_dma' (section 2's "write passthru").
    g_num_write_inputs : positive := 1
  );
  port (
    clk : in std_ulogic;

    --# {{}}
    -- Read requesters, index 0 highest priority in the crossbar's own
    -- lowest-index-first scan.
    input_read_m2s : in axi_read_m2s_vec_t(0 to g_num_read_inputs - 1);
    input_read_s2m : out axi_read_s2m_vec_t(0 to g_num_read_inputs - 1) :=
      (others => axi_read_s2m_init);

    --# {{}}
    -- Write requesters.
    input_write_m2s : in axi_write_m2s_vec_t(0 to g_num_write_inputs - 1);
    input_write_s2m : out axi_write_s2m_vec_t(0 to g_num_write_inputs - 1) :=
      (others => axi_write_s2m_init);

    --# {{}}
    -- The IP's single external AXI4 master port.
    m_axi_m2s : out axi_m2s_t := axi_m2s_init;
    m_axi_s2m : in axi_s2m_t;

    --# {{}}
    -- Per-cycle byte increments for the spec section 8 traffic counters;
    -- '0' on a cycle with no accepted data beat. Accumulated by
    -- 'cnn_accel_cmd_proc' (see the entity-level comment).
    rd_bytes : out unsigned(7 downto 0) := (others => '0');
    wr_bytes : out unsigned(7 downto 0) := (others => '0')
  );
end entity cnn_accel_axi_mux;

architecture a of cnn_accel_axi_mux is

  constant c_beat_bytes : positive := g_axi_data_width / 8;

  signal read_m2s : axi_read_m2s_t := axi_read_m2s_init;
  signal read_s2m : axi_read_s2m_t;
  signal write_m2s : axi_write_m2s_t := axi_write_m2s_init;
  signal write_s2m : axi_write_s2m_t;

begin

  assert c_beat_bytes <= 255
    report "cnn_accel_axi_mux: g_axi_data_width/8 must fit the 8-bit byte-increment ports"
    severity failure;

  m_axi_m2s.read <= read_m2s;
  m_axi_m2s.write <= write_m2s;
  read_s2m <= m_axi_s2m.read;
  write_s2m <= m_axi_s2m.write;

  ------------------------------------------------------------------------
  -- Read side.
  ------------------------------------------------------------------------

  read_bypass_gen : if g_num_read_inputs = 1 generate
    read_m2s <= input_read_m2s(0);
    input_read_s2m(0) <= read_s2m;
  end generate;

  read_crossbar_gen : if g_num_read_inputs > 1 generate
    read_crossbar_inst : entity axi.axi_simple_read_crossbar
      generic map (
        num_inputs => g_num_read_inputs
      )
      port map (
        clk => clk,
        input_ports_m2s => input_read_m2s,
        input_ports_s2m => input_read_s2m,
        output_m2s => read_m2s,
        output_s2m => read_s2m
      );
  end generate;

  ------------------------------------------------------------------------
  -- Write side.
  ------------------------------------------------------------------------

  write_bypass_gen : if g_num_write_inputs = 1 generate
    write_m2s <= input_write_m2s(0);
    input_write_s2m(0) <= write_s2m;
  end generate;

  write_crossbar_gen : if g_num_write_inputs > 1 generate
    write_crossbar_inst : entity axi.axi_simple_write_crossbar
      generic map (
        num_inputs => g_num_write_inputs
      )
      port map (
        clk => clk,
        input_ports_m2s => input_write_m2s,
        input_ports_s2m => input_write_s2m,
        output_m2s => write_m2s,
        output_s2m => write_s2m
      );
  end generate;

  ------------------------------------------------------------------------
  -- Traffic measurement. Combinational: one cycle's worth of bytes, to be
  -- accumulated by the consumer.
  ------------------------------------------------------------------------

  count_bytes : process(all)
    variable strb_bytes : natural range 0 to axi_w_strb_sz;
  begin
    if read_s2m.r.valid = '1' and read_m2s.r.ready = '1' then
      rd_bytes <= to_unsigned(c_beat_bytes, rd_bytes'length);
    else
      rd_bytes <= (others => '0');
    end if;

    strb_bytes := 0;
    if write_m2s.w.valid = '1' and write_s2m.w.ready = '1' then
      for lane in 0 to axi_w_strb_sz - 1 loop
        if write_m2s.w.strb(lane) = '1' then
          strb_bytes := strb_bytes + 1;
        end if;
      end loop;
    end if;
    wr_bytes <= to_unsigned(strb_bytes, wr_bytes'length);
  end process;

end architecture a;
