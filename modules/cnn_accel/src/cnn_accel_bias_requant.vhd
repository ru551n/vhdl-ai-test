library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library axi_stream;
use axi_stream.axi_stream_pkg.all;

library math;

library cnn_accel;
use cnn_accel.cnn_accel_pkg.all;

-- Shared output-quantization stage. See
-- modules/cnn_accel/doc/cnn_accel_bias_requant_req.md and
-- modules/cnn_accel/doc/cnn_accel_bias_requant_proposal.md.
--
-- Per lane (one per output-channel PE row, 'g_pe_rows' lanes total), per
-- accepted 's_accum' beat:
--   total  = accum + (bias when cfg_bias_en else 0)
--   scaled = round_half_up(total * cfg_requant_scale, shift = 15 + cfg_requant_shift)
--            -- = floor((product + 2**(shift-1)) / 2**shift): ties round
--            -- towards +infinity, matching TOSA's `apply_scale_32`
--            -- (SINGLE_ROUND) so the TOSA->cnn_accel compiler can emit
--            -- bit-exact programs (HW milestone H0). Combined single
--            -- rounding step: the Q15 fractional shift (15) and
--            -- cfg_requant_shift are folded into one shift amount, per the
--            -- requirement's documented option, so no double-rounding
--            -- error versus two separate rounding steps.
--   relu   = max(scaled, 0) when cfg_relu_en, applied BEFORE the int8 clamp
--   result = saturate_signed(relu, 8)
-- When cfg_requant_en='0': bias/relu are still applied to 'total', then
-- 'total' is saturated to int8 directly (no scaling) -- debug/bypass path.
-- Saturates rather than wraps so both paths share one overflow semantic
-- (architectural decision D2), matching cnn_accel_model.py's golden
-- reference.
--
-- Timing: the datapath is a 7-stage pipeline (still one output beat per
-- accepted input beat, but 7 cycles of latency) and the per-beat 'cfg_*'
-- values are captured together with the beat, so 'cfg_*' may change as
-- soon as a beat has been accepted. See the architecture's "Pipeline"
-- comment for the stage-by-stage split, and for why this module's own
-- out-of-context netlist Fmax is not a usable number.
--
-- bias_rd_addr / bias tiling (v1 design decision, see proposal doc §3):
-- this module always drives 'bias_rd_addr' to all-zeros, i.e. it assumes a
-- single bias row (row 0) covers all 'g_pe_rows' output channels for the
-- whole layer (out_channels <= g_pe_rows). Multi-tile bias iteration would
-- need an explicit tile-index input not present on this module's port list
-- and is out of scope for v1.
entity cnn_accel_bias_requant is
  generic (
    -- Input accumulator width (int32 default).
    g_accum_width : positive := 32;
    -- Output-channel parallelism: number of independent lanes.
    g_pe_rows : positive := 8;
    -- Width of 'bias_rd_addr'. Added by vhdesign (not in the requirement)
    -- so this module's read-address port can be sized to match
    -- cnn_accel_weight_buffer's actual 'bias_rd_addr' width
    -- (num_bits_needed(g_weight_buffer_depth - 1)) at cnn_accel_top
    -- integration time. The value driven is always all-zeros in v1 (see
    -- entity-level comment above) regardless of this generic's value.
    g_bias_addr_width : positive := 1;
    -- Added by vhdesign: upper bound on the runtime-variable
    -- 'cfg_requant_shift' that this module supports at full precision.
    -- 'cfg_requant_shift' values above this bound are clamped to it (a
    -- defensive limit, not expected to be hit by compiled programs -- see
    -- proposal doc §4).
    g_max_requant_shift : natural := 31
  );
  port (
    clk : in std_ulogic;
    reset : in std_ulogic;

    cfg_bias_en : in std_ulogic;
    cfg_requant_en : in std_ulogic;
    cfg_relu_en : in std_ulogic;
    cfg_requant_scale : in std_ulogic_vector(31 downto 0);
    cfg_requant_shift : in std_ulogic_vector(7 downto 0);

    bias_rd_addr : out std_ulogic_vector(g_bias_addr_width - 1 downto 0);
    bias_rd_data : in std_ulogic_vector(g_accum_width * g_pe_rows - 1 downto 0);

    -- One int32-ish (g_accum_width-bit) partial sum per PE row, from
    -- cnn_accel_pe_array's 'm_accum_m2s'. An array of lanes (D15), not a
    -- packed AXI4-Stream payload, so lane 'l' is indexed directly --
    -- see cnn_accel_pkg.vhd's 'accum_m2s_t' doc comment.
    s_accum_m2s : in accum_m2s_t(data(0 to g_pe_rows - 1)(g_accum_width - 1 downto 0));
    s_accum_s2m : out accum_s2m_t;

    m_out_m2s : out axi_stream_m2s_t;
    m_out_s2m : in axi_stream_s2m_t
  );
end entity cnn_accel_bias_requant;

architecture a of cnn_accel_bias_requant is

  -- One extra guard bit so 'accum + bias' cannot overflow (both operands
  -- are 'g_accum_width'-bit signed).
  constant c_sum_width : positive := g_accum_width + 1;
  -- Exact width of 'total * cfg_requant_scale' (signed(a) * signed(b) is
  -- exactly a+b bits wide, no truncation before rounding).
  constant c_product_width : positive := c_sum_width + 32;

  ------------------------------------------------------------------------
  -- Pipeline
  --
  -- This module used to compute the whole bias-add -> multiply -> rounded
  -- shift -> ReLU -> saturate chain in one combinational cone feeding a
  -- single output register. Measured inside 'cnn_accel_conv_core' on
  -- xc7a200tfbg484-2 that cone was 21.337 ns / 43 logic levels
  -- (25 CARRY4 + 2 series DSP48E1), i.e. 46.77 MHz, and it was the
  -- critical path of the entire M6b composition.
  --
  -- NOTE for anyone re-measuring: this entity's *own* out-of-context
  -- netlist build reported 520.02 MHz for the very same RTL, because
  -- synthesis-only register-to-register timing never times a
  -- combinational cone that starts at an input port -- and here the whole
  -- cone is driven from 's_accum_m2s.data' / 'bias_rd_data'. Only the
  -- composition entity gives a real number for this module. Do not
  -- "optimize" against the standalone figure.
  --
  -- The datapath is therefore split into 'c_stages' register stages,
  -- suffix '_<n>' on every signal naming the stage it is registered in:
  --
  --   1  capture 's_accum'/'bias_rd_data' and the per-beat config
  --      (cuts the input-port cone at a register immediately)
  --   2  total = accum + bias        (c_sum_width-bit add)
  --   3  product = total * scale     (DSP48E1 MREG stage)
  --      and, in parallel, the whole bypass path's 8-bit result
  --   4  product pipeline register   (DSP48E1 PREG stage)
  --   5  rounded shift: quotient (barrel shift) + round-up decision
  --   6  quotient + round_up (the c_product_width incrementer)
  --   7  ReLU, saturate, pack -> output register
  --
  -- Stages 6 and 7 are split rather than fused. Fused, this module was
  -- conv_core's critical path at 6.215 ns / 21 levels / 17 CARRY4
  -- (159.72 MHz): the incrementer's carry chain plus the saturate cone
  -- does not fit one 150 MHz cycle with any margin left for
  -- place-and-route. Splitting them moves this module off the critical
  -- path but buys conv_core almost nothing by itself (159.72 ->
  -- 159.95 MHz), because conv_core has a *plateau* of paths around 6 ns
  -- and window_gen's tap-index decode simply took over at 6.012 ns. It is
  -- kept because 522 FFs (0.2% of the device) to retire a near-limit path
  -- is worth it ahead of real place-and-route closure, not because it
  -- raised the synthesis estimate. Do not read the 0.23 MHz as the value
  -- of the change, and do not expect the next such split to pay either
  -- until the whole plateau moves.
  --
  -- Throughput is unchanged at one output beat per accepted input beat;
  -- only latency grows from 1 to 'c_stages' cycles. Latency is not
  -- observable in 'cnn_accel_conv_core''s cycle budget: this module is a
  -- pure streaming stage downstream of the accumulate-and-emit PE array,
  -- so its latency costs once per frame, not once per pixel.
  --
  -- Backpressure: all stages share one 'pipe_en'. When the output stage
  -- holds a beat the consumer will not take, the whole pipeline freezes
  -- and 's_accum_s2m.ready' goes low; nothing is dropped and no bubble is
  -- inserted while the consumer keeps up.
  ------------------------------------------------------------------------

  constant c_stages : positive := 7;

  -- 'combined_shift' is always 15 plus a non-negative clamp, so the
  -- rounding logic may rely on shift >= 15. It really only needs >= 1 (so
  -- that bit 'shift - 1' of the product exists), but encoding the true
  -- lower bound in the subtype also shrinks the barrel shifter and the
  -- guard-bit decoder that synthesis builds from it.
  subtype shift_t is natural range 15 to 15 + g_max_requant_shift;

  type accum_lanes_t is array (0 to g_pe_rows - 1) of signed(g_accum_width - 1 downto 0);
  type sum_lanes_t is array (0 to g_pe_rows - 1) of signed(c_sum_width - 1 downto 0);
  type product_lanes_t is array (0 to g_pe_rows - 1) of signed(c_product_width - 1 downto 0);
  type byte_lanes_t is array (0 to g_pe_rows - 1) of signed(7 downto 0);

  type shift_pipe_t is array (1 to 4) of shift_t;
  type scale_pipe_t is array (1 to 2) of signed(31 downto 0);

  ------------------------------------------------------------------------
  -- Shared (per-beat, all lanes) control signal decoding.
  ------------------------------------------------------------------------

  signal shift_amt_raw : natural range 0 to 255;
  signal shift_amt_clamped : natural range 0 to g_max_requant_shift;
  signal combined_shift : shift_t;

  ------------------------------------------------------------------------
  -- Pipeline control.
  ------------------------------------------------------------------------

  signal pipe_en : std_ulogic;
  signal valid_q : std_ulogic_vector(1 to c_stages) := (others => '0');
  signal last_q : std_ulogic_vector(1 to c_stages) := (others => '0');

  ------------------------------------------------------------------------
  -- Per-beat captured configuration. Captured *with* the beat at stage 1
  -- rather than read live at each stage: the module is 'c_stages' cycles
  -- deep now, so a 'cfg_*' change made right after a beat was accepted
  -- must not retroactively change that beat's result. (The testbench's
  -- 'send_beat' does exactly this -- drive cfg, hand over one beat, then
  -- drive the next beat's cfg.) Each config signal is carried only as far
  -- as the stage that consumes it.
  ------------------------------------------------------------------------

  signal bias_en_1 : std_ulogic := '0';
  -- Consumed at stage 3 (bypass ReLU) and stage 7 (requant ReLU).
  signal relu_en_p : std_ulogic_vector(1 to 6) := (others => '0');
  -- Consumed at stage 7 (final path mux).
  signal requant_en_p : std_ulogic_vector(1 to 6) := (others => '0');
  -- Consumed at stage 3 (the multiply).
  signal scale_p : scale_pipe_t := (others => (others => '0'));
  -- Consumed at stage 5 (the rounded shift).
  signal shift_p : shift_pipe_t := (others => 15);

  ------------------------------------------------------------------------
  -- Per-lane datapath registers.
  ------------------------------------------------------------------------

  signal accum_1 : accum_lanes_t := (others => (others => '0'));
  signal bias_1 : accum_lanes_t := (others => (others => '0'));
  signal total_2 : sum_lanes_t := (others => (others => '0'));
  signal prod_3 : product_lanes_t := (others => (others => '0'));
  signal prod_4 : product_lanes_t := (others => (others => '0'));
  signal quot_5 : product_lanes_t := (others => (others => '0'));
  signal round_up_5 : std_ulogic_vector(0 to g_pe_rows - 1) := (others => '0');
  signal scaled_6 : product_lanes_t := (others => (others => '0'));
  signal bypass_3 : byte_lanes_t := (others => (others => '0'));
  signal bypass_4 : byte_lanes_t := (others => (others => '0'));
  signal bypass_5 : byte_lanes_t := (others => (others => '0'));
  signal bypass_6 : byte_lanes_t := (others => (others => '0'));

  ------------------------------------------------------------------------
  -- Per-lane combinational stage inputs (the '_next' of each register
  -- above) and the packed output word.
  ------------------------------------------------------------------------

  signal total_next : sum_lanes_t;
  signal bypass_next : byte_lanes_t;
  signal quot_next : product_lanes_t;
  signal round_up_next : std_ulogic_vector(0 to g_pe_rows - 1);
  signal scaled_next : product_lanes_t;
  signal final_lane : byte_lanes_t;

  signal next_out_data_full : std_ulogic_vector(axi_stream_data_sz - 1 downto 0);

  signal out_data_q : std_ulogic_vector(axi_stream_data_sz - 1 downto 0) := (others => '0');

begin

  assert 8 * g_pe_rows <= axi_stream_data_sz
    report "cnn_accel_bias_requant: 8*g_pe_rows exceeds axi_stream_pkg's fixed data width"
    severity failure;

  assert g_accum_width >= 8
    report "cnn_accel_bias_requant: g_accum_width must be >= 8 (bypass path's saturate_signed needs input_width >= result_width=8)"
    severity failure;

  assert (15 + g_max_requant_shift) <= (c_product_width - 2)
    report "cnn_accel_bias_requant: g_max_requant_shift too large for g_accum_width"
    severity failure;

  ------------------------------------------------------------------------
  -- Shared (per-beat, all lanes) control signal decoding. Combinational
  -- off the 'cfg_*' ports; the result is captured at stage 1 with the
  -- beat it belongs to.
  ------------------------------------------------------------------------

  shift_amt_raw <= to_integer(unsigned(cfg_requant_shift));
  shift_amt_clamped <= shift_amt_raw when shift_amt_raw <= g_max_requant_shift else g_max_requant_shift;
  combined_shift <= 15 + shift_amt_clamped;

  ------------------------------------------------------------------------
  -- v1 bias addressing: single bias row (row 0) for the whole layer --
  -- see entity-level comment.
  ------------------------------------------------------------------------

  bias_rd_addr <= (others => '0');

  ------------------------------------------------------------------------
  -- Flow control: one enable for the whole pipeline (see the "Pipeline"
  -- comment above).
  ------------------------------------------------------------------------

  pipe_en <= (not valid_q(c_stages)) or m_out_s2m.ready;
  s_accum_s2m.ready <= pipe_en;

  ------------------------------------------------------------------------
  -- Per-lane combinational logic between the register stages.
  ------------------------------------------------------------------------

  lane_gen : for l in 0 to g_pe_rows - 1 generate

    -- Stage 3's bypass path.
    signal relu_clamped_total_l : signed(c_sum_width - 1 downto 0);
    -- Stage 7's requant path.
    signal relu_scaled_l : signed(c_product_width - 1 downto 0);
    signal sat_result_l : signed(7 downto 0);

  begin

    --------------------------------------------------------------------
    -- Into stage 2: total = accum + (bias when enabled).
    --------------------------------------------------------------------

    total_next(l) <= resize(accum_1(l), c_sum_width) + resize(bias_1(l), c_sum_width) when bias_en_1 = '1' else
                     resize(accum_1(l), c_sum_width);

    --------------------------------------------------------------------
    -- Into stage 3: the bypass path (cfg_requant_en='0'), computed in
    -- parallel with the multiply so only 8 bits per lane need carrying
    -- down the rest of the pipeline instead of the full 'total'. Bias
    -- and ReLU are still applied to 'total', then it is saturated to
    -- int8 -- the same overflow semantic (saturate, not wrap) as the
    -- requant path, per architectural decision D2, matching
    -- cnn_accel_model.py's golden reference. Reuses
    -- 'math.saturate_signed' directly, same primitive as the requant
    -- path below, rather than hand-rolling a second saturation.
    --------------------------------------------------------------------

    relu_clamped_total_l <= (others => '0') when (relu_en_p(2) = '1' and total_2(l)(total_2(l)'high) = '1') else
                            total_2(l);

    bypass_saturate_signed_inst : entity math.saturate_signed
      generic map (
        input_width => c_sum_width,
        result_width => 8,
        enable_output_register => false
      )
      port map (
        clk => clk,
        input_valid => '1',
        input_value => relu_clamped_total_l,
        result_valid => open,
        result_value => bypass_next(l),
        result_is_saturated => open
      );

    --------------------------------------------------------------------
    -- Into stage 5: the rounded arithmetic right shift, split across the
    -- stage 5 / stage 6 boundary.
    --
    -- 'shift_right' on a signed value is a floor division by 2**shift,
    -- so the remainder 'product mod 2**shift' is exactly the low 'shift'
    -- bits of the product read as unsigned -- for negative products too.
    -- Round-half-up (ties towards +infinity, H0) is then just the guard
    -- bit:
    --
    --   guard  = product(shift - 1)          -- is remainder >= half?
    --   2*rem <  2**shift  <=>  guard = '0'  -> truncate
    --   2*rem >= 2**shift  <=>  guard = '1'  -> +1   (ties included)
    --   => round_up = guard
    --
    -- i.e. floor((product + 2**(shift-1)) / 2**shift), identical to
    -- TOSA's apply_scale_32 (SINGLE_ROUND). No sticky reduction and no
    -- quotient-parity term are needed any more (the former round-to-even
    -- rule required both).
    --
    -- Verified bit-exact against cnn_accel_model.py's
    -- 'round_shift_right_signed' by the testbench (that is what
    -- 'test_round_half_up_ties' is for), rather than by this comment.
    --
    -- 'math.truncate_round_signed' is still not usable here: its number
    -- of removed LSBs is fixed by generics at elaboration time, whereas
    -- 'shift' is a genuine runtime value derived from
    -- 'cfg_requant_shift' (an ISA field, loaded per instruction) -- see
    -- proposal doc section 4.
    --------------------------------------------------------------------

    round_shift_proc : process(all)
      variable product_u : unsigned(c_product_width - 1 downto 0);
      variable shift_amt : shift_t;
    begin
      shift_amt := shift_p(4);
      product_u := unsigned(prod_4(l));

      quot_next(l) <= shift_right(prod_4(l), shift_amt);
      round_up_next(l) <= product_u(shift_amt - 1);
    end process;

    --------------------------------------------------------------------
    -- Into stage 6: finish the rounding.
    --------------------------------------------------------------------

    scaled_next(l) <= quot_5(l) + 1 when round_up_5(l) = '1' else quot_5(l);

    --------------------------------------------------------------------
    -- Into stage 7: apply ReLU BEFORE the int8 clamp, saturate (genuine
    -- reuse of 'math.saturate_signed'), then select the requant or the
    -- bypass result.
    --------------------------------------------------------------------

    relu_scaled_l <= (others => '0') when (relu_en_p(6) = '1' and scaled_6(l)(c_product_width - 1) = '1') else
                     scaled_6(l);

    saturate_signed_inst : entity math.saturate_signed
      generic map (
        input_width => c_product_width,
        result_width => 8,
        enable_output_register => false
      )
      port map (
        clk => clk,
        input_valid => '1',
        input_value => relu_scaled_l,
        result_valid => open,
        result_value => sat_result_l,
        result_is_saturated => open
      );

    final_lane(l) <= sat_result_l when requant_en_p(6) = '1' else bypass_6(l);

    next_out_data_full(8 * (l + 1) - 1 downto 8 * l) <= std_ulogic_vector(final_lane(l));

  end generate lane_gen;

  data_padding_gen : if 8 * g_pe_rows < axi_stream_data_sz generate
    next_out_data_full(axi_stream_data_sz - 1 downto 8 * g_pe_rows) <= (others => '0');
  end generate;

  ------------------------------------------------------------------------
  -- The pipeline registers themselves. Reset clears the valid chain only
  -- (no completeness contract on stale data content), same as the
  -- single-stage version this replaces.
  ------------------------------------------------------------------------

  pipeline : process(clk)
  begin
    if rising_edge(clk) then
      if reset = '1' then
        valid_q <= (others => '0');
      elsif pipe_en = '1' then
        -- Stage 1: capture the accepted beat, its bias word and its config.
        valid_q(1) <= s_accum_m2s.valid;
        last_q(1) <= s_accum_m2s.last;
        for l in 0 to g_pe_rows - 1 loop
          accum_1(l) <= s_accum_m2s.data(l);
          bias_1(l) <= signed(bias_rd_data(g_accum_width * (l + 1) - 1 downto g_accum_width * l));
        end loop;
        bias_en_1 <= cfg_bias_en;
        relu_en_p(1) <= cfg_relu_en;
        requant_en_p(1) <= cfg_requant_en;
        scale_p(1) <= signed(cfg_requant_scale);
        shift_p(1) <= combined_shift;

        -- Stage 2: bias add.
        valid_q(2) <= valid_q(1);
        last_q(2) <= last_q(1);
        total_2 <= total_next;
        relu_en_p(2) <= relu_en_p(1);
        requant_en_p(2) <= requant_en_p(1);
        scale_p(2) <= scale_p(1);
        shift_p(2) <= shift_p(1);

        -- Stage 3: the requant multiply, plus the finished bypass result.
        valid_q(3) <= valid_q(2);
        last_q(3) <= last_q(2);
        for l in 0 to g_pe_rows - 1 loop
          prod_3(l) <= total_2(l) * scale_p(2);
        end loop;
        bypass_3 <= bypass_next;
        relu_en_p(3) <= relu_en_p(2);
        requant_en_p(3) <= requant_en_p(2);
        shift_p(3) <= shift_p(2);

        -- Stage 4: product pipeline register. Deliberately a bare
        -- register move: it lets Vivado retime it into the DSP48E1
        -- cascade's own PREG instead of spending fabric on it.
        valid_q(4) <= valid_q(3);
        last_q(4) <= last_q(3);
        prod_4 <= prod_3;
        bypass_4 <= bypass_3;
        relu_en_p(4) <= relu_en_p(3);
        requant_en_p(4) <= requant_en_p(3);
        shift_p(4) <= shift_p(3);

        -- Stage 5: quotient and round-up decision.
        valid_q(5) <= valid_q(4);
        last_q(5) <= last_q(4);
        quot_5 <= quot_next;
        round_up_5 <= round_up_next;
        bypass_5 <= bypass_4;
        relu_en_p(5) <= relu_en_p(4);
        requant_en_p(5) <= requant_en_p(4);

        -- Stage 6: the rounding incrementer, on its own so that its
        -- carry chain does not share a cycle with the saturate cone.
        valid_q(6) <= valid_q(5);
        last_q(6) <= last_q(5);
        scaled_6 <= scaled_next;
        bypass_6 <= bypass_5;
        relu_en_p(6) <= relu_en_p(5);
        requant_en_p(6) <= requant_en_p(5);

        -- Stage 7: the output register.
        valid_q(7) <= valid_q(6);
        last_q(7) <= last_q(6);
        out_data_q <= next_out_data_full;
      end if;
    end if;
  end process;

  m_out_m2s.valid <= valid_q(c_stages);
  m_out_m2s.data <= out_data_q;
  m_out_m2s.last <= last_q(c_stages);
  m_out_m2s.user <= (others => '0');

end architecture a;
