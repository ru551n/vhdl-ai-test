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
--   (scale, shift) = lane l's entry of 'scale_rd_data' when cfg_per_channel_en
--                    else (cfg_requant_scale, cfg_requant_shift)   -- ISA v1.2 (H2)
--   total  = accum + (bias when cfg_bias_en else 0)
--   scaled = round_half_up(total * scale, shift = 15 + shift)
--            -- = floor((product + 2**(shift-1)) / 2**shift): ties round
--            -- towards +infinity, matching TOSA's `apply_scale_32`
--            -- (SINGLE_ROUND) so the TOSA->cnn_accel compiler can emit
--            -- bit-exact programs (HW milestone H0). Combined single
--            -- rounding step: the Q15 fractional shift (15) and
--            -- cfg_requant_shift are folded into one shift amount, per the
--            -- requirement's documented option, so no double-rounding
--            -- error versus two separate rounding steps.
--   biased = scaled + cfg_output_offset   -- ISA v1.1 (H1), exact, unbounded
--   result = clamp(biased, lo, hi)
--            where (lo, hi) = (cfg_clamp_min, cfg_clamp_max) when cfg_clamp_en
--                  else (0 when cfg_relu_en else -128, 127)   -- v1.0 legacy
-- With cfg_output_offset=0 and cfg_clamp_en='0' this is bit-identical to
-- the v1.0 epilogue "ReLU (max(scaled, 0)) BEFORE the int8 saturate":
-- max(x, 0) then saturate_signed(., 8) == clamp(x, 0, 127). cfg_clamp_en
-- makes cfg_relu_en irrelevant (a program wanting a ReLU with a general
-- clamp encodes it as clamp_min = 0). If cfg_clamp_min > cfg_clamp_max
-- (rejected by the ISA encoder, so never seen in a compiled program) the
-- result is min(max(biased, lo), hi) = hi, the same order the golden
-- model's bias_requantize_relu uses.
-- When cfg_requant_en='0': bias and offset are still applied to 'total',
-- then the same clamp is applied directly (no scaling) -- debug/bypass
-- path. Saturates rather than wraps so both paths share one overflow
-- semantic (architectural decision D2), matching cnn_accel_model.py's
-- golden reference.
--
-- Per-channel requantization (ISA v1.2, H2, doc/tosa_compiler_plan.md
-- section 5 extension 2): with cfg_per_channel_en='1' every lane takes
-- its own (multiplier, shift) from 'scale_rd_data' -- the row of
-- cnn_accel_weight_buffer's scale region addressed by this module's own
-- 'bias_rd_addr', so it is tiled and timed exactly like 'bias_rd_data'
-- (lane layout: cnn_accel_pkg's 'c_scale_entry_width' comment). With
-- cfg_per_channel_en='0' the descriptor's cfg_requant_scale/
-- cfg_requant_shift are broadcast to every lane, which is the pre-H2
-- datapath bit for bit; 'scale_rd_data' is then ignored entirely (may be
-- stale or unconnected). The per-lane selection happens at stage 1 with
-- the beat capture, so the pipeline depth, throughput and handshake are
-- unchanged from v1.1 -- only the stage-3 multiplier operand and the
-- stage-5 shift amount became per-lane instead of shared.
--
-- Implementation note on the offset (kept out of the requirement text):
-- the offset is folded into the rounding incrementer, i.e. stage 6 adds
-- 'offset + round_up' (a 17-bit value pre-added at stage 5) to the shifted
-- quotient instead of adding the 1-bit 'round_up' alone. Mathematically
-- identical to "round, then add offset" -- both are exact on the full-width
-- quotient -- and it keeps the pipeline at 7 stages. The bypass path
-- saturates 'total' to 17 bits before adding the 16-bit offset, which is
-- also exact with respect to the final int8 clamp: any 'total' outside
-- 17-bit range lands outside int8 after adding any 16-bit offset, so the
-- clamp result is unchanged.
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
    -- ISA v1.1 (H1) epilogue fields, instruction word W13. Signed int16 /
    -- int8 / int8; all-zero (and cfg_clamp_en='0') reproduces v1.0.
    cfg_output_offset : in std_ulogic_vector(15 downto 0) := (others => '0');
    cfg_clamp_en : in std_ulogic := '0';
    cfg_clamp_min : in std_ulogic_vector(7 downto 0) := (others => '0');
    cfg_clamp_max : in std_ulogic_vector(7 downto 0) := (others => '0');
    -- ISA v1.2 (H2) FLAG_PER_CHANNEL_EN: '1' selects lane-wise
    -- (multiplier, shift) from 'scale_rd_data' instead of the two cfg_*
    -- ports above. Sampled per beat like every other cfg_* port.
    cfg_per_channel_en : in std_ulogic := '0';

    bias_rd_addr : out std_ulogic_vector(g_bias_addr_width - 1 downto 0);
    bias_rd_data : in std_ulogic_vector(g_accum_width * g_pe_rows - 1 downto 0);
    -- Per-channel requant table row for the same 'bias_rd_addr' (cnn_accel_
    -- weight_buffer's 'scale_rd_data'), 'c_scale_entry_width' bits per
    -- lane; only read while cfg_per_channel_en='1'.
    scale_rd_data : in std_ulogic_vector(c_scale_entry_width * g_pe_rows - 1 downto 0) := (others => '0');

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
  --   5  rounded shift: quotient (barrel shift) + round-up decision,
  --      and offset + round_up (17-bit add, in parallel with the shifter)
  --   6  quotient + (offset + round_up) (the c_product_width adder; a
  --      full adder with a 17-bit sign-extended operand costs the same
  --      carry chain as the incrementer it replaces)
  --   7  saturate to int8, clamp to [lo, hi], pack -> output register
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

  -- 8, not 7: stage 7 used to do the int8 saturate, the bypass select
  -- AND the [lo, hi] clamp in one cycle. Post-route at 175 MHz that was
  -- 8 logic levels spread over ~8 slices with 74 % of the delay in pure
  -- routing (-0.521 ns, the design's worst path, and the same cone in
  -- BOTH instantiations -- 'pool_requant' and conv's 'bias_requant' --
  -- for 107 failing endpoints between them). Saturate and clamp are two
  -- independent operations chained only by data, so they split cleanly
  -- into one stage each; see the 'Into stage 7'/'Into stage 8' comments
  -- in 'lane_gen'. 'shared/TimingAndResources.md', Fundamentals:
  -- "Budget logic depth per stage, and check it."
  constant c_stages : positive := 8;

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

  -- Bypass path intermediate: 'total' saturated to 17 bits, then the
  -- 16-bit offset added at 18 bits (see the entity-level implementation
  -- note for why 17 bits is exact).
  constant c_bypass_sat_width : positive := 17;
  constant c_bypass_sum_width : positive := c_bypass_sat_width + 1;
  -- offset (16-bit signed) + round_up (0/1) fits 17 bits signed.
  constant c_offs_round_width : positive := 17;

  type bypass_sat_lanes_t is array (0 to g_pe_rows - 1) of signed(c_bypass_sat_width - 1 downto 0);
  type bypass_sum_lanes_t is array (0 to g_pe_rows - 1) of signed(c_bypass_sum_width - 1 downto 0);
  type offs_round_lanes_t is array (0 to g_pe_rows - 1) of signed(c_offs_round_width - 1 downto 0);

  -- Scale and shift are per lane since ISA v1.2 (per-channel
  -- requantization); the remaining config is shared by all lanes.
  type shift_lanes_t is array (0 to g_pe_rows - 1) of shift_t;
  type scale_lanes_t is array (0 to g_pe_rows - 1) of signed(c_scale_entry_mult_width - 1 downto 0);
  type shift_pipe_t is array (1 to 4) of shift_lanes_t;
  type scale_pipe_t is array (1 to 2) of scale_lanes_t;
  type offset_pipe_t is array (1 to 4) of signed(15 downto 0);
  type bound_pipe_t is array (1 to 7) of signed(7 downto 0);

  ------------------------------------------------------------------------
  -- Per-beat control signal decoding: the per-lane (scale, shift) select
  -- (ISA v1.2) and the shared clamp bounds.
  ------------------------------------------------------------------------

  -- Lane-wise multiplier / combined shift, after the per-channel select
  -- and the g_max_requant_shift clamp.
  signal lane_scale : scale_lanes_t;
  signal combined_shift : shift_lanes_t;
  -- The clamp bounds resolved from cfg_clamp_en/cfg_relu_en/cfg_clamp_*.
  signal clamp_lo : signed(7 downto 0);
  signal clamp_hi : signed(7 downto 0);

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
  -- Consumed at stage 7 (final path mux).
  signal requant_en_p : std_ulogic_vector(1 to 6) := (others => '0');
  -- Consumed at stage 3 (the multiply).
  signal scale_p : scale_pipe_t := (others => (others => (others => '0')));
  -- Consumed at stage 5 (the rounded shift).
  signal shift_p : shift_pipe_t := (others => (others => 15));
  -- Consumed at stage 4 (bypass offset add) and stage 5 (offset + round_up).
  signal offset_p : offset_pipe_t := (others => (others => '0'));
  -- Consumed at stage 8 (the clamp).
  signal lo_p : bound_pipe_t := (others => (others => '0'));
  signal hi_p : bound_pipe_t := (others => (others => '0'));

  ------------------------------------------------------------------------
  -- Per-lane datapath registers.
  ------------------------------------------------------------------------

  signal accum_1 : accum_lanes_t := (others => (others => '0'));
  signal bias_1 : accum_lanes_t := (others => (others => '0'));
  signal total_2 : sum_lanes_t := (others => (others => '0'));
  signal prod_3 : product_lanes_t := (others => (others => '0'));
  signal prod_4 : product_lanes_t := (others => (others => '0'));
  signal quot_5 : product_lanes_t := (others => (others => '0'));
  signal offs_round_5 : offs_round_lanes_t := (others => (others => '0'));
  signal scaled_6 : product_lanes_t := (others => (others => '0'));
  signal bypass_3 : bypass_sat_lanes_t := (others => (others => '0'));
  signal bypass_4 : bypass_sum_lanes_t := (others => (others => '0'));
  signal bypass_5 : byte_lanes_t := (others => (others => '0'));
  signal bypass_6 : byte_lanes_t := (others => (others => '0'));

  ------------------------------------------------------------------------
  -- Per-lane combinational stage inputs (the '_next' of each register
  -- above) and the packed output word.
  ------------------------------------------------------------------------

  signal total_next : sum_lanes_t;
  signal bypass_sat_next : bypass_sat_lanes_t;
  signal bypass_sum_next : bypass_sum_lanes_t;
  signal bypass_sat8_next : byte_lanes_t;
  signal quot_next : product_lanes_t;
  signal round_up_next : std_ulogic_vector(0 to g_pe_rows - 1);
  signal offs_round_next : offs_round_lanes_t;
  signal scaled_next : product_lanes_t;
  signal final_lane : byte_lanes_t;
  -- Stage 7: saturated / bypass-selected byte, before the [lo, hi] clamp.
  -- The register that splits the old single-cycle saturate+clamp cone.
  -- 'pre_clamp_next' is its combinational input, at architecture level
  -- (not inside 'lane_gen') so the sequential process can register it.
  signal pre_clamp_next : byte_lanes_t;
  signal pre_clamp_7 : byte_lanes_t := (others => (others => '0'));

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

  assert c_scale_entry_width = c_scale_entry_mult_width + c_scale_entry_shift_width
    report "cnn_accel_bias_requant: c_scale_entry_width does not match multiplier + shift widths"
    severity failure;

  assert c_scale_entry_mult_width = cfg_requant_scale'length
    report "cnn_accel_bias_requant: per-channel multiplier width must equal cfg_requant_scale's"
    severity failure;

  ------------------------------------------------------------------------
  -- Per-beat control signal decoding. Combinational off the 'cfg_*'/
  -- 'scale_rd_data' ports; the result is captured at stage 1 with the beat
  -- it belongs to. Per lane (ISA v1.2): the multiplier and the raw 8-bit
  -- shift come from the lane's table entry when cfg_per_channel_en, else
  -- from the shared cfg_* ports; the g_max_requant_shift clamp and the
  -- '+15' Q15 fold are then applied to whichever was selected, so both
  -- modes see exactly the same shift arithmetic.
  ------------------------------------------------------------------------

  lane_cfg_gen : for l in 0 to g_pe_rows - 1 generate
    constant c_lane_lo : natural := c_scale_entry_width * l;
    signal shift_raw_l : std_ulogic_vector(7 downto 0);
    signal shift_amt_raw_l : natural range 0 to 255;
    signal shift_amt_clamped_l : natural range 0 to g_max_requant_shift;
  begin
    lane_scale(l) <=
      signed(scale_rd_data(c_lane_lo + c_scale_entry_mult_width - 1 downto c_lane_lo))
      when cfg_per_channel_en = '1' else signed(cfg_requant_scale);
    shift_raw_l <=
      scale_rd_data(c_lane_lo + c_scale_entry_width - 1 downto c_lane_lo + c_scale_entry_mult_width)
      when cfg_per_channel_en = '1' else cfg_requant_shift;

    shift_amt_raw_l <= to_integer(unsigned(shift_raw_l));
    shift_amt_clamped_l <= shift_amt_raw_l when shift_amt_raw_l <= g_max_requant_shift else g_max_requant_shift;
    combined_shift(l) <= 15 + shift_amt_clamped_l;
  end generate lane_cfg_gen;

  -- Clamp bounds: general clamp (ISA v1.1 CLAMP_EN) or the v1.0 legacy
  -- pair, where ReLU is just a lower bound of 0 (see entity comment).
  clamp_lo <= signed(cfg_clamp_min) when cfg_clamp_en = '1' else
              to_signed(0, 8) when cfg_relu_en = '1' else
              to_signed(-128, 8);
  clamp_hi <= signed(cfg_clamp_max) when cfg_clamp_en = '1' else to_signed(127, 8);

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

    -- Stage 7: requant path saturated to int8, then the path mux and the
    -- general clamp.
    signal sat_result_l : signed(7 downto 0);
    signal pre_clamp_l : signed(7 downto 0);

  begin

    --------------------------------------------------------------------
    -- Into stage 2: total = accum + (bias when enabled).
    --------------------------------------------------------------------

    total_next(l) <= resize(accum_1(l), c_sum_width) + resize(bias_1(l), c_sum_width) when bias_en_1 = '1' else
                     resize(accum_1(l), c_sum_width);

    --------------------------------------------------------------------
    -- Into stages 3..5: the bypass path (cfg_requant_en='0'), computed in
    -- parallel with the multiply so only a narrow value per lane needs
    -- carrying down the rest of the pipeline instead of the full 'total'.
    -- Bias and the output offset are still applied to 'total', then it
    -- is saturated to int8 -- the same overflow semantic (saturate, not
    -- wrap) as the requant path, per architectural decision D2, matching
    -- cnn_accel_model.py's golden reference. The clamp (which subsumes
    -- the v1.0 ReLU) is shared with the requant path at stage 7.
    --   stage 3: total saturated to 17 bits (exact w.r.t. the final
    --            clamp, see the entity-level implementation note)
    --   stage 4: + offset, at 18 bits
    --   stage 5: saturated to int8
    -- Reuses 'math.saturate_signed' directly, same primitive as the
    -- requant path below, rather than hand-rolling saturations.
    --------------------------------------------------------------------

    bypass_saturate_wide_inst : entity math.saturate_signed
      generic map (
        input_width => c_sum_width,
        result_width => c_bypass_sat_width,
        enable_output_register => false
      )
      port map (
        clk => clk,
        input_valid => '1',
        input_value => total_2(l),
        result_valid => open,
        result_value => bypass_sat_next(l),
        result_is_saturated => open
      );

    bypass_sum_next(l) <= resize(bypass_3(l), c_bypass_sum_width) + resize(offset_p(3), c_bypass_sum_width);

    bypass_saturate_signed_inst : entity math.saturate_signed
      generic map (
        input_width => c_bypass_sum_width,
        result_width => 8,
        enable_output_register => false
      )
      port map (
        clk => clk,
        input_valid => '1',
        input_value => bypass_4(l),
        result_valid => open,
        result_value => bypass_sat8_next(l),
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
      shift_amt := shift_p(4)(l);
      product_u := unsigned(prod_4(l));

      quot_next(l) <= shift_right(prod_4(l), shift_amt);
      round_up_next(l) <= product_u(shift_amt - 1);
    end process;

    -- The output offset folded into the rounding term (entity-level
    -- implementation note): offset + round_up, 17 bits.
    offs_round_next(l) <= resize(offset_p(4), c_offs_round_width) + 1 when round_up_next(l) = '1' else
                          resize(offset_p(4), c_offs_round_width);

    --------------------------------------------------------------------
    -- Into stage 6: finish the rounding and add the offset in one adder.
    --------------------------------------------------------------------

    scaled_next(l) <= quot_5(l) + resize(offs_round_5(l), c_product_width);

    --------------------------------------------------------------------
    -- Into stage 7: saturate to int8 (genuine reuse of
    -- 'math.saturate_signed'), select the requant or the bypass result,
    -- then clamp to [lo, hi]. clamp(saturate8(x), lo, hi) == clamp(x, lo,
    -- hi) for int8 bounds: anything saturated was already beyond the
    -- bound on that side. The 'lo > hi' term makes the result equal to
    -- min(max(x, lo), hi) also for that (encoder-rejected) case, so the
    -- RTL and the golden model agree everywhere, without chaining the
    -- two comparisons.
    --------------------------------------------------------------------

    saturate_signed_inst : entity math.saturate_signed
      generic map (
        input_width => c_product_width,
        result_width => 8,
        enable_output_register => false
      )
      port map (
        clk => clk,
        input_valid => '1',
        input_value => scaled_6(l),
        result_valid => open,
        result_value => sat_result_l,
        result_is_saturated => open
      );

    pre_clamp_l <= sat_result_l when requant_en_p(6) = '1' else bypass_6(l);
    pre_clamp_next(l) <= pre_clamp_l;

    --------------------------------------------------------------------
    -- Into stage 8: the [lo, hi] clamp, off stage 7's register. Reading
    -- 'pre_clamp_7' rather than 'pre_clamp_l' is the whole split -- the
    -- saturate cone above and the clamp's compare/carry chain below no
    -- longer share a cycle. Bit-exact: same expression, same operands,
    -- one cycle later.
    --------------------------------------------------------------------

    final_lane(l) <= hi_p(7) when (pre_clamp_7(l) > hi_p(7) or lo_p(7) > hi_p(7)) else
                     lo_p(7) when pre_clamp_7(l) < lo_p(7) else
                     pre_clamp_7(l);

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
        requant_en_p(1) <= cfg_requant_en;
        scale_p(1) <= lane_scale;
        shift_p(1) <= combined_shift;
        offset_p(1) <= signed(cfg_output_offset);
        lo_p(1) <= clamp_lo;
        hi_p(1) <= clamp_hi;

        -- Stage 2: bias add.
        valid_q(2) <= valid_q(1);
        last_q(2) <= last_q(1);
        total_2 <= total_next;
        requant_en_p(2) <= requant_en_p(1);
        scale_p(2) <= scale_p(1);
        shift_p(2) <= shift_p(1);
        offset_p(2) <= offset_p(1);
        lo_p(2) <= lo_p(1);
        hi_p(2) <= hi_p(1);

        -- Stage 3: the requant multiply, plus the bypass path's 17-bit
        -- saturated total.
        valid_q(3) <= valid_q(2);
        last_q(3) <= last_q(2);
        for l in 0 to g_pe_rows - 1 loop
          prod_3(l) <= total_2(l) * scale_p(2)(l);
        end loop;
        bypass_3 <= bypass_sat_next;
        requant_en_p(3) <= requant_en_p(2);
        shift_p(3) <= shift_p(2);
        offset_p(3) <= offset_p(2);
        lo_p(3) <= lo_p(2);
        hi_p(3) <= hi_p(2);

        -- Stage 4: product pipeline register. Deliberately a bare
        -- register move: it lets Vivado retime it into the DSP48E1
        -- cascade's own PREG instead of spending fabric on it. The
        -- bypass path adds its offset here.
        valid_q(4) <= valid_q(3);
        last_q(4) <= last_q(3);
        prod_4 <= prod_3;
        bypass_4 <= bypass_sum_next;
        requant_en_p(4) <= requant_en_p(3);
        shift_p(4) <= shift_p(3);
        offset_p(4) <= offset_p(3);
        lo_p(4) <= lo_p(3);
        hi_p(4) <= hi_p(3);

        -- Stage 5: quotient and (offset + round_up); the bypass path's
        -- int8 saturate.
        valid_q(5) <= valid_q(4);
        last_q(5) <= last_q(4);
        quot_5 <= quot_next;
        offs_round_5 <= offs_round_next;
        bypass_5 <= bypass_sat8_next;
        requant_en_p(5) <= requant_en_p(4);
        lo_p(5) <= lo_p(4);
        hi_p(5) <= hi_p(4);

        -- Stage 6: the rounding/offset adder, on its own so that its
        -- carry chain does not share a cycle with the saturate cone.
        valid_q(6) <= valid_q(5);
        last_q(6) <= last_q(5);
        scaled_6 <= scaled_next;
        bypass_6 <= bypass_5;
        requant_en_p(6) <= requant_en_p(5);
        lo_p(6) <= lo_p(5);
        hi_p(6) <= hi_p(5);

        -- Stage 7: the int8 saturate and the bypass select, registered
        -- before the clamp. See 'c_stages' for why this is its own stage.
        valid_q(7) <= valid_q(6);
        last_q(7) <= last_q(6);
        pre_clamp_7 <= pre_clamp_next;
        lo_p(7) <= lo_p(6);
        hi_p(7) <= hi_p(6);

        -- Stage 8: the clamp result, in the output register.
        valid_q(8) <= valid_q(7);
        last_q(8) <= last_q(7);
        out_data_q <= next_out_data_full;
      end if;
    end if;
  end process;

  m_out_m2s.valid <= valid_q(c_stages);
  m_out_m2s.data <= out_data_q;
  m_out_m2s.last <= last_q(c_stages);
  m_out_m2s.user <= (others => '0');

end architecture a;
