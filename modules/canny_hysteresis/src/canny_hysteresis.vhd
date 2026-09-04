library ieee;
use ieee.std_logic_1164.all;

library common;

-- Last stage of the canny pipeline: local (3x3) hysteresis over a
-- pre-classified window. See modules/canny_hysteresis/doc/canny_hysteresis_req.md
-- and modules/canny_hysteresis/doc/canny_hysteresis_proposal.md.
entity canny_hysteresis is
  port (
    clk   : in std_logic;
    rst_n : in std_logic;

    s_axis_tvalid : in  std_logic;
    s_axis_tready : out std_logic;
    -- 3x3 classification window, row-major MSB-to-LSB, 2 bits/tap:
    -- w_tl=17:16, w_tc=15:14, w_tr=13:12, w_ml=11:10, w_mm=9:8, w_mr=7:6,
    -- w_bl=5:4, w_bc=3:2, w_br=1:0.
    s_axis_tdata  : in  std_logic_vector(17 downto 0);
    -- bit0 = SOF, bit1 = border.
    s_axis_tuser  : in  std_logic_vector(1 downto 0);
    s_axis_tlast  : in  std_logic;

    m_axis_tvalid : out std_logic;
    m_axis_tready : in  std_logic;
    -- final edge byte: bit0 = edge, bits 7:1 = '0'.
    m_axis_tdata  : out std_logic_vector(7 downto 0);
    -- bit0 = SOF only -- top-level boundary, border consumed, not re-exposed.
    m_axis_tuser  : out std_logic_vector(0 downto 0);
    m_axis_tlast  : out std_logic
  );
end entity canny_hysteresis;

architecture rtl of canny_hysteresis is

  -- Classification codes (2 bits/tap).
  constant c_strong : std_logic_vector(1 downto 0) := "10";
  constant c_weak   : std_logic_vector(1 downto 0) := "01";

  -- Pipeline payload packed as (sof & edge_byte): bit8=sof, bits 7:0=edge_byte.
  constant c_pipeline_width : positive := 9;

  signal edge_byte : std_logic_vector(7 downto 0);
  signal pipeline_input_data : std_logic_vector(c_pipeline_width - 1 downto 0);
  signal pipeline_output_data : std_logic_vector(c_pipeline_width - 1 downto 0);

  signal pipeline_input_valid, pipeline_input_ready : std_logic;
  signal pipeline_output_valid, pipeline_output_ready : std_logic;
  signal pipeline_output_last : std_logic;

  -- Extracts a 2-bit tap from the packed 3x3 classification window.
  function tap(
    window   : std_logic_vector(17 downto 0);
    tap_idx  : natural range 0 to 8
  ) return std_logic_vector is
  begin
    return window(17 - 2 * tap_idx downto 16 - 2 * tap_idx);
  end function;

  -- Combinational hysteresis compare: edge = '1' if the center tap is
  -- strong, or weak with at least one strong neighbor; border forces
  -- edge = '0' regardless of the window comparison.
  function compute_edge(
    window : std_logic_vector(17 downto 0);
    border : std_logic
  ) return std_logic is
    variable any_neighbor_strong : std_logic := '0';
    variable center : std_logic_vector(1 downto 0);
    variable result : std_logic;
  begin
    for tap_idx in 0 to 8 loop
      if tap_idx /= 4 and tap(window, tap_idx) = c_strong then
        any_neighbor_strong := '1';
      end if;
    end loop;

    center := tap(window, 4);

    if center = c_strong then
      result := '1';
    elsif center = c_weak and any_neighbor_strong = '1' then
      result := '1';
    else
      result := '0';
    end if;

    if border = '1' then
      result := '0';
    end if;

    return result;
  end function;

begin

  -- Reset-gating glue around common.handshake_pipeline (which has no reset
  -- port of its own) -- see doc/canny_hysteresis_proposal.md "Clock/reset
  -- behavior": force-drain the pipeline during reset by keeping its output
  -- side always ready, and mask valid on both AXI-Stream boundaries so no
  -- stale/garbage transaction is ever presented while rst_n = '0'.
  pipeline_input_valid <= s_axis_tvalid and rst_n;
  pipeline_output_ready <= m_axis_tready or not rst_n;
  s_axis_tready <= pipeline_input_ready and rst_n;
  m_axis_tvalid <= pipeline_output_valid and rst_n;

  edge_byte <= "0000000" & compute_edge(window => s_axis_tdata, border => s_axis_tuser(1));

  pipeline_input_data <= s_axis_tuser(0) & edge_byte;

  ------------------------------------------------------------------------------
  handshake_pipeline_inst : entity common.handshake_pipeline
    generic map (
      data_width => c_pipeline_width
    )
    port map (
      clk => clk,
      --
      input_ready => pipeline_input_ready,
      input_valid => pipeline_input_valid,
      input_last  => s_axis_tlast,
      input_data  => pipeline_input_data,
      --
      output_ready => pipeline_output_ready,
      output_valid => pipeline_output_valid,
      output_last  => pipeline_output_last,
      output_data  => pipeline_output_data
    );

  m_axis_tdata    <= pipeline_output_data(7 downto 0);
  m_axis_tuser(0) <= pipeline_output_data(8);
  m_axis_tlast    <= pipeline_output_last;

end architecture rtl;
