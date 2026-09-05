library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library common;

-- Double-threshold classification of the suppressed magnitude into
-- none/weak/strong. See modules/canny/doc/canny_threshold_req.md
-- and modules/canny/doc/canny_threshold_proposal.md.
entity canny_threshold is
  generic (
    thresh_low  : natural;
    thresh_high : natural
  );
  port (
    clk : in std_logic;

    s_axis_tvalid : in  std_logic;
    s_axis_tready : out std_logic;
    s_axis_tdata  : in  std_logic_vector(10 downto 0);
    s_axis_tuser  : in  std_logic_vector(1 downto 0);
    s_axis_tlast  : in  std_logic;

    m_axis_tvalid : out std_logic;
    m_axis_tready : in  std_logic;
    m_axis_tdata  : out std_logic_vector(1 downto 0);
    m_axis_tuser  : out std_logic_vector(1 downto 0);
    m_axis_tlast  : out std_logic
  );
end entity canny_threshold;

architecture a of canny_threshold is

  -- Lane packed through the single handshake_pipeline instance:
  -- compare_result(1:0) & tuser(1:0) & tlast, MSB to LSB.
  constant lane_width : positive := 5;

  -- Pure combinational double-threshold classification. Border forces
  -- "00" (checked first, unconditional); otherwise "10"
  -- (strong, magnitude >= thresh_high), "01" (weak, thresh_low <=
  -- magnitude < thresh_high), or "00" (none).
  function classify(
    magnitude  : std_logic_vector(10 downto 0);
    border     : std_logic;
    thresh_low, thresh_high : natural
  ) return std_logic_vector is
    variable mag : natural;
  begin
    mag := to_integer(unsigned(magnitude));

    if border = '1' then
      return "00";
    end if;

    if mag >= thresh_high then
      return "10";
    elsif mag >= thresh_low then
      return "01";
    else
      return "00";
    end if;
  end function;

  signal compare_result : std_logic_vector(1 downto 0);
  signal input_data, output_data : std_logic_vector(lane_width - 1 downto 0);

begin

  ------------------------------------------------------------------------------
  assert thresh_low <= thresh_high
    report "canny_threshold: thresh_low (" & natural'image(thresh_low) &
      ") must be <= thresh_high (" & natural'image(thresh_high) & ")"
    severity failure;

  ------------------------------------------------------------------------------
  compare_result <= classify(s_axis_tdata, s_axis_tuser(1), thresh_low, thresh_high);

  input_data <= compare_result & s_axis_tuser & s_axis_tlast;

  handshake_pipeline_inst : entity common.handshake_pipeline
    generic map (
      data_width => lane_width
    )
    port map (
      clk => clk,
      --
      input_ready => s_axis_tready,
      input_valid => s_axis_tvalid,
      input_data => input_data,
      --
      output_ready => m_axis_tready,
      output_valid => m_axis_tvalid,
      output_data => output_data
    );

  m_axis_tdata <= output_data(4 downto 3);
  m_axis_tuser <= output_data(2 downto 1);
  m_axis_tlast <= output_data(0);

end architecture a;
