library ieee;
use ieee.std_logic_1164.all;

library common;

-- Generic 2-input AXI4-Stream rendezvous. See
-- modules/axi_stream_join/doc/axi_stream_join_req.md and
-- modules/axi_stream_join/doc/axi_stream_join_proposal.md.
entity axi_stream_join is
  generic (
    data_width_a : positive;
    data_width_b : positive
  );
  port (
    clk : in std_logic;

    s_axis_a_tvalid : in  std_logic;
    s_axis_a_tready : out std_logic;
    s_axis_a_tdata  : in  std_logic_vector(data_width_a - 1 downto 0);
    s_axis_a_tuser  : in  std_logic_vector(1 downto 0);
    s_axis_a_tlast  : in  std_logic;

    s_axis_b_tvalid : in  std_logic;
    s_axis_b_tready : out std_logic;
    s_axis_b_tdata  : in  std_logic_vector(data_width_b - 1 downto 0);
    s_axis_b_tuser  : in  std_logic_vector(1 downto 0);
    s_axis_b_tlast  : in  std_logic;

    m_axis_tvalid : out std_logic;
    m_axis_tready : in  std_logic;
    m_axis_tdata  : out std_logic_vector(data_width_a + data_width_b - 1 downto 0);
    m_axis_tuser  : out std_logic_vector(1 downto 0);
    m_axis_tlast  : out std_logic
  );
end entity axi_stream_join;

architecture a of axi_stream_join is

  signal input_ready : std_logic_vector(0 to 1);
  signal input_valid  : std_logic_vector(0 to 1);

begin

  input_valid <= s_axis_a_tvalid & s_axis_b_tvalid;
  s_axis_a_tready <= input_ready(0);
  s_axis_b_tready <= input_ready(1);

  handshake_merger_inst : entity common.handshake_merger
    generic map (
      num_interfaces => 2
    )
    port map (
      clk => clk,
      --
      input_ready => input_ready,
      input_valid => input_valid,
      --
      result_ready => m_axis_tready,
      result_valid => m_axis_tvalid
    );

  m_axis_tdata <= s_axis_a_tdata & s_axis_b_tdata;
  m_axis_tuser(1) <= s_axis_a_tuser(1) or s_axis_b_tuser(1);
  m_axis_tuser(0) <= s_axis_a_tuser(0);
  m_axis_tlast <= s_axis_a_tlast;

  -- Simulation-only: on every cycle where the join fires, the two lanes
  -- must agree on frame boundaries (sof via tuser(0), and tlast). This is
  -- deliberately not handshake_merger's own last-mismatch assertion (its
  -- input_last ports are left unconnected/defaulted above), since that
  -- would misreport this class of bug as a generic packet-length mismatch.
  fork_synchronization_check : process
  begin
    wait until rising_edge(clk);

    if s_axis_a_tvalid and s_axis_b_tvalid and m_axis_tready then
      assert s_axis_a_tuser(0) = s_axis_b_tuser(0) and s_axis_a_tlast = s_axis_b_tlast
        report "axi_stream_join: fork desynchronization -- s_axis_a and " &
          "s_axis_b tuser(0)/tlast disagree on a jointly-consumed beat"
        severity error;
    end if;
  end process;

end architecture a;
