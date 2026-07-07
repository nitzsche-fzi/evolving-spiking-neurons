//////////////////////////////////////////////////////////////////////////////////
//
// Company: FZI Forschungszentrum Informatik
// Engineer: Sven Nitzsche
//
// Create Date: 25/05/2025
//
// Module Name: tb_neuron
//
// Description: Test neuron while ideling, receiving a single spike and generating a spike.
//              Used to extract latency and switching activity in these scenarios.
//              We use Q notation in the TI version to represent fixed point numbers.
//
//////////////////////////////////////////////////////////////////////////////////

`include "assertions.vh"
`include "neuron_config.vh"

`timescale 1ns / 1ps


module tb_neuron ();
  // -- GENERATED PARAMETERS, AUTOMATICALLY ADJUSTED BY PREPARATION SCRIPT --
  localparam STATE_VARIABLES  = `STATE_VARIABLES;

  // -------------------- DEFINE LOCAL PARAMS -------------------------------
  localparam STATE_WIDTH      = 32;
  localparam WEIGHT_SUM_WIDTH = 32;

  // -------------------- DYNAMIC SIGNAL DECLARATIONS -----------------------
  reg  [STATE_WIDTH-1 : 0] uut_state_i_r      [0:STATE_VARIABLES-1] = '{default:0};
  wire [STATE_WIDTH-1 : 0] uut_state_o        [0:STATE_VARIABLES-1];
  wire                     uut_state_valid_o  [0:STATE_VARIABLES-1];
  reg  [STATE_WIDTH-1 : 0] uut_state_next     [0:STATE_VARIABLES-1] = '{default:0};
  reg                      uut_state_next_vld [0:STATE_VARIABLES-1] = '{default:0};
  reg  [STATE_WIDTH-1 : 0] last_uut_state_i_r [0:STATE_VARIABLES-1] = '{default:0};
  // -------------------- INSTANTIATE THE UUT -------------------------------
  // Define signals to connect to UUT
  reg clk_tb = 0;
  reg reset_tb = 0;
  // AP Control from HLS
  reg  uut_ap_start = 0;
  wire uut_ap_done;
  wire uut_ap_idle;
  wire uut_ap_ready;

  reg [WEIGHT_SUM_WIDTH-1 : 0] uut_weight_sum_i_r = 0;
  reg [WEIGHT_SUM_WIDTH-1 : 0] last_uut_weight_sum_i_r = 0;

  wire uut_spike_out_o;

  // python3: rand_input = torch.randn((1, 256)) + 1.0 # mean 1 and stddev 1
  shortreal rand_input [0:255] = '{
    -6.8705e-01, -3.4127e-02, -4.1575e-01,  3.0120e-01,  1.7142e+00,
    1.2849e+00,  8.2532e-01,  9.3981e-01,  1.0181e+00,  1.4487e+00,
    1.3512e+00,  1.0819e+00, -8.1702e-01,  1.8178e+00,  2.2639e-01,
    1.2570e+00,  1.9460e+00,  3.2065e-01,  1.8143e+00,  1.8616e+00,
    1.9758e+00,  1.7195e+00,  1.1950e+00,  6.9752e-01,  3.3760e-01,
    9.0142e-02, -1.9020e-01,  1.3441e+00,  1.7448e+00, -4.9804e-02,
    1.7878e+00,  9.2494e-01,  1.7802e+00,  3.8853e+00, -8.1296e-01,
    1.0796e+00,  8.5759e-01,  2.9362e-01,  1.1330e+00,  2.7380e+00,
    1.6484e+00,  1.5119e+00, -5.0425e-01,  1.4807e+00,  1.1582e+00,
    4.4458e-01, -2.1649e-01,  4.5839e-01,  1.3065e+00, -1.5209e+00,
    4.9189e-01,  1.5601e+00,  9.9226e-01,  2.5112e+00, -3.5607e-01,
    1.3383e+00,  1.8674e-01,  1.3046e+00,  2.0371e+00,  9.7697e-01,
    2.7165e-01,  1.1441e+00,  2.2520e+00,  2.0285e-01,  2.4918e+00,
    4.8972e-01,  1.3683e+00,  1.5618e+00,  4.3529e-01,  1.8931e+00,
    2.1791e+00,  2.0387e+00,  1.2528e+00,  3.3396e-01,  1.7792e+00,
    9.0620e-01,  1.9626e+00,  1.2281e+00,  1.6840e+00, -8.5291e-02,
    -1.1006e-01,  7.8095e-01,  1.8078e+00,  2.6352e+00,  8.7393e-01,
    4.2110e-01,  1.3351e+00,  1.0344e+00,  6.7092e-01, -1.8238e-01,
    1.3432e+00,  7.9750e-02,  1.1750e+00, -3.9013e-01,  1.1669e+00,
    7.7188e-01,  1.7743e-01, -3.6671e-01,  1.3285e+00,  1.7805e+00,
    5.1012e-02, -1.0868e+00,  1.1640e+00,  5.2116e-01,  5.1298e-01,
    -3.2958e-01,  1.4328e+00, -4.4337e-01,  1.0182e+00,  1.8770e+00,
    2.7364e-01,  2.1800e+00,  1.5571e+00, -9.2384e-01,  1.2468e+00,
    -2.5218e-01,  1.9438e+00,  6.1512e-02,  1.2257e-01,  9.5897e-01,
    2.2514e+00, -7.1907e-02,  9.8287e-01,  2.9129e-01,  2.0305e+00,
    1.5968e+00, -1.4681e+00,  5.7994e-01,  2.4826e+00,  1.3902e+00,
    1.5266e+00,  5.7866e-01, -8.4928e-01,  1.9381e+00,  1.6338e+00,
    2.3251e+00,  1.7177e+00,  1.7376e+00, -3.3929e-01,  8.2953e-01,
    1.3465e+00,  2.3875e-01,  1.7811e-01, -4.8913e-01,  5.6072e-01,
    4.3699e-01, -5.0663e-01,  3.2016e+00,  7.8843e-01,  1.2716e+00,
    2.0280e+00,  9.1101e-01,  8.8413e-01,  2.4482e+00, -7.1468e-01,
    1.9763e+00,  9.0949e-01,  2.2886e+00, -1.5106e+00, -3.9274e-01,
    2.2008e+00,  2.2222e+00,  1.1034e+00,  3.0519e+00,  1.9825e+00,
    9.1616e-01,  4.9042e-01,  5.9151e-01,  7.2288e-01,  6.0971e-01,
    2.1013e+00, -6.2170e-01,  7.4955e-01,  1.7292e+00,  7.4807e-01,
    1.8938e+00,  1.0511e+00,  7.5379e-01,  1.5658e+00,  6.8723e-01,
    1.2718e+00, -7.2535e-02,  5.6748e-01,  1.9686e+00, -3.8127e-01,
    3.2450e+00,  1.6680e+00, -3.1262e-01,  2.0420e+00,  1.4451e-01,
    -7.6387e-01,  1.9089e+00, -7.9394e-01,  2.6602e+00,  1.0481e+00,
    -2.0142e-01,  2.3560e+00,  5.7802e-01,  5.7222e-01,  1.6640e+00,
    1.8415e+00,  8.9842e-01, -8.2254e-02,  1.6878e+00,  1.3338e+00,
    5.1231e-01,  4.3876e-01,  4.1877e-01,  1.5787e+00,  1.4180e+00,
    1.0814e+00, -3.9593e-01,  2.9205e+00,  5.8977e-01,  1.2460e+00,
    2.1260e-01,  5.5086e-01,  1.5894e+00,  1.9629e+00,  1.3888e+00,
    6.8943e-01, -2.8596e-01,  2.1265e-03,  6.8400e-01,  1.5494e+00,
    1.6138e+00,  1.4378e+00,  1.7710e+00,  2.2800e+00,  1.3460e+00,
    9.5691e-01,  5.0842e-01,  1.8323e+00, -1.1872e-01,  1.0587e+00,
    1.6117e+00,  9.9350e-01,  9.3215e-01,  8.4505e-01,  1.0395e+00,
    1.4702e+00, -8.9765e-01,  1.3715e-01,  2.7792e+00,  2.0460e+00,
    -5.0677e-01,  2.6988e+00,  1.5869e+00, -3.5511e-01,  1.7345e+00,
    1.8788e+00,  6.7647e-01, -7.7066e-01, -7.8844e-01, -5.1592e-01,
    6.8159e-01
  };


  // Instantiate UUT
  design_1_wrapper #() uut (
      .clk_100MHz(clk_tb),
      .reset_rtl(reset_tb),

      .ap_ctrl_0_done(uut_ap_done),
      .ap_ctrl_0_idle(uut_ap_idle),
      .ap_ctrl_0_ready(uut_ap_ready),
      .ap_ctrl_0_start(uut_ap_start),

      `ALL_S_I_PORTS,  // Expands to .s0_i_0(uut_state_i[0]), .s1_i_0(...), ...
      `INPUT_PORT,

      `ALL_S_O_PORTS,  // Expands to .s0_o_0(uut_state_o[0]), .s0_o_ap_vld_0(uut_state_valid_o[0]), ...
      .ap_return_0(uut_spike_out_o)
  );

  // Capture updated states and feed them back into the state array once the IP is done reading the current input states.
  generate
    genvar i;
    for (i = 0; i < STATE_VARIABLES; i = i + 1) begin : state_update
      always @(posedge clk_tb) begin
        if (uut_state_valid_o[i]) begin
          if (uut_ap_ready) begin
            uut_state_i_r[i]      <= uut_state_o[i];
          end else begin
            uut_state_next[i]     <= uut_state_o[i];
            uut_state_next_vld[i] <= 1;
          end
        end else if (uut_ap_ready && uut_state_next_vld[i]) begin
          uut_state_i_r[i]      <= uut_state_next[i];
          uut_state_next_vld[i] <= 0;
        end
      end
    end
  endgenerate





  // -------------------- DEFINE CLOCK -------------------------------

  localparam CLOCK_PERIOD = 10;  // in ns
  always #(CLOCK_PERIOD/2) clk_tb <= ~clk_tb;  // 100MHz clock


  // -------------------- DEFINE REGISTERS -------------------------------

  // To be able to jump to each case in the wave file
  integer test_case = 0;

  // -------------------- RUN TESTBENCH -------------------------------------------

  initial begin
    #(CLOCK_PERIOD) $display("Testbench started");

    // Call tests
    test_1_check_idle_logic();
    test_2_check_single_calculation();
    test_3_check_spike_and_reset(); 
    test_4_random_nospike(); 
    test_5_random_spike(); 
    test_1_check_idle_logic(); // dummy test to workaround some weird vivado bug that causes wrong switching activity for the last test

    #(CLOCK_PERIOD) $display("\nTestbench completed");
    $finish;
  end


  // -------------------- UTIL TASKS ----------------------------------------------

  task wait_n(integer n);
    begin
      repeat (n) @(posedge clk_tb);
    end
  endtask

  task reset_uut();
    begin
      wait_n(1);
      uut_ap_start        <= 0;
      uut_weight_sum_i_r  <= 0;
      for (int i = 0; i < STATE_VARIABLES; i++) begin
        uut_state_i_r[i] <= $shortrealtobits(INIT_VALUES[i]);
      end

      reset_tb <= 1;
      wait_n(5);
      reset_tb <= 0;
      wait_n(1);
    end
  endtask

  task initialize_ip();
    begin
      wait (uut_ap_idle == 1'b1);
      uut_ap_start <= 1;
      wait (uut_ap_done);
      uut_ap_start <= 0;
    end
  endtask

  task start_test(reg [4:0] test_num);
    string s;
    begin
      test_case <= test_num;

      reset_uut();
      initialize_ip();

      s.itoa(test_num);
      $display("\n\n--- Test Case %s ---\n", s);
      @(posedge clk_tb);
      $stop;
    end
  endtask

  task start_test_from_last_state(reg [4:0] test_num);
    string s;
    begin
      test_case <= test_num;

      wait_n(1);
      uut_ap_start        <= 0;
      uut_weight_sum_i_r  <= last_uut_weight_sum_i_r;
      for (int i = 0; i < STATE_VARIABLES; i++) begin
        uut_state_i_r[i] <= last_uut_state_i_r[i];
      end
      reset_tb <= 1;
      wait_n(5);
      reset_tb <= 0;
      wait_n(1);

      initialize_ip();

      s.itoa(test_num);
      $display("\n\n--- Test Case %s ---\n", s);
      @(posedge clk_tb);
      $stop;
    end
  endtask

  task run_ip();
    integer i;
    integer cycle_count;
    begin
      uut_ap_start = 1;
      cycle_count = 0;

      // Wait for IP to accept new inputs (ap_ready)
      // @(posedge clk_tb iff uut_ap_ready);
      @(posedge clk_tb);
      while (!uut_ap_ready) begin
        cycle_count++;
         @(posedge clk_tb);
      end

      // Disable new inputs and wait for completion. Usually uut_ap_ready == uut_ap_done so its instantaneous
      uut_ap_start = 0;
      // wait (uut_ap_done);

      while (!uut_ap_done) begin
        cycle_count++;
        @(posedge clk_tb);
      end
      $display("run_ip_uut_latency %0d", cycle_count);
      $display("Debug: spike out = %0d", uut_spike_out_o);

    end
  endtask

  task run_ip_til_ready();
    integer i;
    begin
      uut_ap_start = 1;
      // Wait for IP to accept new inputs (ap_ready)
      @(posedge clk_tb iff uut_ap_ready);
    end
  endtask

  task run_ip_multiple(integer num_iters);
    integer i;
    begin
      for (i = 0; i < num_iters; i = i + 1) begin
        uut_ap_start = 1'b1;

        // Wait for IP to accept new inputs (ap_ready)
        @(posedge clk_tb iff uut_ap_ready);
      end
      // This was the last iteration, disable new inputs and wait for completion
      uut_ap_start = 1'b0;
      wait (uut_ap_done == 1'b1);
    end
  endtask


  // -------------------- TEST CASES ----------------------------------------------

  // TEST 1
  // IP is idle.

  task test_1_check_idle_logic();
    begin
      start_test(1);

      wait_n(1);
      `assert_eq_b(uut_ap_idle,  1'b1);
      `assert_eq_b(uut_ap_ready, 1'b0);
      `assert_eq_b(uut_ap_done,  1'b0);

      wait_n(3);
      `assert_eq_b(uut_ap_idle,  1'b1);
      `assert_eq_b(uut_ap_ready, 1'b0);
      `assert_eq_b(uut_ap_done,  1'b0);

      $stop;
    end
  endtask


  // TEST 2
  // IP processes a single spike.

  task test_2_check_single_calculation();
    begin
      start_test(2);
  
      uut_weight_sum_i_r <= $shortrealtobits(1.0);
      run_ip();

      $stop;
    end
  endtask


  // TEST 3
  // Process inputs until we receive a spike out.

  task test_3_check_spike_and_reset();
    begin
      start_test(3);

      uut_weight_sum_i_r <= $shortrealtobits(30.0);
      run_ip();
      run_ip();
      run_ip();

      $stop;
    end
  endtask

  // runs ip until uut_spike_out_o = target_spikevalue and saves state
  task run_until_spikevalue(reg target_spikevalue);
    begin
      reset_uut();
      initialize_ip();
      @(posedge clk_tb);

      // Loop over inputs until uut_spike_out_o = target_spikevalue
      for (int rand_idx = 0; rand_idx < $size(rand_input); rand_idx++) begin
        uut_weight_sum_i_r = $shortrealtobits(rand_input[rand_idx]);
        
        // save state
        last_uut_weight_sum_i_r = uut_weight_sum_i_r;
        for (int i = 0; i < STATE_VARIABLES; i = i + 1) begin : state_update
            last_uut_state_i_r[i] = uut_state_i_r[i];
        end

        run_ip();

        if (uut_spike_out_o == target_spikevalue) begin
          if (uut_spike_out_o == 1)
            $display("Debug: Neuron spiked at rand_idx = %0d", rand_idx);
          else
            $display("Debug: Neuron didn't spike at rand_idx = %0d", rand_idx);
          return;
        end
      end

      $display("Warning: Neuron didnt match target_spikevalue");
      // TODO: abort test since neuron doesn't behave as expected?
    end
  endtask

  task test_4_random_nospike();
    begin
      run_until_spikevalue(0);
      start_test_from_last_state(4);

      run_ip(); // test
     
      $stop;
    end
  endtask

  task test_5_random_spike();
    begin
      run_until_spikevalue(1);
      start_test_from_last_state(5);

      run_ip(); // test
     
      $stop;
    end
  endtask




endmodule
