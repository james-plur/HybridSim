#include "hybridsim/engine/engine.hpp"
#include "hybridsim/network.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace hybridsim::network;
using namespace hybridsim::engine;

namespace {

constexpr double kEps = 1e-9;

void expect_near(double got, double want, const char *what) {
  if (std::abs(got - want) > 1e-6 &&
      std::abs(got - want) / std::max(1.0, std::abs(want)) > 1e-6) {
    std::cerr << "FAIL " << what << ": got " << got << " want " << want << "\n";
    throw std::runtime_error(what);
  }
}

simcpp20::process<> wait_recv(simcpp20::simulation<> &sim, NetworkAdapter *na,
                              int64_t conn, double *done_at) {
  co_await na->recv_event(conn);
  *done_at = sim.now();
}

} // namespace

void test_single_flow_one_layer() {
  simcpp20::simulation<> sim;
  NetworkBuildConfig cfg;
  cfg.layers = 1;
  cfg.link_bandwidth_bps = 1000.0;
  cfg.link_delay_s = 0.01;
  cfg.bw_policy = BwPolicyKind::MaxMin;

  auto net = Network::build(sim, cfg, {NetworkAddr{0, 0}, NetworkAddr{0, 1}});
  net->start();

  const double size = 100.0;
  double done = -1.0;
  wait_recv(sim, net->adapter(0, 1), 7, &done);
  net->adapter(0, 0)->inject(NetworkAddr{0, 1}, 7, 0, size);

  sim.run();
  net->rethrow_if_error();

  // Two links (adapter->switch, switch->adapter): 2 * delay + size/bw
  const double want = 2.0 * cfg.link_delay_s + size / cfg.link_bandwidth_bps;
  expect_near(done, want, "single_flow_one_layer");
  std::cout << "PASS: single_flow_one_layer now=" << sim.now() << "\n";
  net->stop();
}

void test_two_flows_share_bw() {
  simcpp20::simulation<> sim;
  NetworkBuildConfig cfg;
  cfg.layers = 1;
  cfg.link_bandwidth_bps = 1000.0;
  cfg.link_delay_s = 0.0;
  cfg.bw_policy = BwPolicyKind::MaxMin;

  auto net = Network::build(sim, cfg, {NetworkAddr{0, 0}, NetworkAddr{0, 1}});
  net->start();

  const double size = 100.0;
  double done_a = -1.0;
  double done_b = -1.0;
  wait_recv(sim, net->adapter(0, 1), 1, &done_a);
  wait_recv(sim, net->adapter(0, 1), 2, &done_b);
  net->adapter(0, 0)->inject(NetworkAddr{0, 1}, 1, 0, size);
  net->adapter(0, 0)->inject(NetworkAddr{0, 1}, 2, 0, size);

  sim.run();
  net->rethrow_if_error();

  const double want = 2.0 * size / cfg.link_bandwidth_bps;
  expect_near(done_a, want, "two_flows_a");
  expect_near(done_b, want, "two_flows_b");
  std::cout << "PASS: two_flows_share_bw now=" << sim.now() << "\n";
  net->stop();
}

simcpp20::process<> delayed_inject(simcpp20::simulation<> &sim,
                                   NetworkAdapter *src, NetworkAddr dst,
                                   int64_t conn, double size, double delay) {
  co_await sim.timeout(delay);
  src->inject(dst, conn, 0, size);
}

void test_stale_flow_end() {
  simcpp20::simulation<> sim;
  NetworkBuildConfig cfg;
  cfg.layers = 1;
  cfg.link_bandwidth_bps = 100.0;
  cfg.link_delay_s = 0.0;
  cfg.bw_policy = BwPolicyKind::MaxMin;

  auto net = Network::build(sim, cfg, {NetworkAddr{0, 0}, NetworkAddr{0, 1}});
  net->start();

  double done1 = -1.0;
  double done2 = -1.0;
  wait_recv(sim, net->adapter(0, 1), 1, &done1);
  wait_recv(sim, net->adapter(0, 1), 2, &done2);

  net->adapter(0, 0)->inject(NetworkAddr{0, 1}, 1, 0, 100.0);
  delayed_inject(sim, net->adapter(0, 0), NetworkAddr{0, 1}, 2, 50.0, 0.5);

  sim.run();
  net->rethrow_if_error();

  // Flow1: 0.5s at 100 B/s -> 50 remaining, then share 50 B/s -> +1.0s = 1.5s
  // Flow2 starts at 0.5, 50 bytes at 50 B/s -> ends at 1.5s
  expect_near(done1, 1.5, "stale_flow1");
  expect_near(done2, 1.5, "stale_flow2");
  std::cout << "PASS: stale_flow_end now=" << sim.now() << "\n";
  net->stop();
}

void test_two_layer_routing() {
  simcpp20::simulation<> sim;
  NetworkBuildConfig cfg;
  cfg.layers = 2;
  cfg.leaf_downlinks = 1;
  cfg.leaf_uplinks = 1;
  cfg.num_leaf = 2;
  cfg.num_spine = 1;
  cfg.link_bandwidth_bps = 1000.0;
  cfg.link_delay_s = 0.001;
  cfg.lb_policy = LbPolicyKind::EcmpHash;

  auto net = Network::build(sim, cfg, {NetworkAddr{0, 0}, NetworkAddr{1, 0}});
  net->start();
  assert(net->num_switches() == 3); // 2 leaf + 1 spine

  double done = -1.0;
  wait_recv(sim, net->adapter(1, 0), 9, &done);
  net->adapter(0, 0)->inject(NetworkAddr{1, 0}, 9, 0, 100.0);
  sim.run();
  net->rethrow_if_error();

  // adapter-leaf, leaf-spine, spine-leaf, leaf-adapter: 4 delays + size/bw
  const double want = 4.0 * cfg.link_delay_s + 100.0 / cfg.link_bandwidth_bps;
  expect_near(done, want, "two_layer_routing");
  std::cout << "PASS: two_layer_routing now=" << sim.now() << "\n";
  net->stop();
}

void test_ecmp_stable() {
  simcpp20::simulation<> sim;
  NetworkBuildConfig cfg;
  cfg.layers = 2;
  cfg.leaf_downlinks = 1;
  cfg.leaf_uplinks = 2;
  cfg.num_leaf = 2;
  cfg.num_spine = 2;
  cfg.link_bandwidth_bps = 8000.0;
  cfg.link_delay_s = 0.0;
  cfg.lb_policy = LbPolicyKind::EcmpHash;

  auto net = Network::build(sim, cfg, {NetworkAddr{0, 0}, NetworkAddr{0, 1}});
  net->start();

  double d1 = -1.0;
  double d2 = -1.0;
  wait_recv(sim, net->adapter(0, 1), 3, &d1);
  net->adapter(0, 0)->inject(NetworkAddr{0, 1}, 3, 0, 80.0);
  sim.run();
  net->rethrow_if_error();

  // Rebuild same topology, same flow_id prefix may differ by adapter index but
  // single flow still completes. Just check it finishes with size/bw (no delay).
  expect_near(d1, 80.0 / cfg.link_bandwidth_bps, "ecmp_first");

  // Second identical inject after drain: still completes.
  wait_recv(sim, net->adapter(0, 1), 3, &d2);
  net->adapter(0, 0)->inject(NetworkAddr{0, 1}, 3, 0, 80.0);
  sim.run();
  net->rethrow_if_error();
  expect_near(d2 - d1, 80.0 / cfg.link_bandwidth_bps, "ecmp_second");
  std::cout << "PASS: ecmp_stable\n";
  net->stop();
}

void test_loopback() {
  simcpp20::simulation<> sim;
  NetworkBuildConfig cfg;
  cfg.layers = 1;
  cfg.link_bandwidth_bps = 1000.0;
  cfg.link_delay_s = 0.05;

  auto net = Network::build(sim, cfg, {NetworkAddr{0, 0}});
  net->start();
  double done = -1.0;
  wait_recv(sim, net->adapter(0, 0), 1, &done);
  net->adapter(0, 0)->inject(NetworkAddr{0, 0}, 1, 0, 999.0);
  sim.run();
  net->rethrow_if_error();
  expect_near(done, 0.0, "loopback");
  std::cout << "PASS: loopback\n";
  net->stop();
}

void test_put_wait_two_engines() {
  simcpp20::simulation<> sim;
  NetworkBuildConfig cfg;
  cfg.layers = 1;
  cfg.link_bandwidth_bps = 2000.0;
  cfg.link_delay_s = 0.0;

  auto net = Network::build(sim, cfg, {NetworkAddr{0, 0}, NetworkAddr{0, 1}});
  net->start();

  engine_actor src(sim);
  engine_actor dst(sim);
  src.install_network(net, NetworkAddr{0, 0});
  dst.install_network(net, NetworkAddr{0, 1});

  int64_t src_done = -1;
  int64_t dst_done = -1;
  src.set_on_workload_complete(
      [&](const WorkloadDoneMsg &msg) { src_done = msg.workload_id; });
  dst.set_on_workload_complete(
      [&](const WorkloadDoneMsg &msg) { dst_done = msg.workload_id; });
  src.start();
  dst.start();

  kernel_spec put;
  put.name = "put";
  put.type = kKernelPut;
  put.params.set_string("dst_addr", "0:1");
  put.params.set_int("conn_id", 11);
  put.params.set_double("payload_bytes", 200.0);

  kernel_spec wait;
  wait.name = "wait";
  wait.type = kKernelWait;
  wait.params.set_int("conn_id", 11);

  src.send(WorkloadMsg{.spec = workload_spec{1, {put}}});
  dst.send(WorkloadMsg{.spec = workload_spec{2, {wait}}});
  sim.run();
  src.rethrow_if_error();
  dst.rethrow_if_error();
  net->rethrow_if_error();

  assert(src_done == 1);
  assert(dst_done == 2);
  expect_near(sim.now(), 200.0 / cfg.link_bandwidth_bps, "put_wait");
  std::cout << "PASS: put_wait_two_engines now=" << sim.now() << "\n";
  net->stop();
}

void test_get_auto_reply() {
  simcpp20::simulation<> sim;
  NetworkBuildConfig cfg;
  cfg.layers = 1;
  cfg.link_bandwidth_bps = 1000.0;
  cfg.link_delay_s = 0.0;

  auto net = Network::build(sim, cfg, {NetworkAddr{0, 0}, NetworkAddr{0, 1}});
  net->start();

  engine_actor client(sim);
  engine_actor server(sim);
  client.install_network(net, NetworkAddr{0, 0});
  server.install_network(net, NetworkAddr{0, 1});
  server.start();
  client.start();

  int64_t done = -1;
  client.set_on_workload_complete(
      [&](const WorkloadDoneMsg &msg) { done = msg.workload_id; });

  kernel_spec getk;
  getk.name = "get";
  getk.type = kKernelGet;
  getk.params.set_string("dst_addr", "0:1");
  getk.params.set_int("conn_id", 5);
  getk.params.set_double("payload_bytes", 100.0);

  client.send(WorkloadMsg{.spec = workload_spec{3, {getk}}});
  sim.run();
  client.rethrow_if_error();
  net->rethrow_if_error();

  assert(done == 3);
  // Fetch 64B then reverse 100B, each size/bw, no delay, pipelined hops.
  // Control flow 64/1000 then data 100/1000 sequential (fetch must complete
  // before reply injects).
  const double want = kDefaultSignalBytes / cfg.link_bandwidth_bps +
                      100.0 / cfg.link_bandwidth_bps;
  expect_near(sim.now(), want, "get_auto_reply");
  std::cout << "PASS: get_auto_reply now=" << sim.now() << "\n";
  net->stop();
}

void test_manual_wire() {
  simcpp20::simulation<> sim;
  auto net = Network::create(sim, BwPolicyKind::MaxMin, LbPolicyKind::EcmpHash, 0);
  const int a0 = net->add_adapter(NetworkAddr{0, 0}, 2);
  const int a1 = net->add_adapter(NetworkAddr{0, 1}, 2);
  const int sw = net->add_switch(2);
  net->link(a0, 1, sw, 0, 1000.0, 0.0);
  net->link(a1, 1, sw, 1, 1000.0, 0.0);
  net->install_shortest_path_routes();
  net->start();

  double done = -1.0;
  wait_recv(sim, net->adapter(0, 1), 4, &done);
  net->adapter(0, 0)->inject(NetworkAddr{0, 1}, 4, 0, 50.0);
  sim.run();
  net->rethrow_if_error();
  expect_near(done, 50.0 / 1000.0, "manual_wire");
  std::cout << "PASS: manual_wire now=" << sim.now() << "\n";
  net->stop();
}

int main() {
  test_single_flow_one_layer();
  test_two_flows_share_bw();
  test_stale_flow_end();
  test_two_layer_routing();
  test_ecmp_stable();
  test_loopback();
  test_put_wait_two_engines();
  test_get_auto_reply();
  test_manual_wire();
  std::cout << "All network tests passed.\n";
  return 0;
}
