#include "hybridsim/engine/engine.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace hybridsim;
using namespace hybridsim::engine;

struct kernel_timing {
  std::string name;
  double start = -1.0;
  double end = -1.0;
};

class TimingKernel : public Kernel {
public:
  TimingKernel(kernel_spec spec, std::vector<kernel_timing> &log)
      : spec_{std::move(spec)}, log_{log} {}

  simcpp20::process<> run(simcpp20::simulation<> &sim) override {
    const auto idx = log_.size();
    log_.push_back({spec_.name, sim.now(), -1.0});
    co_await sim.timeout(spec_.duration);
    log_[idx].end = sim.now();
  }

private:
  kernel_spec spec_;
  std::vector<kernel_timing> &log_;
};

KernelFactory make_timing_factory(std::vector<kernel_timing> &log) {
  KernelFactory factory;
  factory.register_creator(0, [&log](const kernel_spec &spec) {
    return std::make_unique<TimingKernel>(spec, log);
  });
  return factory;
}

simcpp20::process<> run_workload(simcpp20::simulation<> &sim,
                                 const workload_spec &spec,
                                 KernelFactory &factory) {
  co_await schedule_dag(sim, spec, factory);
}

void test_linear_chain() {
  simcpp20::simulation<> sim;
  std::vector<kernel_timing> timings;

  workload_spec spec{
      .workload_id = 1,
      .kernels = {{"A", 0, 1.0, {}}, {"B", 0, 2.0, {0}}, {"C", 0, 3.0, {1}}},
  };

  auto factory = make_timing_factory(timings);
  run_workload(sim, spec, factory);
  sim.run();

  assert(timings.size() == 3);
  assert(timings[0].start == 0.0 && timings[0].end == 1.0);
  assert(timings[1].start == 1.0 && timings[1].end == 3.0);
  assert(timings[2].start == 3.0 && timings[2].end == 6.0);
  assert(sim.now() == 6.0);
  std::cout << "PASS: linear_chain\n";
}

void test_diamond_parallel() {
  simcpp20::simulation<> sim;
  std::vector<kernel_timing> timings;

  workload_spec spec{
      .workload_id = 2,
      .kernels = {{"A", 0, 1.0, {}},
                  {"B", 0, 2.0, {0}},
                  {"C", 0, 3.0, {0}},
                  {"D", 0, 4.0, {1, 2}}},
  };

  auto factory = make_timing_factory(timings);
  run_workload(sim, spec, factory);
  sim.run();

  const kernel_timing *a = nullptr;
  const kernel_timing *b = nullptr;
  const kernel_timing *c = nullptr;
  const kernel_timing *d = nullptr;
  for (const auto &t : timings) {
    if (t.name == "A") {
      a = &t;
    } else if (t.name == "B") {
      b = &t;
    } else if (t.name == "C") {
      c = &t;
    } else if (t.name == "D") {
      d = &t;
    }
  }

  assert(a != nullptr && b != nullptr && c != nullptr && d != nullptr);
  assert(a->start == 0.0 && a->end == 1.0);
  assert(b->start == 1.0 && b->end == 3.0);
  assert(c->start == 1.0 && c->end == 4.0);
  assert(d->start == 4.0 && d->end == 8.0);
  assert(sim.now() == 8.0);
  std::cout << "PASS: diamond_parallel\n";
}

void test_workload_complete_handler() {
  simcpp20::simulation<> sim;
  engine_actor engine(sim);

  int64_t done_id = -1;
  engine.set_on_workload_complete(
      [&done_id](const WorkloadDoneMsg &msg) { done_id = msg.workload_id; });

  engine.start();

  workload_spec spec{
      .workload_id = 42,
      .kernels = {{"A", 0, 1.0, {}}, {"B", 0, 2.0, {0}}},
  };

  engine.send(WorkloadMsg{.spec = spec});
  sim.run();

  assert(done_id == 42);
  std::cout << "PASS: workload_complete_handler\n";
}

simcpp20::process<> delayed_workload(simcpp20::simulation<> &sim,
                                     engine_actor &engine, workload_spec spec,
                                     double delay) {
  co_await sim.timeout(delay);
  engine.send(WorkloadMsg{.spec = std::move(spec)});
}

void test_workload_done_priority() {
  simcpp20::simulation<> sim;
  engine_actor engine(sim);
  std::vector<std::pair<double, int64_t>> dones;

  engine.set_on_workload_complete([&](const WorkloadDoneMsg &msg) {
    dones.emplace_back(sim.now(), msg.workload_id);
  });
  engine.start();

  workload_spec first{
      .workload_id = 1,
      .kernels = {{"A", 0, 1.0, {}}},
  };
  workload_spec second{
      .workload_id = 2,
      .kernels = {{"B", 0, 1.0, {}}},
  };

  engine.send(WorkloadMsg{.spec = first});
  delayed_workload(sim, engine, second, 0.5);
  sim.run();

  assert(dones.size() == 2);
  assert(dones[0] == std::make_pair(1.0, int64_t{1}));
  assert(dones[1] == std::make_pair(2.0, int64_t{2}));
  std::cout << "PASS: workload_done_priority\n";
}

void test_invalid_cycle() {
  workload_spec spec{
      .workload_id = 3,
      .kernels = {{"A", 0, 1.0, {1}}, {"B", 0, 1.0, {0}}},
  };

  bool caught = false;
  try {
    validate_dag(spec);
  } catch (const std::invalid_argument &) {
    caught = true;
  }
  assert(caught);
  std::cout << "PASS: invalid_cycle\n";
}

void test_kernel_params() {
  kernel_spec spec{
      .name = "gemm",
      .type = 0,
      .duration = 1.0,
  };
  spec.params.set_int("tile_size", 32);
  spec.params.set_string("dtype", "fp16");

  assert(spec.params.get_int("tile_size") == 32);
  assert(spec.params.get_string("dtype") == "fp16");
  assert(!spec.params.get_bool("tile_size").has_value());
  std::cout << "PASS: kernel_params\n";
}

int main() {
  test_linear_chain();
  test_diamond_parallel();
  test_workload_complete_handler();
  test_workload_done_priority();
  test_invalid_cycle();
  test_kernel_params();
  std::cout << "All engine actor tests passed.\n";
  return 0;
}
