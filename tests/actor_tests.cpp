#include "hybridsim/hybridsim.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace hybridsim;

struct IncrementMsg {
  int delta = 1;
};

struct SetMsg {
  int value = 0;
};

struct UnknownMsg {};

struct PingMsg {};

struct PongMsg {
  int count = 0;
};

void expect_exception(const std::function<void()> &fn,
                      const std::string &expected_substr) {
  bool caught = false;
  try {
    fn();
  } catch (const std::runtime_error &ex) {
    caught = true;
    assert(std::string(ex.what()).find(expected_substr) != std::string::npos);
  }
  assert(caught);
}

void test_sync_handler() {
  simcpp20::simulation<> sim;
  int counter = 0;

  actor a(sim);
  a.on<IncrementMsg>([&counter](actor &, IncrementMsg &msg) {
    counter += msg.delta;
  });
  a.start();
  a.send(IncrementMsg{2});
  a.send(IncrementMsg{3});
  sim.run();

  assert(counter == 5);
  std::cout << "PASS: sync_handler\n";
}

simcpp20::process<> delayed_increment(simcpp20::simulation<> &sim, actor &,
                                      IncrementMsg &msg, int &counter,
                                      double &handled_at) {
  co_await sim.timeout(2);
  counter += msg.delta;
  handled_at = sim.now();
}

void test_async_handler() {
  simcpp20::simulation<> sim;
  int counter = 0;
  double handled_at = -1;

  actor a(sim);
  a.on<IncrementMsg>([&](simcpp20::simulation<> &sim, actor &self,
                         IncrementMsg &msg) -> simcpp20::process<> {
    return delayed_increment(sim, self, msg, counter, handled_at);
  });
  a.start();
  a.send(IncrementMsg{7});
  sim.run();

  assert(counter == 7);
  assert(handled_at == 2);
  std::cout << "PASS: async_handler\n";
}

void test_multi_type_dispatch() {
  simcpp20::simulation<> sim;
  int counter = 0;

  actor a(sim);
  a.on<IncrementMsg>([&counter](actor &, IncrementMsg &msg) {
    counter += msg.delta;
  });
  a.on<SetMsg>([&counter](actor &, SetMsg &msg) { counter = msg.value; });
  a.start();
  a.send(SetMsg{10});
  a.send(IncrementMsg{4});
  sim.run();

  assert(counter == 14);
  std::cout << "PASS: multi_type_dispatch\n";
}

void test_unknown_message_throws() {
  simcpp20::simulation<> sim;

  actor a(sim);
  a.on<IncrementMsg>([](actor &, IncrementMsg &) {});
  a.start();
  a.send(UnknownMsg{});
  sim.run();

  assert(a.has_error());
  expect_exception([&a]() { a.rethrow_if_error(); }, "unhandled message type");
  std::cout << "PASS: unknown_message_throws\n";
}

void test_mailbox_ordering() {
  simcpp20::simulation<> sim;
  std::vector<int> order;

  actor a(sim);
  a.on<SetMsg>([&order](actor &, SetMsg &msg) { order.push_back(msg.value); });
  a.start();
  a.send(SetMsg{1});
  a.send(SetMsg{2});
  a.send(SetMsg{3});
  sim.run();

  assert(order == std::vector<int>({1, 2, 3}));
  std::cout << "PASS: mailbox_ordering\n";
}

void test_ping_pong() {
  simcpp20::simulation<> sim;
  int pong_count = 0;

  actor ping(sim);
  actor pong(sim);

  ping.on<PingMsg>([&pong](actor &, PingMsg &) { pong.send(PongMsg{1}); });
  pong.on<PongMsg>([&pong_count](actor &, PongMsg &msg) {
    pong_count = msg.count;
  });

  ping.start();
  pong.start();
  ping.send(PingMsg{});
  sim.run();

  assert(pong_count == 1);
  std::cout << "PASS: ping_pong\n";
}

int main() {
  test_sync_handler();
  test_async_handler();
  test_multi_type_dispatch();
  test_unknown_message_throws();
  test_mailbox_ordering();
  test_ping_pong();
  std::cout << "All actor tests passed.\n";
  return 0;
}
