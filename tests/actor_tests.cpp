#include "hybridsim/hybridsim.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
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

struct QueryMsg {
  int id = 0;
};

struct QueryResult {
  int id = 0;
  int value = 0;
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

void test_send_at() {
  simcpp20::simulation<> sim;
  std::vector<std::pair<double, int>> events;

  actor a(sim);
  a.on<IncrementMsg>([&](actor &, IncrementMsg &msg) {
    events.emplace_back(a.sim().now(), msg.delta);
  });
  a.start();
  a.send_at(5.0, IncrementMsg{2});
  a.send_at(2.0, IncrementMsg{1});
  a.send_at(0.0, IncrementMsg{0});
  sim.run();

  assert(events.size() == 3);
  assert(events[0] == std::make_pair(0.0, 0));
  assert(events[1] == std::make_pair(2.0, 1));
  assert(events[2] == std::make_pair(5.0, 2));
  assert(sim.now() == 5.0);
  std::cout << "PASS: send_at\n";
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

void test_request_explicit_reply() {
  simcpp20::simulation<> sim;
  QueryResult got{-1, -1};

  actor server(sim);
  actor client(sim);

  server.on<QueryMsg>([](actor &self, QueryMsg &msg) {
    self.reply(*self.current_request(), QueryResult{msg.id, msg.id * 10});
  });

  client.on<PingMsg>([&](simcpp20::simulation<> &, actor &,
                         PingMsg &) -> simcpp20::process<> {
    auto ev = server.request<QueryResult>(QueryMsg{7});
    got = co_await ev;
  });

  server.start();
  client.start();
  client.send(PingMsg{});
  sim.run();

  assert(got.id == 7);
  assert(got.value == 70);
  std::cout << "PASS: request_explicit_reply\n";
}

void test_request_auto_empty_reply() {
  simcpp20::simulation<> sim;
  bool resumed = false;

  actor server(sim);
  actor client(sim);

  server.on<QueryMsg>([](actor &, QueryMsg &) {
    // no reply → auto empty
  });

  client.on<PingMsg>([&](simcpp20::simulation<> &, actor &,
                         PingMsg &) -> simcpp20::process<> {
    co_await server.request(QueryMsg{1});
    resumed = true;
  });

  server.start();
  client.start();
  client.send(PingMsg{});
  sim.run();

  assert(resumed);
  std::cout << "PASS: request_auto_empty_reply\n";
}

void test_request_at_and_timeout_reply() {
  simcpp20::simulation<> sim;
  double replied_at = -1;
  QueryResult got{-1, -1};

  actor server(sim);
  actor client(sim);

  server.on<QueryMsg>([](simcpp20::simulation<> &sim, actor &self,
                         QueryMsg &msg) -> simcpp20::process<> {
    co_await sim.timeout(2.0);
    self.reply(*self.current_request(), QueryResult{msg.id, 1});
  });

  client.on<PingMsg>([&](simcpp20::simulation<> &sim, actor &,
                         PingMsg &) -> simcpp20::process<> {
    auto ev = server.request_at<QueryResult>(1.0, QueryMsg{3});
    got = co_await ev;
    replied_at = sim.now();
  });

  server.start();
  client.start();
  client.send(PingMsg{});
  sim.run();

  assert(got.id == 3);
  assert(replied_at == 3.0);
  std::cout << "PASS: request_at_and_timeout_reply\n";
}

void test_send_and_request_mixed() {
  simcpp20::simulation<> sim;
  int sends = 0;
  int replies = 0;

  actor server(sim);
  actor client(sim);

  server.on<SetMsg>([&sends](actor &, SetMsg &) { ++sends; });
  server.on<QueryMsg>([&replies](actor &self, QueryMsg &) {
    ++replies;
    self.reply(*self.current_request());
  });

  client.on<PingMsg>([&](simcpp20::simulation<> &, actor &,
                         PingMsg &) -> simcpp20::process<> {
    server.send(SetMsg{1});
    co_await server.request(QueryMsg{1});
    server.send(SetMsg{2});
  });

  server.start();
  client.start();
  client.send(PingMsg{});
  sim.run();

  assert(sends == 2);
  assert(replies == 1);
  std::cout << "PASS: send_and_request_mixed\n";
}

void test_send_and_request_delay() {
  simcpp20::simulation<> sim;
  std::vector<std::pair<double, std::string>> events;

  actor server(sim);
  actor client(sim);

  server.on<SetMsg>([&](actor &, SetMsg &msg) {
    events.emplace_back(sim.now(), "set:" + std::to_string(msg.value));
  });
  server.on<QueryMsg>([&](actor &self, QueryMsg &msg) {
    events.emplace_back(sim.now(), "query:" + std::to_string(msg.id));
    self.reply(*self.current_request(), QueryResult{msg.id, 1});
  });

  client.on<PingMsg>([&](simcpp20::simulation<> &sim, actor &,
                         PingMsg &) -> simcpp20::process<> {
    server.send(SetMsg{1}, 1.0);
    QueryResult got = co_await server.request<QueryResult>(QueryMsg{9}, 2.0);
    events.emplace_back(sim.now(), "client_done:" + std::to_string(got.id));
  });

  server.start();
  client.start();
  client.send(PingMsg{});
  sim.run();

  assert(events.size() == 3);
  assert(events[0] == std::make_pair(1.0, std::string("set:1")));
  assert(events[1] == std::make_pair(2.0, std::string("query:9")));
  assert(events[2] == std::make_pair(2.0, std::string("client_done:9")));
  std::cout << "PASS: send_and_request_delay\n";
}

int main() {
  test_sync_handler();
  test_async_handler();
  test_send_at();
  test_multi_type_dispatch();
  test_unknown_message_throws();
  test_mailbox_ordering();
  test_ping_pong();
  test_request_explicit_reply();
  test_request_auto_empty_reply();
  test_request_at_and_timeout_reply();
  test_send_and_request_mixed();
  test_send_and_request_delay();
  std::cout << "All actor tests passed.\n";
  return 0;
}
