#include "hybridsim/hybridsim.hpp"

#include <cstdio>

struct Ping {};

struct Pong {
  int round = 0;
};

int main() {
  simcpp20::simulation<> sim;
  int rounds = 0;
  const int max_rounds = 3;

  hybridsim::actor ping(sim);
  hybridsim::actor pong(sim);

  ping.on<Ping>([&pong, &rounds, max_rounds](hybridsim::actor &, Ping &) {
    if (rounds < max_rounds) {
      pong.send(Pong{rounds + 1});
    }
  });

  ping.on<Pong>([&rounds, max_rounds](hybridsim::actor &self, Pong &msg) {
    printf("[%.0f] ping received pong round %d\n", self.sim().now(),
           msg.round);
    rounds = msg.round;
    if (rounds < max_rounds) {
      self.send(Ping{});
    }
  });

  pong.on<Pong>([&ping](hybridsim::actor &self, Pong &msg) {
    printf("[%.0f] pong received round %d\n", self.sim().now(), msg.round);
    ping.send(msg);
  });

  ping.start();
  pong.start();
  ping.send(Ping{});

  sim.run();

  printf("completed %d rounds\n", rounds);
  return rounds == max_rounds ? 0 : 1;
}
