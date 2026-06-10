#include "hybridsim/dynamic_actor.hpp"
#include "hybridsim/dynamic_message.hpp"

#include <cassert>
#include <iostream>

int main() {
  simcpp20::simulation<> sim;
  hybridsim::dynamic_actor actor(sim);

  const auto id = hybridsim::message_registry::instance().register_type("Test");
  actor.on(id, [](simcpp20::simulation<> &, hybridsim::dynamic_actor &self,
                  std::shared_ptr<hybridsim::dynamic_message> msg)
             -> simcpp20::process<> {
    assert(msg->type_id() == hybridsim::message_registry::instance().lookup("Test"));
    self.stop();
    co_return;
  });

  actor.start();
  actor.send(hybridsim::make_dynamic_message(id, std::any()));
  sim.run();
  actor.rethrow_if_error();

  std::cout << "dynamic_actor_test passed\n";
  return 0;
}
