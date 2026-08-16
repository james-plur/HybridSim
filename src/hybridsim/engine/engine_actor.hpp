#pragma once

#include "hybridsim/actor.hpp"
#include "hybridsim/engine/dag_scheduler.hpp"
#include "hybridsim/engine/kernel_factory.hpp"
#include "hybridsim/engine/messages.hpp"

#include <functional>

namespace hybridsim::engine {

using workload_complete_handler = std::function<void(const WorkloadDoneMsg &)>;

class engine_actor : public actor {
public:
  explicit engine_actor(simcpp20::simulation<> &sim) : actor(sim) {
    on<WorkloadMsg>([this](simcpp20::simulation<> &sim, actor &self,
                           WorkloadMsg &msg) -> simcpp20::process<> {
      co_await schedule_dag(sim, msg.spec, factory_);
      self.send(WorkloadDoneMsg{msg.spec.workload_id}, 0.0, kMsgPriorityHigh);
    });

    on<WorkloadDoneMsg>([this](actor &, WorkloadDoneMsg &msg) {
      on_workload_complete(msg);
    });
  }

  void set_on_workload_complete(workload_complete_handler handler) {
    on_workload_complete_ = std::move(handler);
  }

protected:
  virtual void on_workload_complete(const WorkloadDoneMsg &msg) {
    if (on_workload_complete_) {
      on_workload_complete_(msg);
    }
  }

private:
  KernelFactory factory_;
  workload_complete_handler on_workload_complete_;
};

} // namespace hybridsim::engine
