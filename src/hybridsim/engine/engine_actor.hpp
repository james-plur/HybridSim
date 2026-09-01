#pragma once

#include "hybridsim/actor.hpp"
#include "hybridsim/engine/comm_kernel.hpp"
#include "hybridsim/engine/dag_scheduler.hpp"
#include "hybridsim/engine/kernel_factory.hpp"
#include "hybridsim/engine/kernel_types.hpp"
#include "hybridsim/engine/messages.hpp"
#include "hybridsim/network/addr.hpp"
#include "hybridsim/network/network.hpp"

#include <functional>
#include <memory>
#include <stdexcept>

namespace hybridsim::engine {

using workload_complete_handler = std::function<void(const WorkloadDoneMsg &)>;

class engine_actor : public actor {
public:
  explicit engine_actor(simcpp20::simulation<> &sim) : actor(sim) {
    register_comm_creators();
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

  void install_network(std::shared_ptr<network::Network> net,
                       network::NetworkAddr addr) {
    if (!net) {
      throw std::invalid_argument("install_network requires a Network");
    }
    network_ = std::move(net);
    addr_ = addr;
    adapter_ = network_->adapter(addr_);
  }

  network::NetworkAdapter *adapter() const noexcept { return adapter_; }
  network::NetworkAddr addr() const noexcept { return addr_; }

protected:
  virtual void on_workload_complete(const WorkloadDoneMsg &msg) {
    if (on_workload_complete_) {
      on_workload_complete_(msg);
    }
  }

private:
  network::NetworkAdapter *require_adapter() const {
    if (adapter_ == nullptr) {
      throw std::runtime_error(
          "comm kernel requires engine_actor::install_network");
    }
    return adapter_;
  }

  void register_comm_creators() {
    factory_.register_creator(kKernelPut, [this](const kernel_spec &spec) {
      return std::make_unique<PutKernel>(spec, require_adapter(), addr_);
    });
    factory_.register_creator(kKernelSignal, [this](const kernel_spec &spec) {
      return std::make_unique<SignalKernel>(spec, require_adapter(), addr_);
    });
    factory_.register_creator(kKernelWait, [this](const kernel_spec &spec) {
      return std::make_unique<WaitKernel>(spec, require_adapter(), addr_);
    });
    factory_.register_creator(kKernelGet, [this](const kernel_spec &spec) {
      return std::make_unique<GetKernel>(spec, require_adapter(), addr_);
    });
  }

  KernelFactory factory_;
  workload_complete_handler on_workload_complete_;
  std::shared_ptr<network::Network> network_;
  network::NetworkAdapter *adapter_ = nullptr;
  network::NetworkAddr addr_{};
};

} // namespace hybridsim::engine
