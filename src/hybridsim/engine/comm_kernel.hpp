#pragma once

#include "hybridsim/engine/kernel.hpp"
#include "hybridsim/engine/kernel_spec.hpp"
#include "hybridsim/engine/kernel_types.hpp"
#include "hybridsim/network/adapter.hpp"
#include "hybridsim/network/addr.hpp"
#include "hybridsim/network/types.hpp"

#include <stdexcept>
#include <string>

namespace hybridsim::engine {

struct CommKernelArgs {
  network::NetworkAdapter *adapter = nullptr;
  network::NetworkAddr self{};
  network::NetworkAddr dst{};
  int64_t conn_id = 0;
  int qos = 0;
  double payload_bytes = 0.0;
};

inline CommKernelArgs comm_args_from_spec(const kernel_spec &spec,
                                          network::NetworkAdapter *adapter,
                                          network::NetworkAddr self) {
  if (adapter == nullptr) {
    throw std::runtime_error("comm kernel requires install_network");
  }
  CommKernelArgs args;
  args.adapter = adapter;
  args.self = self;
  if (auto v = spec.params.get_string("dst_addr")) {
    args.dst = network::NetworkAddr::parse(*v);
  }
  if (auto v = spec.params.get_int("conn_id")) {
    args.conn_id = *v;
  }
  if (auto v = spec.params.get_int("qos")) {
    args.qos = static_cast<int>(*v);
  }
  if (auto v = spec.params.get_double("payload_bytes")) {
    args.payload_bytes = *v;
  } else if (auto i = spec.params.get_int("payload_bytes")) {
    args.payload_bytes = static_cast<double>(*i);
  }
  return args;
}

class PutKernel : public Kernel {
public:
  PutKernel(kernel_spec spec, network::NetworkAdapter *adapter,
            network::NetworkAddr self)
      : args_{comm_args_from_spec(spec, adapter, self)} {}

  simcpp20::process<> run(simcpp20::simulation<> &) override {
    args_.adapter->inject(args_.dst, args_.conn_id, args_.qos,
                          args_.payload_bytes);
    co_return;
  }

private:
  CommKernelArgs args_;
};

class SignalKernel : public Kernel {
public:
  SignalKernel(kernel_spec spec, network::NetworkAdapter *adapter,
               network::NetworkAddr self)
      : args_{comm_args_from_spec(spec, adapter, self)} {
    if (args_.payload_bytes <= 0.0) {
      args_.payload_bytes = network::kDefaultSignalBytes;
    }
  }

  simcpp20::process<> run(simcpp20::simulation<> &) override {
    args_.adapter->inject(args_.dst, args_.conn_id, args_.qos,
                          args_.payload_bytes);
    co_return;
  }

private:
  CommKernelArgs args_;
};

class WaitKernel : public Kernel {
public:
  WaitKernel(kernel_spec spec, network::NetworkAdapter *adapter,
             network::NetworkAddr self)
      : args_{comm_args_from_spec(spec, adapter, self)} {}

  simcpp20::process<> run(simcpp20::simulation<> &) override {
    co_await args_.adapter->recv_event(args_.conn_id);
  }

private:
  CommKernelArgs args_;
};

class GetKernel : public Kernel {
public:
  GetKernel(kernel_spec spec, network::NetworkAdapter *adapter,
            network::NetworkAddr self)
      : args_{comm_args_from_spec(spec, adapter, self)} {}

  simcpp20::process<> run(simcpp20::simulation<> &) override {
    args_.adapter->inject(args_.dst, args_.conn_id, args_.qos,
                          network::kDefaultSignalBytes, /*is_fetch=*/true,
                          args_.payload_bytes);
    co_await args_.adapter->recv_event(args_.conn_id);
  }

private:
  CommKernelArgs args_;
};

} // namespace hybridsim::engine
