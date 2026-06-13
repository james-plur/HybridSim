#pragma once

#include <pybind11/pybind11.h>

namespace hybridsim::python {

void bind_engine(pybind11::module_ &m);

} // namespace hybridsim::python
