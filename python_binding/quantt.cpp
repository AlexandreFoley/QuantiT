/*
 * File: quantt.cpp
 * Project: quantt
 * File Created: Thursday, 11th November 2021 11:17:35 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * Copyright (c) 2021 Alexandre Foley
 * All rights reserved
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdexcept>

#include <torch/extension.h>
#include "blockTensor/btensor.h"



/*
 * torch must also be already loaded in the module.
 * For the module to load we need to define TORCH_USE_RTLD_GLOBAL=YES environment variable, otherwise missing symbol prevent loading
 * Seems like that env_var is a stopgap solution, correct approach is to link with PYTHON_TORCH_LIBRARIES or TORCH_PYTHON_LIBRARIES (which?) cmake variable.
*/
namespace
{

namespace py = pybind11;

torch::Tensor test2(){
    auto label_map = torch::zeros({100, 100}).to(torch::kFloat32);
    return label_map;
}

} // namespace
void init_conserved_qtt(py::module &m);
// The first argument needs to match the name of the *.so in the BUILD file.
PYBIND11_MODULE(QuantiT, m)
{
    auto PYTORCH = py::module_::import("torch"); 
    m.doc() = "quantiT";
    init_conserved_qtt(m);
    m.def("test2",

          &test2,
          "test2"
        );

}