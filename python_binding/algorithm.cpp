

#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/stl.h>

#include "utilities.h"
#include "dmrg_logger.h"
#include "dmrg_options.h"
#include <dmrg.h>

using namespace quantit;
namespace py = pybind11;

void register_loggers(py::module_ &alg);

void init_algorithms(py::module &m)
{
	auto alg = m.def_submodule("algorithms");
	register_loggers(alg);
	auto dummy_logger = dmrg_default_logger();
	py::class_<dmrg_options>(alg, "dmrg_options")
	    .def_readwrite("cutoff", &dmrg_options::cutoff)
	    .def_readwrite("convergence_criterion", &dmrg_options::convergence_criterion)
	    .def_readwrite("maximum_bond", &dmrg_options::maximum_bond)
	    .def_readwrite("minimum_bond", &dmrg_options::minimum_bond)
	    .def_readwrite("maximum_iterations", &dmrg_options::maximum_iterations)
	    .def_readwrite("state_gradient", &dmrg_options::state_gradient)
	    .def_readwrite("hamil_gradient", &dmrg_options::hamil_gradient)
	    .def(py::init<double, double, size_t, size_t, size_t, bool, bool>(),
	         py::kw_only(),
	         py::arg("cutoff") = dmrg_options::def_cutoff,
	         py::arg("convergence_criterion") = dmrg_options::def_conv_crit,
	         py::arg("max_bond") = dmrg_options::def_max_bond, py::arg("min_bond") = dmrg_options::def_min_bond,
	         py::arg("maximum_iterations") = dmrg_options::def_max_it,
	         py::arg("state_gradient") = dmrg_options::def_pytorch_gradient,
	         py::arg("hamil_gradient") = dmrg_options::def_pytorch_gradient);

	/**
	 * Apply the DMRG algorithm to solve the ground state of the input hamiltonian given as a MPO.
	 * Uses the supplied MPS in_out_state as a starting point, and store the optimized MPS there.
	 * The associated energy is the return value.
	 */
	// torch::Tensor dmrg( MPO &hamiltonian, MPS &in_out_state, const dmrg_options &options,
	//                    dmrg_logger &logger = dummy_logger);
	alg.def("dmrg",[](MPO& mpo, MPS& mps, const dmrg_options& opt,dmrg_default_logger& logger){return dmrg(mpo,mps,opt,logger);},"perform dmrg on the supplied MPS and MPO",py::arg("MPO"),py::arg("MPS"),py::arg("dmrg_options"),py::arg("dmrg_logger")=dummy_logger);
	// btensor dmrg( bMPO &hamiltonian, bMPS &in_out_state, const dmrg_options &options,
	//                    dmrg_logger &logger = dummy_logger);
	alg.def("dmrg",[](bMPO& mpo, bMPS& mps, const dmrg_options& opt,dmrg_default_logger& logger){return dmrg(mpo,mps,opt,logger);},"perform dmrg on the supplied MPS and MPO",py::arg("MPO"),py::arg("MPS"),py::arg("dmrg_options"),py::arg("dmrg_logger")=dummy_logger);

	/**
	 * Apply the DMRG algorithm to solve the ground state of the input hamiltonian given as a MPO.
	 * uses a random starting MPS with minimum_bond bond dimension.
	 * return the ground state energy and optimized MPS.
	 */
	// std::tuple<torch::Tensor, MPS> dmrg( MPO &hamiltonian, const dmrg_options &options,
	//                                     dmrg_logger &logger = dummy_logger);
	alg.def("dmrg",[](MPO& mpo, const dmrg_options& opt,dmrg_default_logger& logger){return dmrg(mpo,opt,logger);},"perform dmrg on a random starting MPS and the supplied MPO",py::arg("MPO"),py::arg("dmrg_options"),py::arg("dmrg_logger")=dummy_logger);
	// std::tuple<btensor, bMPS> dmrg( bMPO &hamiltonian, any_quantity_cref state_constraint , const dmrg_options
	// &options,
	//                                     dmrg_logger &logger = dummy_logger);
	alg.def("dmrg",[](bMPO& mpo,any_quantity qt, const dmrg_options& opt,dmrg_default_logger& logger){return dmrg(mpo,qt,opt,logger);},"perform dmrg on a random starting MPS with the specified constraint and the supplied MPO",py::arg("MPO"),py::arg("constraint"),py::arg("dmrg_options"),py::arg("dmrg_logger")=dummy_logger);
}

template <class logger_base = dmrg_default_logger>
class py_default_logger : public logger_base
{
  public:
	using logger_base::logger_base;
	// /**
	//  * A default logger that does nothing.
	//  */
	// class dmrg_default_logger : public dmrg_logger
	// {
	//   public:
	// 	void log_step(size_t) override {}
	void log_step(size_t step) override { PYBIND11_OVERRIDE(void, logger_base, log_step, step); }
	// 	void log_energy(const torch::Tensor&) override {}
	void log_energy(const torch::Tensor &E) override { PYBIND11_OVERRIDE(void, logger_base, log_energy, E); }
	// 	void log_bond_dims(const MPS &) override {}
	void log_bond_dims(const MPS &state) override { PYBIND11_OVERRIDE(void, logger_base, log_bond_dims, state); }
	// 	void log_energy(const btensor&) override {}
	void log_energy(const btensor &E) override { PYBIND11_OVERRIDE(void, logger_base, log_energy, E); }
	// 	void log_bond_dims(const bMPS &) override {}
	void log_bond_dims(const bMPS &state) override { PYBIND11_OVERRIDE(void, logger_base, log_bond_dims, state); }

	void init(const dmrg_options &opt) override { PYBIND11_OVERRIDE(void, logger_base, init, opt); }

	// 	virtual void it_log_all(size_t step_num,const torch::Tensor& E, const MPS &state) { log_all(step_num, E, state);
	// }
	void it_log_all(size_t step_num, const torch::Tensor &E, const MPS &state) override
	{
		PYBIND11_OVERRIDE(void, logger_base, it_log_all, step_num, E, state);
	}
	// 	virtual void it_log_all(size_t step_num,const btensor& E, const bMPS &state) { log_all(step_num, E, state); }
	void it_log_all(size_t step_num, const btensor &E, const bMPS &state) override
	{
		PYBIND11_OVERRIDE(void, logger_base, it_log_all, step_num, E, state);
	}
	// 	virtual void end_log_all(size_t step_num, const torch::Tensor& E, const MPS &state) { log_all(step_num, E,
	// state); }
	void end_log_all(size_t step_num, const torch::Tensor &E, const MPS &state) override
	{
		PYBIND11_OVERRIDE(void, logger_base, end_log_all, step_num, E, state);
	}
	// 	virtual void end_log_all(size_t step_num, const btensor& E, const bMPS &state) { log_all(step_num, E, state); }
	void end_log_all(size_t step_num, const btensor &E, const bMPS &state) override
	{
		PYBIND11_OVERRIDE(void, logger_base, end_log_all, step_num, E, state);
	}

	// 	virtual void log_all(size_t step_num, torch::Tensor E, const MPS &state)
	void log_all(size_t step_num, const torch::Tensor &E, const MPS &state) override
	{
		PYBIND11_OVERRIDE(void, logger_base, end_log_all, step_num, E, state);
	}

	// 	virtual void log_all(size_t step_num, btensor E, const bMPS &state)
	void log_all(size_t step_num, const btensor &E, const bMPS &state) override
	{
		PYBIND11_OVERRIDE(void, logger_base, end_log_all, step_num, E, state);
	}
};

void register_loggers(py::module_ &alg)
{
	using ddlogger = dmrg_default_logger;
	py::class_<ddlogger, py_default_logger<>>(
	    alg, "dmrg_logger",
	    "base logger for quantit.algorithm.dmrg. Does nothing. Derive a class from this if you need specfic behavior "
	    "that differ from either dmrg_simple_logger or dmrg_sweeptime_logger.")
	    .def(py::init<>())
	    // 	void log_step(size_t) override {}
	    .def("log_step", &ddlogger::log_step,
	         "action to take to log the step number, default implementation does nothing", py::arg("step"))
	    // 	void log_energy(const torch::Tensor&) override {}
	    .def("log_energy", py::overload_cast<const torch::Tensor &>(&ddlogger::log_energy),
	         "action to take to log the energy, default implementation does nothing", py::arg("energy"))
	    // 	void log_bond_dims(const MPS &) override {}
	    .def("log_bond_dims", py::overload_cast<const MPS &>(&ddlogger::log_bond_dims),
	         "action to take to log the bond dimensions, default implementation does nothing", py::arg("mps"))
	    // 	void log_energy(const btensor&) override {}
	    .def("log_energy", py::overload_cast<const btensor &>(&ddlogger::log_energy),
	         "action to take to log the energy, default implementation does nothing", py::arg("energy"))
	    // 	void log_bond_dims(const bMPS &) override {}
	    .def("log_bond_dims", py::overload_cast<const bMPS &>(&ddlogger::log_bond_dims),
	         "action to take to log the bond dimensions, default implementation does nothing", py::arg("mps"))
	    //  void init(const dmrg_options & opt) override
	    .def("init", &ddlogger::init,
	         "logging action to take during DMRG initialization, default implementation does nothing")
	    // 	virtual void it_log_all(size_t step_num,const torch::Tensor& E, const MPS &state) { log_all(step_num, E,
	    // state); }
	    .def("it_log_all", py::overload_cast<size_t, const torch::Tensor &, const MPS &>(&ddlogger::it_log_all),
	         "logging action take at every dmrg sweep. default implementation calls log_all.", py::arg("step"),
	         py::arg("energy"), py::arg("state"))
	    // 	virtual void it_log_all(size_t step_num,const btensor& E, const bMPS &state) { log_all(step_num, E, state);
	    // }
	    .def("it_log_all", py::overload_cast<size_t, const btensor &, const bMPS &>(&ddlogger::it_log_all),
	         "logging action take at every dmrg sweep. default implementation calls log_all.", py::arg("step"),
	         py::arg("energy"), py::arg("state"))
	    // 	virtual void end_log_all(size_t step_num, const torch::Tensor& E, const MPS &state) { log_all(step_num, E,
	    // state); }
	    .def("end_log_all", py::overload_cast<size_t, const torch::Tensor &, const MPS &>(&ddlogger::end_log_all),
	         "logging action taken at the end of the dmrg optimization. default implementation calls log_all.",
	         py::arg("step"), py::arg("energy"), py::arg("state"))
	    // 	virtual void end_log_all(size_t step_num, const btensor& E, const bMPS &state) { log_all(step_num, E,
	    // state); }
	    .def("end_log_all", py::overload_cast<size_t, const btensor &, const bMPS &>(&ddlogger::end_log_all),
	         "logging action taken at the end of the dmrg optimization. default implementation calls log_.",
	         py::arg("step"), py::arg("energy"), py::arg("state"))
	    // 	virtual void log_all(size_t step_num, torch::Tensor E, const MPS &state)
	    .def("end_log_all", py::overload_cast<size_t, const torch::Tensor &, const MPS &>(&ddlogger::log_all),
	         "default logging action. default implementation calls log_energy, log_bond_dim and log_step.",
	         py::arg("step"), py::arg("energy"), py::arg("state"))
	    // 	virtual void log_all(size_t step_num, btensor E, const bMPS &state)
	    .def("end_log_all", py::overload_cast<size_t, const btensor &, const bMPS &>(&ddlogger::log_all),
	         "default logging action. default implementation calls log_energy, log_bond_dim and log_step.",
	         py::arg("step"), py::arg("energy"), py::arg("state"));

	py::class_<dmrg_log_simple, ddlogger>(alg, "dmrg_simple_logger",
	                                      "logs the final middle bond dimension and number of dmrg sweeps")
	    .def(py::init<>())
	    .def_property_readonly("sweep_number", [](const dmrg_log_simple &self) { return self.it_num; })
	    .def_property_readonly("final_mid_bond_dim", [](const dmrg_log_simple &self) { return self.middle_bond_dim; });

	py::class_<dmrg_log_sweeptime, ddlogger>(
	    alg, "dmrg_sweeptime_logger",
	    "log the sweeptimes and the evolution of the middle bond dimension function of sweep number")
	    .def(py::init<>())
	    .def_property_readonly(
	        "time_list", [](const dmrg_log_sweeptime &self) { return self.time_list; }, "time of each sweep in seconds")
	    .def_property_readonly(
	        "bond_list", [](const dmrg_log_sweeptime &self) { return self.bond_list; },
	        "bond dimension as of function of sweep number");
}