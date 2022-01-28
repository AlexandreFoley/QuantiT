
#include <pybind11/pybind11.h>

#include "operators.h"
#include "models.h"
#include "utilities.h"

using namespace quantt;
namespace py = pybind11;

void init_operators(py::module &m)
{

	auto sub = m.def_submodule("operators");

	sub.def("fermions",py::overload_cast<>(&fermions),"Generate spin half fermion operators. returns c_up,c_dn,F,id: up spin annihilation operator, down spin annihilation operator, the non-local fermionic phase operator and the identity" );
	sub.def("pauli", py::overload_cast<>(&pauli),"Generate operator basis for spin half system. return Sx,iSy,Sz,lo,id: Pauli X, i times Pauli Y, Pauli Z, the spin lowering operator and the identity. iSy is given instead of Sy because it avoid introducing complex number in time reversible models. ");

	sub.def("fermions",py::overload_cast<const btensor&>(&fermions),"Generate spin half fermion operators with conserved quantities specified by the supplied shape tensor. returns c_up,c_dn,F,id: up spin annihilation operator, down spin annihilation operator, the non-local fermionic phase operator and the identity" ,py::arg("shape"));
	sub.def("pauli", py::overload_cast<const btensor&>(&pauli),"Generate operator basis for spin half system with conserved quantities specified by the supplied shape tensor. return Sx,iSy,Sz,lo,id: Pauli X, i times Pauli Y, Pauli Z, the spin lowering operator and the identity. iSy is given instead of Sy because it avoid introducing complex number in time reversible models. ",py::arg("shape"));
	
	// MPO Heisenberg(torch::Tensor J, size_t lenght)
	sub.def("Heisenberg",py::overload_cast<torch::Tensor,size_t>(&Heisenberg),"build a MPO for a linear Heisenberg model with coupling J and a given length",py::arg("J"),py::arg("length"));
	// bMPO Heisenberg(torch::Tensor J, size_t lenght,btensor local_shape)
	sub.def("Heisenberg",py::overload_cast<torch::Tensor,size_t,const btensor&>(&Heisenberg),"build a MPO for a linear Heisenberg model with coupling J, a given length and conservation law specified by physical_shape",py::arg("J"),py::arg("length"),py::arg("physical_shape"));
	// MPO Hubbard(torch::Tensor U, torch::Tensor mu, size_t lenght)
	sub.def("Hubbard",py::overload_cast<torch::Tensor,torch::Tensor,size_t>(&Hubbard),"build a MPO for a linear Hubbard model with interaction U,chemical potential mu and a given length",py::arg("U"),py::arg("mu"),py::arg("length"));
	// bMPO Hubbard(torch::Tensor U, torch::Tensor mu, size_t lenght,btensor local_shape)
	sub.def("Hubbard",py::overload_cast<torch::Tensor,torch::Tensor,size_t,const btensor&>(&Hubbard),"build a MPO for a linear Hubbard model with interaction U,chemical potential mu, a given length and conservation law specified by physical_shape",py::arg("U"),py::arg("mu"),py::arg("length"),py::arg("physical_shape"));
	// MPO Heisenberg(torch::Tensor J, size_t lenght)
	sub.def("Heisenberg",utils::wrap_scalar([](torch::Scalar J,size_t l){return Heisenberg(torch::full({},J),l);}),"build a MPO for a linear Heisenberg model with coupling J and a given length",py::arg("J"),py::arg("length"));
	// bMPO Heisenberg(torch::Tensor J, size_t lenght,btensor local_shape)
	sub.def("Heisenberg",utils::wrap_scalar([](torch::Scalar J,size_t l,const btensor& shape){return Heisenberg(torch::full({},J),l,shape);}),"build a MPO for a linear Heisenberg model with coupling J, a given length and conservation law specified by physical_shape",py::arg("J"),py::arg("length"),py::arg("physical_shape"));
	// MPO Hubbard(torch::Tensor U, torch::Tensor mu, size_t lenght)
	sub.def("Hubbard",utils::wrap_scalar([](torch::Scalar U,torch::Scalar mu,size_t l){return Hubbard(torch::full({},U),torch::full({},mu),l);}),"build a MPO for a linear Hubbard model with interaction U,chemical potential mu and a given length",py::arg("U"),py::arg("mu"),py::arg("length"));
	// bMPO Hubbard(torch::Tensor U, torch::Tensor mu, size_t lenght,btensor local_shape)
	sub.def("Hubbard",utils::wrap_scalar([](torch::Scalar U,torch::Scalar mu,size_t l,const btensor& shape){return Hubbard(torch::full({},U),torch::full({},mu),l,shape);}),"build a MPO for a linear Hubbard model with interaction U,chemical potential mu, a given length and conservation law specified by physical_shape",py::arg("U"),py::arg("mu"),py::arg("length"),py::arg("physical_shape"));
}