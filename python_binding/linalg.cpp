

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "blockTensor/LinearAlgebra.h"
#include "utilities.h"

using namespace quantit;
using namespace utils;
void init_linalg_qtt(py::module &m)
{
	auto linalg_sub = m.def_submodule("linalg");

	// std::tuple<btensor, btensor, btensor> svd(const btensor &tensor, BOOL some = true, BOOL compute_uv = true);
	linalg_sub.def(
	    "svd", [](const btensor &tens, bool some, bool cpt_UV) { return svd(tens, some, cpt_UV); },
	    "compute the batched singular value decomposition, default to greedy and computes the U and V matrices",
	    py::arg("tensor"), py::kw_only(), py::arg("some") = true, py::arg("compute_UV") = true);
	// std::tuple<btensor, btensor, btensor> svd(const btensor &tensor, size_t split);
	linalg_sub.def(
	    "svd", [](const btensor &tens, int split) { return svd(tens, split); },
	    "compute the tensor singular value decomposition, implicitly reshape to rank 2 according to the split value.",
	    py::arg("tensor"), py::arg("split"));
	// std::tuple<btensor, btensor, btensor> svd(const btensor &A, size_t split, btensor::Scalar tol, size_t min_size,ˇ
	//   size_t max_size, btensor::Scalar pow = 2);
	// inline std::tuple<btensor, btensor, btensor> svd(const btensor &A, int split, btensor::Scalar tol, size_t
	// min_size,
	//  size_t max_size, btensor::Scalar pow = 2)
	// std::tuple<btensor, btensor, btensor> svd(const btensor &A, size_t split, btensor::Scalar tol, btensor::Scalar
	// pow = 2); inline std::tuple<btensor, btensor, btensor> svd(const btensor &A, int split, btensor::Scalar tol,
	//  btensor::Scalar pow = 2)
	linalg_sub.def("svd",
	               wrap_scalar([](const btensor &tens, int split, btensor::Scalar tol, size_t min_size, size_t max_size,
	                              btensor::Scalar pow) { return svd(tens, split, tol, min_size, max_size, pow); }),
	               "compute the tensor singular value decomposition, implicitly reshape to rank 2 according to the "
	               "split value. truncates the smallest singular values without introducing an absolute reconstruction error larger than "
	               "the tolerence with a pow-norm. The number of singular value kept is always in [min_size,max_size]",
	               py::arg("tensor"), py::arg("split"), py::arg("tol"), py::kw_only(), py::arg("min_size") = 1,
	               py::arg("max_size") = std::numeric_limits<size_t>::max(), py::arg("pow") = 2);
	// std::tuple<btensor, btensor> eigh(const btensor &tensor, BOOL upper = false);
	linalg_sub.def(
	    "eigh", [](const btensor &tens, bool upper) { return eigh(tens, upper); },
	    "compute the batched hermitian eigenvalue decomposition, consider only the lower (or upper) triangle of each "
	    "matrices.",
	    py::arg("tensor"), py::kw_only(), py::arg("upper") = false);
	// std::tuple<btensor, btensor> eigh(const btensor &tensor, size_t split);
	linalg_sub.def(
	    "eigh", [](const btensor &tens, int split) { return eigh(tens, split); },
	    "compute the tensor eigenvalue decomposition, implicitly reshape to rank 2 according to the split value.",
	    py::arg("tensor"), py::arg("split"));
	// std::tuple<btensor, btensor> eigh(const btensor &A, size_t split, btensor::Scalar tol, size_t min_size,
	// size_t max_size, btensor::Scalar pow = 1);
	// std::tuple<btensor, btensor> eigh(const btensor &A, size_t split, btensor::Scalar tol, btensor::Scalar pow = 1);
	linalg_sub.def("eigh",
	               wrap_scalar([](const btensor &tens, int split, btensor::Scalar tol, size_t min_size, size_t max_size,
	                              btensor::Scalar pow) { return eigh(tens, split, tol, min_size, max_size, pow); }),
	               "compute the tensor eigenvalue decomposition, implicitly reshape to rank 2 according to the "
	               "split value. truncates the eigenvalues closest to zero without introducing an absolute reconstruction error larger than "
	               "the tolerence with a pow-norm. The number of eigenvalues kept is always in [min_size,max_size]",
	               py::arg("tensor"), py::arg("split"), py::arg("tol"), py::kw_only(), py::arg("min_size") = 1,
	               py::arg("max_size") = std::numeric_limits<size_t>::max(), py::arg("pow") = 1);
}