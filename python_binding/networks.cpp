

#include "utilities.h"
#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include "MPT.h"

using namespace quantit;
using namespace utils;
namespace py = pybind11;

template <class S>
void common_core(py::class_<S> &pyclass)
{
	// tensorlift common stuff
	//  vector_lift() : tensors() {}
	pyclass.def(py::init());
	// vector_lift(size_type size) : tensors(size) {}
	pyclass.def(py::init<size_t>());
	// vector_lift(size_type size, const Tens &val) : tensors(size, val) {}
	pyclass.def(py::init<size_t, const typename S::Tens &>());
	// vector_lift(std::vector<Tens> &other) : tensors(other) {}
	pyclass.def(py::init<std::vector<typename S::Tens>>());
	// vector_lift &operator=(vector_lift other) noexcept

	// pyclass.def(py::self = py::self) //unecessary?

	// interface function from std::vector
	// reference at(size_t i) { return tensors.at(i); }
	pyclass.def("at", [](S& self, size_t i){return self.at(i);},py::return_value_policy::reference_internal);
	// reference operator[](size_t i) { return tensors[i]; }
	pyclass.def("__getitem__",[](S& self, size_t i){return self[i];}
	, py::return_value_policy::reference_internal);
	// reference front() { return tensors.front(); }
	pyclass.def("front", [](S& self){return self.front();}, py::return_value_policy::reference_internal);
	// reference back() { return tensors.back(); }
	pyclass.def("back", [](S& self){return self.back();}, py::return_value_policy::reference_internal);
	// iterators
	pyclass.def(
	    "__iter__", [](S &obj) { return py::make_iterator(obj.begin(), obj.end()); });
	// capacity
	// [[nodiscard]] auto empty() const noexcept { return tensors.empty(); }
	pyclass.def("empty", &S::empty);
	// [[nodiscard]] auto size() const noexcept { return tensors.size(); }
	pyclass.def("size", &S::size);
	// [[nodiscard]] auto max_size() const noexcept { return tensors.max_size(); }
	pyclass.def("max_size", &S::max_size);
	// void reserve(size_t new_cap) { tensors.reserve(new_cap); }
	pyclass.def("reserve", &S::reserve);
	// [[nodiscard]] auto capacity() const noexcept { return tensors.capacity(); }
	pyclass.def("capacity", &S::capacity);
	// void shrink_to_fit() { tensors.shrink_to_fit(); }
	pyclass.def("shrink_to_fit", &S::shrink_to_fit);
	// modifiers
	// void clear() noexcept { tensors.clear(); }
	pyclass.def("clear", &S::clear);
	// iterator insert(const_iterator pos, const Tens &val) { return tensors.insert(pos, val); }
	// iterator insert(const_iterator pos, Tens &&val) { return tensors.insert(pos, val); }
	// iterator insert(const_iterator pos, size_type count, Tens &&val) { return tensors.insert(pos, count, val); }
	pyclass
	    .def("insert",
	         [](S &self, size_t index, const typename S::Tens &val)
	         {
		         if (index > self.size())
			         throw std::invalid_argument("index is beyond the end of the list");
		         self.insert(self.begin() + index, val);
	         });
	    // template <class InputIT>
	    // iterator insert(const_iterator pos, InputIT first, InputIT last)
	    // iterator insert(const_iterator pos, std::initializer_list<Tens> list) { return tensors.insert(pos, list); }
	    // iterator erase(const_iterator pos) { return tensors.erase(pos); }
	    pyclass.def("erase",
	                [](S &self, size_t index)
	                {
		                if (index > self.size())
			                throw std::invalid_argument("Index is beyond the end of the list");
		                self.erase(self.begin() + index);
	                });
	// iterator erase(const_iterator first, const_iterator last) { return tensors.erase(first, last); }
	pyclass.def("erase",
	            [](S &self, size_t first, size_t last)
	            {
		            if (first < last)
			            throw std::invalid_argument("first index larger than last index");
		            if (first > self.size())
			            throw std::invalid_argument("First index is beyond the end of the list");
		            if (last > self.size())
			            throw std::invalid_argument("Last index is beyond the end of the list");
		            self.erase(self.begin() + first,self.begin()+last);
	            });
	// void push_back(const Tens &val) { tensors.push_back(val); }
	// void push_back(Tens &&val) { tensors.push_back(val); }
	pyclass.def("append", [](S& self,const typename S::Tens& tens){return self.push_back(tens);});
	// void pop_back() { tensors.pop_back(); }
	pyclass.def("pop", &S::pop_back);
	pyclass.def("pop",
	            [](S &self, size_t index)
	            {
		            auto val = self[index];
		            self.erase(self.begin() + index);
		            return val;
	            });
	// void resize(size_type count) { tensors.resize(count); }
	pyclass.def("resize", py::overload_cast<typename S::size_type>(&S::resize));
	// void resize(size_type count, const Tens &value) { tensors.resize(count, value); }
	pyclass.def("resize", py::overload_cast<typename S::size_type, const typename S::Tens &>(&S::resize));
	// stuff about the tensors, written as covariant functions
	// S to(const torch::TensorOptions &options = {}, bool non_blocking = false, bool copy = false,
	//  c10::optional<c10::MemoryFormat> memory_format = c10::nullopt) const
	pyclass.def(
	    "to",
	    [](const S &self, torch::ScalarType dtype, c10::optional<torch::Device> dev, bool non_blocking, bool copy,
	       c10::MemoryFormat fmt) { return self.to(torch::TensorOptions(dtype).device(dev), non_blocking, copy, fmt); },
	    "perform tensors options conversion", py::arg("dtype"), py::kw_only(),
	    py::arg("device") = c10::optional<torch::Device>(), py::arg("non_blocking") = false, py::arg("copy") = false,
	    py::arg("memory_format") = c10::MemoryFormat::Preserve);
	// inplace version of to, will resolve to any equivalent out-of-place equivalent.i
	pyclass.def(
	    "to_",
	    [](S &self, torch::ScalarType dtype, c10::optional<torch::Device> dev, bool non_blocking, bool copy,
	       c10::MemoryFormat fmt)
	    { return self.to_(torch::TensorOptions(dtype).device(dev), non_blocking, copy, fmt); },
	    "perform tensors options conversion", py::arg("dtype"), py::kw_only(),
	    py::arg("device") = c10::optional<torch::Device>(), py::arg("non_blocking") = false, py::arg("copy") = false,
	    py::arg("memory_format") = c10::MemoryFormat::Preserve, py::return_value_policy::reference_internal);
	// void print_dims(const T &mps) //T = MPS,MPT,MPO,bMPS,bMPT,bMPO
	pyclass.def("print_dims",[](const S& self){print_dims(self);},"print the bond dimensions of the tensors in the network");
}
template<class S>
void MPS_MPO(py::class_<S>& pyclass)
{
	// common to (b)MPS and (b)MPO
	//  bool check_ranks() const; MPO,MPS
	pyclass.def("check_ranks",&S::check_ranks);
	//  static bool check_one(const Tens &tens);  MPO MPS
	pyclass.def_static("check_one", &S::check_one);
}
template<class S>
void MPS_only(py::class_<S>& pyclass)
{
	// (b)MPS
	// move_oc
	pyclass.def("move_oc",&S::move_oc);
	// property : orthogonality center.
	pyclass.def_property_readonly("orthogonality_center",[](const S& self){return static_cast<size_t>(self.orthogonality_center);});

}

void init_networks(py::module &m)
{
	auto sub = m.def_submodule("networks");
	auto pyMPT = py::class_<quantit::MPT>(sub, "MPT");
	auto pyMPS = py::class_<quantit::MPS>(sub, "MPS");
	auto pyMPO = py::class_<quantit::MPO>(sub, "MPO");
	auto pybMPT = py::class_<quantit::bMPT>(sub, "bMPT");
	auto pybMPS = py::class_<quantit::bMPS>(sub, "bMPS");
	auto pybMPO = py::class_<quantit::bMPO>(sub, "bMPO");
	common_core(pyMPT);
	common_core(pyMPS);
	common_core(pyMPO);
	common_core(pybMPT);
	common_core(pybMPS);
	common_core(pybMPO);
	MPS_MPO(pyMPS);
	MPS_MPO(pyMPO);
	MPS_MPO(pybMPS);
	MPS_MPO(pybMPO);
	MPS_only(pyMPS);
	MPS_only(pybMPS);
	//bMPO only.
	pybMPO.def("coalesce", wrap_scalar([](bMPO &self, btensor::Scalar cutoff) { return self.coalesce(cutoff); }),
	           "simplify the block representation of the tensors with a gauge transform. Can introduce an "
	           "approximation smaller or equal to the cutoff on each tensors",
	           py::arg("cutoff") = 0);

	// MPS random_MPS(size_t length, size_t bond_dim, size_t phys_dim, torch::TensorOptions opt = {});
	sub.def("random_MPS", TOPT_binder<size_t, size_t, size_t>::bind(&quantit::random_MPS),
	      "Generate a random MPS",
	      py::arg("length"), py::arg("bond_dim"), py::arg("phys_dim"), py::kw_only(),
	      py::arg("dtype") = opt<stype>(), py::arg("device") = opt<tdev>(), py::arg("requires_grad") = opt<bool>(),
	      py::arg("pin_memory") = opt<bool>());
	// MPS random_MPS(size_t bond_dim, const MPO &hamil, torch::TensorOptions opt = {});
	sub.def("random_MPS", TOPT_binder<size_t, const MPO&>::bind(&quantit::random_MPS),
	      "Generate a random MPS that can be contracted with the given MPO",
	      py::arg("bond_dim"), py::arg("mpo"), py::kw_only(),
	      py::arg("dtype") = opt<stype>(), py::arg("device") = opt<tdev>(), py::arg("requires_grad") = opt<bool>(),
	      py::arg("pin_memory") = opt<bool>());
	// MPS random_MPS(size_t bond_dim, const std::vector<int64_t> &phys_dims,
	//                torch::TensorOptions opt = {}); 
	sub.def("random_MPS", TOPT_binder<size_t, const std::vector<int64_t>& >::bind(&quantit::random_MPS),
	      "Generate a random MPS with the specified physical dimensions",
	      py::arg("bond_dim"), py::arg("phys_dim_list"), py::kw_only(),
	      py::arg("dtype") = opt<stype>(), py::arg("device") = opt<tdev>(), py::arg("requires_grad") = opt<bool>(),
	      py::arg("pin_memory") = opt<bool>());


	// torch::Tensor contract(const MPS &a, const MPS &b, const MPO &obs, torch::Tensor left_edge,
	//    const torch::Tensor &right_edge);
	sub.def("contract",py::overload_cast<const MPS& , const MPS&, const MPO&,torch::Tensor,const torch::Tensor&>(&quantit::contract),"contract to a scalar the two given MPS with the MPO and edge tensors",py::arg("bra"),py::arg("ket"),py::arg("operator"),py::arg("left_edge"),py::arg("right_edge"));
	// torch::Tensor contract(const MPS &a, const MPS &b, const MPO &obs);
	sub.def("contract",py::overload_cast<const MPS& , const MPS&, const MPO&>(&quantit::contract),"contract to a scalar the two given MPS with the MPO",py::arg("bra"),py::arg("ket"),py::arg("operator"));
	// torch::Tensor contract(const MPS &a, const MPS &b);
	sub.def("contract",py::overload_cast<const MPS& , const MPS&>(&quantit::contract),"contract to a scalar the two given MPS",py::arg("bra"),py::arg("ket"));
	// torch::Tensor contract(const MPS &a, const MPS &b, torch::Tensor left_edge, const torch::Tensor &right_edge);
	sub.def("contract",py::overload_cast<const MPS& , const MPS&,torch::Tensor,const torch::Tensor&>(&quantit::contract),"contract to a scalar the two given MPS with the edge tensors",py::arg("bra"),py::arg("ket"),py::arg("left_edge"),py::arg("right_edge"));

	// btensor contract(const bMPS &a, const bMPS &b, const bMPO &obs);
	sub.def("contract",py::overload_cast<const bMPS& , const bMPS&, const bMPO&>(&quantit::contract),"contract to a scalar the two given MPS with the MPO",py::arg("bra"),py::arg("ket"),py::arg("operator"));
	// btensor contract(const bMPS &a, const bMPS &b, const bMPO &obs, btensor left_edge, const btensor &right_edge);
	sub.def("contract",py::overload_cast<const bMPS& , const bMPS&, const bMPO&,quantit::btensor,const quantit::btensor&>(&quantit::contract),"contract to a scalar the two given MPS with the MPO and edge tensors",py::arg("bra"),py::arg("ket"),py::arg("operator"),py::arg("left_edge"),py::arg("right_edge"));
	// btensor contract(const bMPS &a, const bMPS &b, btensor left_edge, const btensor &right_edge);
	sub.def("contract",py::overload_cast<const bMPS& , const bMPS&,btensor,const btensor&>(&quantit::contract),"contract to a scalar the two given MPS with the edge tensors",py::arg("bra"),py::arg("ket"),py::arg("left_edge"),py::arg("right_edge"));
	// btensor contract(const bMPS &a, const bMPS &b);
	sub.def("contract",py::overload_cast<const bMPS& , const bMPS&>(&quantit::contract),"contract to a scalar the two given MPS",py::arg("bra"),py::arg("ket"));


	// bMPS random_bMPS(size_t length, size_t bond_dim, const btensor &phys_dim_spec, any_quantity_cref q_num,
	//                  unsigned int seed = (std::random_device())(), torch::TensorOptions opt = {});
	auto bound_randomBMPSA = TOPT_binder<size_t,size_t,const btensor &,any_quantity>::bind_fl([](size_t length, size_t bond_dim, const btensor &phys_dim_spec, any_quantity q_num, torch::TensorOptions opt)
	{
		return random_bMPS(length,bond_dim,phys_dim_spec,q_num,std::random_device()(),opt);
	});
	sub.def("random_bMPS",bound_randomBMPSA,"generate a MPS constrained by a conservation law, with the specified lenght, bond dimensions, and physical index. The physical index is specified by a rank 1 btensor given in argument",
	py::arg("length"),py::arg("bond_dim"),py::arg("physical_dim"),py::arg("conservation_law") ,py::kw_only(),
	      py::arg("dtype") = opt<stype>(), py::arg("device") = opt<tdev>(), py::arg("requires_grad") = opt<bool>(),
	      py::arg("pin_memory") = opt<bool>());
	// bMPS random_bMPS(size_t bond_dim, const bMPO &Hamil, any_quantity_cref q_num,
	//                  unsigned int seed = (std::random_device())(), torch::TensorOptions opt = {});
	auto bound_randomBMPSB = TOPT_binder<size_t,const bMPO &,any_quantity>::bind_fl([](size_t bond_dim,  const bMPO &OP, any_quantity q_num, torch::TensorOptions opt)
	{
		return random_bMPS(bond_dim,OP,q_num,std::random_device()(),opt);
	});
	sub.def("random_bMPS",bound_randomBMPSB,"generate a MPS constrained by a conservation law, with the specified bond dimensions. The physical dimensions are specifed by the index 3 of the tensors in the MPO, the length is the number of tensors in the MPO",
	py::arg("bond_dim"),py::arg("MPO"),py::arg("conservation_law") ,py::kw_only(),
	      py::arg("dtype") = opt<stype>(), py::arg("device") = opt<tdev>(), py::arg("requires_grad") = opt<bool>(),
	      py::arg("pin_memory") = opt<bool>());
	// bMPS random_bMPS(size_t bond_dim, const std::vector<btensor> &phys_dim_spec, any_quantity_cref q_num,
	//                  unsigned int seed = (std::random_device())(), torch::TensorOptions opt = {});
	auto bound_randomBMPSC = TOPT_binder<size_t,const std::vector<btensor> &,any_quantity>::bind_fl([]( size_t bond_dim, const std::vector<btensor> &phys_dim_spec, any_quantity q_num, torch::TensorOptions opt)
	{
		return random_bMPS(bond_dim,phys_dim_spec,q_num,std::random_device()(),opt);
	});
	sub.def("random_bMPS",bound_randomBMPSC,"generate a MPS constrained by a conservation law, with the specified bond dimensions. The physical dimensions are specifed by a list of rank 1 tensors, the length is the number of tensors in the list",
	py::arg("bond_dim"),py::arg("physical_dims"),py::arg("conservation_law") ,py::kw_only(),
	      py::arg("dtype") = opt<stype>(), py::arg("device") = opt<tdev>(), py::arg("requires_grad") = opt<bool>(),
	      py::arg("pin_memory") = opt<bool>());
	// bMPS random_MPS(size_t length, size_t bond_dim, const btensor &phys_dim_spec, any_quantity_cref q_num,
	//                 unsigned int seed = (std::random_device())(), torch::TensorOptions opt = {});
	sub.def("random_MPS",bound_randomBMPSA,"generate a MPS constrained by a conservation law, with the specified lenght, bond dimensions, and physical index. The physical index is specified by a rank 1 btensor given in argument",
	py::arg("length"),py::arg("bond_dim"),py::arg("physical_dim"),py::arg("conservation_law") ,py::kw_only(),
	      py::arg("dtype") = opt<stype>(), py::arg("device") = opt<tdev>(), py::arg("requires_grad") = opt<bool>(),
	      py::arg("pin_memory") = opt<bool>());
	// bMPS random_MPS(size_t bond_dim, const bMPO &Hamil, any_quantity_cref q_num,
	//                 unsigned int seed = (std::random_device())(), torch::TensorOptions opt = {});
	sub.def("random_MPS",bound_randomBMPSB,"generate a MPS constrained by a conservation law, with the specified bond dimensions. The physical dimensions are specifed by the index 3 of the tensors in the MPO, the length is the number of tensors in the MPO",
	py::arg("bond_dim"),py::arg("MPO"),py::arg("conservation_law") ,py::kw_only(),
	      py::arg("dtype") = opt<stype>(), py::arg("device") = opt<tdev>(), py::arg("requires_grad") = opt<bool>(),
	      py::arg("pin_memory") = opt<bool>());
	// bMPS random_MPS(size_t bond_dim, const std::vector<btensor> &phys_dim_spec, any_quantity_cref q_num,
	//                 unsigned int seed = (std::random_device())(), torch::TensorOptions opt = {});
	sub.def("random_MPS",bound_randomBMPSC,"generate a MPS constrained by a conservation law, with the specified bond dimensions. The physical dimensions are specifed by a list of rank 1 tensors, the length is the number of tensors in the list",
	py::arg("bond_dim"),py::arg("physical_dims"),py::arg("conservation_law") ,py::kw_only(),
	      py::arg("dtype") = opt<stype>(), py::arg("device") = opt<tdev>(), py::arg("requires_grad") = opt<bool>(),
	      py::arg("pin_memory") = opt<bool>());


}
