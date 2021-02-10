/*
 * File: MPT.h
 * Project: QuanTT
 * File Created: Thursday, 23rd July 2020 9:38:33 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Thursday, 23rd July 2020 10:46:02 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#ifndef ADA5A359_8ACF_448D_91BC_09C085F510CC
#define ADA5A359_8ACF_448D_91BC_09C085F510CC

#include "property.h"
#include <algorithm>
#include <fmt/core.h>
#include <torch/torch.h>
#include <type_traits>
#include <vector>

#include "doctest/doctest_proxy.h"


namespace quantt
{

// forward declaration for dmrg_impl.
class env_holder;
class dmrg_options;
class dmrg_logger;
class MPS;
class MPO;
class MPT;
namespace details
{
/**
 * @brief implementation for dmrg algorithms
 * 
 * @param hamiltonian 
 * @param two_sites_hamil MPT which contain the contraction of neighbouring sites of the MPO
 * @param in_out_state the state to optimize
 * @param options 
 * @param Env local environment container
 * @param logger 
 * @return torch::Scalar optimized Energy
 */
torch::Scalar dmrg_impl(const MPO &hamiltonian, const MPT &two_sites_hamil, MPS &in_out_state,
                        const dmrg_options &options, env_holder &Env, dmrg_logger &logger);
}
// matrix product tensors base type. require concrete derived class to implement MPT empty_copy(const S&)
/**
 * @brief base type for MPT, MPS, MPO.
 * 
 * Give us the ability to reuse the interface of std::vector multiple without inheriting from it and with minimal boiler plate.
 * 
 * @tparam S the derived type
 */
template <class S> class vector_lift
{
	std::vector<torch::Tensor> tensors;

  public:
	// dependent types
	using Tens = torch::Tensor;
	using VTens = std::vector<Tens>;
	using size_type = VTens::size_type;
	using iterator = VTens::iterator;
	using const_iterator = VTens::const_iterator;
	using reverse_iterator = VTens::reverse_iterator;
	using const_reverse_iterator = VTens::const_reverse_iterator;
	using const_reference = VTens::const_reference;
	using reference = VTens::reference;
	// constructors
	vector_lift() : tensors() {}
	vector_lift(size_type size) : tensors(size) {}
	vector_lift(size_type size, const Tens &val) : tensors(size, val) {}
	template <class T> friend class vector_lift;
	template <class T> // can copy and move a vector_lift no matter the derived class.
	vector_lift(const vector_lift<T> &other) : tensors(other.tensors)
	{
	}
	template <class T> vector_lift(vector_lift<T> &&other) noexcept : tensors(std::move(other.tensors)) {}
	vector_lift(VTens &other) : tensors(other) {}
	vector_lift(std::initializer_list<Tens> initl) : tensors(initl) {}
	virtual ~vector_lift() = default;

	explicit operator S()
	{
		S out;
		vector_lift *view = &out;
		view->tensors = this->tensors;
		return out;
	}

	void swap(vector_lift &other) noexcept { this->tensors.swap(other.tensors); }
	friend void swap(vector_lift &lhs, vector_lift &rhs) noexcept { lhs.swap(rhs); }
	vector_lift &operator=(vector_lift other) noexcept
	{
		swap(other);
		return *this;
	}
	// interface function from std::vector
	// elements access
	reference at(size_t i) { return tensors.at(i); }
	const_reference at(size_t i) const { return tensors.at(i); }
	reference operator[](size_t i) { return tensors[i]; }
	const_reference operator[](size_t i) const { return tensors[i]; }
	reference front() { return tensors.front(); }
	const_reference front() const { return tensors.front(); }
	reference back() { return tensors.back(); }
	const_reference back() const { return tensors.back(); }
	Tens *data() noexcept { return tensors.data(); }
	const Tens *data() const noexcept { return tensors.data(); }
	// iterators
	iterator begin() noexcept { return tensors.begin(); }
	const_iterator begin() const noexcept { return tensors.begin(); }
	const_iterator cbegin() const noexcept { return tensors.cbegin(); }
	iterator end() noexcept { return tensors.end(); }
	const_iterator end() const noexcept { return tensors.end(); }
	const_iterator cend() const noexcept { return tensors.cend(); }
	reverse_iterator rbegin() noexcept { return tensors.rbegin(); }
	const_reverse_iterator rbegin() const noexcept { return tensors.rbegin(); }
	const_reverse_iterator crbegin() const noexcept { return tensors.crbegin(); }
	reverse_iterator rend() noexcept { return tensors.rend(); }
	const_reverse_iterator rend() const noexcept { return tensors.rend(); }
	const_reverse_iterator crend() const noexcept { return tensors.crend(); }
	// capacity
	[[nodiscard]] auto empty() const noexcept { return tensors.empty(); }
	[[nodiscard]] auto size() const noexcept { return tensors.size(); }
	[[nodiscard]] auto max_size() const noexcept { return tensors.max_size(); }
	void reserve(size_t new_cap) { tensors.reserve(new_cap); }
	[[nodiscard]] auto capacity() const noexcept { return tensors.capacity(); }
	void shrink_to_fit() { tensors.shrink_to_fit(); }
	// modifiers
	void clear() noexcept { tensors.clear(); }
	iterator insert(const_iterator pos, const Tens &val) { return tensors.insert(pos, val); }
	iterator insert(const_iterator pos, Tens &&val) { return tensors.insert(pos, val); }
	iterator insert(const_iterator pos, size_type count, Tens &&val) { return tensors.insert(pos, count, val); }
	template <class InputIT> iterator insert(const_iterator pos, InputIT first, InputIT last)
	{
		return tensors.insert(pos, first, last);
	}
	iterator insert(const_iterator pos, std::initializer_list<Tens> list) { return tensors.insert(pos, list); }
	template <class... Args> iterator emplace(const_iterator pos, Args &&... args)
	{
		return tensors.emplace(pos, std::forward<Args>(args)...);
	}
	iterator erase(const_iterator pos) { return tensors.erase(pos); }
	iterator erase(const_iterator first, const_iterator last) { return tensors.erase(first, last); }
	void push_back(const Tens &val) { tensors.push_back(val); }
	void push_back(Tens &&val) { tensors.push_back(val); }
	template <class... Args> auto emplace_back(Args &&... args)
	{
		return tensors.emplace_back(std::forward<Args>(args)...);
	}
	void pop_back() { tensors.pop_back(); }
	void resize(size_type count) { tensors.resize(count); }
	void resize(size_type count, const Tens &value) { tensors.resize(count, value); }

	// stuff about the tensors, written as covariant functions
	S to(const torch::TensorOptions &options = {}, bool non_blocking = false, bool copy = false,
	     c10::optional<c10::MemoryFormat> memory_format = c10::nullopt) const
	{ // those function can't really exist outside the derived class if empty_copy is private, which it is for MPT, MPS
	  // and MPO.
		S out = empty_copy(static_cast<S const &>(*this));
		auto out_it = out.begin();
		std::for_each(this->cbegin(), this->cend(),
		              [&out_it, &options, non_blocking, copy, memory_format](const auto &atensor) {
			              *(out_it++) = atensor.to(options, non_blocking, copy, memory_format);
		              });
		return out;
	}
	S to(torch::Device device, torch::ScalarType dtype, bool non_blocking = false, bool copy = false,
	     c10::optional<c10::MemoryFormat> memory_format = c10::nullopt) const
	{
		S out = empty_copy(static_cast<S const &>(*this));
		auto out_it = out.begin();
		std::for_each(this->cbegin(), this->cend(),
		              [&out_it, device, dtype, non_blocking, copy, memory_format](const auto &atensor) {
			              *(out_it++) = atensor.to(device, dtype, non_blocking, copy, memory_format);
		              });
		return out;
	}
	S to(torch::ScalarType dtype, bool non_blocking = false, bool copy = false,
	     c10::optional<c10::MemoryFormat> memory_format = c10::nullopt) const
	{
		S out = empty_copy(static_cast<S const &>(*this));
		auto out_it = out.begin();
		std::for_each(this->cbegin(), this->cend(),
		              [&out_it, dtype, non_blocking, copy, memory_format](const auto &atensor) {
			              *(out_it++) = atensor.to(dtype, non_blocking, copy, memory_format);
		              });
		return out;
	}
	S to(caffe2::TypeMeta type_meta, bool non_blocking = false, bool copy = false,
	     c10::optional<c10::MemoryFormat> memory_format = c10::nullopt) const
	{
		S out = empty_copy(static_cast<S const &>(*this));
		auto out_it = out.begin();
		std::for_each(this->cbegin(), this->cend(),
		              [&out_it, type_meta, non_blocking, copy, memory_format](const auto &atensor) {
			              *(out_it++) = atensor.to(type_meta, non_blocking, copy, memory_format);
		              });
		return out;
	}
	S to(const torch::Tensor &other, bool non_blocking = false, bool copy = false,
	     c10::optional<c10::MemoryFormat> memory_format = c10::nullopt) const
	{
		S out = empty_copy(static_cast<S const &>(*this));
		auto out_it = out.begin();
		std::for_each(this->cbegin(), this->cend(),
		              [&out_it, &other, non_blocking, copy, memory_format](const auto &atensor) {
			              *(out_it++) = atensor.to(other, non_blocking, copy, memory_format);
		              });
		return out;
	}
	// inplace version of to, will resolve to any equivalent out-of-place equivalent.
	template <class... Args> void to_(Args &&... args)
	{
		// torch::Tensor is really a pointer type, so copying it is pretty cheap: only the meta data get copied and
		// shuffled around
		// perhaps this could be done more efficiently, but the gains would almost certainly be incommensurate to the
		// efforts.
		auto other = this->to(std::forward<Args>(args)...);
		swap(other);
	}
};
/**
 * @brief tensor train type
 * 
 */
class MPT final : public vector_lift<MPT>
{
  public:
	MPT() : vector_lift<MPT>() {}
	MPT(size_type size) : vector_lift<MPT>(size) {}
	MPT(size_type size, const Tens &val) : vector_lift<MPT>(size, val) {}
	MPT(const MPT &other) : vector_lift<MPT>(other) {}
	MPT(MPT &&other) noexcept : vector_lift<MPT>(std::move(other)) {}
	MPT(std::initializer_list<Tens> initl) : vector_lift<MPT>(initl) {}
	virtual ~MPT() = default;

	void swap(MPT &other) { vector_lift<MPT>::swap(other); }

	friend void swap(MPT &lhs, MPT &rhs) { lhs.swap(rhs); }
	MPT &operator=(MPT other)
	{
		swap(other);
		return *this;
	}

  private:
	static MPT empty_copy(const MPT &in) { return MPT(in.size()); }
};

/*!
 * @brief Class for the matrix product state. A tensor train of rank three tensors, with an orthogonality center.
 *
 * Class for the matrix product state. A tensor train of rank three tensors, with an orthogonality center. 
 */
class MPS final
    : public vector_lift<MPS> // specialization for rank 3 tensors, addionnaly can have an orthogonality center.
{
  public:
	property<size_t, MPS> orthogonality_center; // read only for users
	/**
	 * Check that all the tensors are rank three, and that the size of the bond dimension of neighbouring tensors
	 * matches.
	 */
	bool check_ranks() const;
	static bool check_one(const Tens &tens);
	MPS() : vector_lift<MPS>(), orthogonality_center(0) {}
	MPS(size_type size) : vector_lift<MPS>(size), orthogonality_center(0) {}
	MPS(const MPS &other) : vector_lift(other), orthogonality_center(other.orthogonality_center) {}
	MPS(MPS &&other) noexcept : vector_lift(std::move(other)), orthogonality_center(other.orthogonality_center) {}
	MPS(std::initializer_list<Tens> initl, size_t oc = 0) : vector_lift<MPS>(initl), orthogonality_center(oc)
	{
		bool ok = check_ranks();
		if (not ok)
			throw std::invalid_argument("one or more input Tensors has rank differing from 3 and/or a bond dimension "
			                            "mismatch with its neighbor (dims 0 and 2).");
		if (oc >= size() and oc != 0)
			throw std::invalid_argument("orthogonality center position greater than the number of defined tensors.");
	}

	MPS(size_type lenght, const Tens &val, size_t oc = 0) : vector_lift(lenght, val), orthogonality_center(oc)
	{
		bool ok = check_one(val);
		if (oc >= size() and oc != 0)
			throw std::invalid_argument("orthogonality center position greater than the number of defined tensors.");
		if (not ok)
			throw std::invalid_argument(
			    "Input tensor must be of rank 3 and have equal bond dimensions (dims 0 and 2).");
	}

	MPS(const MPT &other, size_t oc = 0) : vector_lift<MPS>(other), orthogonality_center(oc)
	{
		bool ok = check_ranks();
		if (not ok)
			throw std::invalid_argument("Input MPT is an invalid MPS: one or more Tensors has rank differing from 3 "
			                            "and/or a bond dimension mismatch with its neighbor (dims 0 and 2).");
		if (oc >= size() and oc != 0)
			throw std::invalid_argument("orthogonality center position greater than the number of defined tensors.");
	}
	MPS(MPT &&other, size_t oc = 0) : vector_lift<MPS>(std::move(other)), orthogonality_center(oc)
	{
		bool ok = check_ranks();
		if (not ok)
			throw std::invalid_argument("Input MPT is an invalid MPS: one or more Tensors has rank differing from 3 "
			                            "and/or a bond dimension mismatch with its neighbor (dims 0 and 2).");
		if (oc >= size() and oc != 0)
			throw std::invalid_argument("orthogonality center position greater than the number of defined tensors.");
	}

	virtual ~MPS() = default;

	void swap(MPS &other)
	{
		vector_lift<MPS>::swap(other);
		using std::swap;
		swap(other.orthogonality_center.value, orthogonality_center.value);
	}

	MPS &operator=(MPS other)
	{
		swap(other);
		return *this;
	}

	/**
	 * explicit conversion to a MPT. discards the position of the orthogonality center and lift the MPS constraints.
	 * Be careful, this function does not create a copy of the underlying tensors.
	 */
	explicit operator MPT()
	{ // careful! the underlying data is shared.
		return MPT(vector_lift<MPT>(static_cast<vector_lift<MPS> &>(*this)));
	}

	/**
	 * move the orthogonality center to the position i on the chain.
	 */
	void move_oc(int i);
	friend torch::Scalar details::dmrg_impl(const MPO &hamiltonian, const MPT &two_sites_hamil, MPS &in_out_state,
	                                        const dmrg_options &options, env_holder &Env,
	                                        dmrg_logger &logger); // allow dmrg to manipulate the oc.
  private:
	size_t &oc = orthogonality_center.value; // private direct access to the value variable.
	static MPS empty_copy(const MPS &in) { return MPS(in.size(), in.oc); }
};
inline void swap(MPS &lhs, MPS &rhs) { lhs.swap(rhs); }

/**
 * @brief Class for tensor trains of rank 4.
 *
 */
class MPO final : public vector_lift<MPO> // specialization for rank 4 tensors
{
  public:
	bool check_ranks() const;
	static bool check_one(const Tens &tens);

	MPO() : vector_lift<MPO>() {}
	MPO(size_type size) : vector_lift<MPO>(size) {}
	MPO(const MPO &other) : vector_lift<MPO>(other) {}
	MPO(MPO &&other) noexcept : vector_lift<MPO>(std::move(other)) {}
	MPO(std::initializer_list<Tens> initl) : vector_lift<MPO>(initl)
	{
		bool ok = check_ranks();
		// @cond
		if (not ok)
		{
			throw std::invalid_argument("one or more input Tensors has rank differing from 4 and/or a bond dimension "
			                            "mismatch with its neighbor (dims 0 and 2).");
		}
		// @endcond
	}

	MPO(size_type size, const Tens &val) : vector_lift<MPO>(size, val)
	{
		bool ok = check_one(val);
		if (not ok)
			throw std::invalid_argument(
			    "Input tensor must be of rank 4 and have equal bond dimensions (dims 0 and 2).");
	}

	MPO(const MPT &other) : vector_lift<MPO>(other)
	{
		bool ok = check_ranks();
		if (not ok)
			throw std::invalid_argument("Input MPT is an invalid MPO: one or more Tensors has rank differing from 4 "
			                            "and/or a bond dimension mismatch with its neighbor (dims 0 and 2).");
	}
	MPO(MPT &&other) : vector_lift<MPO>(std::move(other))
	{
		bool ok = check_ranks();
		if (not ok)
			throw std::invalid_argument("Input MPT is an invalid MPO: one or more Tensors has rank differing from 4 "
			                            "and/or a bond dimension mismatch with its neighbor (dims 0 and 2).");
	}

	virtual ~MPO() = default;

	void swap(MPO &other) noexcept { vector_lift<MPO>::swap(other); }

	explicit operator MPT()
	{ // careful! the underlying data is shared.
		return MPT(vector_lift<MPT>(static_cast<vector_lift<MPO> &>(*this)));
	}

	MPO &operator=(MPO other) noexcept
	{
		swap(other);
		return *this;
	}

  private:
	static MPO empty_copy(const MPO &in) { return MPT(in.size()); }
};
inline void swap(MPO &lhs, MPO &rhs) noexcept { lhs.swap(rhs); }

template <class T,
          class Z = std::enable_if_t<std::is_base_of_v<vector_lift<T>, T>>> // Z only serves to prevent call on
                                                                            // something else than MPT,MPS and MPO.
                                                                            void print_dims(const T &mps)
{
	fmt::print("MPS size: ");
	for (const auto &i : mps)
	{
		fmt::print("{},", i.sizes());
	}
	fmt::print("\n");
}

torch::Tensor contract(const MPS &a, const MPS &b, const MPO &obs);
torch::Tensor contract(const MPS &a, const MPS &b);

qtt_TEST_CASE("MPT manipulations")
{
	MPT amps({torch::rand({1, 2, 3}), torch::rand({3, 2, 6}), torch::rand({6, 2, 4})});
	MPT ampo({torch::rand({1, 2, 3, 2}), torch::rand({3, 2, 6, 2}), torch::rand({6, 2, 4, 2})});

	qtt_REQUIRE(amps.size() == 3);
	qtt_REQUIRE(amps.capacity() >= 3);
	qtt_REQUIRE(ampo.size() == 3);
	qtt_REQUIRE(ampo.capacity() >= 3);

	qtt_CHECK_NOTHROW(auto M1 = MPS(amps)); // must assign somewhere so the optimizer doesn't play tricks on us.
	qtt_CHECK_NOTHROW(auto M2 = MPO(ampo));
	qtt_CHECK_THROWS_AS(auto M3 = MPS(ampo), std::invalid_argument);
	qtt_CHECK_THROWS_AS(auto M4 = MPO(amps), std::invalid_argument);
	qtt_CHECK_THROWS_AS(auto M5 = MPS(amps, -1), std::invalid_argument);
}

qtt_TEST_CASE("MPS basic manipulation")
{
	torch::set_default_dtype(torch::scalarTypeToTypeMeta(
	    torch::kFloat64)); // otherwise the type promotion always goes to floats when promoting a tensor
	                       // we must make sure side effects don't leak out of the test when compiling executable other
	                       // than the test itself.

	qtt_SUBCASE("Bad constructions")
	{
		qtt_CHECK_THROWS_AS(MPS(5, torch::rand({1, 3, 1}), 10), std::invalid_argument);
		qtt_CHECK_THROWS_AS(MPS(5, torch::rand({1, 3, 5}), 1), std::invalid_argument);
	}
	qtt_SUBCASE("Construction")
	{
		MPS A({torch::rand({1, 4, 3}), torch::rand({3, 4, 1})});

		qtt_REQUIRE(A.size() == 2);
		qtt_REQUIRE(A.capacity() >= 2);
		qtt_CHECK(A.orthogonality_center == 0);
		auto size_0 = std::vector<int64_t>{1, 4, 3};
		auto size_1 = std::vector<int64_t>{3, 4, 1};
		qtt_CHECK(A[0].sizes() == size_0);
		qtt_CHECK(A[1].sizes() == size_1);
		// size_0 = A[0].sizes().vec(); //copy the current size of A[0];
		qtt_SUBCASE("moving orthogonality center")
		{
			auto norm2 =
			    torch::tensordot(torch::tensordot(torch::tensordot(A[0], A[0].conj(), {0, 1}, {0, 1}), A[1], {0}, {0}),
			                     A[1].conj(), {0, 1, 2}, {0, 1, 2});
			// qtt_REQUIRE(norm2.sizes().size()==1);
			// qtt_REQUIRE(norm2.sizes()[0] == 1);
			qtt_CHECK_THROWS_AS(A.move_oc(2), std::invalid_argument);
			A.move_oc(1);
			qtt_CHECK(A.orthogonality_center == 1);
			qtt_CHECK(A[0].sizes() == size_0);
			qtt_CHECK(A[1].sizes() == size_1);
			auto CC = torch::tensordot(A[0], A[0].conj(), {0, 1}, {0, 1});
			qtt_CHECK(torch::allclose(CC, torch::eye(3)));
			// the default tolerances for the close() familly of function is strange. It's too low for float comparison,
			// but very high for double comparison.
			auto canon_norm2 = torch::tensordot(A[1], A[1].conj(), {0, 1, 2}, {0, 1, 2});
			// qtt_REQUIRE(canon_norm2.sizes().size()==1);
			// qtt_REQUIRE(canon_norm2.sizes()[0] == 1);
			qtt_CHECK(
			    norm2.item().to<double>() ==
			    doctest::Approx(canon_norm2.item().to<double>())); // the dynamical nature of the type make extracting a
			                                                       // concrete value a bit complicated.
		}
		qtt_SUBCASE("conversion to MPT")
		{
			auto B = MPT(A);
			qtt_CHECK(torch::equal(B[0], A[0]));
		}
	}
}
qtt_TEST_CASE("MPO basic manipulation")
{
	qtt_SUBCASE("Bad constructions") { qtt_CHECK_THROWS_AS(MPO(5, torch::rand({1, 3, 5, 3})), std::invalid_argument); }
	qtt_SUBCASE("Construction")
	{
		MPO A({torch::rand({1, 4, 3, 4}), torch::rand({3, 4, 1, 4})});

		qtt_REQUIRE(A.size() == 2);
		qtt_REQUIRE(A.capacity() >= 2);

		auto size_0 = std::vector<int64_t>{1, 4, 3, 4};
		auto size_1 = std::vector<int64_t>{3, 4, 1, 4};
		qtt_CHECK(A[0].sizes() == size_0);
		qtt_CHECK(A[1].sizes() == size_1);
		size_0 = A[0].sizes().vec(); // copy the current size of A[0];
		qtt_SUBCASE("conversion to MPT")
		{
			auto B = MPT(A);
			qtt_CHECK(torch::equal(B[0], A[0]));
			qtt_CHECK(torch::equal(B[1], A[1]));
		}
	}
}
qtt_TEST_CASE("contraction equivalence tests")
{
	torch::set_default_dtype(torch::scalarTypeToTypeMeta(
	    torch::kFloat64)); // otherwise the type promotion always goes to floats when promoting a tensor
	MPS state(2, torch::rand({2, 2, 2}));
	state[0] = state[0].reshape({1, 4, 2});
	state[1] = state[1].reshape({2, 4, 1});
	MPO op(2, torch::rand({2, 4, 2, 4}));
	op[0] = op[0].reshape({1, 4, 4, 4});
	op[1] = op[1].reshape({4, 4, 1, 4});

	MPO idtt(2, torch::diag(torch::ones({4})).reshape({1, 4, 1, 4}));

	auto conc_idtt = torch::tensordot(idtt[0], idtt[1], {2}, {0}).permute({0, 1, 3, 4, 2, 5}).reshape({16, 16});
	qtt_REQUIRE(torch::allclose(conc_idtt, torch::diag(torch::ones({16}))));
	auto conc_op = torch::tensordot(op[0], op[1], {2}, {0}).permute({0, 1, 3, 4, 2, 5}).reshape({16, 16});
	auto conc_state = torch::tensordot(state[0], state[1], {2}, {0}).reshape({16});

	auto norm2 = torch::tensordot(conc_state, conc_state, {0}, {0});
	auto norm2_test = contract(state, state);
	auto norm2_idtt = contract(state, state, idtt);
	// fmt::print("{}\n",norm2);
	// fmt::print("{}\n",norm2_test);
	// fmt::print("{}\n",norm2_idtt);
	qtt_CHECK(torch::allclose(norm2, norm2_test));
	qtt_CHECK(torch::allclose(norm2, norm2_idtt));

	auto aver = torch::tensordot(conc_state, torch::tensordot(conc_op, conc_state, {1}, {0}), {0}, {0});
	auto aver_norm2 = torch::tensordot(conc_state, torch::tensordot(conc_idtt, conc_state, {1}, {0}), {0}, {0});

	auto aver_test = contract(state, state, op);
	// fmt::print("{}\n", aver);
	// fmt::print("{}\n", aver_test);
	// fmt::print("{}\n", aver_norm2);
	qtt_CHECK(torch::allclose(aver, aver_test));
}
} // namespace quantt
#endif /* ADA5A359_8ACF_448D_91BC_09C085F510CC */
