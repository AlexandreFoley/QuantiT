/*
 * File: btensor.h
 * Project: quantt
 * File Created: Thursday, 1st October 2020 10:54:53 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Friday, 8th January 2021 12:02:08 pm
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2021 Alexandre Foley
 * All rights reserved
 */

#ifndef D49FFA60_85C4_431A_BA62_9B1D30D67E86
#define D49FFA60_85C4_431A_BA62_9B1D30D67E86

#include "Conserved/Composite/quantity.h"
#include "Conserved/Composite/quantity_vector.h"
#include "Conserved/quantity.h"
#include "blockTensor/flat_map.h"
#include "boost/stl_interfaces/iterator_interface.hpp"
#include "boost/stl_interfaces/view_interface.hpp"
#include "property.h"
#include "torch_formatter.h"
#include <algorithm>
#include <exception>
#include <torch/torch.h>
#include <vector>

namespace quantt
{
	class block_qtt_view;
	/**
 * @brief btensor is a type meant to represent block sparse tensor with conservation laws.
 * The conservation law determines which block can or cannot be non-nul.
 * Each block is itself a tensor of the same rank as the overall tensor.
 * 
 * Each dimensions of the tensor are separated in sections with independent sizes, each of those section 
 * has an associated conserved quantity. The blocks are formed by the intersection of those section.
 * The only block that can contain non-zero values are those the satisfy the selection rule: the sum over the
 * dimension of the conserved quantity must equal a specififed value (the selection rule).
 * 
 * 
 * exemple with a rank 2 tensor (matrix) of the inner structure
 * of this type:
 * \verbatim
 *             S0,0 │ S0,1 │ S0,2 │ S0,3 
 *            ╔═════╪══════╪══════╪═════╗
 *            ║     │      │      │     ║
 *        S1,0║(0,0)│ (0,1)│ (0,2)│(0,3)║
 *            ║     │      │      │     ║
 *           ─╫─────┼──────┼──────┼─────╢
 *            ║     │      │      │     ║
 *            ║     │      │      │     ║
 *        S1,1║(1,0)│ (1,1)│ (1,2)│(1,3)║
 *            ║     │      │      │     ║
 *            ║     │      │      │     ║
 *            ║     │      │      │     ║
 *           ─╫─────┼──────┼──────┼─────╢
 *            ║     │      │      │     ║
 *            ║     │      │      │     ║
 *        S1,2║(2,0)│ (2,1)│ (2,2)│(2,3)║
 *            ║     │      │      │     ║
 *            ║     │      │      │     ║
 *            ║     │      │      │     ║
 *            ╚═════╧══════╧══════╧═════╝
 * 
 * \endverbatim
 * In the preceding exemple, the rows are separated in 4 sections, and the columns in 3 sections.
 * This make up to 12 blocks, that we label by section.
 * Let's consider that the conserved quantity is simply an integer under the addition,that the column sections [-2,-1,1], the row sections 
 * have the conserved quantity [1,2,3,-1] and the selection rule is 0.
 * In that case, only the blocks [(1,0),(0,1),(2,3)] can be non-zero.
 * 
 */
	class btensor
	{
	public:
		using index_list = std::vector<int64_t>;
		using block_list_t = flat_map<index_list, torch::Tensor>;
		using init_list_t = std::initializer_list<std::initializer_list<std::tuple<size_t, any_quantity>>>;
		using Scalar = torch::Scalar;
		property<any_quantity, btensor, any_quantity_cref> selection_rule; //dmrjulia equiv: the flux.
		/**
	 * @brief Construct a new btensor object
	 * 
	 * @param dir_block_size_cqtt a nested list of pair of section size and conserved quantities. The number of element 
	 * in the first is level is the rank of the tensor. The number of elements in the second level is the number of 
	 * section for that dimension of the tensor
	 * @param selection_rule 
	 */
		btensor(init_list_t dir_block_size_cqtt, any_quantity_cref selection_rule);
		btensor(init_list_t dir_block_size_cqtt, any_quantity_cref selection_rule, size_t num_blocks);

		btensor(index_list _sections_by_dim, any_quantity_vector _c_vals, index_list _section_sizes, any_quantity _sel_rule);
		btensor(const btensor &other) : selection_rule((other.selection_rule.value)), rank(other.rank),
										sections_by_dim((other.sections_by_dim)), sections_sizes((other.sections_sizes)),
										blocks((other.blocks)), c_vals((other.c_vals)) {}
		btensor(btensor &&other) : selection_rule(std::move(other.selection_rule.value)), rank(other.rank),
								   sections_by_dim(std::move(other.sections_by_dim)), sections_sizes(std::move(other.sections_sizes)),
								   blocks(std::move(other.blocks)), c_vals(std::move(other.c_vals)) {}
		btensor &operator=(btensor other)
		{
			swap(other);
			return *this;
		}
		btensor() = default;
	/**
	 * @brief Construct a new btensor object. construct from raw structure elements. Avoid using this constructor if you can.
	 * 
	 * @param _rank : the number of dimension of the tensor
	 * @param _blocks : list of pair<position,sub-tensor>, the position is stored in a block index
	 * @param _sections_by_dims : number of section for each dimension of the tensor
	 * @param _section_sizes : number of element for each section of each dimension
	 * @param _c_vals : conserved quantity associated to each of the section in each of the dimension
	 * @param _sel_rule : overall selection rule, the sum over the dimension of the conserved quantities of a given block must equal this value for a block to be allowed to differ from zero. 
	 */
		btensor(size_t _rank, block_list_t _blocks, index_list _sections_by_dims, index_list _sections_sizes,
				any_quantity_vector _c_vals, any_quantity _sel_rule);

		/**
	 * @brief increment a block index for this tensor
	 * 
	 * @param block_index reference to the block index to increment.
	 */
		void block_increment(btensor::index_list &block_index) const;
		static size_t btensor_compute_max_size(const btensor &btens, size_t max = std::numeric_limits<size_t>::max());
		static void add_tensor_check(const btensor &a, const btensor &b);

		void swap(btensor &);
		//utility classes
		template <class val_iter>
		struct block_prop_iter;
		template <class const_val_iter, class val_iter>
		struct const_block_prop_iter;
		template <class iter>
		struct block_prop_view;
		template <class const_iterator, class iterator>
		struct const_block_prop_view;
		using block_qtt_iter = block_prop_iter<any_quantity_vector::iterator>;
		using const_block_qtt_iter = const_block_prop_iter<any_quantity_vector::const_iterator, any_quantity_vector::iterator>;
		using block_qtt_view = block_prop_view<block_qtt_iter>;
		using const_block_qtt_view = const_block_prop_view<const_block_qtt_iter, block_qtt_iter>;
		using block_size_iter = block_prop_iter<index_list::iterator>;
		using const_block_size_iter = const_block_prop_iter<index_list::const_iterator, index_list::iterator>;
		using block_size_view = block_prop_view<block_size_iter>;
		using const_block_size_view = const_block_prop_view<const_block_size_iter, block_size_iter>;

		bool block_conservation_rule_test(index_list block_index) const;

		size_t section_size(size_t dim, size_t section) const;
		std::tuple<index_list::const_iterator, index_list::const_iterator> section_sizes(size_t dim) const;
		std::tuple<any_quantity_vector::const_iterator, any_quantity_vector::const_iterator> section_cqtts(size_t dim) const;
		std::tuple<index_list::const_iterator, index_list::const_iterator, any_quantity_vector::const_iterator, any_quantity_vector::const_iterator>
		section_sizes_cqtts(size_t dim) const;

		any_quantity_cref section_conserved_qtt(size_t dim, size_t section) const;
		std::tuple<any_quantity_vector::const_iterator, any_quantity_vector::const_iterator> section_conserved_qtt_range(size_t index) const;
		std::tuple<size_t, any_quantity_cref> section_size_cqtt(size_t dim, size_t section) const;
		//block accessor
		torch::Tensor &block_at(const index_list &); //throws if the block isn't present.
		torch::Tensor &block(const index_list &);	 //create the block if it isn't present and allowed
		const_block_qtt_view block_quantities(index_list block_index) const;
		const_block_size_view block_sizes(index_list block_index) const;
		size_t dim() const { return rank; }
		size_t section_number(size_t dim) const { return sections_by_dim[dim]; }
		// block_qtt_view block_quantities(index_list block_index);

		//iterator
		block_list_t::const_iterator begin() const { return blocks.begin(); }
		block_list_t::const_iterator end() const { return blocks.end(); }
		block_list_t::const_iterator cbegin() const { return blocks.cbegin(); }
		block_list_t::const_iterator cend() const { return blocks.cend(); }
		block_list_t::iterator begin() { return blocks.begin(); }
		block_list_t::iterator end() { return blocks.end(); }
		block_list_t::reverse_iterator rbegin() { return blocks.rbegin(); }
		block_list_t::reverse_iterator rend() { return blocks.rend(); }
		block_list_t::const_reverse_iterator rbegin() const { return blocks.rbegin(); }
		block_list_t::const_reverse_iterator rend() const { return blocks.rend(); }
		block_list_t::const_reverse_iterator crbegin() const { return blocks.crbegin(); }
		block_list_t::const_reverse_iterator crend() const { return blocks.crend(); }

		torch::Tensor to_dense() const;
		//properties check
		static std::string check_tensor(const btensor &);
		static void throw_bad_tensor(const btensor &);
		//Algebra !!Attention!! most of this stuff isn't implemented.
		btensor add(const btensor &other, Scalar alpha = 1) const;
		btensor add(btensor &&other, Scalar alpha = 1) const;
		btensor &add_(const btensor &other, Scalar alpha = 1);
		btensor &add_(btensor &&other, Scalar alpha = 1);
		btensor &operator+=(const btensor &other) { return add_(other); }
		btensor &operator+=(btensor &&other) { return add_(std::move(other)); }
		btensor &operator-=(const btensor &other) { return add_(other, -1); }
		btensor &operator-=(btensor &&other) { return add_(std::move(other), -1); }
		btensor addmv(const btensor &mat, const btensor &vec, Scalar beta = 1, Scalar alpha = 1) const;
		btensor &addmv(const btensor &mat, const btensor &vec, Scalar beta = 1);
		btensor addmm(const btensor &mat, const btensor &mat2, Scalar beta = 1, Scalar alpha = 1) const;
		btensor &addmm_(const btensor &mat, const btensor &mat2, Scalar beta = 1, Scalar alpha = 1);
		btensor addbmm(const btensor &mat, const btensor &mat2, Scalar beta = 1, Scalar alpha = 1) const;
		btensor &addbmm_(const btensor &mat, const btensor &mat2, Scalar beta = 1, Scalar alpha = 1);
		btensor &addcdiv_(const btensor &tensor1, const btensor &tensor2, Scalar beta = 1);
		btensor addcdiv(const btensor &tensor1, const btensor &tensor2, Scalar beta = 1);
		btensor &addcmul_(const btensor &tensor1, const btensor &tensor2, Scalar beta = 1);
		btensor addcmul(const btensor &tensor1, const btensor &tensor2, Scalar beta = 1);
		btensor baddbmm(const btensor &bathc1, const btensor &batch2, Scalar beta = 1, Scalar alpha = 1) const;
		btensor &baddbmm_(const btensor &bathc1, const btensor &batch2, Scalar beta = 1, Scalar alpha = 1);
		btensor bmm(const btensor &mat) const;
		btensor dot(const btensor &other) const;
		btensor vdot(const btensor &other) const;
		btensor kron(const btensor &other) const;
		btensor matmul(const btensor &other) const;
		btensor mm(const btensor &other) const;
		btensor mul(const btensor &other) const;
		btensor &mul_(const btensor &other);
		btensor mul(Scalar other) const;
		btensor &mul_(Scalar other);
		btensor multiply(const btensor &other) const;
		btensor &multiply_(const btensor &other);
		btensor multiply(Scalar other) const {return mul(other);}
		btensor &multiply_(Scalar other) {return mul_(other);}
		btensor mv(const btensor &vec) const;
		btensor permute(torch::IntArrayRef) const;
		btensor &permute_(torch::IntArrayRef);
		btensor reshape(torch::IntArrayRef shape) const;
		btensor reshape_as(torch::IntArrayRef shape) const;
		btensor transpose(int64_t dim0, int64_t dim1) const;
		btensor transpose(torch::Dimname dim0, torch::Dimname dim1) const;
		btensor &transpose_(int64_t dim0, int64_t dim1);
		btensor sub(const btensor &other, Scalar alpha = 1) const{return sub(other,-alpha);}
		btensor &sub_(const btensor &other, Scalar alpha = 1){return sub_(other,-alpha);}
		btensor sub(btensor &&other, Scalar alpha = 1) const {return sub(std::move(other),-alpha);}
		btensor &sub_(btensor &&other, Scalar alpha = 1){return sub_(std::move(other),-alpha);}
		btensor sub(Scalar other, Scalar alpha = 1) const;
		btensor &sub_(Scalar other, Scalar alpha = 1);
		btensor subtract(const btensor &other, Scalar alpha = 1) const { return sub(other, alpha); }
		btensor &subtract_(const btensor &other, Scalar alpha = 1) { return sub_(other, alpha); }
		btensor subtract(Scalar other, Scalar alpha = 1) const { return sub(other, alpha); }
		btensor &subtract_(Scalar other, Scalar alpha = 1) { return sub_(other, alpha); }
		btensor tensordot(const btensor &other, torch::IntArrayRef dim_self, torch::IntArrayRef dims_other) const;
		btensor tensorgdot(const btensor &mul1, const btensor &mul2, torch::IntArrayRef dims1, torch::IntArrayRef dims2, Scalar beta = 1, Scalar alpha = 1) const;
		btensor &tensorgdot_(const btensor &mul1, const btensor &mul2, torch::IntArrayRef dims1, torch::IntArrayRef dims2, Scalar beta = 1, Scalar alpha = 1);

	private:
		size_t rank;
		index_list sections_by_dim;
		index_list sections_sizes; //for non-empty slices, this is strictly redundent: the information could be found by inspecting the blocks
		//truncation should remove any and all empty slices, but user-written tensor could have empty slices.
		block_list_t blocks;		//
		any_quantity_vector c_vals; //dmrjulia equiv: QnumSum in the QTensor class. This structure doesn't need the full list (QnumMat)
		friend struct fmt::formatter<quantt::btensor>;
	};

	size_t get_refcount(const torch::Tensor &tens);

	inline void swap(btensor &a, btensor &b)
	{
		a.swap(b);
	}

	qtt_TEST_CASE("btensor")
	{
		qtt_SUBCASE("contruction")
		{
			using cqt = conserved::C<5>;
			using index = btensor::index_list;
			any_quantity flux(cqt(0));

			btensor A({{{2, cqt(0)}, {3, cqt(1)}},
					   {{2, cqt(0)}, {3, cqt(1).inverse()}}},
					  flux);
			qtt_CHECK(A.end() - A.begin() == 0);
			auto A00 = torch::rand({2, 2});
			auto A11 = torch::rand({3, 3});
			A.block({0, 0}) = A00;
			A.block({1, 1}) = A11;
			qtt_REQUIRE_NOTHROW(btensor::throw_bad_tensor(A));
			qtt_CHECK(A.end() - A.begin() == 2);
			qtt_CHECK_NOTHROW(A.block_at({0, 0}));
			qtt_CHECK_THROWS_AS(A.block_at({1, 0}), std::out_of_range);	 //there's no block here.
			qtt_CHECK_THROWS_AS(A.block({1, 0}), std::invalid_argument); //and we can't create one.
			qtt_CHECK(btensor::check_tensor(A) == "");
			// fmt::print("{}", A);
			qtt_SUBCASE("tensor contraction")
			{
				btensor B({{{3, cqt(4)}, {2, cqt(0)}}, {{2, cqt(0)}, {3, cqt(1)}}, {{1, cqt(1)}, {3, cqt(0)}}}, any_quantity(cqt(1)));
				auto B100 = torch::rand({2, 2, 1});
				auto B010 = torch::rand({3, 3, 1});
				auto B111 = torch::rand({2, 3, 3});
				B.block({0, 1, 0}) = B010;
				B.block({1, 1, 1}) = B111;
				B.block({1, 0, 0}) = B100;
				qtt_REQUIRE_NOTHROW(btensor::throw_bad_tensor(B));
				btensor C;
				auto C100 = torch::tensordot(A11, B010, {1}, {1});
				auto C010 = torch::tensordot(A00, B100, {1}, {1});
				auto C111 = torch::tensordot(A11, B111, {1}, {1});
				qtt_CHECK_NOTHROW(C = A.tensordot(B, {1}, {1}));
				qtt_REQUIRE_NOTHROW(C.block_at({0, 1, 0}));
				qtt_REQUIRE_NOTHROW(C.block_at({1, 0, 0}));
				qtt_REQUIRE_NOTHROW(C.block_at({1, 1, 1}));
				qtt_CHECK(torch::allclose(C.block_at({0, 1, 0}), C010));
				qtt_CHECK(torch::allclose(C.block_at({1, 0, 0}), C100));
				qtt_CHECK(torch::allclose(C.block_at({1, 1, 1}), C111));
			}
			qtt_SUBCASE("addition")
			{
				btensor AA00({{{2, cqt(0)}, {3, cqt(1)}},
							  {{2, cqt(0)}, {3, cqt(1).inverse()}}},
							 flux);
				btensor AA11({{{2, cqt(0)}, {3, cqt(1)}},
							  {{2, cqt(0)}, {3, cqt(1).inverse()}}},
							 flux);
				AA00.block({0,0})=A00;
				AA11.block({1,1})=A11;
				AA00.add_(AA11);
				auto B00 = 2 * A00;
				auto B11 = 2 * A11;
				auto AP00 = 3 * A00; //A post add_
				auto AP11 = 3 * A11; //A post add_
				auto C00 = 5 * A00;
				auto C11 = 5 * A11;
				auto B = A.add(A);
				auto C = A.add(B);
				C.add_(B);
				qtt_CHECK(torch::allclose(A.block_at({0, 0}), A00));
				qtt_CHECK(torch::allclose(A.block_at({1, 1}), A11)); //A unchanged
				qtt_CHECK_THROWS(A.block_at({0, 1}));
				qtt_CHECK_THROWS(A.block_at({1, 0}));
				qtt_CHECK_THROWS(B.block_at({0, 1}));
				qtt_CHECK_THROWS(B.block_at({1, 0}));
				qtt_CHECK_THROWS(C.block_at({0, 1}));
				qtt_CHECK_THROWS(C.block_at({1, 0}));
				qtt_REQUIRE_NOTHROW(A.block_at({0, 0}));
				qtt_REQUIRE_NOTHROW(A.block_at({1, 1}));
				qtt_REQUIRE_NOTHROW(B.block_at({0, 0}));
				qtt_REQUIRE_NOTHROW(B.block_at({1, 1}));
				qtt_REQUIRE_NOTHROW(C.block_at({0, 0}));
				qtt_REQUIRE_NOTHROW(C.block_at({1, 1}));
				qtt_CHECK(torch::allclose(B.block_at({0, 0}), B00));
				qtt_CHECK(torch::allclose(B.block_at({1, 1}), B11));
				qtt_CHECK(torch::allclose(C.block_at({0, 0}), C00));
				qtt_CHECK(torch::allclose(C.block_at({1, 1}), C11));
				qtt_REQUIRE_NOTHROW(AA00.block_at({0,0}));
				qtt_CHECK_THROWS(AA00.block_at({1,0}));
				qtt_REQUIRE_NOTHROW(AA00.block_at({1,1}));
				qtt_CHECK_THROWS(AA00.block_at({0,1}));
				A.add_(std::move(B)); //this destroys B, so any state verification on it must be done.
				qtt_CHECK(torch::allclose(A.block_at({0, 0}), AP00));
				qtt_CHECK(torch::allclose(A.block_at({1, 1}), AP11));
				qtt_CHECK(get_refcount(AA00.block_at({1,1})) == 1);
			}
		}
	}
	template <class value_iterator>
	struct btensor::block_prop_iter
		: boost::stl_interfaces::iterator_interface<
			  block_prop_iter<value_iterator>, std::bidirectional_iterator_tag, typename value_iterator::value_type,
			  typename value_iterator::reference, typename value_iterator::pointer, typename value_iterator::difference_type>
	{
		using il_iter = typename btensor::index_list::const_iterator;
		using ValueIterator = value_iterator;

	private:
		value_iterator val_iter;
		il_iter section_by_dim;
		il_iter block_index;

	public:
		block_prop_iter(value_iterator val_it, il_iter sect_by_dim, il_iter block_ind)
			: val_iter(val_it), section_by_dim(sect_by_dim), block_index(block_ind) {}
		block_prop_iter() : val_iter(), section_by_dim(), block_index() {}
		using base_type = boost::stl_interfaces::iterator_interface<
			block_prop_iter, std::bidirectional_iterator_tag, typename value_iterator::value_type,
			typename value_iterator::reference, typename value_iterator::pointer, typename value_iterator::difference_type>;
		typename base_type::reference operator*()
		{
			return *(val_iter + *block_index);
		}
		bool operator==(const block_prop_iter &other)
		{ //comparison between the iterators of 2 different view object will alway return not equal, enev if constructed from the same block index
			return val_iter == other.val_iter &&
				   section_by_dim == other.section_by_dim &&
				   block_index == other.block_index;
		}
		block_prop_iter &operator++()
		{
			++block_index;
			val_iter += *section_by_dim;
			++section_by_dim;
			return *this;
		}
		block_prop_iter &operator--()
		{
			--section_by_dim;
			val_iter -= *section_by_dim;
			--block_index;
			return *this;
		}
		using base_type::operator++;
		using base_type::operator--;

		const value_iterator &get_val_iter() const
		{
			return val_iter;
		}
		const il_iter &get_section() const
		{
			return section_by_dim;
		}
		const il_iter &get_bi() const
		{
			return block_index;
		}
	};
	template <class const_value_iter, class value_iter>
	struct btensor::const_block_prop_iter : btensor::block_prop_iter<const_value_iter>
	{
		const_block_prop_iter(const block_prop_iter<value_iter> &other)
			: block_prop_iter<const_value_iter>(other.get_val_iter(), other.get_section(), other.get_bi())
		{
		}
		using block_prop_iter<const_value_iter>::block_prop_iter;
		const_block_prop_iter &operator++()
		{
			block_prop_iter<const_value_iter>::operator++();
			return *this;
		}
		const_block_prop_iter &operator--()
		{
			block_prop_iter<const_value_iter>::operator--();
			return *this;
		}
		const_block_prop_iter &operator++(int)
		{
			auto out = *this;
			block_prop_iter<const_value_iter>::operator++();
			return out;
		}
		const_block_prop_iter &operator--(int)
		{
			auto out = *this;
			block_prop_iter<const_value_iter>::operator--();
			return out;
		}
	};

	template <class iterator>
	struct btensor::block_prop_view : boost::stl_interfaces::view_interface<block_prop_view<iterator>>
	{
		block_prop_view() = default;
		block_prop_view(typename iterator::ValueIterator val_first, typename iterator::ValueIterator val_last,
						index_list::const_iterator section_by_dim_begin,
						index_list::const_iterator section_by_dim_end, index_list _block_index)
			: block_index(std::move(_block_index)), first(val_first, section_by_dim_begin, block_index.begin()),
			  last(val_last, section_by_dim_end, block_index.end()) {}
		block_prop_view(typename iterator::ValueIterator val_first, typename iterator::ValueIterator val_last,
						const index_list &section_by_dim, index_list _block_index)
			: block_index(std::move(_block_index)), first(val_first, section_by_dim.begin(), block_index.begin()), last(val_last, section_by_dim.end(), block_index.end()) {}
		auto begin() const { return first; }
		auto end() const { return last; }

		const index_list &get_index() const
		{
			return block_index;
		}

	private:
		btensor::index_list block_index;
		iterator first;
		iterator last;
	};
	template <class const_iterator, class iterator>
	struct btensor::const_block_prop_view : btensor::block_prop_view<const_iterator>
	{
		using block_prop_view<const_iterator>::block_prop_view;
		const_block_prop_view(const btensor::block_prop_view<iterator> &other)
			: block_prop_view<const_iterator>(other.begin().get_val_iter(), other.end().get_val_iter(), other.begin().get_section(),
											  other.end().get_section(), other.get_index()) {}
	};

} // namespace quantt

template <>
struct fmt::formatter<quantt::btensor>
{
	constexpr auto parse(format_parse_context &ctx)
	{
		auto it = ctx.begin(), end = ctx.end();
		if (it){
		if (it != end and *it != '}')
			throw format_error("invalid format, no formatting option for quantt::btensor");
		if (*it != '}')
			throw format_error("invalid format,closing brace missing");
}
		// Return an iterator past the end of the parsed range:
		return it;
	}

	template <class FormatContext>
	auto format(const quantt::btensor &t, FormatContext &ctx)
	{
		constexpr auto btensor_fmt_string =
			"btensor rank {}\n selection rule {}\n number of sections by dim {}\n "
			"sections sizes {}\n sections conserved quantity {}\n";
		constexpr auto btensor_fmt_blocks = "block at {}\n {}\n";
		auto out = format_to(ctx.out(), btensor_fmt_string, t.rank, t.selection_rule,
							 t.sections_by_dim, t.sections_sizes, t.c_vals);
		for (const auto &b : t.blocks)
		{
			out = format_to(out, btensor_fmt_blocks, b.first, b.second);
		}
		return out;
	}
};

#endif /* D49FFA60_85C4_431A_BA62_9B1D30D67E86 */
