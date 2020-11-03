/*
 * File: blocktensor.h
 * Project: quantt
 * File Created: Thursday, 1st October 2020 10:54:53 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Thursday, 1st October 2020 10:54:53 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#ifndef D49FFA60_85C4_431A_BA62_9B1D30D67E86
#define D49FFA60_85C4_431A_BA62_9B1D30D67E86

#include "Conserved/Composite/quantity.h"
#include "Conserved/Composite/quantity_vector.h"
#include "Conserved/quantity.h"
#include "blockTensor/flat_map.h"
#include "property.h"
#include "boost/stl_interfaces/iterator_interface.hpp"
#include "boost/stl_interfaces/view_interface.hpp"
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
 * Let's consider that the conserved quantity is simply an integer,that the column section [-2,-1,1], the row section 
 * have the conserved quantity [1,2,3,-1] and the selection rule is 0.
 * In that case, only the blocks [(1,0),(0,1),(2,3)] can be non-zero.
 * 
 */
class btensor
{
public:
	using index_list = std::vector<size_t>;
	using block_list_t = flat_map<index_list, torch::Tensor>;
	using init_list_t = std::initializer_list<std::initializer_list<std::tuple<size_t, any_quantity>>>;
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

	/**
	 * @brief Construct a new btensor object. construct from raw structure elements. Avoid using this constructor if you can.
	 * 
	 * @param _rank : the number of dimension of the tensor
	 * @param _blocks : list of pair<position,sub-tensor>, the position is stored in a block index
	 * @param _block_shapes : number of section for each dimension of the tensor
	 * @param _block_sizes : number of element for each section of each dimension
	 * @param _c_vals : conserved quantity associated to each of the section in each of the dimension
	 * @param _sel_rule : overall selection rule, the sum over the dimension of the conserved quantities of a given block must equal this value for a block to be allowed to differ from zero. 
	 */
	btensor(size_t _rank, block_list_t _blocks, index_list _sections_by_dim, index_list _block_shapes, index_list _block_sizes,
	        any_quantity_vector _c_vals, any_quantity _sel_rule);

	size_t section_size(size_t index, size_t block) const;
	any_quantity_cref section_conserved_qtt(size_t index, size_t block) const;
	std::tuple<size_t, any_quantity_cref> section_size_cqtt(size_t index, size_t block) const;

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
	using const_block_size_iter = const_block_prop_iter<index_list::const_iterator,index_list::iterator>;
	using block_size_view = block_prop_view<block_size_iter>;
	using const_block_size_view = const_block_prop_view<const_block_size_iter,block_size_iter>;


	//accessor
	torch::Tensor block_at(const index_list&); //throws if the block isn't present.
	torch::Tensor block(const index_list&);    //create the block if it isn't present and allowed
	const_block_qtt_view block_quantities(index_list block_index) const;
	const_block_size_view block_size(index_list block_index) const;
	// block_qtt_view block_quantities(index_list block_index);

private:
	size_t rank;
	index_list sections_by_dim;
	index_list sections_sizes; //for non-empty slices, this is strictly redundent: the information could be found by inspecting the blocks
	//truncation should remove any and all empty slices, but user-written tensor could have empty slices.
	block_list_t blocks;        //
	any_quantity_vector c_vals; //dmrjulia equiv: QnumSum in the QTensor class. This structure doesn't need the full list (QnumMat)
	static std::string check_tensor(const btensor&);
};

qtt_TEST_CASE("btensor")
{
	qtt_SUBCASE("contruction")
	{
		using cqt = conserved::C<5>;
		any_quantity flux(cqt(0));

		btensor A({{{2, cqt(0)}, {3, cqt(1)}},
		           {{2, cqt(0)}, {3, cqt(-1)}}},
		          flux);
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
	bool operator==(const block_prop_iter& other)
	{
		return val_iter == other.val_iter && section_by_dim == other.section_by_dim &&
		       block_index == other.block_index;
	}
	block_prop_iter& operator++()
	{
		++block_index;
		val_iter += *section_by_dim;
		++section_by_dim;
		return *this;
	}
	block_prop_iter& operator--()
	{
		--section_by_dim;
		val_iter -= *section_by_dim;
		--block_index;
		return *this;
	}
	using base_type::operator++;
	using base_type::operator--;

	const value_iterator& get_val_iter() const
	{
		return val_iter;
	}
	const il_iter& get_section() const
	{
		return section_by_dim;
	}
	const il_iter& get_bi() const
	{
		return block_index;
	}
};
template <class const_value_iter, class value_iter>
struct btensor::const_block_prop_iter : btensor::block_prop_iter<const_value_iter>
{
	const_block_prop_iter(const block_prop_iter<value_iter>& other)
	    : block_prop_iter<const_value_iter>(other.get_val_iter(), other.get_section(), other.get_bi())
	{
	}
	using block_prop_iter<const_value_iter>::block_prop_iter;
	const_block_prop_iter& operator++()
	{
		block_prop_iter<const_value_iter>::operator++(); 
		return *this;
	}
	const_block_prop_iter& operator--()
	{
		block_prop_iter<const_value_iter>::operator--(); 
		return *this;
	}
	const_block_prop_iter& operator++(int)
	{
		auto out = *this;
		block_prop_iter<const_value_iter>::operator++(); 
		return out;
	}
	const_block_prop_iter& operator--(int)
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
	                const index_list& section_by_dim, index_list _block_index)
	    : block_index(std::move(_block_index)), first(val_first, section_by_dim.begin(), block_index.begin()), last(val_last, section_by_dim.end(), block_index.end()) {}
	auto begin() const { return first; }
	auto end() const { return last; }

	const index_list& get_index() const
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
	const_block_prop_view(const btensor::block_prop_view<iterator>& other)
	    : block_prop_view<const_iterator>(other.begin().get_val_iter(), other.end().get_val_iter(), other.begin().get_section(),
	                      other.end().get_section(), other.get_index()) {}
};

} // namespace quantt

#endif /* D49FFA60_85C4_431A_BA62_9B1D30D67E86 */
