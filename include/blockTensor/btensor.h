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
#include "property.h"
#include <algorithm>
#include <exception>
#include <torch/torch.h>
#include <vector>
namespace quantt
{

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
	using block_list_t = std::vector<std::pair<index_list, torch::Tensor>>;

	property<any_quantity, btensor, any_quantity_cref> selection_rule; //dmrjulia equiv: the flux.
	/**
	 * @brief Construct a new btensor object
	 * 
	 * @param dir_block_size_cqtt a nested list of pair of section size and conserved quantities. The number of element 
	 * in the first is level is the rank of the tensor. The number of elements in the second level is the number of 
	 * section for that dimension of the tensor
	 * @param selection_rule 
	 */
	btensor(std::initializer_list<std::initializer_list<std::tuple<size_t&, any_quantity_cref>>> dir_block_size_cqtt,
	        any_quantity_cref selection_rule);
	btensor(std::initializer_list<std::initializer_list<std::tuple<size_t&, any_quantity_cref>>> dir_block_size_cqtt,
	        any_quantity_cref selection_rule, size_t num_blocks);

	/**
	 * @brief Construct a new btensor object. construct from raw structure. Avoid using this constructor if you can.
	 * 
	 * @param _rank : the number of dimension of the tensor
	 * @param _blocks : list of pair<position,sub-tensor>, the position is stored in a block index
	 * @param _block_shapes : number of section for each dimension of the tensor
	 * @param _block_sizes : number of element for each section of each dimension
	 * @param _c_vals : conserved quantity associated to each of the section in each of the dimension
	 * @param _sel_rule : overall selection rule, the sum over the dimension of the conserved quantities of a given block must equal this value for a block to be allowed to differ from zero. 
	 */
	btensor(size_t _rank, block_list_t _blocks, index_list _block_shapes, index_list _block_sizes,
	        any_quantity_vector _c_vals, any_quantity _sel_rule);

	size_t block_size(size_t index, size_t block) const;
	any_quantity_cref block_conserved_qtt(size_t index, size_t block) const;
	std::tuple<size_t, any_quantity_cref> block_size_cqtt(size_t index, size_t block) const;

	//acessors
	

private:
	size_t rank;
	index_list sections_by_dim;
	index_list sections_sizes; //for non-empty slices, this is strictly redundent: the information could be found by inspecting the blocks
	//truncation should remove any and all empty slices, but user-written tensor could have empty slices.
	std::vector<std::pair<index_list, torch::Tensor>> blocks; //
	any_quantity_vector c_vals;                               //dmrjulia equiv: QnumSum in the QTensor class. This structure doesn't need the full list (QnumMat)
	static std::string check_tensor(const btensor&);
};




} // namespace quantt

#endif /* D49FFA60_85C4_431A_BA62_9B1D30D67E86 */
