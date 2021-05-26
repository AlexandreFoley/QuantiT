/*
 * File: LinearAlgebra.h
 * Project: quantt
 * File Created: Thursday, 13th May 2021 11:22:38 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * Copyright (c) 2021 Alexandre Foley
 * All rights reserved
 */

#ifndef BTENSORLINEARALGEBRA_H
#define BTENSORLINEARALGEBRA_H

#include "blockTensor/btensor.h"
#include "doctest/doctest_proxy.h"
#include <ATen/Functions.h>
#include <ATen/TensorIndexing.h>
#include <fmt/core.h>
#include <fmt/ranges.h>

namespace quantt
{
/**
 * 
 * @return std::tuple<btensor,btensor,btensor> U,D,V the matrices of the singular value decomposition
 */

/**
 * @brief Compute the batched singular value decomposition on the input btensor: \f$ I = U.D.V^\dagger \f$. 
 * Only non-nul elements (the diagonnal) of the diagonnal matrix are returned.
 *
 * The matrix decomposition is computed on the last two indices of the tensor, all other indices are treated as indexing otherwise independent matrices.
 * When doing such a decomposition, the conserved value on the batched index act as modification of the selection rule of each individual matrices and
 * the selection rule can be distributed on the output tensors in an arbitrary manner.
 * For the sake of simplicity, the conserved values on the batch indices and the selection rule are all put on the first tensor of the output tuple.
 * The two other tensor have all neutral element on their batch indices and selection rule.
 * Because we reduce the matrix of singular value to a vector of the value, we are force to assign the neutral element on the blocks.
 * 
 * The matrix mulitplication of the diagonnal matrix \f$D \f$, represented by its list of elements \f$d\f$, with \f$U\f$ or \f$V\f$ can be done like so:
 * \f$[
	 \sum_x U_{ijx}.D_{ixk} = U_{ijk}d_{ijk}
 * \f]
 *
 * @param tensor the tensor to decompose 
 * @return std::tuple<btensor,btensor,btensor> U, d and V
 */
std::tuple<btensor,btensor,btensor> svd(const btensor& tensor,bool some=true,bool compute_uv=true);

#ifndef NDEBUG
namespace LA_helpers
{
/**
 * @brief reorder the block list of the tensor by their conserved values
 *
 * All of the blocks of each of the tensor given in output have the same conserved value for each dimensions.
 * The sum of the output tensors equals the input tensor.
 *
 * Consider only the last 2 dimensions when determining the blocks. This is so batched linear algebra benefit from the
 * block structure too. For tensor networks need, reshape to a rank 2 tensor first.
 *
 * @param tensor
 * @return std::vector<btensor::block_list_t>
 */
btensor::block_list_t reorder_by_cvals(const btensor &tensor);

/**
 * @brief compactify a range of blocks into a single dense torch::tensor
 *
 * Create the minimal size necessary for the resulting tensor, assumes all blocks are in the same sector for all but the
 * last 2 dimensions. return the compactified
 * @param start
 * @param end
 * @return
 * std::tuple<torch::Tensor,std::vector<std::tuple<int,torch::indexing::Slice>>,std::vector<std::tuple<int,torch::indexing::Slice>>
 * >
 */
std::tuple<torch::Tensor, btensor::index_list, std::vector<std::tuple<int, torch::indexing::Slice>>,
           std::vector<std::tuple<int, torch::indexing::Slice>>>
compact_dense_single(btensor::block_list_t::iterator start, btensor::block_list_t::iterator end);

using Slice = torch::indexing::Slice;
using TensInd = torch::indexing::TensorIndex;
std::tuple<btensor::index_list,std::array<TensInd,3>> build_index_slice(const btensor::index_list& other_indices, const std::tuple<int,Slice>& rb_slices, const std::tuple<int,Slice>& cb_slices);

} // namespace LA_helpers

#endif // NDEBUG

qtt_TEST_CASE("btensor Linear algebra")
{
#ifndef NDEBUG
	// those helpers are not in the header when not in debug mode.
	qtt_SUBCASE("btensor linear algebra helpers")
	{
		using cqt = conserved::C<5>; // don't put negative number in the constructor and expect sensible results.
		using index = btensor::index_list;
		any_quantity selection_rule(cqt(0)); // DMRJulia flux
		btensor A({{{2, cqt(0)}, {3, cqt(1)}},
		           {{1, cqt(1)}, {2, cqt(0)}, {3, cqt(-1)}, {1, cqt(1)}},
		           {{3, cqt(0)}, {2, cqt(-2)}, {2, cqt(-1)}}},
		          selection_rule);
		A.block({0, 0, 2}) = torch::rand({2, 1, 2});
		A.block({0, 1, 0}) = torch::rand({2, 2, 3});
		A.block({0, 3, 2}) = torch::rand({2, 1, 2});
		A.block({1, 0, 1}) = torch::rand({3, 1, 2});
		A.block({1, 1, 2}) = torch::rand({3, 2, 2});
		A.block({1, 2, 0}) = torch::rand({3, 3, 3});
		A.block({1, 3, 1}) = torch::rand({3, 1, 2});
		qtt_CHECK_NOTHROW(btensor::throw_bad_tensor(A));
		{
			auto reordered_block = LA_helpers::reorder_by_cvals(A);
			std::vector<btensor::index_list> exp_blocks = {{0, 1, 0}, {1, 1, 2}, {1, 0, 1}, {1, 3, 1},
			                                               {0, 0, 2}, {0, 3, 2}, {1, 2, 0}};
			qtt_CHECK(exp_blocks.size() == reordered_block.size());
			for (auto [block, expec] = std::make_tuple(reordered_block.begin(), exp_blocks.begin());
			     block != reordered_block.end(); ++block, ++expec)
			{
				// auto cvals_view = A.block_quantities(std::get<0>(*block));
				// fmt::print( "cvals: {}\n",fmt::join(cvals_view.begin(),cvals_view.end(),","));
				// fmt::print("block index: {}\n", std::get<0>(*block));
				qtt_CHECK(std::get<0>(*block) == *expec);
			}
			// reordered_block[2:4] is a set of block with the same c_vals on the last two indices, and with the same
			// block indices for all but the last two dims.
			auto [compact_tensor, other_indices, row_block_slices, col_block_slices] =
			    LA_helpers::compact_dense_single(reordered_block.begin() + 2, reordered_block.begin() + 4);
			qtt_REQUIRE(other_indices.size() == 1);
			for (const auto &rb_slices : row_block_slices)
			{
				for (const auto &cb_slices : col_block_slices)
				{
					auto [block_ind, slice] = LA_helpers::build_index_slice(other_indices,rb_slices,cb_slices);
					// we have rebuilt the block index in block_ind. It would be good to have a function to do that.
					qtt_CHECK(torch::equal(A.block(block_ind), compact_tensor.index(slice)));
					// then we verify that the blocks from the original tensor can be obtained from the stored slincing
					// of the compacted tensor
				}
			}
		}
		{
			btensor B;
			qtt_REQUIRE_NOTHROW(B = A.reshape({1})); // joins dimensions 2 and 1
			// fmt::print("tensor {}\n",B);
			auto reordered_block = LA_helpers::reorder_by_cvals(B);
			std::vector<btensor::index_list> exp_blocks = {{0,2},{0,3},{0,11},{1,1},{1,5},{1,6},{1,10}};
			qtt_CHECK(exp_blocks.size() == reordered_block.size());
			for (auto [block, expec] = std::make_tuple(reordered_block.begin(), exp_blocks.begin());
			     block != reordered_block.end(); ++block, ++expec)
			{
				// auto cvals_view = B.block_quantities(std::get<0>(*block));
				// fmt::print( "cvals: {}\n",fmt::join(cvals_view.begin(),cvals_view.end(),","));
				// fmt::print("block index: {}\n", std::get<0>(*block));
				qtt_CHECK(std::get<0>(*block) == *expec);
			}
			auto [compact_tensor, other_indices, row_block_slices, col_block_slices] =
			    LA_helpers::compact_dense_single(reordered_block.begin() , reordered_block.begin() + 3);
			qtt_REQUIRE(other_indices.size() == 0);
			for (const auto &rb_slices : row_block_slices)
			{
				for (const auto &cb_slices : col_block_slices)
				{
					auto [block_ind, slice] = LA_helpers::build_index_slice(other_indices,rb_slices,cb_slices);
					// fmt::print("block {}, cval {}\n",block_ind,B.block_quantities(block_ind));
					qtt_CHECK(torch::equal(B.block(block_ind), compact_tensor.index(slice)));
					// then we verify that the blocks from the original tensor can be obtained from the stored slincing
					// of the compacted tensor
				}
			}

		}
	}
#endif // NDEBUG
}

} // namespace quantt

#endif // BTENSORLINEARALGEBRA_H
