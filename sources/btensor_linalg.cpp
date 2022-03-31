
#include "LinearAlgebra.h"
#include "blockTensor/LinearAlgebra.h"
#include "torch_formatter.h"
#include <ATen/TensorIndexing.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <tuple>
namespace quantit
{

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
btensor::block_list_t::content_t reorder_by_cvals(const btensor &tensor)
{
	// identify sets of blocks with the same c_vals on every index
	auto [row_start, row_end] = tensor.section_conserved_qtt_range(tensor.dim() - 2);
	auto [col_start, col_end] = tensor.section_conserved_qtt_range(tensor.dim() - 1);
	// auto stable_perm_sort = [r = tensor.dim(), row_qtt = row_start, col_qtt = col_start](auto &&start, auto &&finish)
	// { 	std::vector<int32_t> perm(std::distance(start, finish)); // size N, the number of blocks, vector
	// 	std::iota(perm.begin(), perm.end(), 0);                  // fill perm with {0,1,2,...,N-2,N-1};
	// 	std::stable_sort(
	// 	    perm.begin(), perm.end(),
	// 	    [&start, &row_qtt, col_qtt](auto &&a, auto &&b) { // compare the conserved value of each of the blocks.
	// 		    // suppose start and finish are iterator on the block_list
	// 		const auto row_a_qtt = row_qtt[ start[a]->first)[r-2]];
	// 		const auto row_b_qtt = row_qtt[ start[b]->first)[r-2]];
	// 		const auto col_a_qtt = col_qtt[ start[a]->first)[r-1]];
	// 		const auto col_b_qtt = col_qtt[ start[b]->first)[r-1]];
	// 		bool out = row_a_qtt < row_b_qtt;
	// 		bool out |= (not out) and (col_a_qtt < col_b_qtt);
	// 		return out;
	// 	    });
	// 	return perm;

	// };
	auto out = btensor::block_list_t::content_t(tensor.begin(), tensor.end());
	std::stable_sort(out.begin(), out.end(),
	                 [r = tensor.dim(), row_qtt = row_start, col_qtt = col_start](auto &&a, auto &&b)
	                 {
		                 const auto &row_a_qtt = row_qtt[a.first[r - 2]];
		                 const auto &row_b_qtt = row_qtt[b.first[r - 2]];
		                 const auto &col_a_qtt = col_qtt[a.first[r - 1]];
		                 const auto &col_b_qtt = col_qtt[b.first[r - 1]];

		                 bool out = row_a_qtt < row_b_qtt;
		                 out |= (row_a_qtt == row_b_qtt) and (col_a_qtt < col_b_qtt);
		                 return out;
	                 });
	return out;
}

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
compact_dense_single(btensor::block_list_t::content_t::const_iterator start,
                     btensor::block_list_t::content_t::const_iterator end)
{ // So here, we know for a fact that all the blocks in the span [start,end) belong to the same matrix for the purpose
  // of the linear algebra routine.
	// One sort of optimisation that we might have missed is if something like [0 A;B 0] or [A 0; 0 B] happens. A priori
	// that would be the consequence of an accidental or unenforced (Abelian) symmetry.
	const auto rank = start->first.size();
	std::vector<int64_t> tensor_size(rank);
	auto &row_size = tensor_size[rank - 2];
	row_size = start->second.sizes()[rank - 2];
	auto &col_size = tensor_size[rank - 1];
	btensor::index_list other_indices((rank > 2) * (rank - 2));
	if (other_indices.size())
		std::copy(start->first.begin(), start->first.end() - 2, other_indices.begin());
	std::vector<std::tuple<int, torch::indexing::Slice>> encountered_block_col;
	std::vector<std::tuple<int, torch::indexing::Slice>> encountered_block_row;
	int64_t row_acc = 0;
	int64_t col_acc = 0;
	for (int i = 0; i < rank - 2; ++i)
	{
		tensor_size[i] = start->second.sizes()[i];
	}
	encountered_block_col.reserve(std::distance(start, end));
	encountered_block_row.reserve(std::distance(start, end));
	// Look at all the sector present, from it determine the size of the dense tensor. Could this be done for all the
	// tensors at once while counting the number of tensors?
	int cur_row = start->first[rank - 2];
	auto slice_end = row_acc + start->second.sizes()[rank - 2];
	encountered_block_row.emplace_back(cur_row, torch::indexing::Slice(row_acc, slice_end));
	row_acc = slice_end;
	for (auto it = start; it != end;
	     ++it) // nlogn loop, the n^2 loop might be faster because of easier to predict branches.
	{
		const auto col = it->first[rank - 1];
		auto pos = std::lower_bound(encountered_block_col.begin(), encountered_block_col.end(), col,
		                            [](const auto &a, const auto &b) { return std::get<0>(a) < b; });
		if (pos == encountered_block_col.end() or std::get<0>(*pos) != col)
		{
			auto slice_end = col_acc + it->second.sizes()[rank - 1];
			encountered_block_col.emplace(pos, col, torch::indexing::Slice(col_acc, slice_end));
			col_size += it->second.sizes()[rank - 1];
			col_acc = slice_end;
		}
		if (cur_row != it->first[rank - 2])
		{
			auto slice_end = row_acc + it->second.sizes()[rank - 2];
			cur_row = it->first[rank - 2];
			encountered_block_row.emplace_back(cur_row, torch::indexing::Slice(row_acc, slice_end));
			row_size += it->second.sizes()[rank - 2];
			row_acc = slice_end;
		}
	}
	auto out_tensor = torch::zeros(tensor_size);
	for (auto it = start; it != end;
	     ++it) // stuff the blocks in the dense tensor. can be parallelized, but false sharing can occur.
	{
		auto col_slice_it = std::find_if(encountered_block_col.begin(), encountered_block_col.end(),
		                                 [b = it->first[rank - 1]](auto &a) { return std::get<0>(a) == b; });
		auto row_slice_it = std::find_if(encountered_block_row.begin(), encountered_block_row.end(),
		                                 [b = it->first[rank - 2]](auto &a) { return std::get<0>(a) == b; });
		out_tensor.index_put_({torch::indexing::Ellipsis, std::get<1>(*row_slice_it), std::get<1>(*col_slice_it)},
		                      it->second);
	}
	// if there are blocks of zeros, we must assign them slice index anyway for the undoing. The linear algebra could
	// create values there.
	// since there's no hole, we can simply store as a vectors of slice index and block posisiton.I think as a rule, the
	// index can be {Ellipsis,slice_A,slice_B}
	return std::make_tuple(out_tensor, other_indices, encountered_block_row, encountered_block_col);
}

/**
 * @brief count the number of dense tensor to create for linear algebra call.
 *
 * When of the tensor is 2, it uses a constant time heuristic to compute the number. It is an upper bound.
 * Otherwise it compute the actual number of tensor. It takes a time linear in the number of blocks.
 *
 * @tparam Func
 * @param block_lists
 * @param rank_greater_than_2
 * @param equal_index
 * @return size_t
 */
template <class Func1, class Func2>
size_t compact_tensor_count(btensor::block_list_t::content_t &tensor, bool rank_greater_than_2, size_t rank,
                            Func1 &&equal_index, Func2 &&equal_cval)
{
	size_t out = 0;
	// if (rank_greater_than_2)
	// {
	auto it1 = tensor.begin();
	auto it2 = tensor.begin() + (it1 != tensor.end());
	while (it2 != tensor.end())
	{
		auto ind1 = it1->first;
		auto ind2 = it2->first;
		bool eq_ind = equal_index(it1->first, it2->first);
		bool eq_cval = equal_cval(it1->first, it2->first);
		out += !(eq_ind and eq_cval);
		++it1;
		++it2;
	}
	out += it1 != it2;
	return out;
}
/**
 * @brief make a set of full torch tensor from a btensor, removes lines and columns of zeros in the last 2 dimensions
 *
 * The output contains a list of tuple of torch::tensor, the block index from the first N-2 dimensions, a list
 * containing a tuple of a block-row and the slice to apply on the N-1 dimension to reobtain the section at that
 * block-row, and a likewise list for the last dimension.

 * @param tensor
 * @return std::vector<std::tuple<torch::Tensor, btensor::index_list,
 * std::vector<std::tuple<int,torch::indexing::Slice>>, std::vector<std::tuple<int, torch::indexing::Slice>>>>
 */
std::vector<std::tuple<torch::Tensor, btensor::index_list, std::vector<std::tuple<int, torch::indexing::Slice>>,
                       std::vector<std::tuple<int, torch::indexing::Slice>>>>
compact_dense(const btensor &tensor)
{
	using out_type =
	    std::vector<std::tuple<torch::Tensor, btensor::index_list, std::vector<std::tuple<int, torch::indexing::Slice>>,
	                           std::vector<std::tuple<int, torch::indexing::Slice>>>>;
	auto rank = tensor.dim();
	bool rank_greater_than_2 = rank > 2;
	btensor::block_list_t::content_t block_list = reorder_by_cvals(tensor);
	auto equal_index = [rank](const btensor::index_list &index_a, const btensor::index_list &index_b)
	{
		bool lout = true;
		if (rank > 2)
		{
			auto b_it = index_b.begin();
			for (auto a_it = index_a.begin(); a_it != index_a.end() - 2; ++a_it, ++b_it)
			{
				lout &= *a_it == *b_it;
				// if ( not lout) break; //might be faster not to break, this loop should be fairly short because of the
				// exponential nature of the tensor with the lenght of this. //always flatten into matrices when it gets
				// here, so it could be somewhat long.
			}
		}
		return lout;
	};
	auto [col_cval_start, col_cval_end] = tensor.section_conserved_qtt_range(tensor.dim() - 1);
	auto [row_cval_start, row_cval_end] = tensor.section_conserved_qtt_range(tensor.dim() - 2);
	auto equal_c_vals = [col_cval = col_cval_start, row_cval = row_cval_start, rank](const btensor::index_list &index_a,
	                                                                                 const btensor::index_list &index_b)
	{
		return (col_cval[index_a[rank - 1]] == col_cval[index_b[rank - 1]] and
		        row_cval[index_a[rank - 2]] == row_cval[index_b[rank - 2]]);
	};

	out_type out_tensors(compact_tensor_count(block_list, rank_greater_than_2, rank, equal_index, equal_c_vals));
	// create the list of <tensorIndex, bloc_position> pairs for all the sections involved.
	auto it1 = block_list.begin();
	auto it2 = block_list.begin() + 1;
	auto out_it = out_tensors.begin();
	auto block_list_l = std::distance(it1, block_list.end());
	auto out_l = std::distance(out_it, out_tensors.end());
	// for (; it2 != block_list.end(); ++it2)//skips the last set of blocks.
	while (it1 != block_list.end())
	{
		if (it2 == block_list.end() or
		    not(equal_index(it1->first, it2->first) and equal_c_vals(it1->first, it2->first)))
		{
			bool out_at_end = out_it == out_tensors.end();
			*out_it = compact_dense_single(it1, it2); // each call are independent, can be parallelized
			++out_it;
			it1 = it2;
		}
		++it2;
	}
	// concatenate the blocs according to their bloc position.
	// return the tensor and the list.
	return out_tensors;
}
using Slice = torch::indexing::Slice;
using TensInd = torch::indexing::TensorIndex;
std::tuple<btensor::index_list, std::array<TensInd, 3>> build_index_slice(const btensor::index_list &other_indices,
                                                                          const std::tuple<int, Slice> &rb_slices,
                                                                          const std::tuple<int, Slice> &cb_slices)
{
	auto rank = other_indices.size() + 2;
	btensor::index_list block_ind(rank);
	std::copy(other_indices.begin(), other_indices.end(), block_ind.begin());
	block_ind[rank - 2] = std::get<0>(rb_slices);
	block_ind[rank - 1] = std::get<0>(cb_slices);
	return std::make_tuple(
	    block_ind, std::array<TensInd, 3>{torch::indexing::Ellipsis, std::get<1>(rb_slices), std::get<1>(cb_slices)});
}
void reverse_compact_dense(torch::Tensor &tensor, btensor &out, btensor::index_list other_indices,
                           std::vector<std::tuple<int, torch::indexing::Slice>> rows,
                           std::vector<std::tuple<int, torch::indexing::Slice>> cols)
{
	/*
	 * for SVD, the U blocs have the row of of their bloc and the column given by their ordering relative to other blocs
	            the V blocs have to column of their blocs and to row corresponding to the ordering relative to the other
	 blocs.
	 */
	for (auto &row : rows)
	{
		for (auto &col : cols)
		{
			auto [block_index, slices] = build_index_slice(other_indices, row, col);
			out.block(block_index) = tensor.index(slices);
		}
	}
}

} // namespace LA_helpers
using namespace LA_helpers;

std::string qformat(any_quantity_cref qt) { return fmt::format("{}", qt); }

std::tuple<btensor, btensor> eigh(const btensor &tensor, BOOL upper)
{
	// extract independant btensors
	std::vector<std::tuple<torch::Tensor, btensor::index_list, std::vector<std::tuple<int, torch::indexing::Slice>>,
	                       std::vector<std::tuple<int, torch::indexing::Slice>>>>
	    tensors_n_indices = LA_helpers::compact_dense(tensor);
	// compute the size and conserved values of the diagonnal matrix, most likely to be the same code for the eigenvalue
	// problem.
	auto d_blocks = tensors_n_indices.size();
	any_quantity_vector right_D_cvals(d_blocks, tensor.selection_rule->neutral());
	any_quantity_vector left_D_cvals(d_blocks, tensor.selection_rule->neutral());
	std::vector<int64_t> D_block_sizes(tensors_n_indices.size());
	auto D_rcval_it = right_D_cvals.begin();
	auto D_lcval_it = left_D_cvals.begin();
	auto D_bsize_it = D_block_sizes.begin();
	size_t U_blocks = 0;
	for (auto &[basictensor, other_indices, rows, cols] : tensors_n_indices)
	{
		*D_rcval_it = (tensor.section_conserved_qtt(tensor.dim() - 1, std::get<0>(cols[0])));
		*D_lcval_it = D_rcval_it->inverse();
		U_blocks += rows.size();
		*D_bsize_it = basictensor.sizes()[basictensor.dim() - 1];
		++D_rcval_it;
		++D_lcval_it;
		++D_bsize_it;
	}
	std::vector<int64_t> d_shape(tensor.dim(), -1);
	*(d_shape.end() - 2) = *(d_shape.end() - 1) = 0;
	btensor leftD_shape({static_cast<long>(d_blocks)}, left_D_cvals, D_block_sizes, tensor.selection_rule->neutral());
	btensor rightD_shape({static_cast<long>(d_blocks)}, right_D_cvals, D_block_sizes, tensor.selection_rule->neutral());
	auto d = shape_from(tensor.shape_from(d_shape), rightD_shape).neutral_shape_();
	// fmt::print("d \n {}\n",d);
	std::vector<int64_t> to_U_shape(tensor.dim(), -1);
	to_U_shape.back() = 0;
	btensor U = shape_from(tensor.shape_from(to_U_shape)
	                           .shift_selection_rule_(tensor.section_conserved_qtt(tensor.dim() - 1, 0).inverse()),
	                       rightD_shape);
	// fmt::print("right_D_shape: \n {}\nU: \n{}",rightD_shape,U);
	// fmt::print("V: \n {}\n",V);
	int b_i = 0;
	// preallocate the blocks, with index, so that the SVD calls can be parallelized.
	U.reserve_space_(U_blocks);
	d.reserve_space_(d_blocks);
	// Independent steps if the block list have preallocated blocks
	for (auto &[basictensor, other_indices, rows, cols] : tensors_n_indices)
	{ // The two inner loops are independent from one another. Their respective steps are also independent if the
	  // btensor's block_list already has a block there.
		auto extra_block_slice = std::make_tuple(b_i, torch::indexing::Slice());
		for (auto &row : rows)
		{
			auto [block_ind, slice] = LA_helpers::build_index_slice(other_indices, row, extra_block_slice);
			U.block(block_ind) = torch::Tensor();
		}
		auto [block_ind, slice] = LA_helpers::build_index_slice(other_indices, extra_block_slice, extra_block_slice);
		d.block(btensor::index_list(block_ind.begin(), block_ind.end() - 1)) = torch::Tensor();
		++b_i;
	}
	b_i = 0;
	for (auto &[basictensor, other_indices, rows, cols] : tensors_n_indices)
	{
		auto [bD, bU] = torch::linalg::eigh(basictensor, upper ? "U" : "L");
		auto extra_block_slice = std::make_tuple(b_i, torch::indexing::Slice());
		for (auto &row : rows)
		{
			auto [block_ind, slice] = LA_helpers::build_index_slice(other_indices, row, extra_block_slice);
			U.block(block_ind) = bU.index(slice);
		}
		auto [block_ind, slice] = LA_helpers::build_index_slice(other_indices, extra_block_slice, extra_block_slice);
		d.block(btensor::index_list(block_ind.begin(), block_ind.end() - 1)) = bD;
		++b_i;
	}

	// return output tuple
	return std::make_tuple(d, U);
}
std::tuple<btensor, btensor> eigh(const btensor &tensor, size_t split)
{
	// reshape according to split
	auto rtensor = tensor.reshape({static_cast<int64_t>(split)});
	// call batched SVD
	auto [d, rU] = eigh(rtensor, true); // we want the eigenvectors.
	// undo reshape
	std::vector<int64_t> U_shape(tensor.dim(), -1);
	{
		for (size_t i = split; i < tensor.dim(); ++i)
		{
			U_shape[i] = 0;
		}
	}
	// fmt::print("rV \n{}\n\n",rV);
	// fmt::print("V shape \n{}\n\n",V_shape);
	// fmt::print("tensor \n{}\n\n",tensor);
	auto U = rU.reshape_as(shape_from(tensor.shape_from(U_shape), rU.shape_from({0, -1})));
	// return tuple
	return std::make_tuple(d, U);
}
std::tuple<btensor, btensor, btensor> svd(const btensor &tensor, const BOOL some, const BOOL compute_uv)
{
	// extract independant btensors
	// fmt::print("========SVD input =======\n{}\n\n",tensor);
	std::vector<std::tuple<torch::Tensor, btensor::index_list, std::vector<std::tuple<int, torch::indexing::Slice>>,
	                       std::vector<std::tuple<int, torch::indexing::Slice>>>>
	    tensors_n_indices = LA_helpers::compact_dense(tensor);
	// compute the size and conserved values of the diagonnal matrix, most likely to be the same code for the eigenvalue
	// problem.
	auto d_blocks = tensors_n_indices.size();
	any_quantity_vector right_D_cvals(d_blocks, tensor.selection_rule->neutral());
	any_quantity_vector left_D_cvals(d_blocks, tensor.selection_rule->neutral());
	std::vector<int64_t> D_block_sizes(tensors_n_indices.size());
	auto D_rcval_it = right_D_cvals.begin();
	auto D_lcval_it = left_D_cvals.begin();
	auto D_bsize_it = D_block_sizes.begin();
	size_t U_blocks = 0;
	size_t V_blocks = 0;
	// compute the room needed in the output structure.
	for (auto &[basictensor, other_indices, rows, cols] : tensors_n_indices)
	{
		*D_rcval_it = (tensor.section_conserved_qtt(tensor.dim() - 1, std::get<0>(cols[0])));
		*D_lcval_it = D_rcval_it->inverse();
		U_blocks += rows.size();
		V_blocks += cols.size();
		if (some)
		{
			*D_bsize_it =
			    std::min(basictensor.sizes()[basictensor.dim() - 1], basictensor.sizes()[basictensor.dim() - 2]);
		}
		else
		{
			*D_bsize_it =
			    std::max(basictensor.sizes()[basictensor.dim() - 1], basictensor.sizes()[basictensor.dim() - 2]);
		}
		++D_rcval_it;
		++D_lcval_it;
		++D_bsize_it;
	}
	std::vector<int64_t> d_shape(tensor.dim(), -1);
	*(d_shape.end() - 2) = *(d_shape.end() - 1) = 0;
	btensor leftD_shape({static_cast<long>(d_blocks)}, left_D_cvals, D_block_sizes, tensor.selection_rule->neutral());
	btensor rightD_shape({static_cast<long>(d_blocks)}, right_D_cvals, D_block_sizes, tensor.selection_rule->neutral());
	// small d is just the diagonnal of D, the diagonal matrix of singular values.
	//  because it is a vector, the correct operation to multiply it with the other matrices is a broadcasting
	//  elementwise multiplication. This operation must multiply together the conserved value. We adopt a convention
	//  where D always has the neutral selection rule therefor, all the conserved quantities of d must be the neutral
	//  element.
	auto d = shape_from(tensor.shape_from(d_shape), rightD_shape).neutral_shape_();
	// fmt::print("d \n {}\n",d);
	std::vector<int64_t> to_U_shape(tensor.dim(), -1);
	to_U_shape.back() = 0;
	btensor U = shape_from(tensor.shape_from(to_U_shape), rightD_shape).set_selection_rule_(tensor.selection_rule);
	// fmt::print("right_D_shape: \n {}\nU: \n{}",rightD_shape,U);
	std::vector<int64_t> to_V_others(tensor.dim(), -1);
	*(to_V_others.end() - 2) = to_V_others.back() = 0;
	std::vector<int64_t> to_V_left(tensor.dim(), 0);
	to_V_left.back() = -1;
	// the broadcast part (first shape) is an element by element mulitplication, the conserved quantity are multiplied
	// as well in sucha a situation Any decomposition of the conserved quantities of the input tensor on those would do,
	// we choose the simplest, the input on U
	//  and all neutral element on d and V.
	btensor V = shape_from(tensor.shape_from(to_V_others).neutral_shape_(), tensor.shape_from(to_V_left), leftD_shape);
	V.set_selection_rule_(V.selection_rule->neutral());
	// fmt::print("V: \n {}\n",V);
	// preallocate the blocks, with index, so that the SVD calls can be parallelized.
	U.reserve_space_(U_blocks);
	V.reserve_space_(V_blocks);
	d.reserve_space_(d_blocks);
	// Independent steps if the block list have preallocated blocks
	int b_i = 0;
	for (auto &[basictensor, other_indices, rows, cols] : tensors_n_indices)
	{ // The two inner loops are independent from one another. Their respective steps are also independent if the
	  // btensor's block_list already has a block there.
		auto extra_block_slice = std::make_tuple(b_i, torch::indexing::Slice());
		for (auto &row : rows)
		{
			auto [block_ind, slice] = LA_helpers::build_index_slice(other_indices, row, extra_block_slice);
			U.block(block_ind) = torch::Tensor();
		}
		for (auto &col : cols)
		{
			auto [block_ind, slice] = LA_helpers::build_index_slice(other_indices, col, extra_block_slice);
			V.block(block_ind) = torch::Tensor();
		}
		auto [block_ind, slice] = LA_helpers::build_index_slice(other_indices, extra_block_slice, extra_block_slice);
		d.block(btensor::index_list(block_ind.begin(), block_ind.end() - 1)) = torch::Tensor();
		++b_i;
	}
	b_i = 0;
	for (auto &[basictensor, other_indices, rows, cols] : tensors_n_indices)
	{
		// fmt::print("======basictensor=======\n {}\n\n", basictensor );
		auto [bU, bD, bV] = torch::svd(basictensor, some, compute_uv);
		auto extra_block_slice = std::make_tuple(b_i, torch::indexing::Slice());
		for (auto &row : rows)
		{
			auto [block_ind, slice] = LA_helpers::build_index_slice(other_indices, row, extra_block_slice);
			U.block(block_ind) = bU.index(slice);
		}
		for (auto &col : cols)
		{
			auto [block_ind, slice] = LA_helpers::build_index_slice(other_indices, col, extra_block_slice);
			V.block(block_ind) = bV.index(slice);
		}
		auto [block_ind, slice] = LA_helpers::build_index_slice(other_indices, extra_block_slice, extra_block_slice);
		d.block(btensor::index_list(block_ind.begin(), block_ind.end() - 1)) = bD;
		++b_i;
	}

	// return output tuple
	return std::make_tuple(U, d, V.inverse_cvals_());
}
std::tuple<btensor, btensor, btensor> svd(const btensor &tensor, size_t split)
{
	// reshape according to split
	auto rtensor = tensor.reshape({static_cast<int64_t>(split)});
	// call batched SVD
	auto [rU, d, rV] = svd(rtensor);
	// undo reshape
	std::vector<int64_t> U_shape(tensor.dim(), -1);
	std::vector<int64_t> V_shape(tensor.dim(), -1);
	{
		size_t i = 0;
		for (; i < split; ++i)
		{
			V_shape[i] = 0;
		}
		for (; i < tensor.dim(); ++i)
		{
			U_shape[i] = 0;
		}
	}
	// fmt::print("rV \n{}\n\n",rV);
	// fmt::print("V shape \n{}\n\n",V_shape);
	// fmt::print("tensor \n{}\n\n",tensor);
	auto U = rU.reshape_as(shape_from(tensor.shape_from(U_shape), rU.shape_from({0, -1})));
	auto V_left_part = tensor.shape_from(V_shape).inverse_cvals();
	auto V_right_part = rV.shape_from({0, -1});
	// fmt::print("V_left_part \n{}\n\n", V_left_part);
	// fmt::print("V_right_part \n{}\n\n", V_right_part);
	auto V = rV.reshape_as(shape_from(V_left_part, V_right_part));
	// return tuple
	return std::make_tuple(U, d, V);
}

/**
 * @brief Search for the first index with a value not greater than val in a desccending ordered list of value. For
 * pytorch.
 *
 * Currently a linear search, a branchless binary search might be better.
 *
 * @tparam TorchAccessor
 * @tparam Value
 * @param acc
 * @param val
 * @return int64_t
 */
template <class TorchAccessor, class Value>
int64_t lower_bound_impl2(TorchAccessor &acc, Value &val)
{
	int64_t i = 0;
	for (; i < acc.size(0); ++i)
	{
		if (!(acc[i] > val))
			break;
	}
	return i;
}

template <torch::ScalarType Scal_type, torch::DeviceType Dev_type>
int64_t lower_bound_impl(torch::Tensor &tens, const torch::Scalar &val)
{
	assert(tens.dim() == 1);
	using stype = decltype(c10::impl::ScalarTypeToCPPType<Scal_type>::t);
	auto valA = val.to<stype>();
	if constexpr (Dev_type == torch::kCPU)
	{
		// auto tensA = tens.accessor<double,1>();
		auto tensA = tens.accessor<stype, 1>();
		// torch accessor don't supply iterators... piece of crap.
		return lower_bound_impl2(tensA, valA);
	}
	else if constexpr (Dev_type == torch::kCUDA)
	{
		// the 32 means we're limited to 32bits addressable size. that's a lot, but we could go beyond using
		// packed_accessor64

		auto tensA = tens.packed_accessor32<stype, 1>();
		return lower_bound_impl2(tensA, valA);
	}
	static_assert(Dev_type == torch::kCUDA or Dev_type == torch::kCPU, "unsupported device type for this function.");
}

#define SWITCH_CASE_TORCHDTYPE_NUMERIC_PROP(CPPTYPE, c10ScalarType, numeric_property)                                  \
	case c10::ScalarType::c10ScalarType:                                                                               \
		return torch::Scalar(                                                                                          \
		    std::numeric_limits<                                                                                       \
		        decltype(c10::impl::ScalarTypeToCPPType<c10::ScalarType::c10ScalarType>::t)>::numeric_property());     \
		break;

#define NUMERICAL_PROP_BODY(DTYPE, SUBSTITUTED_MACRO, numeric_property)                                                \
	switch (torch::typeMetaToScalarType(DTYPE))                                                                        \
	{                                                                                                                  \
		AT_FORALL_SCALAR_TYPES(SUBSTITUTED_MACRO)                                                                      \
	default:                                                                                                           \
		throw std::invalid_argument(                                                                                   \
		    fmt::format("unsupported element type {} of tensors for {}, complex numbers are unsupported.",             \
		                DTYPE.name(), #numeric_property));                                                             \
		break;                                                                                                         \
	}

torch::Scalar epsilon(caffe2::TypeMeta dtype){
#define SUB_THIRD(CPPTYPE, C10ScalarType) SWITCH_CASE_TORCHDTYPE_NUMERIC_PROP(CPPTYPE, C10ScalarType, epsilon)
    NUMERICAL_PROP_BODY(dtype, SUB_THIRD, epsilon)
#undef SUB_THIRD
} torch::Scalar epsilon(const torch::Tensor &tens)
{
#define SUB_THIRD(CPPTYPE, C10ScalarType) SWITCH_CASE_TORCHDTYPE_NUMERIC_PROP(CPPTYPE, C10ScalarType, epsilon)
	NUMERICAL_PROP_BODY(tens.dtype(), SUB_THIRD, epsilon)
#undef SUB_THIRD
}
// macro for usage with lower_bound_dev
// create a case for a scalar type supported by pytorch.
// Theres a macro in torch that turn this into a case for each of the supported type
#define SWITCH_CASE_TORCHDTYPE_LOWERBOUND(CPPTYPE, c10ScalarType)                                                      \
	case c10::ScalarType::c10ScalarType:                                                                               \
		return lower_bound_impl<c10::ScalarType::c10ScalarType, Dev_type>(tens, scal);                                 \
		break;
/**
 * @brief dispatch lower_bound_impl based on the scalar type in tens
 */
template <torch::DeviceType Dev_type>
int64_t lower_bound_dev(torch::Tensor &tens, const torch::Scalar &scal)
{

	switch (torch::typeMetaToScalarType(tens.dtype()))
	{
		AT_FORALL_SCALAR_TYPES(SWITCH_CASE_TORCHDTYPE_LOWERBOUND)
	default:
		throw std::invalid_argument(
		    fmt::format("unsupported element type {} of tensors for lowerbound, complex numbers are unsupported.",
		                tens.dtype().name()));
		break;
	}
}

int64_t lower_bound(torch::Tensor &tens, const torch::Scalar &scal)
{
	if (tens.device() == torch::kCUDA)
		return lower_bound_dev<torch::kCUDA>(tens, scal);
	else if (tens.device() == torch::kCPU)
		return lower_bound_dev<torch::kCPU>(tens, scal);
	throw std::logic_error(fmt::format("unsupported backend {} for quantit::lower_bound(torch::Tensor&,torch::Scalar&>",
	                                   c10::DeviceTypeName(tens.device().type())));
}
/**
 * @brief truncation for a decomposition that induce any number of unitary matrix and a list of scalar weights
 *
 * @param d scalar weights for the unitaries
 * @param unitaries unitary tensors, last index is the weighted dimension
 * @param max maximum dimension
 * @param min minimum dimension
 * @param tol tolerance on induced error, max takes precedence
 * @param pow power of the value to use in the error computation
 * @return std::tuple<btensor,btensor,btensor>
 */
template <class... BTENS>
std::tuple<btensor, std::tuple<BTENS...>> truncate_impl(btensor &&d, std::tuple<BTENS...> &&unitaries, size_t max,
                                                        size_t min, btensor::Scalar tol, btensor::Scalar pow)
{
	static_assert(std::conjunction_v<std::is_same<BTENS, btensor>...>, "The unitaries must be btensors!");
	// d is a vector of singular values
	assert(d.dim() == 1);
	// prepare a list containing all the singular value
	auto L = std::accumulate(d.begin(), d.end(), 0, [](auto &&a, auto &&b) { return a + std::get<1>(b).sizes()[0]; });
	auto vd = torch::zeros(
	    {L}, torch::device(torch::kCPU).dtype(torch::kDouble)); // always done on the CPU with double precicion
	size_t n = 0;
	for (const auto &block : d)
	{
		auto &tens = std::get<1>(block);
		auto l = tens.sizes()[0];
		vd.index_put_({torch::indexing::Slice(n, n + l)}, tens); // bug here.
		n += l;
	}
	// quantit::print(vd);
	vd = std::get<0>(
	    vd.sort(-1, true)); // sort the last (only) dimension in descending order //no inplace sort in torch...
	// vd is now sorted in ascending order.
	auto smallest_value = vd.index({torch::indexing::Ellipsis, compute_last_index(vd, tol, pow, min, max)});
	smallest_value -=
	    2 * smallest_value * epsilon(d.options().dtype()); // Better chance to preserve degenerate multiplets that way.
	// fmt::print("epsilon {}\n",epsilon(smallest_value).toDouble());
	// for each block trio, we can remove all the values smaller than the one in smallest_value
	// without inducing an error larger than the tol.
	auto d_it = d.end();
	std::vector<btensor::index_list> to_remove;
	to_remove.reserve(std::distance(d.begin(), d.end()));
	auto remove_unit_blocks = [](btensor &unitary, const auto &d_ind)
	{
		auto U_src = unitary.begin();
		auto U_dest = unitary.begin();
		while (U_dest != unitary.end())
		{
			auto &src_ind = std::get<0>(*U_src);
			auto &dest_ind = std::get<0>(*U_dest);
			U_dest += (dest_ind.back() == d_ind.back());
			if (U_dest != U_src and U_dest != unitary.end())
			{
				swap(*U_dest, *U_src);
			}
			U_dest += U_dest != unitary.end(); // stop incrementing if we reached the end
			++U_src;
		}
		unitary.blocks_list.resize(
		    unitary.blocks_list.size() -
		    std::distance(U_src, U_dest)); // the unwanted blocks are pushed to the end, resizing down destroy them.
	};
	auto trunc_unit_block = [&d](btensor &U, const auto &d_ind, const auto last_index)
	{
		U.sections_sizes[U.sections_sizes.size() - (d.sections_sizes.size() - d_ind.back())] = last_index;
		using namespace torch::indexing;
		auto U_it = U.begin();
		while (U_it != U.end())
		{
			auto &U_ind = std::get<0>(*U_it);
			auto &Ub = std::get<1>(*U_it);
			if (U_ind.back() == d_ind.back())
				Ub = Ub.index({Ellipsis, Slice(0, last_index)});
			++U_it;
		}
	};
	while (d_it != d.begin())
	{
		--d_it;
		auto &d_ind = std::get<0>(*d_it);
		auto &db = std::get<1>(*d_it);
		using namespace torch::indexing;
		// fmt::print("last_index! {}\n\n", db > smallest_value);
		// fmt::print("last_index other order because of implicit casts!! {}\n\n",  smallest_value < db);
		auto last_index = lower_bound(db, smallest_value.item());
		if (last_index == 0)
		{ // remove the whole block...
			// i think, out of laziness and lack of advantages to the converse, i will only erase the blocks without
			// changing the shape of the blocktensor. This mean the resulting tensor will have extra structure, with no
			// elements in there.
			// this will not result in an accumulation of empty sections, because empty section in a tensor do not
			// induce an empty section in the singular values tensor
			// Correctly evaluating the bond dimension of a MPS becomes a bit more involved. But the induced extra
			// structure should disapear as DMRG nears convergence
			for_each(unitaries, [&d_ind, &remove_unit_blocks](btensor &U) { return remove_unit_blocks(U, d_ind); });
			d.blocks_list.erase(d_it);
		}
		else
		{ // truncate d
			db = db.index({Slice(0, last_index)});
			// adjust the section sizes
			d.sections_sizes[d_ind.back()] = last_index;
			// truncate all the blocks in U on the section matching with the current d block.
			for_each(unitaries, [&d_ind, &last_index, &trunc_unit_block](btensor &U)
			         { return trunc_unit_block(U, d_ind, last_index); });
		}
	}
	return std::make_tuple(std::move(d), std::move(unitaries));
}
/**
 * @brief truncation for the tensor network SVD
 *
 * @param U Left unitary matrix
 * @param d Singular values
 * @param V Right unitary matrix
 * @param max maximum dimension
 * @param min minimum dimension
 * @param tol tolerance on induced error, max and min takes precedence
 * @param pow power of the value to use in the error computation
 * @return std::tuple<btensor,btensor,btensor>
 */
std::tuple<btensor, btensor, btensor> truncate(std::tuple<btensor, btensor, btensor> &&U_d_V, size_t max, size_t min,
                                               btensor::Scalar tol, btensor::Scalar pow)
{
	auto [d, unit] = truncate_impl(std::move(std::get<1>(U_d_V)),
	                               std::make_tuple(std::move(std::get<0>(U_d_V)), std::move(std::get<2>(U_d_V))), max,
	                               min, tol, pow);
	return std::make_tuple(std::move(std::get<0>(unit)), std::move(d), std::move(std::get<1>(unit)));
}
/**
 * @brief truncation for the eigenvalue decomposition
 *
 * @param e_S tuple of the eigenvalue vector and eigenunitary
 * @param max maximum number of eignvalue kept
 * @param min minimum number of eigenvalue kept
 * @param tol tolerence on the truncation induced error, max and min takes precedence
 * @param pow
 * @return std::tuple<btensor,btensor>
 */
std::tuple<btensor, btensor> truncate(std::tuple<btensor, btensor> &&e_S, size_t max, size_t min, btensor::Scalar tol,
                                      btensor::Scalar pow)
{
	auto [d, unit] =
	    truncate_impl(std::move(std::get<0>(e_S)), std::make_tuple(std::move(std::get<0>(e_S))), max, min, tol, pow);
	return std::make_tuple(std::move(d), std::move(std::get<0>(unit)));
}
std::tuple<btensor, btensor, btensor> truncate(btensor &&U, btensor &&e, btensor &&S, size_t max, size_t min,
                                               btensor::Scalar tol, btensor::Scalar pow)
{
	return truncate(std::tuple<btensor, btensor, btensor>(std::move(U), std::move(e), std::move(S)), max, min, tol,
	                pow);
}
std::tuple<btensor, btensor> truncate(btensor &&e, btensor &&S, size_t max, size_t min, btensor::Scalar tol,
                                      btensor::Scalar pow)
{
	return truncate(std::tuple<btensor, btensor>(std::move(e), std::move(S)), max, min, tol, pow);
}

std::tuple<btensor, btensor, btensor> svd(const btensor &A, size_t split, btensor::Scalar tol, size_t min_size,
                                          size_t max_size, btensor::Scalar pow)
{
	return truncate(svd(A, split), max_size, min_size, tol, pow);
}
std::tuple<btensor, btensor, btensor> svd(const btensor &A, size_t split, btensor::Scalar tol, btensor::Scalar pow)
{
	size_t min_size = 1;
	size_t max_size = std::numeric_limits<size_t>::max();
	return svd(A, split, tol, min_size, max_size, pow);
}
std::tuple<btensor, btensor> eigh(const btensor &A, size_t split, btensor::Scalar tol, size_t min_size, size_t max_size,
                                  btensor::Scalar pow)
{
	// TODO: truncating doesn't have the same meaning here as it does in SVD.
	// The most meaningful thing we could do is truncate based on the value of exp(-\beta E)/Tr(exp(-\beta E)) where
	// beta is an additionnal user parameter.
	return truncate(eigh(A, split), max_size, min_size, tol, pow);
}
std::tuple<btensor, btensor> eigh(const btensor &A, size_t split, btensor::Scalar tol, btensor::Scalar pow)
{
	size_t min_size = 1;
	size_t max_size = std::numeric_limits<size_t>::max();
	return eigh(A, split, tol, min_size, max_size, pow);
}

} // namespace quantit