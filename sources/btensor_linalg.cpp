
#include "blockTensor/LinearAlgebra.h"
#include <ATen/TensorIndexing.h>
#include <cstdint>
#include <stdexcept>
#include <tuple>

namespace quantt
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
btensor::block_list_t reorder_by_cvals(const btensor &tensor)
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
	auto out = flat_map(tensor.blocks());
	std::stable_sort(out.begin(), out.end(),
	                 [r = tensor.dim(), row_qtt = row_start, col_qtt = col_start](auto &&a, auto &&b)
	                 {
		                 const auto& row_a_qtt = row_qtt[a.first[r - 2]];
		                 const auto& row_b_qtt = row_qtt[b.first[r - 2]];
		                 const auto& col_a_qtt = col_qtt[a.first[r - 1]];
		                 const auto& col_b_qtt = col_qtt[b.first[r - 1]];

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
compact_dense_single(btensor::block_list_t::iterator start, btensor::block_list_t::iterator end)
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
size_t compact_tensor_count(btensor::block_list_t &tensor, bool rank_greater_than_2, size_t rank, Func1 &&equal_index,
                            Func2 &&equal_cval)
{
	size_t out = 0;
	// if (rank_greater_than_2)
	// {
	auto it1 = tensor.begin();
	auto it2 = tensor.begin() + 1;
	for (; it2 != tensor.end(); ++it1, ++it2)
	{
		out += !(equal_index(it1->first, it2->first) and equal_cval(it1->first, it2->first));
	}
	// }
	// else
	// {
	// 	out  = std::min(tensor.section_number(tensor.dim()-2),tensor.section_number(tensor.dim()-1));
	// }
	return out;
}
/**
 * @brief make a full torch tensor from a btensor, removes lines and columns of zeros in the last 2 dimensions
 *
 * @param tensor
 * @return torch::Tensor
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
	btensor::block_list_t block_list = reorder_by_cvals(tensor);
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
				// exponential nature of the tensor with the lenght of this.
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
	for (; it2 != block_list.end(); ++it2)
	{
		if (not(equal_index(it1->first, it2->first) and equal_c_vals(it1->first, it2->first)))
		{

			*out_it = compact_dense_single(it1, it2); // each call are independent, can be parallelized
			++out_it;
			it1 = it2;
		}
	}
	// concatenate the blocs according to their bloc position.
	// return the tensor and the list.
	return out_tensors;
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

} // namespace LA_helpers
using namespace LA_helpers;

std::string qformat(any_quantity_cref qt)
{
	return fmt::format("{}",qt);
}

std::tuple<btensor, btensor, btensor> svd(const btensor &tensor, const bool some,const bool compute_uv)
{
	// extract independant btensors
	std::vector<std::tuple<torch::Tensor, btensor::index_list, std::vector<std::tuple<int, torch::indexing::Slice>>,
	                       std::vector<std::tuple<int, torch::indexing::Slice>>>>
	    tensors_n_indices = LA_helpers::compact_dense(tensor);
	// compute the size and conserved values of the diagonnal matrix, most likely to be the same code for the eigenvalue
	// problem.
	auto d_blocks = tensors_n_indices.size(); //that's not quite right. the number of independent part is not the number of sector in the diagonnal
	any_quantity_vector right_D_cvals(d_blocks, tensor.selection_rule->neutral());
	any_quantity_vector left_D_cvals(d_blocks, tensor.selection_rule->neutral());
	std::vector<int64_t> D_block_sizes(tensors_n_indices.size());
	auto D_rcval_it = right_D_cvals.begin();
	auto D_lcval_it = left_D_cvals.begin();
	auto D_bsize_it = D_block_sizes.begin();
	size_t U_blocks = 0;
	size_t V_blocks = 0;
	for (auto &[basictensor, other_indices, rows, cols] : tensors_n_indices)
	{
		D_rcval_it->operator=(tensor.section_conserved_qtt( // clangd spuriously tags an error on the call to
		    tensor.dim() - 1,                       // section_conserved_qtt... it's ok,
		    std::get<0>(cols[0])));                  // SFINAE and there's a non template overload that is an exact match
		D_lcval_it->operator=(*D_rcval_it);
		D_lcval_it->inverse_();
		// fmt::print("{} {}\n",*D_lcval_it,*D_rcval_it);
		// fmt::print("other indices {}\n", other_indices);
		
		// for (auto& row:rows)
		// {
		// fmt::print("input row {}\n", std::get<0>(row));
		// }
		// for (auto& col:cols)
		// {
		// fmt::print("input row {}\n", std::get<0>(row));
		// }
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
	btensor leftD_shape({static_cast<long>(d_blocks)}, left_D_cvals, D_block_sizes,
	                    tensor.selection_rule->neutral());
	btensor rightD_shape({static_cast<long>(d_blocks)}, right_D_cvals, D_block_sizes,
	                     tensor.selection_rule->neutral());
	auto d = shape_from(tensor.shape_from(d_shape), rightD_shape).neutral_shape();
	// fmt::print("d \n {}\n",d);
	std::vector<int64_t> to_U_shape(tensor.dim(), -1);
	to_U_shape.back() = 0;
	btensor U = shape_from(
	    tensor.shape_from(to_U_shape).shift_selection_rule(tensor.section_conserved_qtt(tensor.dim() - 1, 0).inverse()),
	    rightD_shape);
	// fmt::print("right_D_shape: \n {}\nU: \n{}",rightD_shape,U);
	std::vector<int64_t> to_V_others(tensor.dim(), -1);
	*(to_V_others.end() - 2) = to_V_others.back() =  0;
	std::vector<int64_t> to_V_left(tensor.dim(), 0);
	to_V_left.back() = -1;
	btensor V = shape_from(tensor.shape_from(to_V_others).neutral_shape(),tensor.shape_from(to_V_left),leftD_shape);
	V.shift_selection_rule(V.selection_rule->inverse());
	// fmt::print("V: \n {}\n",V);
	int b_i = 0;
	//preallocate the blocks, with index, so that the SVD calls can be parallelized.
	U.reserve_space(U_blocks);
	V.reserve_space(V_blocks);
	d.reserve_space(d_blocks);
	// Independent steps if the block list have preallocated blocks
	for (auto &[basictensor, other_indices, rows, cols] : tensors_n_indices)
	{//The two inner loops are independent from one another. Their respective steps are also independent if the btensor's block_list already has a block there.
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
		d.block(btensor::index_list( block_ind.begin(),block_ind.end()-1) ) = torch::Tensor();
		++b_i;
	}
	b_i=0;
	for (auto &[basictensor, other_indices, rows, cols] : tensors_n_indices)
	{
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
			V.block(block_ind) = bU.index(slice);
		}
		auto [block_ind, slice] = LA_helpers::build_index_slice(other_indices, extra_block_slice, extra_block_slice);
		d.block(btensor::index_list( block_ind.begin(),block_ind.end()-1) ) = bD;
		++b_i;
	}

	// return output tuple
	return std::make_tuple(U,d,V);
}
std::tuple<btensor, btensor, btensor> svd(const btensor &tensor, size_t split)
{
	// reshape according to split

	// call batched SVD

	// undo reshape

	// return tuple
	return std::make_tuple(btensor(), btensor(), btensor());
}
} // namespace quantt