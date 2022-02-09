/*
 * File: LinearAlgebra.h
 * Project: QuantiT
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
#include <ostream>

namespace quantit
{

/**
 * @brief A bool type with no implicit conversion from arithmetic types.
 *
 * to avoid abiguous overloads between the batch svd and the tensors svd
 *
 */
struct BOOL
{
	bool val;
	BOOL(bool _val) noexcept : val(_val) {}
	operator bool() const noexcept { return val; }
};
/**
 * @brief Compute the batched singular value decomposition on the input btensor: \f$ I = U.D.V^\dagger \f$.
 * Only non-nul elements (the diagonnal) of the diagonnal matrix are returned.
 *
 * The matrix decomposition is computed on the last two indices of the tensor, all other indices are treated as indexing
 otherwise independent matrices.
 * When doing such a decomposition, the conserved value on the batched index act as modification of the selection rule
 of each individual matrices and
 * the selection rule can be distributed on the output tensors in an arbitrary manner.
 * For the sake of simplicity, the conserved values on the batch indices and the selection rule are all put on the first
 tensor of the output tuple.
 * The two other tensor have all neutral element on their batch indices and selection rule.
 * Because we reduce the matrix of singular value to a vector of the value, we are force to assign the neutral element
 on the blocks.
 *
 * The matrix mulitplication of the diagonnal matrix \f$D \f$, represented by its list of elements \f$d\f$, with \f$U\f$
 or \f$V\f$ can be done like so:
 * \f$[
     \sum_x U_{ijx}.D_{ixk} = U_{ijk}d_{ik}
 * \f]
 * btensor::mul and btensor::mul_ accomplish this. Because of the way torch broadcasting works, we must insert a size
 one index between i and k in d.
 * So the correct multiplication is accomplished by
 \code{.cpp}
    auto tensor = btensor();//put some arguements in here specifying a shape.
    // ... assign values into the blocks of the tensor.
    auto [U,d,V] = svd(tensor);
    auto d_r = d.reshape_as(shape_from(d, btensor({{{1, d.selection_rule->neutral()}}}, d.selection_rule->neutral())))
                   .transpose_(-1, -2);
    auto Vt = V.transpose(-1, -2);
    auto A = U.mul(d_r).bmm(Vt) //This should be equal to tensor to numerical accuracy
 \endcode
 * @param tensor the tensor to decompose
 * @param some wether to compute the thin (true) or full svd (false), true by default. Warning, full is currently
 untested. And the full is mostly pointless as the extra singular vectors are random, and the singular values zero.
 * @param compute_UV whether to compute the singular vectors, true by default. If false, U and V are empty tensors.
 * @return std::tuple<btensor,btensor,btensor> U, d and V
 */
std::tuple<btensor, btensor, btensor> svd(const btensor &tensor, BOOL some = true, BOOL compute_uv = true);

/**
 * @brief svd for tensor network methods, treat the tensor as a matrix, with the index before the split indices treated
 * as the row of the matrix and the split indice and the ones after and the column indices.
 *
 * The original tensor can be reconstructed by tensordot(U.mul(d), V.conj(), {U.dim() - 1}, {V.dim()-1})
 *
 * @param tensor tensor to decompose
 * @param split seperation between row and columns, among the dimensions of the tensor. rows are [0,split[ and columns
 * are [split,last]
 * @return std::tuple<btensor,btensor,btensor> U, d and V
 */
std::tuple<btensor, btensor, btensor> svd(const btensor &tensor, size_t split);
/**
 * @brief overload for implicit conversion disambiguation, see svd(const btensor&, size_t)
 *
 */
inline std::tuple<btensor, btensor, btensor> svd(const btensor &tensor, int split)
{
	return svd(tensor, static_cast<size_t>(split));
}

/**
 * @brief truncating svd for tensor network methods, treats the tensor as a matrix, with the index before the split
 * treated as the row of the matrix and the indices at and after the split as the column indices.
 *
 * If the number of singular values kept is smaller than max_size an approximation of the original tensor can be
 * reconstructed by tensordot(U.mul(d),V.conj(),{U.dims()-1},{0}) with an error smaller than tol.
 *
 * At least min_size singular values are kept.
 *
 * @param A tensor to decompose
 * @param split index that split the row indices from the column indices
 * @param tol error on the trace of the singular value to the pow that is tolerated
 * @param min_size minimum number of singular value kept
 * @param max_size maximum number of singular value kept
 * @param pow power used in the computation of the truncation error. When optimizing the energy of a MPS, 2 is the right
 * choice.
 * @return std::tuple<btensor,btensor,btensor> U,d,V
 */
std::tuple<btensor, btensor, btensor> svd(const btensor &A, size_t split, btensor::Scalar tol, size_t min_size,
                                          size_t max_size, btensor::Scalar pow = 2);
/**
 * @brief overload for implicit conversion disambiguation, see svd(cont
 * btensor&,size_t,btensor::Scalar,size_t,size_t,btensor::Scalar)
 */
inline std::tuple<btensor, btensor, btensor> svd(const btensor &A, int split, btensor::Scalar tol, size_t min_size,
                                                 size_t max_size, btensor::Scalar pow = 2)
{
	return svd(A, static_cast<size_t>(split), tol, min_size, max_size, pow);
}
/**
 * @brief truncating svd for tensor network methods, treats the tensor as a matrix, with the index before the split
 * treated as the row of the matrix and the indices at and after the split as the column indices.
 *
 * An approximation of the original tensor can be reconstructed by tensordot(U.mul(d),V.conj(),{U.dims()-1},{0}) with an
 * error smaller than tol.
 *
 * At least one singular value is kept.
 *
 * @param A tensor to decompose
 * @param split index that split the row indices from the column indices
 * @param tol error on the trace of the singular value to the pow that is tolerated
 * @param pow power used in the computation of the truncation error. When optimizing the energy of a MPS, 2 is the right
 * choice.
 * @return std::tuple<btensor,btensor,btensor> U,d,V
 */
std::tuple<btensor, btensor, btensor> svd(const btensor &A, size_t split, btensor::Scalar tol, btensor::Scalar pow = 2);
/**
 * @brief overload for implicit conversion disambiguation, see svd(cont btensor&,size_t,btensor::Scalar,btensor::Scalar)
 */
inline std::tuple<btensor, btensor, btensor> svd(const btensor &A, int split, btensor::Scalar tol,
                                                 btensor::Scalar pow = 2)
{
	return svd(A, static_cast<size_t>(split), tol, pow);
}
/**
 * @brief compute the batched eigenvalue decomposition.
 *
 * @param tensor to solve the eigenvalue problem for
 * @param eigenvectors weither to compute the eigenvectors as well
 * @param upper weither to use only the upper part or only the lower part for the algorithm.
 * @return std::tuple<btensor, btensor> e,S
 */
std::tuple<btensor, btensor> eigh(const btensor &tensor, BOOL upper = false);
/**
 * @brief tensor eigenvalue decomposition
 *
 * @param tensor tensor to decompose
 * @param split index that split the row indices from the column indices
 * @return std::tuple<btensor, btensor> e,S
 */
std::tuple<btensor, btensor> eigh(const btensor &tensor, size_t split);
/**
 * @brief truncating tensor eigenvalue decomposition
 *
 * @param A  tensor to decompose
 * @param split index that split the row indices from the column indices
 * @param tol tolerence on error induced by truncation
 * @param min_size minimum number of eigenvalue kept supersede tol
 * @param max_size maximum number of eigenvalue kept, supersede tol
 * @param pow power used when computing the induced error \f$ sum(e_t^{pow})<tol^{pow}\f$ where \f$e_t\f$ is a rejected
 * eigenvalues.
 * @return std::tuple<btensor, btensor>
 */
std::tuple<btensor, btensor> eigh(const btensor &A, size_t split, btensor::Scalar tol, size_t min_size,
                                    size_t max_size, btensor::Scalar pow = 1);
/**
 * @brief truncating tensor eigenvalue decomposition
 *
 * @param A tensor to decompose
 * @param split index that split the row indices from the column indices
 * @param tol tolerence on error induced by truncation
 * @param pow power used when computing the induced error \f$ sum(e_t^{pow})<tol^{pow}\f$ where \f$e_t\f$ is a rejected
 * eigenvalues.
 * @return std::tuple<btensor, btensor>
 */
std::tuple<btensor, btensor> eigh(const btensor &A, size_t split, btensor::Scalar tol, btensor::Scalar pow = 1);

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
btensor::block_list_t::content_t reorder_by_cvals(const btensor &tensor);

/**
 * @brief compactify a range of blocks into a single dense torch::tensor
 *
 * Create the minimal size necessary for the resulting tensor, assumes all blocks are in the same sector for all but the
 * last 2 dimensions. return the compactified
 * @param start
 * @param end
 * @return
 * std::tuple<torch::Tensor,std::vector<std::tuple<int,torch::indexing::Slice>>,std::vector<std::tuple<int,torch::indexing::Slice>>>
 */
std::tuple<torch::Tensor, btensor::index_list, std::vector<std::tuple<int, torch::indexing::Slice>>,
           std::vector<std::tuple<int, torch::indexing::Slice>>>
compact_dense_single(typename btensor::block_list_t::content_t::const_iterator start,
                     typename btensor::block_list_t::content_t::const_iterator end);

using Slice = torch::indexing::Slice;
using TensInd = torch::indexing::TensorIndex;
std::tuple<btensor::index_list, std::array<TensInd, 3>> build_index_slice(const btensor::index_list &other_indices,
                                                                          const std::tuple<int, Slice> &rb_slices,
                                                                          const std::tuple<int, Slice> &cb_slices);

std::vector<std::tuple<torch::Tensor, btensor::index_list, std::vector<std::tuple<int, torch::indexing::Slice>>,
                       std::vector<std::tuple<int, torch::indexing::Slice>>>>
compact_dense(const btensor &tensor);
} // namespace LA_helpers

inline std::ostream &operator<<(std::ostream &out, any_quantity_cref qt)
{
	out << fmt::format("{}", qt);
	return out;
}

std::string qformat(any_quantity_cref qt);

std::tuple<btensor, btensor, btensor> truncate(btensor &&U, btensor &&d, btensor &&V, size_t max, size_t min,
                                               btensor::Scalar tol, btensor::Scalar pow);

qtt_TEST_CASE("btensor Linear algebra")
{
	qtt_SUBCASE("decompositions")
	{
		using cqt = conserved::C<5>;
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
		qtt_REQUIRE_NOTHROW(btensor::throw_bad_tensor(A));
		auto [U, d, V] = svd(A);
#ifndef NDEBUG
		// those helpers are not in the header when not in debug mode.
		qtt_SUBCASE("btensor linear algebra helpers")
		{
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
				// reordered_block[2:4] is a set of block with the same c_vals on the last two indices, and with the
				// same block indices for all but the last two dims.
				auto [compact_tensor, other_indices, row_block_slices, col_block_slices] =
				    LA_helpers::compact_dense_single(reordered_block.begin() + 2, reordered_block.begin() + 4);
				qtt_REQUIRE(other_indices.size() == 1);
				for (const auto &rb_slices : row_block_slices)
				{
					for (const auto &cb_slices : col_block_slices)
					{
						auto [block_ind, slice] = LA_helpers::build_index_slice(other_indices, rb_slices, cb_slices);
						// we have rebuilt the block index in block_ind. It would be good to have a function to do that.
						qtt_CHECK(torch::equal(A.block(block_ind), compact_tensor.index(slice)));
						// then we verify that the blocks from the original tensor can be obtained from the stored
						// slincing of the compacted tensor
					}
				}
			}
			{
				btensor B;
				qtt_REQUIRE_NOTHROW(B = A.reshape({1})); // joins dimensions 2 and 1
				// fmt::print("tensor {}\n",B);
				auto reordered_block = LA_helpers::reorder_by_cvals(B);
				std::vector<btensor::index_list> exp_blocks = {{0, 2}, {0, 3}, {0, 11}, {1, 1},
				                                               {1, 5}, {1, 6}, {1, 10}};
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
				    LA_helpers::compact_dense_single(reordered_block.begin(), reordered_block.begin() + 3);
				qtt_REQUIRE(other_indices.size() == 0);
				for (const auto &rb_slices : row_block_slices)
				{
					for (const auto &cb_slices : col_block_slices)
					{
						auto [block_ind, slice] = LA_helpers::build_index_slice(other_indices, rb_slices, cb_slices);
						// fmt::print("block {}, cval {}\n",block_ind,B.block_quantities(block_ind));
						qtt_CHECK(torch::equal(B.block(block_ind), compact_tensor.index(slice)));
						// then we verify that the blocks from the original tensor can be obtained from the stored
						// slincing of the compacted tensor
					}
				}
			}
		}
#endif // NDEBUG
       // fmt::print("V {}\n",V);
		qtt_CHECK_NOTHROW(btensor::throw_bad_tensor(U));
		qtt_CHECK_NOTHROW(btensor::throw_bad_tensor(d));
		qtt_REQUIRE_NOTHROW(btensor::throw_bad_tensor(V));
		auto d_r =
		    d.reshape_as(shape_from(d, btensor({{{1, d.selection_rule->neutral()}}}, d.selection_rule->neutral())))
		        .transpose_(-1, -2);
		auto Vt = V.transpose(-1, -2).conj();
		auto AA = U.mul(d_r).bmm(Vt);
		// fmt::print("U {}\n",U);
		// fmt::print("V {}\n",V);
		auto it_AA = AA.begin();
		auto it_A = A.begin();
		qtt_REQUIRE(std::distance(it_AA, AA.end()) == std::distance(it_A, A.end()));
		for (; it_AA != AA.end(); ++it_AA, ++it_A)
		{
			auto &AA_ind = std::get<0>(*it_AA);
			auto &A_ind = std::get<0>(*it_A);
			auto &AA_tens = std::get<1>(*it_AA);
			auto &A_tens = std::get<1>(*it_A);
			qtt_CHECK(AA_ind == A_ind);
			qtt_CHECK(torch::allclose(AA_tens, A_tens));
		}
		auto Ut = U.transpose(-2, -1).conj();
		// fmt::print("Ut {}\n",Ut );
		// fmt::print("U {}\n", U);
		auto ID_u = U.bmm(Ut); // ATTN! In general, Ut.bmm(Ut) != identity
		auto ID_v = Vt.bmm(V); // ATTN! In general, V.bmm(Vt) != Identity
		// fmt::print("ID_U {}", ID_u);
		for (auto &block : ID_u)
		{
			auto &ind = std::get<0>(block);
			auto &tens = std::get<1>(block);
			if (ind[ind.size() - 1] == ind[ind.size() - 2])
			{
				qtt_CHECK(tens.sizes()[tens.dim() - 1] == tens.sizes()[tens.dim() - 2]);
				auto id = torch::eye(tens.sizes()[tens.dim() - 1]);
				qtt_CHECK(torch::allclose(id, tens));
			}
			else
			{
				auto zer = torch::zeros({tens.sizes()[tens.dim() - 2], tens.sizes()[tens.dim() - 1]});
				qtt_CHECK(torch::allclose(zer, tens));
			}
		}
		// fmt::print("ID_V \n{}\n\n",ID_v);
		// fmt::print("Vt \n{}\n\n",Vt);
		// fmt::print("V \n{}\n\n",V);
		for (auto &block : ID_v)
		{
			auto &ind = std::get<0>(block);
			auto &tens = std::get<1>(block);
			if (ind[ind.size() - 1] == ind[ind.size() - 2])
			{
				qtt_CHECK(tens.sizes()[tens.dim() - 1] == tens.sizes()[tens.dim() - 2]);
				auto id = torch::eye(tens.sizes()[tens.dim() - 1]);
				qtt_CHECK(torch::allclose(id, tens));
			}
			else
			{
				auto zer = torch::zeros({tens.sizes()[tens.dim() - 2], tens.sizes()[tens.dim() - 1]});
				qtt_CHECK(torch::allclose(zer, tens));
			}
		}
	}
	qtt_SUBCASE("tensor decomposition")
	{
		using cqt = conserved::C<5>;
		using index = btensor::index_list;
		double tole = 1e-5;
		any_quantity selection_rule(cqt(0)); // DMRJulia flux
		btensor A({{{2, cqt(0)}, {3, cqt(1)}},
		           {{1, cqt(1)}, {2, cqt(0)}, {3, cqt(-1)}, {1, cqt(1)}},
		           {{3, cqt(0)}, {2, cqt(-2)}, {2, cqt(-1)}}},
		          selection_rule);
		A.block({0, 0, 2}) = tole * torch::rand({2, 1, 2});
		A.block({0, 1, 0}) = tole * torch::rand({2, 2, 3});
		A.block({0, 3, 2}) = 0.1 * tole * torch::rand({2, 1, 2});
		A.block({1, 0, 1}) = tole * torch::rand({3, 1, 2});
		A.block({1, 1, 2}) = tole * torch::rand({3, 2, 2});
		A.block({1, 2, 0}) = 0.1 * tole * torch::rand({3, 3, 3});
		A.block({1, 3, 1}) = tole * torch::rand({3, 1, 2});
		// fmt::print("A \n{}\n\n",A);
		qtt_REQUIRE_NOTHROW(btensor::throw_bad_tensor(A));
		qtt_SUBCASE("tensor singular decomposition")
		{
			btensor U, d, V;
			qtt_REQUIRE_NOTHROW(std::tie(U, d, V) = svd(A, 1));
			// fmt::print("U \n{}\n\n",U);
			// fmt::print("d \n{}\n\n",d);
			// fmt::print("V \n{}\n\n",V);
			auto Ud = U.mul(d);
			// fmt::print("Ud \n{}\n\n",Ud);
			auto AA = tensordot(Ud, V.conj(), {U.dim() - 1}, {V.dim() - 1});
			// fmt::print("AA \n{}\n\n",AA);
			auto AA_it = AA.begin();
			auto A_it = A.begin();
			qtt_REQUIRE(std::distance(AA_it, AA.end()) == std::distance(A_it, A.end()));
			while (AA_it != AA.end())
			{
				auto AA_ind = std::get<0>(*AA_it);
				auto A_ind = std::get<0>(*A_it);
				auto AA_tens = std::get<1>(*AA_it);
				auto A_tens = std::get<1>(*A_it);
				qtt_CHECK(AA_ind == A_ind);
				qtt_CHECK(torch::allclose(AA_tens, A_tens));
				++AA_it;
				++A_it;
			}
		}
		qtt_SUBCASE("random tensor decomposition")
		{
			using cqt = conserved::C<2>;
			btensor dummy = rand({}, cqt(0));
			btensor X = quantit::rand({{{2, cqt(-2)}, {2, cqt(0)}, {2, cqt(2)}},
			                          {{1, cqt(1)}, {1, cqt(-1)}},
			                          {{1, cqt(1)}, {1, cqt(-1)}},
			                          {{2, cqt(2)}, {2, cqt(0)}, {2, cqt(-2)}}},
			                         cqt(0));
			auto [U, d, V] = svd(X, 2);
			qtt_CHECK(tensordot(U, U.conj(), {0, 1, 2}, {0, 1, 2}).item().toDouble() == doctest::Approx(d.sizes()[0]));
			qtt_CHECK(tensordot(V, V.conj(), {0, 1, 2}, {0, 1, 2}).item().toDouble() == doctest::Approx(d.sizes()[0]));
			qtt_CHECK(tensordot(U, U.conj(), {2, 0, 1}, {2, 0, 1}).item().toDouble() == doctest::Approx(d.sizes()[0]));
			qtt_CHECK(tensordot(V, V.conj(), {2, 0, 1}, {2, 0, 1}).item().toDouble() == doctest::Approx(d.sizes()[0]));
			// fmt::print("U shape {}\n\nV shape {}\n\n d {}\n\n",shape_from(U,dummy),shape_from(V,dummy),d);
			auto U2 = U.reshape({2});
			auto V2 = V.reshape({2});
			// fmt::print("U {}\n\nV {}\n\n", tensordot(U2,U2.conj(), {1,0},{1,0}),tensordot(V2,V2.conj(),{1,0},{1,0}));
			qtt_CHECK(allclose(tensordot(U.mul(d), V.conj(), {2}, {2}), X));
			// auto XX = tensordot(U.mul(d), V.conj(), {2}, {2});
			// fmt::print("X\n{}\n\n", shape_from(X, dummy));
			// fmt::print("reconstituded X\n{}\n\n", shape_from(XX, dummy));
			// fmt::print("U\n{}\n\n", shape_from(U, dummy));
			// fmt::print("d\n{}\n\n", shape_from(d, dummy));
			// fmt::print("V\n{}\n\n", shape_from(V, dummy));
		}
		qtt_SUBCASE("truncating tensor singular decomposition")
		{
			btensor U, d, V;
			qtt_REQUIRE_NOTHROW(std::tie(U, d, V) = svd(A, 1, tole));
			// fmt::print("U \n{}\n\n",U);
			// fmt::print("d \n{}\n\n",d);
			// fmt::print("V \n{}\n\n",V);
			auto AA = tensordot(U.mul(d), V.conj(), {U.dim() - 1}, {V.dim() - 1});
			// fmt::print("AA \n{}\n\n",AA);
			auto AA_it = AA.begin();
			auto A_it = A.begin();
			qtt_REQUIRE(std::distance(AA_it, AA.end()) <= std::distance(A_it, A.end()));
			while (AA_it != AA.end())
			{
				auto AA_ind = std::get<0>(*AA_it);
				auto A_ind = std::get<0>(*A_it);
				auto AA_tens = std::get<1>(*AA_it);
				auto A_tens = std::get<1>(*A_it);
				if (AA_ind == A_ind)
				{
					qtt_CHECK(torch::all(torch::less(torch::abs(A_tens - AA_tens), tole)).item().to<bool>());
					// if (not torch::all(torch::less(torch::abs(A_tens-AA_tens),tol)).item().to<bool>())
					// {
					// 	fmt::print("reduction check failed: ind {}\n",A_ind);
					// 	fmt::print("absolute difference \n{}\n\n",torch::abs(A_tens-AA_tens));
					// }
					// qtt_CHECK(torch::allclose(AA_tens, A_tens, tol, tol));
					++AA_it;
					++A_it;
				}
				else
				{
					// if a block from A is literally not present in AA, then that block must be all zeros to the
					// tol.
					qtt_CHECK(torch::all(torch::less(torch::abs(A_tens), tole)).item().to<bool>());
					// if (not torch::all(torch::less(torch::abs(A_tens),tol)).item().to<bool>())
					// {
					// 	fmt::print("block removed check failed: ind {}\n",A_ind);
					// 	fmt::print("a_tens \n{}\n\n",A_tens);
					// }
					// qtt_CHECK(torch::allclose(A_tens,torch::zeros_like(A_tens),tol,tol));
					bool AAlessA = AA_ind < A_ind;
					AA_it += AAlessA;
					A_it += !AAlessA;
				}
			}
		}
		qtt_SUBCASE("truncating smaller tensor singular decomposition")
		{
			btensor U, d, V;
			A.mul_(0.3);
			qtt_REQUIRE_NOTHROW(std::tie(U, d, V) = svd(A, 1, tole));
			// fmt::print("U \n{}\n\n",U);
			// fmt::print("d \n{}\n\n",d);
			// fmt::print("V \n{}\n\n",V);
			auto AA = tensordot(U.mul(d), V.conj(), {U.dim() - 1}, {V.dim() - 1});
			// fmt::print("AA \n{}\n\n",AA);
			auto AA_it = AA.begin();
			auto A_it = A.begin();
			qtt_REQUIRE(std::distance(AA_it, AA.end()) <= std::distance(A_it, A.end()));
			while (AA_it != AA.end())
			{
				auto AA_ind = std::get<0>(*AA_it);
				auto A_ind = std::get<0>(*A_it);
				auto AA_tens = std::get<1>(*AA_it);
				auto A_tens = std::get<1>(*A_it);
				if (AA_ind == A_ind)
				{
					qtt_CHECK(torch::all(torch::less(torch::abs(A_tens - AA_tens), tole)).item().to<bool>());
					// if (not torch::all(torch::less(torch::abs(A_tens-AA_tens),tol)).item().to<bool>())
					// {
					// 	fmt::print("reduction check failed: ind {}\n",A_ind);
					// 	fmt::print("absolute difference \n{}\n\n",torch::abs(A_tens-AA_tens));
					// }
					// qtt_CHECK(torch::allclose(AA_tens, A_tens, tol, tol));
					++AA_it;
					++A_it;
				}
				else
				{
					// if a block from A is literally not present in AA, then that block must be all zeros to the
					// tol.
					qtt_CHECK(torch::all(torch::less(torch::abs(A_tens), tole)).item().to<bool>());
					// if (not torch::all(torch::less(torch::abs(A_tens),tol)).item().to<bool>())
					// {
					// 	fmt::print("block removed check failed: ind {}\n",A_ind);
					// 	fmt::print("a_tens \n{}\n\n",A_tens);
					// }
					// qtt_CHECK(torch::allclose(A_tens,torch::zeros_like(A_tens),tol,tol));
					bool AAlessA = AA_ind < A_ind;
					AA_it += AAlessA;
					A_it += !AAlessA;
				}
			}
		}
	}
}

} // namespace quantit

#endif // BTENSORLINEARALGEBRA_H
