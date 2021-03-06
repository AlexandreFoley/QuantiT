/*
 * File: btensor.h
 * Project: QuantiT
 * File Created: Thursday, 1st October 2020 10:54:53 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Friday, 8th January 2021 12:02:08 pm
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2021 Alexandre Foley
 * Licensed under GPL v3
 */

#ifndef D49FFA60_85C4_431A_BA62_9B1D30D67E86
#define D49FFA60_85C4_431A_BA62_9B1D30D67E86

#include "Conserved/Composite/cquantity.h"
#include "Conserved/Composite/quantity_vector.h"
#include "Conserved/quantity.h"
#include "blockTensor/flat_map.h"
#include "boost/stl_interfaces/iterator_interface.hpp"
#include "boost/stl_interfaces/view_interface.hpp"
#include "property.h"
#include "torch_formatter.h"
#include <ATen/Functions.h>
#include <algorithm>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/core/TensorOptions.h>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <torch/csrc/utils/variadic.h>
#include <torch/torch.h>
#include <type_traits>
#include <vector>

#include "doctest/doctest_proxy.h"

namespace quantit
{
class block_qtt_view;
enum class btensor_size
{
	max
};
enum class reshape_mode
{
	dims_only,
	overwrite_c_vals
};
/**
 * @brief if any of the element in the range convert to true, return true.
 *
 * @return true at least one element converts to true
 * @return false no element convert to true
 */
template <class T>
bool any_truth(const T &in)
{
	bool out = false;
	for (auto &it : in)
	{
		out |= bool(it);
		if (out)
			break;
	}
	return out;
}
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
              S0,0 ??? S0,1 ??? S0,2 ??? S0,3
             ?????????????????????????????????????????????????????????????????????????????????
             ???     ???      ???      ???     ???
         S1,0???(0,0)??? (0,1)??? (0,2)???(0,3)???
             ???     ???      ???      ???     ???
            ????????????????????????????????????????????????????????????????????????????????????
             ???     ???      ???      ???     ???
         S1,1???(1,0)??? (1,1)??? (1,2)???(1,3)???
             ???     ???      ???      ???     ???
            ????????????????????????????????????????????????????????????????????????????????????
             ???     ???      ???      ???     ???
         S1,2???(2,0)??? (2,1)??? (2,2)???(2,3)???
             ???     ???      ???      ???     ???
             ?????????????????????????????????????????????????????????????????????????????????
 \endverbatim
 * In the preceding exemple, the rows are separated in 4 sections, and the columns in 3 sections.
 * This make up to 12 blocks, that we label by section.
 * Let's consider that the conserved quantity is simply an integer under the addition,that the column sections
 * [-2,-1,1], the row sections have the conserved quantity [1,2,3,-1] and the selection rule is 0. In that case, only
 * the blocks [(1,0),(0,1),(2,3)] can be non-zero.
 *
 */
class btensor
{
  public:
	using index_list = std::vector<int64_t>;
	using block_list_t = flat_map<index_list, torch::Tensor>;
	using init_list_t = std::initializer_list<std::initializer_list<std::tuple<size_t, any_quantity>>>;
	using vec_list_t = std::vector<std::vector<std::tuple<size_t, any_quantity>>>;

	using Scalar = torch::Scalar;
	property<any_quantity, btensor, any_quantity_cref> selection_rule; // dmrjulia equiv: the flux.
	/**
	 * @brief Construct a new btensor object
	 *
	 * @param dir_block_size_cqtt a nested list of pair of section size and conserved quantities. The number of element
	 * in the first is level is the rank of the tensor. The number of elements in the second level is the number of
	 * section for that dimension of the tensor
	 * @param selection_rule determine which blocks are allowed to be non-zero.
	 */
	btensor(const vec_list_t &dir_block_size_cqtt, any_quantity_cref selection_rule, c10::TensorOptions opt = {});
	btensor(const vec_list_t &dir_block_size_cqtt, any_quantity_cref selection_rule, size_t num_blocks,
	        c10::TensorOptions opt = {});
	// btensor(vec_list_t, any_quantity,c10::TensorOptions opt = {}); //for python
	/**
	 * @brief Construct a new btensor object from a subset of the raw structure. use carefully.
	 *
	 * @param _sections_by_dim
	 * @param _c_vals
	 * @param _section_sizes
	 * @param _sel_rule
	 */
	btensor(index_list _sections_by_dim, any_quantity_vector _c_vals, index_list _section_sizes, any_quantity _sel_rule,
	        c10::TensorOptions opt = {});
	btensor(const btensor &other)
	    : selection_rule((other.selection_rule.value)), rank(other.rank), sections_by_dim((other.sections_by_dim)),
	      sections_sizes((other.sections_sizes)), blocks_list((other.blocks_list)), c_vals((other.c_vals)),
	      _options(other._options)
	{
	}
	btensor(btensor &&other)
	    : selection_rule(std::move(other.selection_rule.value)), rank(other.rank),
	      sections_by_dim(std::move(other.sections_by_dim)), sections_sizes(std::move(other.sections_sizes)),
	      blocks_list(std::move(other.blocks_list)), c_vals(std::move(other.c_vals)),
	      _options(std::move(other._options))
	{
	}
	btensor &operator=(btensor other)
	{
		swap(other);
		return *this;
	}
	explicit btensor() = default;
	/**
	 * @brief Construct a new btensor object. construct from raw structure elements. Avoid using this constructor if you
	 * can.
	 *
	 * @param _rank : the number of dimension of the tensor
	 * @param _blocks : list of pair<position,sub-tensor>, the position is stored in a block index
	 * @param _sections_by_dims : number of section for each dimension of the tensor
	 * @param _section_sizes : number of element for each section of each dimension
	 * @param _c_vals : conserved quantity associated to each of the section in each of the dimension
	 * @param _sel_rule : overall selection rule, the sum over the dimension of the conserved quantities of a given
	 * block must equal this value for a block to be allowed to differ from zero.
	 */
	btensor(size_t _rank, block_list_t _blocks, index_list _sections_by_dims, index_list _sections_sizes,
	        any_quantity_vector _c_vals, any_quantity _sel_rule, c10::TensorOptions opt = {});

	/**
	 * @brief return the list of the sizes along each dimensions of the tensor
	 *
	 * @return std::vector<int64_t>
	 */
	std::vector<int64_t> sizes() const;
	/**
	 * @brief if the tensor contain a single element, return a scalar object.
	 *
	 * @return btensor::Scalar
	 */
	btensor::Scalar item() const;
	/**
	 * @brief increment a block index for this tensor
	 *
	 * @param block_index reference to the block index to increment.
	 */
	void block_increment(btensor::index_list &block_index) const;
	static size_t btensor_compute_max_size(const btensor &btens, size_t max = std::numeric_limits<size_t>::max());
	static void add_tensor_check(const btensor &a, const btensor &b);

	void swap(btensor &);
	// utility classes
	template <class val_iter>
	struct block_prop_iter;
	template <class const_val_iter, class val_iter>
	struct const_block_prop_iter;
	template <class iter>
	struct block_prop_view;
	template <class const_iterator, class iterator>
	struct const_block_prop_view;
	using block_qtt_iter = block_prop_iter<any_quantity_vector::iterator>;
	using const_block_qtt_iter =
	    const_block_prop_iter<any_quantity_vector::const_iterator, any_quantity_vector::iterator>;
	using block_qtt_view = block_prop_view<block_qtt_iter>;
	using const_block_qtt_view = const_block_prop_view<const_block_qtt_iter, block_qtt_iter>;
	using block_size_iter = block_prop_iter<index_list::iterator>;
	using const_block_size_iter = const_block_prop_iter<index_list::const_iterator, index_list::iterator>;
	using block_size_view = block_prop_view<block_size_iter>;
	using const_block_size_view = const_block_prop_view<const_block_size_iter, block_size_iter>;

	bool block_conservation_rule_test(index_list block_index) const;

	size_t section_size(size_t dim, size_t section) const;
	std::tuple<index_list::const_iterator, index_list::const_iterator> section_sizes(size_t dim) const;
	std::tuple<any_quantity_vector::const_iterator, any_quantity_vector::const_iterator> section_cqtts(
	    size_t dim) const;
	std::tuple<index_list::const_iterator, index_list::const_iterator, any_quantity_vector::const_iterator,
	           any_quantity_vector::const_iterator>
	section_sizes_cqtts(size_t dim) const;

	any_quantity_cref section_conserved_qtt(size_t dim, size_t section) const;
	any_quantity_cref element_conserved_qtt(size_t dim, size_t element) const;
	std::tuple<any_quantity_vector::const_iterator, any_quantity_vector::const_iterator> section_conserved_qtt_range(
	    size_t index) const;
	std::tuple<size_t, any_quantity_cref> section_size_cqtt(size_t dim, size_t section) const;
	// block accessor
	/**
	 * @brief access the block at the block index given in argument.
	 *
	 * Throws a std::out_of_range if the block isn't allocated or allowed.
	 *
	 * @return torch::Tensor&
	 */
	torch::Tensor &block_at(const index_list &);
	/**
	 * @brief access the block at the index given in argument. Allocate space for the block if necessary.
	 *
	 * Throws a std::bad_argument if the block isn't allowed by the conservation law.
	 *
	 * @return torch::Tensor&
	 */
	torch::Tensor &block(const index_list &); // create the block if it isn't present and allowed

	/**
	 * @brief const reference to the raw block list.
	 *
	 * @return const block_list_t&
	 */
	const block_list_t &blocks() const;

	/**
	 * @brief obtain a view on the conserved quantities of each indices of a block with the block index given in
	 * argument
	 *
	 * the conserved quantity for any block can be accessed that way, whether non-zero values are allowed or not.
	 *
	 * @param block_index index of the block
	 * @return const_block_qtt_view
	 */
	const_block_qtt_view block_quantities(const index_list &block_index) const;
	/**
	 * @brief obtain a view on the size of the block.
	 *
	 * The size of any block can be accessed in this manner, whether non-zero values are allowed or not.
	 *
	 * @param block_index
	 * @return const_block_size_view
	 */
	const_block_size_view block_sizes(const index_list &block_index) const;
	/**
	 * @brief Return the rank of the tensor
	 *
	 * @return int64_t
	 */
	int64_t dim() const { return rank; }
	/**
	 * @brief return the number of sections in a given dimension
	 *
	 * @param dim
	 * @return size_t
	 */
	size_t section_number(size_t dim) const { return sections_by_dim[dim]; }
	/**
	 * @brief return the number of section for all the dimensions, in order.
	 *
	 * @return const auto&
	 */
	const auto &section_numbers() const { return sections_by_dim; }
	// block_qtt_view block_quantities(index_list block_index);

	const auto &get_cvals() const { return c_vals; }

	/**
	 * @brief create an empty tensor from selected dimensions of this. Minimum necessary set of feature for tensor
	 * network reshape.
	 *
	 * Basic version of index, it can only discard whole dimensions.
	 *
	 * @param dims List of dimensions, put -1 to keep the dimension, specify the index to keep otherwise.
	 * @return btensor
	 */
	btensor shape_from(const std::vector<int64_t> &dims) const;
	/**
	 * @brief Create a view object on this tensor. Minimum necessary set of feature for tensor network reshape.
	 *
	 * Basic version of index, it can only discard whole dimensions.
	 *
	 * @param dims List of dimensions, put -1 to keep the dimension, specify the index to keep otherwise.
	 * @return btensor&
	 */
	btensor basic_create_view(const std::vector<int64_t> &dims, bool preserve_rank = false);
	/**
	 * @brief make all the the conserved value and the conservation rule the neutral element of the group. works only on
	 * empty tensors.
	 *
	 * @return btensor&
	 */

	btensor &basic_index_put_(const std::vector<int64_t> &dims, const btensor &value);
	btensor &basic_index_put_(const std::vector<int64_t> &dims, const torch::Tensor &value);

	btensor neutral_shape() const;
	/**

	 * @param other
	 * @return btensor
	 */
	btensor &neutral_shape_();
	/**
	 * @brief compute the shape of the tensor product of this with the other tensor. store the shape information in an
	 * empty btensor
	 *
	 * If you want to actually proceed to the tensor product, use tensordot with no contracted
	 *
	 * @param other
	 * @return btensor
	 */
	btensor tensor_product_shape(const btensor &other) const;

	/**
	 * @brief create a view on the block tensor.
	 *
	 * The view on the btensor is itself a btensor. The underlying non-zero blocks are shared, new blocks cannot be
	 * added to the original tensor this way.
	 *
	 * @param indices torch's index class, allow for slices, ellipsis, boolean, tensors, and simple index.
	 * @return btensor view on the original tensor
	 */
	btensor index(torch::ArrayRef<torch::indexing::TensorIndex> indices) const;
	/**
	 * @brief create a view on the block tensor.
	 *
	 * The view on the btensor is itself a btensor. The underlying non-zero blocks are shared, new blocks cannot be
	 * added to the original tensor this way.
	 *
	 * @param indicesa list of torch's index class, allow for slices, ellipsis, and index.
	 * @return btensor view on the original tensor
	 */
	btensor index(std::initializer_list<torch::indexing::TensorIndex> indices) const;
	/**
	 * @brief Elements insertion operator, for basic torch tensor
	 *
	 * The input tensor is sliced into blocks. elements of the input tensors disallowed by the conservation law are
	 * silently droped.
	 *
	 * @param indices  list of torch's index class, allow for slices, ellipsis, and simple index.
	 * @param rhs tensor of values to insert, it's shape must match the view described by the indices
	 * @return btensor& reference to this
	 */
	btensor &index_put_(torch::ArrayRef<torch::indexing::TensorIndex> indices, const torch::Tensor &rhs);
	/**
	 * @brief Elements insertion operator, for block tensor
	 *
	 * The input tensor block structure must match the view's block structure
	 *
	 * @param indices  list of torch's index class, allow for slices, ellipsis, and simple index.
	 * @param rhs tensor of values to insert, it's shape must match the view described by the indices
	 * @return btensor& reference to this
	 */
	btensor &index_put_(torch::ArrayRef<torch::indexing::TensorIndex> indices, const btensor &rhs);
	/**
	 * @brief Elements insertion operator, for scalars
	 *
	 * If the index describes multiple elements, all the element that respect the conservation law are set to the
	 * supplied value.
	 *
	 * @param indices list of torch's index class, allow for slices, ellipsis, and simple index.
	 * @param rhs tensor of values to insert, it's shape must match the view described by the indices
	 * @return btensor& reference to this
	 */
	btensor &index_put_(torch::ArrayRef<torch::indexing::TensorIndex> indices, const Scalar &v);
	/**
	 * @brief Elements insertion operator, for basic torch tensor
	 *
	 * The input tensor is sliced into blocks. elements of the input tensors disallowed by the conservation law are
	 * silently droped.
	 *
	 * @param indices  list of torch's index class, allow for slices, ellipsis, and simple index.
	 * @param rhs tensor of values to insert, it's shape must match the view described by the indices
	 * @return btensor& reference to this
	 */
	btensor &index_put_(std::initializer_list<torch::indexing::TensorIndex> indices, const torch::Tensor &rhs);
	/**
	 * @brief Elements insertion operator, for block tensor
	 *
	 * The input tensor block structure must match the view's block structure
	 *
	 * @param indices  list of torch's index class, allow for slices, ellipsis, and simple index.
	 * @param rhs tensor of values to insert, it's shape must match the view described by the indices
	 * @return btensor& reference to this
	 */
	btensor &index_put_(std::initializer_list<torch::indexing::TensorIndex> indices, const btensor &rhs);
	/**
	 * @brief Elements insertion operator, for scalars
	 *
	 * If the index describes multiple elements, all the element that respect the conservation law are set to the
	 * supplied value.
	 *
	 * @param indices  list of torch's index class, allow for slices, ellipsis, and simple index.
	 * @param rhs tensor of values to insert, it's shape must match the view described by the indices
	 * @return btensor& reference to this
	 */
	btensor &index_put_(std::initializer_list<torch::indexing::TensorIndex> indices, const Scalar &v);

	// iterator
	block_list_t::const_iterator begin() const { return blocks_list.begin(); }
	block_list_t::const_iterator end() const { return blocks_list.end(); }
	block_list_t::const_iterator cbegin() const { return blocks_list.cbegin(); }
	block_list_t::const_iterator cend() const { return blocks_list.cend(); }
	block_list_t::iterator begin() { return blocks_list.begin(); }
	block_list_t::iterator end() { return blocks_list.end(); }
	block_list_t::reverse_iterator rbegin() { return blocks_list.rbegin(); }
	block_list_t::reverse_iterator rend() { return blocks_list.rend(); }
	block_list_t::const_reverse_iterator rbegin() const { return blocks_list.rbegin(); }
	block_list_t::const_reverse_iterator rend() const { return blocks_list.rend(); }
	block_list_t::const_reverse_iterator crbegin() const { return blocks_list.crbegin(); }
	block_list_t::const_reverse_iterator crend() const { return blocks_list.crend(); }

	/**
	 * @brief Convert the block tensor to a regular torch tensor
	 *
	 * @return torch::Tensor
	 */
	torch::Tensor to_dense() const;
	/**
	 * @brief Check that the tensor is correct.
	 *
	 * No forbidden element allocated, torch tensor sizes matches their sectors sizes, etc.
	 *
	 * return a string explaining violations.
	 *
	 * @return std::string
	 */
	static std::string check_tensor(const btensor &);
	/**
	 * @brief throw an error in any situation where check_tensor return a non-empty string
	 *
	 */
	static void throw_bad_tensor(const btensor &);
	// Algebra !!Attention!! most of this stuff isn't implemented.
	btensor add(const btensor &other, Scalar alpha = 1) const;
	btensor add(btensor &&other, Scalar alpha = 1) const;
	btensor &add_(const btensor &other, Scalar alpha = 1);
	btensor &add_(btensor &&other, Scalar alpha = 1);
	btensor add(Scalar other, Scalar alpha = 1) const;
	btensor &add_(Scalar other, Scalar alpha = 1);
	btensor &operator+=(Scalar other) { return add_(other); };
	btensor &operator-=(Scalar other) { return sub_(other); };
	btensor &operator+=(const btensor &other) { return add_(other); }
	btensor &operator+=(btensor &&other) { return add_(std::move(other)); }
	btensor &operator-=(const btensor &other) { return add_(other, -1); }
	btensor &operator-=(btensor &&other) { return add_(std::move(other), -1); }
	btensor addmv(const btensor &mat, const btensor &vec, Scalar beta = 1, Scalar alpha = 1) const;
	btensor &addmv(const btensor &mat, const btensor &vec, Scalar beta = 1);
	// addmm do fused matrix multiply-add on the last two dimensions, with broadcasting on the other dimensions
	btensor addmm(const btensor &mat, const btensor &mat2, Scalar beta = 1, Scalar alpha = 1) const;
	btensor &addmm_(const btensor &mat, const btensor &mat2, Scalar beta = 1, Scalar alpha = 1);
	// addbmm function do fused matrix multiply-add, with a reduction on the first tensor index. See baddbmm if
	// reduction is not desired
	btensor addbmm(const btensor &mat, const btensor &mat2, Scalar beta = 1, Scalar alpha = 1) const;
	btensor &addbmm_(const btensor &mat, const btensor &mat2, Scalar beta = 1, Scalar alpha = 1);
	btensor &addcdiv_(const btensor &tensor1, const btensor &tensor2, Scalar beta = 1);
	btensor addcdiv(const btensor &tensor1, const btensor &tensor2, Scalar beta = 1);
	btensor &addcmul_(const btensor &tensor1, const btensor &tensor2, Scalar beta = 1);
	btensor addcmul(const btensor &tensor1, const btensor &tensor2, Scalar beta = 1);
	// baddbmm is the batched fused mutiply add.
	btensor baddbmm(const btensor &bathc1, const btensor &batch2, Scalar beta = 1, Scalar alpha = 1) const;
	btensor &baddbmm_(const btensor &bathc1, const btensor &batch2, Scalar beta = 1, Scalar alpha = 1);
	// batched matrix-multiply, no broadcast
	btensor bmm(const btensor &mat) const;
	btensor dot(const btensor &other) const;
	btensor vdot(const btensor &other) const;
	btensor kron(const btensor &other) const;
	// broadcasting matmul (batched)
	btensor matmul(const btensor &other) const;
	// matmul no broadcast
	btensor mm(const btensor &other) const;
	btensor sum() const;

	btensor t() const { return transpose(dim() - 1, dim() - 2); }
	btensor &t_() { return transpose_(dim() - 1, dim() - 2); }
	btensor sqrt() const;
	btensor &sqrt_();
	btensor abs() const;
	btensor &abs_();
	btensor pow(btensor::Scalar exponent) const;
	btensor &pow_(btensor::Scalar exponent);
	btensor pow(const btensor &exponent) const;
	btensor &pow_(const btensor &exponent);
	btensor ge(btensor::Scalar other) const;
	btensor ge(const btensor &other) const;
	btensor le(btensor::Scalar other) const;
	btensor le(const btensor &other) const;
	btensor less(const btensor &other) const;
	btensor less(btensor::Scalar other) const;
	btensor greater(const btensor &other) const;
	btensor greater(btensor::Scalar other) const;
	btensor eq(btensor::Scalar other) const;
	btensor eq(const btensor &other) const;
	btensor not_equal(btensor::Scalar other) const;
	btensor not_equal(const btensor &other) const;

	btensor div(btensor::Scalar) const;
	btensor &div_(btensor::Scalar);
	btensor div(const btensor &other) const;
	btensor &div_(const btensor &other);

	btensor &operator*=(btensor::Scalar val) { return mul_(val); }
	btensor &operator*=(const btensor &other) { return mul_(other); }
	btensor &operator/=(btensor::Scalar val) { return div_(val); }
	btensor &operator/=(const btensor &other) { return div_(other); }
	btensor operator-() { return mul(-1); }

	/**
	 * @brief in-place element wise product, with broadcasting on size 1 dimensions.
	 *
	 * Will throw an error if the rank of this is smaller than the would be output.
	 * I suspect this is a bug in torch. There's no reason not to adapt the rank of the tensor,
	 * from a storage perspective this isn't different from increasing the size of one dimenion, which this function can
	 * do.
	 *
	 * @param other
	 * @return btensor&
	 */
	btensor &mul_(const btensor &other);
	btensor mul(const btensor &other) const;
	btensor mul(Scalar other) const;
	btensor &mul_(Scalar other);
	btensor multiply(const btensor &other) const;
	btensor &multiply_(const btensor &other);
	btensor multiply(Scalar other) const { return mul(other); }
	btensor &multiply_(Scalar other) { return mul_(other); }
	btensor mv(const btensor &vec) const;
	btensor permute(torch::IntArrayRef) const;
	btensor &permute_(torch::IntArrayRef);
	/**
	 * @brief Reshape the btensor into a btensor of a lower rank
	 *
	 * This function is significantly different from torch's equivalent, both in the required input and the resulting
	 * tensor. The reshaping is done once on the block structure and once on the block content. Consequently, the
	 * content is permuted relative to the same reshape done on a regular tensor.
	 *
	 * @param index_group each integer mark the first element of the next bundle of index to group
	 * for exemple: [3,4,5] would group index 0,1,2 into a single new index, leave 3 and 4 as they are, and
	 * group 5..rank-1 together
	 * @return btensor reshaped tensor
	 */
	btensor reshape(torch::IntArrayRef index_group) const;
	/**
	 * @brief reshape the tensor into the shape of the supplied tensor.
	 *
	 * The supplied tensor must have conserved quantities compatible with this.
	 * Therefore, the conserved quantities of the blocks of this must factorize into the conserved quantity of the
	 * matching block of the arguement. Can reshape into a tensor of greater rank.
	 *
	 * When overwrite_c_vals is true, the selection rule will be overwritten with the one of the input tensor.
	 * The non-zero blocks of this must satisfy the selection rule of the proposed output shape.
	 *
	 * @tparam reshaping mode, modify the conserved quantities on the indices only by default, overwrite the selection
	 * rule with reshape_mode::overwrite_cvals
	 * @param other Tensor with the target shape.
	 * @return btensor Reshaped tensor
	 */
	template <reshape_mode mode = reshape_mode::dims_only>
	btensor reshape_as(const btensor &other) const;

	/**
	 * @brief Reshape the btensor into a btensor of a lower rank
	 *
	 * This function is significantly different from torch's equivalent, both in the required input and the resulting
	 * tensor. The reshaping is done once on the block structure and once on the block content. Consequently, the
	 * content is permuted relative to the same reshape done on a regular tensor.
	 *
	 * @param index_group each integer mark the first element of the next bundle of index to group
	 * for exemple: [3,4,5] would group index 0,1,2 into a single new index, leave 3 and 4 as they are, and
	 * group 5..rank-1 together
	 * @return btensor reshaped tensor
	 */
	btensor reshape(std::initializer_list<int64_t> a) { return reshape(torch::IntArrayRef(a)); }
	/**
	 * @brief exchange the order of two indices.
	 *
	 * Simplify to a matrix transpose for tensors of rank 2.
	 *
	 * @param dim0
	 * @param dim1
	 * @return btensor
	 */
	btensor transpose(int64_t dim0, int64_t dim1) const;
	// btensor transpose(torch::Dimname dim0, torch::Dimname dim1) const;
	btensor &transpose_(int64_t dim0, int64_t dim1);
	btensor sub(const btensor &other, Scalar alpha = 1) const { return add(other, -alpha); }
	btensor &sub_(const btensor &other, Scalar alpha = 1) { return add_(other, -alpha); }
	btensor sub(btensor &&other, Scalar alpha = 1) const { return add(std::move(other), -alpha); }
	btensor &sub_(btensor &&other, Scalar alpha = 1) { return add_(std::move(other), -alpha); }
	btensor sub(Scalar other, Scalar alpha = 1) const { return add(other, -alpha); }
	btensor &sub_(Scalar other, Scalar alpha = 1) { return add_(other, -alpha); };
	btensor subtract(const btensor &other, Scalar alpha = 1) const { return sub(other, alpha); }
	btensor &subtract_(const btensor &other, Scalar alpha = 1) { return sub_(other, alpha); }
	btensor subtract(Scalar other, Scalar alpha = 1) const { return sub(other, alpha); }
	btensor &subtract_(Scalar other, Scalar alpha = 1) { return sub_(other, alpha); }
	btensor tensordot(const btensor &other, torch::IntArrayRef dim_self, torch::IntArrayRef dims_other) const;
	btensor tensorgdot(const btensor &mul1, const btensor &mul2, torch::IntArrayRef dims1, torch::IntArrayRef dims2,
	                   Scalar beta = 1, Scalar alpha = 1) const;
	btensor &tensorgdot_(const btensor &mul1, const btensor &mul2, torch::IntArrayRef dims1, torch::IntArrayRef dims2,
	                     Scalar beta = 1, Scalar alpha = 1);
	btensor squeeze() const;
	btensor squeeze(int64_t dim) const;
	btensor &squeeze_(int64_t dim);
	btensor &squeeze_();
	btensor isnan() const;
	torch::Tensor any() const;
	bool anynan() const;
	/**
	 * @brief return the complex conjugate of this tensor and inverse the conserved quantities
	 *
	 * For non complex type, the only the conserved quantities are modified.
	 * Those two operation are grouped together because in the context of tensor networks for quantum mechanics.
	 * the adjoint operation conjugate the values and inverse the conserved quantities.
	 * The adjoint also transpose the opterator or state, but the effect of the transposition of a network on its
	 * constituant tensor depends storngly on the structure of the network and must be done on a case by case basis.
	 *
	 * grouping those two operation together is necessary to bring the textual difference in an implementation of an
	 * algorithm for torch::Tensor and quantit::btensor
	 *
	 * @return btensor
	 */
	btensor conj() const;
	/**
	 * @brief return the complex conjugate of this tensor
	 *
	 * For non complex type, the output is identical to the input.
	 * The conserved quantities are unaffected.
	 *
	 * @return btensor
	 */
	btensor conj_only() const;

	/**
	 * @brief create a new tensor with its section rule and all its conserved quantities inversed.
	 *
	 * Conserved quantities must be inversed when doing the hermitian conjugation of an operator.
	 *
	 * Caution: The blocks of the new tensors are shallow copies of the original.
	 *
	 * @return btensor
	 */
	btensor inverse_cvals() const;
	/**
	 * @brief inverse the selection rule and all the conserved quantities of this btensor.
	 *
	 * Conserved quantities must be inversed when doing the hermitian conjugation of an operator.
	 *
	 * @return btensor&
	 */
	btensor &inverse_cvals_();
	/**
	 * @brief Shifts the conserved quantities of one dimension of the tensor, applies the opposite shift to the
	 * conservation rule.
	 *
	 * @param shift shift to apply
	 * @param dim dimension to which the shift is applied
	 */
	btensor cval_shift(any_quantity_cref shift, int64_t dim) const;
	/**
	 * @brief Shifts the conserved quantities of one dimension of the tensor, applies the opposite shift to the
	 * conservation rule.
	 *
	 * @param shift shift to apply
	 * @param dim dimension to which the shift is applied
	 */
	btensor &cval_shift_(any_quantity_cref shift, int64_t dim);
	/**
	 * @brief Shifts the conserved quantities of one dimension of the tensor without regards for the conservation laws.
	 *
	 * Can only be applied to empty tensors.
	 *
	 * @param shift shift to apply
	 * @param dim dimension to which the shift is applied
	 */
	btensor &non_conserving_cval_shift_(any_quantity_cref shift, int64_t dim);
	/**
	 * @brief Modify the selection rule by the value of shift. Can only be done on empty tensors .
	 *
	 * @param shift
	 * @return btensor&
	 */
	btensor &shift_selection_rule_(any_quantity_cref shift);

	/**
	 * @brief Reserve space in the block list
	 *
	 * @param N number of blocks for which to reserve space
	 */
	void reserve_space_(size_t N);
	void reserve_space_(btensor_size);

	/**
	 * @brief Set the selection rule of the block tensor
	 *
	 * Only works on empty btensor.
	 *
	 * @param value
	 * @return btensor&
	 */
	btensor &set_selection_rule_(any_quantity_cref value);
	btensor &neutral_selection_rule_() { return set_selection_rule_(selection_rule->neutral()); }
	btensor neutral_selection_rule() const
	{
		btensor out = *this;
		return out.set_selection_rule_(selection_rule->neutral());
	}
	template <class... BTENS>
	friend std::tuple<btensor, std::tuple<BTENS...>> truncate_impl(btensor &&d, std::tuple<BTENS...> &&unitaries,
	                                                               size_t max, size_t min, btensor::Scalar tol,
	                                                               btensor::Scalar pow);

	btensor to(const torch::TensorOptions &options = {}, bool non_blocking = false, bool copy = false,
	           c10::optional<c10::MemoryFormat> memory_format = c10::nullopt) const
	{
		btensor::block_list_t out_list =
		    new_block_list_apply_to_all_blocks([&options, non_blocking, copy, memory_format](const auto &atensor)
		                                       { return atensor.to(options, non_blocking, copy, memory_format); });
		auto out = btensor(*this, std::move(out_list));
		out._options = out.begin()!= out.end() ? out.begin()->second.options() : torch::empty({},options).options();
		return out;
	}
	btensor to(const btensor &other, bool non_blocking = false, bool copy = false,
	           c10::optional<c10::MemoryFormat> memory_format = c10::nullopt) const
	{
		auto options = other.options();
		return to(options, non_blocking, copy, memory_format);
	}
	btensor to(torch::Device device, torch::ScalarType dtype, bool non_blocking = false, bool copy = false,
	           c10::optional<c10::MemoryFormat> memory_format = c10::nullopt) const
	{
		btensor::block_list_t out_list = new_block_list_apply_to_all_blocks(
		    [device, dtype, non_blocking, copy, memory_format](const auto &atensor)
		    { return atensor.to(device, dtype, non_blocking, copy, memory_format); });
		auto out = btensor(*this, std::move(out_list));
		out._options = out.begin()!= out.end() ? out.begin()->second.options() : torch::empty({},torch::TensorOptions().device(device).dtype(dtype).memory_format(memory_format)).options();
		return out;
	}
	btensor to(torch::ScalarType dtype, bool non_blocking = false, bool copy = false,
	           c10::optional<c10::MemoryFormat> memory_format = c10::nullopt) const
	{
		btensor::block_list_t out_list =
		    new_block_list_apply_to_all_blocks([dtype, non_blocking, copy, memory_format](const auto &atensor)
		                                       { return atensor.to(dtype, non_blocking, copy, memory_format); });
		auto out = btensor(*this, std::move(out_list));
		out._options = out.begin()!= out.end() ? out.begin()->second.options() :torch::empty({},torch::TensorOptions().dtype(dtype).memory_format(memory_format)).options();
		return out;
	}
	btensor to(caffe2::TypeMeta type_meta, bool non_blocking = false, bool copy = false,
	           c10::optional<c10::MemoryFormat> memory_format = c10::nullopt) const
	{
		btensor::block_list_t out_list =
		    new_block_list_apply_to_all_blocks([type_meta, non_blocking, copy, memory_format](const auto &atensor)
		                                       { return atensor.to(type_meta, non_blocking, copy, memory_format); });
		auto out = btensor(*this, std::move(out_list));
		out._options = out.begin()!= out.end() ? out.begin()->second.options() : torch::empty({},torch::TensorOptions().dtype(torch::typeMetaToScalarType(type_meta)).memory_format(memory_format)).options();
		return out;
	}
	btensor to(const torch::Tensor &other, bool non_blocking = false, bool copy = false,
	           c10::optional<c10::MemoryFormat> memory_format = c10::nullopt) const
	{
		btensor::block_list_t out_list =
		    new_block_list_apply_to_all_blocks([&other, non_blocking, copy, memory_format](const auto &atensor)
		                                       { return atensor.to(other, non_blocking, copy, memory_format); });
		auto out = btensor(*this, std::move(out_list));
		out._options = out.begin()!= out.end() ? out.begin()->second.options() : torch::empty_like(other,{},memory_format).options();
		return out;
	}
	c10::TensorOptions options() const { return _options; }
	friend inline btensor sparse_zeros_like(const btensor &tens, c10::TensorOptions opt);
	template <bool Throws = false>
	static bool check_product_compat(const btensor &in1, const btensor &in2, torch::IntArrayRef dims1,
	                                 torch::IntArrayRef dims2) noexcept(!Throws);
	/**
	 * @brief compute the slices associated with a block in a full tensor with the same shape.
	 *
	 * @param tensor
	 * @param block
	 * @return std::vector<torch::indexing::TensorIndex>
	 */
	static std::vector<torch::indexing::TensorIndex> full_slice(const btensor &tensor,
	                                                            const btensor::index_list &block);

	static bool test_same_shape(const btensor &a, const btensor &b);

	/**
	 * @brief split an index adressing an element within the full tensor into a block index, block-element index pair.
	 *
	 * @param element_index
	 * @return std::tuple<index_list,index_list>
	 */
	std::tuple<index_list, index_list> element_index_decompose(const index_list &element_index) const;

  private:
	int rank;
	/**
	 * @brief number of section for each of the dimensions of the tensor
	 *
	 */
	index_list sections_by_dim;
	/**
	 * @brief packed list of the size of each section along all dimensions.
	 *
	 */
	index_list sections_sizes; // for non-empty slices, this is strictly redundent: the information could be found by
	                           // inspecting the blocks
	// truncation should remove any and all empty slices, but user-written tensor could have empty slices.
	block_list_t blocks_list; //
	any_quantity_vector
	    c_vals; // dmrjulia equiv: QnumSum in the QTensor class. This structure doesn't need the full list (QnumMat)
	c10::TensorOptions _options;
	friend struct fmt::formatter<quantit::btensor>;
	friend class mul_helpers;
	friend btensor eye_like(const btensor& shape,c10::TensorOptions opt);

	/**
	 * @brief Construct a new btensor object, copy the shape another btensor, uses the supplied block list
	 *
	 * non-default options specified in argument overwrite the tensor options stored in the shape for the resulting
	 * btensor.
	 * @param block_list
	 */
	btensor(const btensor &shape, block_list_t &&block_list, c10::TensorOptions opt = {})
	    : selection_rule(shape.selection_rule.value), rank(shape.rank), sections_by_dim(shape.sections_by_dim),
	      sections_sizes(shape.sections_sizes), blocks_list(std::move(block_list)), c_vals(shape.c_vals),
	      _options(shape._options.merge_in(opt))
	{
	}
	/**
	 * @brief private, mutable version of the public function. With some const cast they can share implementation.
	 *
	 * @param index
	 * @return std::tuple<any_quantity_vector::iterator, any_quantity_vector::iterator>
	 */
	std::tuple<any_quantity_vector::iterator, any_quantity_vector::iterator> section_conserved_qtt_range(size_t index);
	/**
	 * @brief common part of conserving and non conserving shift.
	 *
	 * @param shift
	 * @param dim
	 */
	void shift_impl(any_quantity_cref shift, int64_t dim);
	/**
	 * @brief apply a function to all torch tensors contained in this
	 *
	 * only useful for in place operations...
	 *
	 * @param a function like object to apply to the block index, takes the block index by reference. ATTENTION: you can
	 * mess up ordering. it is your responsability to reorder the flat_map if the operation affect block index ordering.
	 * @param f function (member) reference/pointer to apply to the blovk tensors
	 * @param args argument to pass to f
	 */
	template <class A, class F, class... Args>
	void apply_to_all_blocks_mod_index(A &&a, F &&f, Args &&...args)
	{
		for (auto &b : blocks_list)
		{
			a(std::get<0>(b));
			std::invoke(std::forward<F>(f), std::get<1>(b), std::forward<Args>(args)...);
		}
	}
	/**
	 * @brief apply a function to all torch tensors contained in this
	 *
	 * only useful for in place operations...
	 *
	 * @param f function (member) reference/pointer to apply to the blovk tensors
	 * @param args argument to pass to f
	 */
	template <class F, class... Args>
	void apply_to_all_blocks(F &&f, Args &&...args)
	{
		apply_to_all_blocks_mod_index([](auto &x) {}, std::forward<F>(f), std::forward<Args>(args)...);
	}
	/**
	 * @brief apply a function to all torch tensors contained in this
	 *
	 * for out-of-place operations, asign the result to the input tensors
	 *
	 * @param a function like object to apply to the block index, takes the block index by reference. ATTENTION: you can
	 * mess up ordering. it is your responsability to reorder the flat_map if the operation affect block index ordering.
	 * @param f function (member) reference/pointer to apply to the blovk tensors
	 * @param args argument to pass to f
	 */
	template <class A, class F, class... Args>
	void force_inplace_apply_to_all_blocks_mod_index(A &&a, F &&f, Args &&...args)
	{
		for (auto &b : blocks_list)
		{
			a(std::get<0>(b));
			std::get<1>(b) = std::invoke(std::forward<F>(f), std::get<1>(b), std::forward<Args>(args)...);
		};
	}
	/**
	 * @brief apply a function to all torch tensors contained in this
	 *
	 * for out-of-place operations, asign the result to the input tensors
	 *
	 * @param f function (member) reference/pointer to apply to the blovk tensors
	 * @param args argument to pass to f
	 */
	template <class F, class... Args>
	void force_inplace_apply_to_all_blocks(F &&f, Args &&...args)
	{
		force_inplace_apply_to_all_blocks_mod_index([](auto &) {}, std::forward<F>(f), std::forward<Args>(args)...);
	}

	/**
	 * @brief apply a function to all torch tensors contained in this
	 *
	 * return a block_list containing the result of the operation
	 *
	 * @param a function like object to apply to the block index, takes the block index by reference. ATTENTION: you can
	 * mess up ordering. it is your responsability to reorder the flat_map if the operation affect block index ordering.
	 * @param f the function reference or method pointer
	 * @param args argument pack for the function to apply, excluding the torch::tensor
	 * @return block_list_t block_list containing the transformed tensors
	 */
	template <class A, class F, class... Args>
	block_list_t new_block_list_apply_to_all_blocks_mod_index(A &&a, F &&f, Args &&...args) const
	{
		block_list_t new_blocks;
		new_blocks.reserve(blocks_list.size());
		for (auto &b : blocks_list)
		{
			new_blocks.emplace(new_blocks.end(), std::get<0>(b),
			                   std::invoke(std::forward<F>(f), std::get<1>(b), std::forward<Args>(args)...));
			a(std::get<0>(*(new_blocks.end() - 1)));
		}
		return new_blocks;
	}
	/**
	 * @brief apply a function to all torch tensors contained in this
	 *
	 * return a block_list containing the result of the operation
	 *
	 * @param f the function reference or method pointer
	 * @param args argument pack for the function to apply, excluding the torch::tensor
	 * @return block_list_t block_list containing the transformed tensors
	 */
	template <class F, class... Args>
	block_list_t new_block_list_apply_to_all_blocks(F &&f, Args &&...args) const
	{
		return new_block_list_apply_to_all_blocks_mod_index([](auto &) {}, std::forward<F>(f),
		                                                    std::forward<Args>(args)...);
	}
	template <class F, bool promote = true>
	btensor broadcast_operation(const btensor &other, F &&f) const;
	template <class F, class F_>
	btensor &broadcast_operation_(const btensor &other, F &&f, F_ &&f_);
	btensor &impl_basic_index_put_(const std::vector<int64_t> &dims, const btensor &value);
};

/**
 * @brief create an empty (no allocatred blocks) btensor with the same shape and selection rule as the imput tensor
 * 
 * @param tens shape specifying tensor
 * @param opt tensor options, specified option overwrite those copied from tens
 * @return empty btensor
 */
inline btensor sparse_zeros_like(const btensor &tens, c10::TensorOptions opt = {})
{
	auto out = btensor(tens,btensor::block_list_t());
	out._options = out._options.merge_in(opt);
	return out;
}
/**
 * @brief Construct an empty block tensor from the supplied btensors.
 *
 * The output structure is the same as that of the tensor product of the supplied tensors.
 *
 * @param btens_list list of block tensors
 * @return btensor
 */
inline btensor shape_from(const std::vector<btensor> &btens_list)
{
	// now what's missing is a way to make view on btensors. For that i will almost definitly need to reproduce the
	// equivalent part of pytorch. I had hoped to make that at a much later point, but it needed now. This function will
	// be very useful to implement the tensor (kronecker) product
	auto out = quantit::sparse_zeros_like(*btens_list.begin());
	auto tens_it = btens_list.begin();
	++tens_it;
	for (; tens_it != btens_list.end(); ++tens_it)
	{
		out = out.tensor_product_shape(*tens_it);
	}
	return out;
}
/**
 * @brief When in a context where the method btensor::shape_from is accessible without explicitly specifying the object,
 * the compiler has trouble selecting the correct function.
 *
 * @param btens_list
 * @return btensor
 */
inline btensor disambiguated_shape_from(const std::vector<btensor> &btens_list) { return shape_from(btens_list); }
#if __cplusplus <= 201703L
template <class A>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<A>>;
#else
template <class A>
using remove_cvref_t = std::remove_cvref_t<A>;
#endif

/**
 * @brief create a shape (empty btensor) from the input tensor and the list of index.
 *
 * In the list of index, dimensions can be kept as they are by specifying -1, otherwise only the element with the
 * specified index value are kept along that dimensions, reducing the output tensor's rank.
 * The quantum number of the reduced index modifies the selection rule of the resulting shape.
 * Since this function result in an empty btensor, the selection rule of the tensor can be modified freely after the it
 * is created but before elements are put into it.
 *
 * @param tens
 * @param inds
 * @return btensor
 */
inline btensor shape_from(const btensor &tens, const std::vector<int64_t> &inds) { return tens.shape_from(inds); }

struct torch_shape
{
	std::vector<int64_t> _sizes;
	torch::TensorOptions opt;
	torch_shape() = default;
	torch_shape(const torch::Tensor &tens) : _sizes(tens.sizes().begin(), tens.sizes().end()), opt(tens.options()) {}
	torch_shape(std::vector<int64_t> _sizes, torch::TensorOptions _opt)
	    : _sizes(std::move(_sizes)), opt(std::move(_opt))
	{
	}
	int64_t dim() const { return _sizes.size(); }
	torch::IntArrayRef sizes() const { return _sizes; }
	operator torch::Tensor() const { return torch::empty(_sizes, opt); }
	torch_shape neutral_shape() { return *this; }
	torch_shape &neutral_shape_() { return *this; }
	torch_shape &inverse_cvals() { return *this; }
	torch_shape inverse_cvals_() { return *this; }
	torch_shape neutral_selection_rule() { return *this; }
	torch_shape &neutral_selection_rule_() { return *this; }
	torch_shape &set_selection_rule_(any_quantity_cref) { return *this; }
};
inline any_quantity get_section_cval(const torch_shape &, size_t, size_t) { return quantity<conserved::C<1>>(0); }
inline any_quantity_cref get_section_cval(const btensor &tens, size_t dim, size_t section)
{
	return tens.section_conserved_qtt(dim, section);
}
torch_shape shape_from(std::initializer_list<torch_shape> shapes);
/**
 * @brief  Construct an empty block tensor or torch_shape from the supplied shapes.
 *
 * if all the input are btensors, the output an empty btensors.
 * if all the input are torch_shape, the output is a torch_shape object.
 * The output shape is the same as that of the tensor product of the supplied tensors.
 *
 * @tparam Args btensors or torch_shape type
 * @param args shapes
 * @return shape
 */
template <class... Args, class Enabled = std::enable_if_t<
                             std::conjunction_v<std::is_convertible<remove_cvref_t<Args>, torch_shape>...> or
                             std::conjunction_v<std::is_same<remove_cvref_t<Args>, btensor>...>>>
inline auto shape_from(const Args &...args)
{
	static_assert(std::conjunction_v<std::is_convertible<Args, torch_shape>...> or
	                  std::conjunction_v<std::is_same<remove_cvref_t<Args>, btensor>...>,
	              "All the arguments must be either torch_shape or btensor to get into this function, don't try to "
	              "side-step the enable if.");
	return shape_from({args...});
}
torch_shape shape_from(const torch_shape &shape, const std::vector<int64_t> inds);

size_t get_refcount(const torch::Tensor &tens);

inline void swap(btensor &a, btensor &b) { a.swap(b); }

template <class value_iterator>
struct btensor::block_prop_iter
    : boost::stl_interfaces::iterator_interface<block_prop_iter<value_iterator>, std::bidirectional_iterator_tag,
                                                typename value_iterator::value_type, typename value_iterator::reference,
                                                typename value_iterator::pointer,
                                                typename value_iterator::difference_type>
{
	using il_iter = typename btensor::index_list::const_iterator;
	using ValueIterator = value_iterator;

  private:
	value_iterator val_iter;
	il_iter section_by_dim;
	il_iter block_index;

  public:
	block_prop_iter(value_iterator val_it, il_iter sect_by_dim, il_iter block_ind)
	    : val_iter(val_it), section_by_dim(sect_by_dim), block_index(block_ind)
	{
	}
	block_prop_iter() : val_iter(), section_by_dim(), block_index() {}
	using base_type = boost::stl_interfaces::iterator_interface<
	    block_prop_iter, std::bidirectional_iterator_tag, typename value_iterator::value_type,
	    typename value_iterator::reference, typename value_iterator::pointer, typename value_iterator::difference_type>;
	typename base_type::reference operator*() { return *(val_iter + *block_index); }
	bool operator==(const block_prop_iter &other)
	{ // comparison between the iterators of 2 different view object will alway return not equal, enev if constructed
	  // from the same block index
		return val_iter == other.val_iter && section_by_dim == other.section_by_dim && block_index == other.block_index;
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

	const value_iterator &get_val_iter() const { return val_iter; }
	const il_iter &get_section() const { return section_by_dim; }
	const il_iter &get_bi() const { return block_index; }
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
	const_block_prop_iter operator++(int)
	{
		auto out = *this;
		block_prop_iter<const_value_iter>::operator++();
		return out;
	}
	const_block_prop_iter operator--(int)
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
	                index_list::const_iterator section_by_dim_begin, index_list::const_iterator section_by_dim_end,
	                index_list _block_index)
	    : block_index(std::move(_block_index)), first(val_first, section_by_dim_begin, block_index.begin()),
	      last(val_last, section_by_dim_end, block_index.end())
	{
	}
	block_prop_view(typename iterator::ValueIterator val_first, typename iterator::ValueIterator val_last,
	                const index_list &section_by_dim, index_list _block_index)
	    : block_index(std::move(_block_index)), first(val_first, section_by_dim.begin(), block_index.begin()),
	      last(val_last, section_by_dim.end(), block_index.end())
	{
	}
	auto begin() const { return first; }
	auto end() const { return last; }

	const index_list &get_index() const { return block_index; }

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
	    : block_prop_view<const_iterator>(other.begin().get_val_iter(), other.end().get_val_iter(),
	                                      other.begin().get_section(), other.end().get_section(), other.get_index())
	{
	}
};
/**
 * @brief tensor contraction, or tensor dot product.
 *
 * @param left the left tensor
 * @param right the right tensor
 * @param dims_left dimensions to contract on the left tensor
 * @param dims_right dimensions to contract on the right tensor
 * @return btensor tensor resulting from the contraction
 */
inline btensor tensordot(const btensor &left, const btensor &right, torch::IntArrayRef dims_left,
                         torch::IntArrayRef dims_right)
{
	return left.tensordot(right, dims_left, dims_right);
}
/**
 * @brief generalized tensor dot product, adds the result of the contraction to a copy of a specified tensor
 *
 * @param add tensor into which the dot product is added
 * @param mul1 right tensor in the contraction
 * @param mul2 left tensor in the contraction
 * @param dims1 dimensions of mul1 to contract
 * @param dims2 dimension of mul2 to contract
 * @param beta scalar factor to apply to the result of the contraction
 * @param alpha scalar factor to apply to add
 * @return btensor
 */
inline btensor tensorgdot(const btensor &add, const btensor &mul1, const btensor &mul2, torch::IntArrayRef dims1,
                          torch::IntArrayRef dims2, btensor::Scalar beta = 1, btensor::Scalar alpha = 1)
{
	return add.tensorgdot(mul1, mul2, dims1, dims2, beta, alpha);
}
/**
 * @brief in-place generalized tensor dot product, adds the result of the contraction to a specified tensor
 *
 * @param add tensor into which the dot product is added
 * @param mul1 right tensor in the contraction
 * @param mul2 left tensor in the contraction
 * @param dims1 dimensions of mul1 to contract
 * @param dims2 dimension of mul2 to contract
 * @param beta scalar factor to apply to the result of the contraction
 * @param alpha scalar factor to apply to add
 * @return btensor reference to the modified added to tensor
 */
inline btensor &tensorgdot_(btensor &add, const btensor &mul1, const btensor &mul2, torch::IntArrayRef dims1,
                            torch::IntArrayRef dims2, btensor::Scalar beta = 1, btensor::Scalar alpha = 1)
{
	return add.tensorgdot_(mul1, mul2, dims1, dims2, beta, alpha);
}

inline btensor sparse_zeros(const btensor::vec_list_t &shape_spec, any_quantity selection_rule,
                            c10::TensorOptions opt = {})
{
	return btensor(shape_spec, std::move(selection_rule), std::move(opt) );
}
btensor zeros(const btensor::vec_list_t &shape_spec, any_quantity selection_rule, c10::TensorOptions opt = {});
btensor zeros_like(const btensor &tens, c10::TensorOptions opt = {});
btensor ones(const btensor::vec_list_t &shape_spec, any_quantity selection_rule, c10::TensorOptions opt = {});
btensor ones_like(const btensor &tens, c10::TensorOptions opt = {});
btensor empty(const btensor::vec_list_t &shape_spec, any_quantity selection_rule, c10::TensorOptions opt = {});
btensor empty_like(const btensor &tens, c10::TensorOptions opt = {});
btensor rand(const btensor::vec_list_t &shape_spec, any_quantity selection_rule, c10::TensorOptions opt = {});
btensor rand_like(const btensor &tens, c10::TensorOptions opt = {});
btensor full(const btensor::vec_list_t &shape_spec, any_quantity selection_rule, btensor::Scalar fill_value,
             c10::TensorOptions opt = {});
btensor full_like(const btensor &tens, btensor::Scalar fill_value, c10::TensorOptions opt = {});
btensor randint(int64_t low, int64_t high, const btensor::vec_list_t &shape_spec, any_quantity selection_rule,
                c10::TensorOptions opt = {});
inline btensor randint(const btensor::vec_list_t &shape_spec, any_quantity selection_rule, int64_t low, int64_t high,
                       c10::TensorOptions opt = {})
{
	return randint(low, high, shape_spec, selection_rule, opt);
}
btensor randint_like(int64_t low, int64_t high, const btensor &tens, c10::TensorOptions opt = {});
inline btensor randint_like(const btensor &shape, int64_t low, int64_t high, c10::TensorOptions opt = {})
{
	return randint_like(low, high, shape, opt);
}
inline btensor randint(int64_t high, const btensor::vec_list_t &shape_spec, any_quantity selection_rule,
                       c10::TensorOptions opt = {})
{
	return randint(0, high, shape_spec, selection_rule, opt);
}
inline btensor randint_like(int64_t high, const btensor &tens, c10::TensorOptions opt = {})
{
	return randint_like(0, high, tens, opt);
}

inline btensor randint_like(const btensor &tens, int64_t high, c10::TensorOptions opt = {})
{
	return randint_like(high, tens, opt);
}
inline btensor randint(const btensor::vec_list_t &shape_spec, any_quantity selection_rule, int64_t high,
                       c10::TensorOptions opt = {})
{
	return randint(high, shape_spec, selection_rule, opt);
}
/**
 * @brief create a matrix with the identity matrix for every permited block on the diagonnal.
 * 
 * @param shape_n shape descriptor, either rank 1 or 2. when the input is rank 1, it describes the row of an hermitian matrix
 * @param opt tensor options
 * @return btensor 
 */
btensor eye(const btensor::vec_list_t& shape_n,c10::TensorOptions opt ={});
/**
 * @brief create a matrix with the identity matrix for every permited block on the diagonnal.
 * 
 * @param shape shape of the matrix to construct
 * @param opt tensor options
 * @return btensor 
 */
btensor eye_like(const btensor& shape,c10::TensorOptions opt ={});

btensor randn(const btensor::vec_list_t &shape_spec, any_quantity selection_rule, c10::TensorOptions opt = {});
btensor randn_like(const btensor &tens, c10::TensorOptions opt = {});

btensor from_basic_tensor(const btensor::vec_list_t &shape_spec, any_quantity selection_rule,
                          const torch::Tensor &values, const torch::Scalar cutoff = 1e-16, c10::TensorOptions = {});
btensor from_basic_tensor_like(const btensor &shape, const torch::Tensor &values, const torch::Scalar cutoff = 1e-16,
                               c10::TensorOptions = {});

inline torch::Tensor zeros_like(const torch_shape &shape, c10::TensorOptions opt = {})
{
	return torch::zeros(shape._sizes, shape.opt.merge_in(opt));
}
inline torch::Tensor eye_like(const torch_shape &shape, c10::TensorOptions opt = {})
{
	if (shape._sizes.size() != 2) throw std::invalid_argument("eye_like only accept rank 2 shapes.");
	return torch::eye(shape._sizes[0],shape._sizes[1], shape.opt.merge_in(opt));
}
inline torch::Tensor ones_like(const torch_shape &shape, c10::TensorOptions opt = {})
{
	return torch::ones(shape._sizes, shape.opt.merge_in(opt));
}
inline torch::Tensor empty_like(const torch_shape &shape, c10::TensorOptions opt = {})
{
	return torch::empty(shape._sizes, shape.opt.merge_in(opt));
}
inline torch::Tensor rand_like(const torch_shape &shape, c10::TensorOptions opt = {})
{
	return torch::rand(shape._sizes, shape.opt.merge_in(opt));
}
inline torch::Tensor full_like(const torch_shape &shape, btensor::Scalar fill, c10::TensorOptions opt = {})
{
	return torch::full(shape._sizes, fill, shape.opt.merge_in(opt));
}
inline torch::Tensor randint_like(int64_t low, int64_t high, const torch_shape &shape, c10::TensorOptions opt = {})
{
	return torch::randint(low, high, shape._sizes, shape.opt.merge_in(opt));
}
inline torch::Tensor randint_like(int64_t high, const torch_shape &shape, c10::TensorOptions opt = {})
{
	return torch::randint(0, high, shape._sizes, shape.opt.merge_in(opt));
}
inline torch::Tensor randn_like(const torch_shape &shape, c10::TensorOptions opt = {})
{
	return torch::randn(shape._sizes, shape.opt.merge_in(opt));
}

inline btensor operator+(const btensor &A, const btensor &B) { return A.add(B); }
inline btensor operator-(const btensor &A, const btensor &B) { return A.sub(B); }
inline btensor operator-(const btensor &A) { return A.mul(-1); }
inline btensor operator*(const btensor &A, const btensor &B) { return A.mul(B); }
inline btensor operator*(const btensor &A, const btensor::Scalar &B) { return A.mul(B); }
inline btensor operator*(const btensor::Scalar &B, const btensor &A) { return A.mul(B); }
inline btensor operator/(const btensor &A, const btensor &B) { return A.div(B); }
inline btensor operator/(const btensor &A, const btensor::Scalar &B) { return A.div(B); }
btensor operator/(const btensor::Scalar &A, const btensor &B);

inline btensor sqrt(const btensor &A) { return A.sqrt(); }
inline btensor &sqrt_(btensor &A) { return A.sqrt_(); }
inline btensor pow(const btensor &A, btensor::Scalar p) { return A.pow(p); }
inline btensor pow(const btensor &A, const btensor &p) { return A.pow(p); }
inline btensor &pow_(btensor &A, btensor::Scalar p) { return A.pow_(p); }
inline btensor &pow_(btensor &A, const btensor &p) { return A.pow_(p); }

inline btensor ge(const btensor &A, btensor::Scalar other) { return A.ge(other); }
inline btensor ge(const btensor &A, const btensor &other) { return A.ge(other); }
inline btensor le(const btensor &A, btensor::Scalar other) { return A.le(other); }
inline btensor le(const btensor &A, const btensor &other) { return A.le(other); }
inline btensor less(const btensor &A, const btensor &other) { return A.less(other); }
inline btensor less(const btensor &A, btensor::Scalar other) { return A.less(other); }
inline btensor greater(const btensor &A, const btensor &other) { return A.greater(other); }
inline btensor greater(const btensor &A, btensor::Scalar other) { return A.greater(other); }

inline btensor greater(btensor::Scalar other, const btensor &A) { return A.less(other); }
inline btensor less(btensor::Scalar other, const btensor &A) { return A.greater(other); }
inline btensor le(btensor::Scalar other, const btensor &A) { return A.ge(other); }
inline btensor ge(btensor::Scalar other, const btensor &A) { return A.le(other); }

inline btensor eq(const btensor &A, const btensor &B) { return A.eq(B); }
inline btensor eq(const btensor &A, btensor::Scalar B) { return A.eq(B); }
inline btensor eq(btensor::Scalar B, const btensor &A) { return A.eq(B); }
inline btensor not_equal(const btensor &A, const btensor &B) { return A.not_equal(B); }
inline btensor not_equal(const btensor &A, btensor::Scalar B) { return A.not_equal(B); }
inline btensor not_equal(btensor::Scalar B, const btensor &A) { return A.not_equal(B); }

inline btensor operator>(const btensor &A, btensor::Scalar other) { return greater(A, other); }
inline btensor operator>(const btensor &A, const btensor &other) { return greater(A, other); }
inline btensor operator>(btensor::Scalar A, const btensor &other) { return greater(A, other); }
inline btensor operator<(const btensor &A, btensor::Scalar other) { return less(A, other); }
inline btensor operator<(const btensor &A, const btensor &other) { return less(A, other); }
inline btensor operator<(btensor::Scalar A, const btensor &other) { return less(A, other); }
inline btensor operator>=(const btensor &A, btensor::Scalar other) { return ge(A, other); }
inline btensor operator>=(const btensor &A, const btensor &other) { return ge(A, other); }
inline btensor operator>=(btensor::Scalar A, const btensor &other) { return ge(A, other); }
inline btensor operator<=(const btensor &A, btensor::Scalar other) { return le(A, other); }
inline btensor operator<=(const btensor &A, const btensor &other) { return le(A, other); }
inline btensor operator<=(btensor::Scalar A, const btensor &other) { return le(A, other); }
inline btensor operator==(const btensor &A, btensor::Scalar other) { return eq(A, other); }
inline btensor operator==(const btensor &A, const btensor &other) { return eq(A, other); }
inline btensor operator==(btensor::Scalar A, const btensor &other) { return eq(A, other); }
inline btensor operator!=(const btensor &A, btensor::Scalar other) { return not_equal(A, other); }
inline btensor operator!=(const btensor &A, const btensor &other) { return not_equal(A, other); }
inline btensor operator!=(btensor::Scalar A, const btensor &other) { return not_equal(A, other); }

bool allclose(const btensor &a, const btensor &b, double rtol = 1e-5, double atol = 1e-8, bool equal_nan = false);

inline btensor sum(const btensor &t) { return t.sum(); }

using torch::pow;
using torch::sqrt;
using torch::tensordot;

/**
 * @brief create a new tensor with its section rule and all its conserved quantities inversed.
 *
 * Conserved quantities must be inversed when doing the hermitian conjugation of an operator.
 *
 * Caution: The blocks of the new tensors are shallow copies of the original.
 *
 * @return btensor
 */
inline btensor inverse_cvals(const btensor &tens) { return tens.inverse_cvals(); }
inline btensor inverse_cvals(btensor &&tens) { return std::move(tens.inverse_cvals_()); }
inline torch::Tensor inverse_cvals(const torch::Tensor &tens) { return tens; }
inline torch::Tensor inverse_cvals(torch::Tensor &&tens) { return std::move(tens); }
/**
 * @brief inverse the selection rule and all the conserved quantities of this btensor.
 *
 * Conserved quantities must be inversed when doing the hermitian conjugation of an operator.
 *
 * @return btensor&
 */
inline btensor &inverse_cvals_(btensor &tens) { return tens.inverse_cvals_(); }
inline torch::Tensor &inverse_cvals_(torch::Tensor &tens) { return tens; }
/**
 * @brief Shifts the conserved quantities of one dimension of the tensor, applies the opposite shift to the
 * conservation rule.
 *
 * @param shift shift to apply
 * @param dim dimension to which the shift is applied
 */
inline btensor &cval_shift_(btensor &tens, any_quantity_cref shift, int64_t dim)
{
	return tens.cval_shift_(shift, dim);
}
/**
 * @brief Shifts the conserved quantities of one dimension of the tensor without regards for the conservation laws.
 *
 * Can only be applied to empty tensors.
 *
 * @param shift shift to apply
 * @param dim dimension to which the shift is applied
 */
inline btensor &non_conserving_cval_shift_(btensor &tens, any_quantity_cref shift, int64_t dim)
{
	return tens.non_conserving_cval_shift_(shift, dim);
}

/**
 * @brief Modify the selection rule by the value of shift. Can only be done on empty tensors .
 *
 * @param shift
 * @return btensor&
 */
inline btensor &shift_selection_rule_(btensor &tens, any_quantity_cref shift)
{
	return tens.shift_selection_rule_(shift);
}

inline btensor squeeze(const btensor &tens) { return tens.squeeze(); }
inline btensor squeeze(const btensor &tens, int64_t dim) { return tens.squeeze(dim); }
inline btensor &squeeze_(btensor &tens) { return tens.squeeze_(); }
inline btensor &squeeze_(btensor &tens, int64_t dim) { return tens.squeeze_(dim); }

/**
 * @brief find the selection rule for rank 2 torch tensors
 *
 * throws if no selection rule can be found
 *
 * @param tens tensor.
 * @param shape btensor specifying the conserved quantities on each dimensions. its selection rule does not matter.
 * @return any_quantity selection r
 */
any_quantity find_selection_rule(const torch::Tensor &tens, const btensor &shape, btensor::Scalar cutoff = 0);

void increment_index_right(btensor::index_list &index, torch::IntArrayRef sizes, size_t rank);
void increment_index_left(btensor::index_list &index, torch::IntArrayRef max_index, size_t rank);
void print(const btensor &x);
std::string to_string(const btensor &x);

/**
 * @brief Set the selection rule of the block tensor
 *
 * Only works on empty btensor.
 *
 * @param value
 * @return btensor&
 */
inline btensor &set_selection_rule_(btensor &tens, any_quantity_cref value) { return tens.set_selection_rule_(value); }
qtt_TEST_CASE("btensor")
{
	using cqt = conserved::C<5>; // don't put negative number in the constructor and expect sensible results.
	using index = btensor::index_list;
	any_quantity selection_rule(cqt(0)); // DMRJulia flux
	btensor A({{{2, cqt(0)}, {3, cqt(1)}}, {{2, cqt(0)}, {3, cqt(1).inverse()}}}, selection_rule);
	volatile auto x = to_string(A); // trying to force to_string into existance for debugging.
	qtt_CHECK(A.end() - A.begin() == 0);
	auto A00 = torch::rand({2, 2});
	auto A11 = torch::rand({3, 3});
	A.block({0, 0}) = A00;
	A.block({1, 1}) = A11;
	qtt_REQUIRE_NOTHROW(btensor::throw_bad_tensor(A));
	qtt_CHECK(A.end() - A.begin() == 2);
	qtt_CHECK_NOTHROW(A.block_at({0, 0}));
	qtt_CHECK_THROWS_AS(A.block_at({1, 0}), std::out_of_range);  // there's no block here.
	qtt_CHECK_THROWS_AS(A.block({1, 0}), std::invalid_argument); // and we can't create one.
	qtt_CHECK(btensor::check_tensor(A) == "");
	// fmt::print("{}", A);
	qtt_SUBCASE("tensordot trace, index order independence")
	{
		auto X = rand_like(shape_from(A, A.permute({1, 0})));
		// fmt::print("{}\n\n",X);
		auto XX = X.reshape({});
		std::vector<int64_t> inds = {0, 1, 2, 3};
		double correct_trace = 0;
		for (auto &x : X)
		{
			auto &T = std::get<1>(x);
			correct_trace += tensordot(T, T.conj(), inds, inds).item().toDouble();
		}
		double trXX = tensordot(XX, XX.conj(), {0}, {0}).item().toDouble();
		double correct_trace2 = 0;
		for (auto &x : XX)
		{
			auto &T = std::get<1>(x);
			correct_trace2 += tensordot(T, T.conj(), {0}, {0}).item().toDouble();
		}
		qtt_CHECK(trXX == doctest::Approx(correct_trace));
		qtt_CHECK(correct_trace2 == doctest::Approx(correct_trace));
		// fmt::print(" correct traces: {} {}\n",correct_trace2, correct_trace);
		for (int i = 0; i < 16; ++i)
		{
			auto tr2 = tensordot(X, X.conj(), inds, inds);
			double trace2 = tr2.item().toDouble();
			qtt_CHECK_MESSAGE(doctest::Approx(correct_trace) == trace2, fmt::format("index order: {}", inds));
			std::next_permutation(inds.begin(), inds.end());
		}
	}
	qtt_SUBCASE("tensor contraction")
	{
		btensor B({{{3, cqt(4)}, {2, cqt(0)}}, {{2, cqt(0)}, {3, cqt(1)}}, {{1, cqt(1)}, {3, cqt(0)}}},
		          any_quantity(cqt(1)));
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

		btensor A_trace;
		qtt_CHECK_NOTHROW(A_trace = tensordot(A, A.inverse_cvals(), {0, 1}, {0, 1}));
		torch::allclose(A_trace.block_at({}),
		                tensordot(A00, A00, {0, 1}, {0, 1}) + tensordot(A11, A11, {0, 1}, {0, 1}));
	}
	qtt_SUBCASE("rank 0 tensors")
	{
		auto x = zeros({}, selection_rule); // if this one work all the factories works. Their guts is shared.
		auto empty_index_list = btensor::index_list{};
		auto x_begin_first = x.begin()->first;
		qtt_CHECK(x_begin_first == empty_index_list);
		qtt_CHECK(torch::allclose(x.begin()->second, torch::zeros({})));
	}
	qtt_SUBCASE("Tensorproduct from tensordot")
	{
		btensor C;
		qtt_CHECK_NOTHROW(C = A.tensordot(A, {}, {}));
		qtt_CHECK(C.dim() == A.dim() * 2);
		qtt_CHECK(C.block_at({0, 0, 0, 0}).equal(torch::tensordot(A00, A00, {}, {})));
		qtt_CHECK(C.block_at({0, 0, 1, 1}).equal(torch::tensordot(A00, A11, {}, {})));
		qtt_CHECK(C.block_at({1, 1, 0, 0}).equal(torch::tensordot(A11, A00, {}, {})));
		qtt_CHECK(C.block_at({1, 1, 1, 1}).equal(torch::tensordot(A11, A11, {}, {})));
		// fmt::print("{}\n", A);
		// fmt::print("{}\n", C);
	}
	qtt_SUBCASE("addition")
	{
		btensor AA00({{{2, cqt(0)}, {3, cqt(1)}}, {{2, cqt(0)}, {3, cqt(1).inverse()}}}, selection_rule);
		btensor AA11({{{2, cqt(0)}, {3, cqt(1)}}, {{2, cqt(0)}, {3, cqt(1).inverse()}}}, selection_rule);
		AA00.block({0, 0}) = A00;
		AA11.block({1, 1}) = A11;
		AA00.add_(AA11);
		auto B00 = 2 * A00;
		auto B11 = 2 * A11;
		auto AP00 = 3 * A00; // A post add_
		auto AP11 = 3 * A11; // A post add_
		auto C00 = 5 * A00;
		auto C11 = 5 * A11;
		auto B = A.add(A);
		auto C = A.add(B);
		C.add_(B);
		qtt_CHECK(torch::allclose(A.block_at({0, 0}), A00));
		qtt_CHECK(torch::allclose(A.block_at({1, 1}), A11)); // A unchanged
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
		qtt_REQUIRE_NOTHROW(AA00.block_at({0, 0}));
		qtt_CHECK_THROWS(AA00.block_at({1, 0}));
		qtt_REQUIRE_NOTHROW(AA00.block_at({1, 1}));
		qtt_CHECK_THROWS(AA00.block_at({0, 1}));
		A.add_(std::move(B)); // this destroys B, so any state verification on it must be done.
		qtt_CHECK(torch::allclose(A.block_at({0, 0}), AP00));
		qtt_CHECK(torch::allclose(A.block_at({1, 1}), AP11));
		qtt_CHECK(get_refcount(AA00.block_at({1, 1})) == 1);
	}
	qtt_SUBCASE("substraction")
	{
		btensor A = rand({}, quantity(cqt(0)));
		btensor B = rand({}, quantity(cqt(0)));
		auto C = A - B;
		qtt_CHECK_NOTHROW(A.block_at({}));
		qtt_CHECK_NOTHROW(B.block_at({}));
		qtt_CHECK_NOTHROW(C.block_at({}));
		qtt_CHECK(torch::allclose(C.block_at({}), A.block_at({}) - B.block({})));
	}
	qtt_SUBCASE("Reshape")
	{
		auto B = A.reshape({});             // reshape into a vector.
		qtt_CHECK_NOTHROW(B.block_at({0})); // on diagonnal block of A
		qtt_CHECK_NOTHROW(B.block_at({3})); // on-diagonal block of A
		qtt_CHECK_THROWS(B.block_at({1}));  // off-diagonal block of A, empty
		qtt_CHECK_THROWS(B.block_at({2}));  // off-diagonal block of A, empty
		qtt_CHECK((*B.block_quantities({0}).begin()) == any_quantity(selection_rule));
		qtt_CHECK((*B.block_quantities({3}).begin()) == any_quantity(selection_rule));
		qtt_CHECK((*B.block_quantities({1}).begin()) == any_quantity(cqt(4))); // This is the correct value indeed.
		qtt_CHECK((*B.block_quantities({2}).begin()) == any_quantity(cqt(1)));
		qtt_CHECK((*B.block_sizes({0}).begin()) == 4);
		qtt_CHECK((*B.block_sizes({3}).begin()) == 9);
		qtt_CHECK((*B.block_sizes({1}).begin()) == 6);
		qtt_CHECK((*B.block_sizes({2}).begin()) == 6);
		qtt_CHECK((B.block_at({0}).sizes()) == std::vector<int64_t>{4});
		qtt_CHECK((B.block_at({3}).sizes()) == std::vector<int64_t>{9});
		auto C = B.reshape_as(A); // reshape it back to its' original shape.
		qtt_REQUIRE_NOTHROW(btensor::throw_bad_tensor(C));
		qtt_CHECK(C.end() - C.begin() == 2);
		qtt_CHECK_NOTHROW(C.block_at({0, 0}));
		qtt_CHECK_NOTHROW(C.block_at({1, 1}));
		qtt_CHECK_THROWS_AS(C.block_at({1, 0}), std::out_of_range);  // there's no block here.
		qtt_CHECK_THROWS_AS(C.block({1, 0}), std::invalid_argument); // and we can't create one.
		qtt_CHECK(C.block_at({0, 0}).equal(A.block_at({0, 0})));
		qtt_CHECK(C.block_at({1, 1}).equal(A.block_at({1, 1})));
		qtt_CHECK(btensor::check_tensor(C) == "");
		{
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
			btensor B;
			qtt_CHECK_NOTHROW(B = A.reshape({1}));
		}
	}
	qtt_SUBCASE("Shape building tools")
	{
		btensor B;
		btensor C;
		qtt_CHECK_NOTHROW(B = A.shape_from({-1, 0}));
		qtt_CHECK_NOTHROW(C = shape_from(A, B));
		int split = 1;
		A = btensor({{{2, cqt(0)}, {3, cqt(1)}},
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
		btensor rV({{{3, cqt(1)},
		             {2, cqt(4)},
		             {2, cqt(0)},
		             {6, cqt(0)},
		             {4, cqt(3)},
		             {4, cqt(4)},
		             {9, cqt(4)},
		             {6, cqt(2)},
		             {6, cqt(3)},
		             {3, cqt(1)},
		             {2, cqt(4)},
		             {2, cqt(0)}},
		            {{2, cqt(0)}, {3, cqt(1)}}},
		           selection_rule);
		rV.block({2, 0}) = torch::rand({2, 2});
		rV.block({3, 0}) = torch::rand({6, 2});
		rV.block({11, 0}) = torch::rand({2, 2});
		rV.block({1, 1}) = torch::rand({2, 3});
		rV.block({5, 1}) = torch::rand({4, 3});
		rV.block({6, 1}) = torch::rand({9, 3});
		rV.block({10, 1}) = torch::rand({2, 3});
		qtt_REQUIRE(btensor::check_tensor(A) == "");
		qtt_REQUIRE(btensor::check_tensor(rV) == "");

		std::vector<int64_t> V_shape(A.dim(), -1);
		{
			size_t i = 0;
			for (; i < split; ++i)
			{
				V_shape[i] = 0;
			}
		}
		auto V_left_part = A.shape_from(V_shape);
		auto V_right_part = rV.shape_from({0, -1});
		// fmt::print("V_left_part \n{}\n\n", V_left_part);
		// fmt::print("V_right_part \n{}\n\n", V_right_part);
		qtt_REQUIRE_NOTHROW(rV.reshape_as(shape_from(V_left_part, V_right_part)));
	}
	qtt_SUBCASE("elementwise multiplication with broadcasting")
	{

		btensor B({{{2, cqt(0)}, {3, cqt(0)}}}, selection_rule);
		auto B0 = torch::rand({2});
		auto B1 = torch::rand({3});
		B.block({0}) = B0;
		B.block({1}) = B1;
		auto C = A.mul(B);
		qtt_CHECK(torch::allclose(C.block({0, 0}), A00.mul(B0)));
		qtt_CHECK(torch::allclose(C.block({1, 1}), A11.mul(B1)));
		C = A.mul(1); // create an independant copy?
		qtt_CHECK(torch::allclose(C.block({0, 0}), A.block({0, 0})));
		qtt_CHECK(torch::allclose(C.block({1, 1}), A.block({1, 1})));
		C.mul_(B);
		qtt_CHECK(torch::allclose(C.block({0, 0}), A00.mul(B0))); // The numerical result commutes.
		qtt_CHECK(
		    torch::allclose(C.block({1, 1}), A11.mul(B1))); // it's the details of the allocation that depends on order
		qtt_WARN_THROWS(B.mul_(A));                         // failure on pytorch side.
	}
	qtt_SUBCASE("Basic index put")
	{
		btensor B({{{3, cqt(1)}, {4, cqt(4)}, {1, cqt(3)}},
		           {{2, cqt(-1)}, {4, cqt(1)}},
		           {{4, cqt(-3)}, {4, cqt(-2)}, {2, cqt(0)}}},
		          any_quantity(cqt(0)));
		btensor C = quantit::rand({{{3, cqt(1)}, {4, cqt(4)}, {1, cqt(3)}},
		           {{4, cqt(-3)}, {4, cqt(-2)}, {2, cqt(0)}}},
		          any_quantity(cqt(1)));
		qtt_CHECK_NOTHROW(B.basic_index_put_({-1,0,-1},C));
	}
	qtt_SUBCASE("batched matrix multiply")
	{
		// B and C are compatible
		btensor B({{{3, cqt(1)}, {4, cqt(4)}, {1, cqt(3)}},
		           {{2, cqt(-1)}, {4, cqt(1)}},
		           {{4, cqt(-3)}, {4, cqt(-2)}, {2, cqt(0)}}},
		          any_quantity(cqt(0)));
		btensor C({{{3, cqt(1)}, {4, cqt(2)}, {1, cqt(0)}},
		           {{4, cqt(-3).inverse()}, {4, cqt(-2).inverse()}, {2, cqt(0)}},
		           {{2, cqt(0)}, {2, cqt(-3)}, {2, cqt(-2)}}},
		          any_quantity(cqt(1)));
		auto B002 = torch::rand({3, 2, 2});
		auto B011 = torch::rand({3, 4, 4});
		B.block({0, 0, 2}) = B002;
		B.block({0, 1, 1}) = B011;
		auto C020 = torch::rand({3, 2, 2});
		auto C001 = torch::rand({3, 4, 2});
		auto C012 = torch::rand({3, 4, 2});
		auto C111 = torch::rand({4, 4, 2});
		auto C202 = torch::rand({1, 4, 2});
		C.block({0, 2, 0}) = C020;
		C.block({0, 0, 1}) = C001;
		C.block({0, 1, 2}) = C012;
		C.block({1, 1, 1}) = C111;
		C.block({2, 0, 2}) = C202;
		btensor BC;
		// D, E, F, G are incompatible with B
		// incompatible batch section number
		btensor D({{{4, cqt(2)}, {1, cqt(0)}}, {{4, cqt(-3)}, {4, cqt(-2)}, {2, cqt(0)}}, {{2, cqt(0)}, {2, cqt(1)}}},
		          any_quantity(cqt(-1)));
		qtt_CHECK_THROWS(B.bmm(D));
		// incompatible batch sizes
		btensor G({{{7, cqt(1)}, {4, cqt(2)}, {1, cqt(0)}},
		           {{4, cqt(-3)}, {4, cqt(-2)}, {2, cqt(0)}},
		           {{2, cqt(0)}, {2, cqt(1)}}},
		          any_quantity(cqt(-1)));
		qtt_CHECK_THROWS(B.bmm(G));
		// incompatible conserved quantities
		btensor E({{{3, cqt(1)}, {4, cqt(2)}, {1, cqt(0)}},
		           {{4, cqt(-3)}, {4, cqt(-5)}, {2, cqt(0)}},
		           {{2, cqt(0)}, {2, cqt(1)}}},
		          any_quantity(cqt(-1)));
		qtt_CHECK_THROWS(B.bmm(E));
		// incompatible matrix sizes
		btensor F({{{3, cqt(1)}, {4, cqt(2)}, {1, cqt(0)}},
		           {{4, cqt(-3)}, {6, cqt(-2)}, {2, cqt(0)}},
		           {{2, cqt(0)}, {2, cqt(1)}}},
		          any_quantity(cqt(-1)));
		qtt_CHECK_THROWS(B.bmm(F));
		qtt_REQUIRE_NOTHROW(BC = B.bmm(C));
		// All the (no)throw check are fine.
		// remains to test the correctness of BC.
		{ // number of block and their indices.
			size_t n = 0;
			for (auto &a : BC)
			{
				auto &ind = std::get<0>(a);
				bool ok = ind == std::vector<int64_t>{0, 0, 0} or ind == std::vector<int64_t>{0, 1, 2};
				qtt_REQUIRE(ok);
				++n;
			}
			qtt_REQUIRE(n == 2);
		}
		qtt_CHECK(torch::allclose(BC.block_at({0, 0, 0}), B002.matmul(C020)));
		qtt_CHECK(torch::allclose(BC.block_at({0, 1, 2}), B011.matmul(C012)));

		auto b = B.basic_create_view({1, -1, -1});
		auto c = C.basic_create_view({1, -1, -1});
		qtt_REQUIRE(btensor::check_tensor(b) == "");
		qtt_REQUIRE(btensor::check_tensor(c) == "");
		qtt_CHECK_NOTHROW(b.bmm(c));
	}
}

class bad_selection_rule : public std::invalid_argument {using std::invalid_argument::invalid_argument;};
class non_matching_cvals : public std::invalid_argument {using std::invalid_argument::invalid_argument;};
class non_matching_sizes : public std::invalid_argument {using std::invalid_argument::invalid_argument;};
} // namespace quantit

template <>
struct fmt::formatter<quantit::btensor>
{
	constexpr auto parse(format_parse_context &ctx)
	{
		auto it = ctx.begin(), end = ctx.end();
		if (it)
		{
			if (it != end and *it != '}')
				throw format_error("invalid format, no formatting option for quantit::btensor");
			if (*it != '}')
				throw format_error("invalid format,closing brace missing");
		}
		// Return an iterator past the end of the parsed range:
		return it;
	}

	template <class FormatContext>
	auto format(const quantit::btensor &t, FormatContext &ctx) const
	{
		constexpr auto btensor_fmt_string = "btensor rank {}\n selection rule {}\n number of sections by dim {}\n "
		                                    "sections sizes {}\n sections conserved quantity {}\n";
		constexpr auto btensor_fmt_blocks = "block at {}\n {}\n";
		auto out = format_to(ctx.out(), btensor_fmt_string, t.rank, t.selection_rule, t.sections_by_dim,
		                     t.sections_sizes, t.c_vals);
		for (const auto &b : t.blocks_list)
		{
			out = format_to(out, btensor_fmt_blocks, b.first, b.second);
		}
		return out;
	}
};

#endif /* D49FFA60_85C4_431A_BA62_9B1D30D67E86 */
