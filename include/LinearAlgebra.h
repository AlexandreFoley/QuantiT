/*
 * File: LinearAlgebra.h
 * Project: quantt
 * File Created: Wednesday, 5th August 2020 11:38:57 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Wednesday, 5th August 2020 3:34:16 pm
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#include <torch/torch.h>




//doctest always last. its' macro must work and conflict with pytorch's.
#include "doctest_redef.h" // makes the redefinition appear without compiler warnings.
// we don't use pytorch's macro so its fine to redefine them. 
#include "doctest.h"

namespace quantt
{

/**
 * truncate the input tensors according to the small values of d.
 * truncate all the values of d, d_i, such that \sum_i abs(d_i)^pow <= tol.
 * Assumes the entry of d are ordered in decreasing order.
 * The last dimension of u and v are the one that are truncated along with d.
 * Output is in the same order as input
 */
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> truncate(torch::Tensor u,torch::Tensor d, torch::Tensor v, torch::Scalar tol=0, torch::Scalar pow=2);
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> truncate(std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> tensors, torch::Scalar tol=0, torch::Scalar pow=2);

/**
 * truncate the input tensors according to the small values of e.
 * truncate all the values of e, e_i, such that \sum_i abs(e_i)^pow <= tol.
 * Assumes the entry of e are ordered in decreasing order.
 * The last dimension of u is the one that is truncated along with e.
 * Output is in the same order as input
 */
std::tuple<torch::Tensor,torch::Tensor> truncate(torch::Tensor u,torch::Tensor e, torch::Scalar tol=0, torch::Scalar pow=1);
std::tuple<torch::Tensor,torch::Tensor> truncate(std::tuple<torch::Tensor,torch::Tensor> tensors, torch::Scalar tol=0, torch::Scalar pow=1);

/**
 * perform the singular value decomposition of the rank N tensor A. 
 * The tensor A is implicitly reshape as a matrix: all the index [0,split] become the left index and ]split,N] is the right index.
 * This behavior is different from the batching done by pytorch's functions.
 * can additionnaly truncate the output tensors, if tol != 0. Consult truncate for mor details.
 * output U,D,V.t() such that A = U*diag(D)*V
 */
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> svd(torch::Tensor A, size_t split,torch::Scalar tol, torch::Scalar pow = 2);
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> svd(torch::Tensor A, size_t split);

std::tuple<torch::Tensor,torch::Tensor> eig(torch::Tensor A, size_t split);
std::tuple<torch::Tensor,torch::Tensor> eig(torch::Tensor A, size_t split,torch::Scalar tol, torch::Scalar pow = 1);




}//namespace quantt