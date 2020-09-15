/*
 * File: groups_utils.h
 * Project: quantt
 * File Created: Tuesday, 15th September 2020 1:16:37 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Tuesday, 15th September 2020 1:16:37 pm
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#ifndef D241DFD2_9200_4C66_8225_2C3BBD27EDE4
#define D241DFD2_9200_4C66_8225_2C3BBD27EDE4
#include "templateMeta.h"
#include <utility>
namespace quantt
{
namespace groups
{

// this template must work with your implemented group.
// try to implement Grp& op(Grp&& other) in such a way that this template
// is efficient.
template <class T>
T op(T lhs, const T& rhs) { return lhs.op(rhs); }

// custom simple greoup must satisfy the following type traits to work with the
// composite group type cgoups.
template <class T>
using op2_sig = decltype(
    groups::op(std::declval<const T&>(),
               std::declval<const T&>())); // this template must work with your
                                           // implemented group.
template <class T>
using op_sig = decltype(std::declval<T&>().op(std::declval<T&>()));
template <class T>
using has_op =
    and_<is_detected_exact<T&, op_sig, T>, is_detected_exact<T, op2_sig, T>>;
template <class T>
using inverse__sig = decltype(std::declval<T&>().inverse_());
template <class T>
using has_inverse_ = is_detected_exact<T&, inverse__sig, T>;
template <class T>
using commute__sig = decltype(std::declval<T&>().commute_(std::declval<T&>()));
template <class T>
using has_commute_ = is_detected_exact<void, commute__sig, T>;
template <class T>
using commute_sig =
    decltype(std::declval<const T&>().commute(std::declval<T&>()));
template <class T>
using has_commute = is_detected_exact<void, commute_sig, T>;
template <class T>
using comparatorequal_sig =
    decltype(operator==(std::declval<const T&>(), std::declval<const T&>()));
template <class T>
using comparatorequal_member_sig =
    decltype(std::declval<const T&>().operator==(std::declval<const T&>()));
template <class T>
using has_comparatorequal =
    or_<is_detected_exact<bool, comparatorequal_member_sig, T>,
        is_detected_exact<bool, comparatorequal_sig, T>>;
template <class T>
using comparatornotequal_member_sig =
    decltype(std::declval<const T&>().operator!=(std::declval<const T&>()));
template <class T>
using comparatornotequal_sig =
    decltype(operator!=(std::declval<const T&>(), std::declval<const T&>()));
template <class T>
using has_comparatornotequal =
    or_<is_detected_exact<bool, comparatornotequal_member_sig, T>,
        is_detected_exact<bool, comparatornotequal_sig, T>>;

// the following compile time template constant is true iff the template
// parameter satisfy the constraint for a group that will work with cgroup
template <class T>
using is_group =
    and_<has_op<T>, has_inverse_<T>, has_commute<T>, has_commute_<T>,
         has_comparatorequal<T>, has_comparatornotequal<T>>;

template <class T>
constexpr bool is_group_v = is_group<T>::value;

template <class... T>
constexpr bool all_group_v = and_<is_group<T>...>::value;
} // namespace groups
} // namespace quantt

#endif /* D241DFD2_9200_4C66_8225_2C3BBD27EDE4 */
