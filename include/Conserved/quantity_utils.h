/*
 * File:quantity_utils.h
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
#include <type_traits>
#include <utility>
namespace quantt
{
    class vquantity;
namespace conserved
{

// this template must work with your implemented group.
// try to implement Grp& op(Grp&& other) in such a way that this template
// is efficient.
template <class T>
constexpr T op(T lhs, const T& rhs) { return lhs.op(rhs); }

// custom simple greoup must satisfy the following type traits to work with the
// composite group type cgoups.
template <class T>
using op2_sig = decltype(
    conserved::op(std::declval<const T&>(),
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
using comparatorequal_sig =
    decltype(operator==(std::declval<const T&>(), std::declval<const T&>()));
template <class T>
using comparatorequal_member_sig =
    decltype(std::declval<const T&>().operator==(std::declval<const T&>()));

template <class subject, class E = void>
struct constexprequal_membertest
{
	constexpr static bool call() { return false; }
};
template <class subject>
struct constexprequal_membertest<subject, std::enable_if_t<is_detected_v<comparatorequal_member_sig, subject>>>
{
	template <int Value = subject().operator==(subject())>
	constexpr static std::true_type do_call(int) { return std::true_type(); }

	constexpr static std::false_type do_call(...) { return std::false_type(); }

	constexpr static bool call() { return do_call(0); }
};
template <class T>
using has_constexpr_equal_member = std::bool_constant<constexprequal_membertest<T>::call()>;
template <class subject, class E = void>
struct constexprequal_test
{
	constexpr static bool call() { return false; }
};
template <class subject>
struct constexprequal_test<subject, std::enable_if_t<is_detected_v<comparatorequal_sig, subject>>>
{
	template <int Value = operator==(subject(), subject())>
	constexpr static std::true_type do_call(int) { return std::true_type(); }

	constexpr static std::false_type do_call(...) { return std::false_type(); }

	constexpr static bool call() { return do_call(0); }
};
template<>
struct constexprequal_test<quantt::vquantity>
{

	constexpr static bool call() { return std::false_type(); }
};

template <class T>
using has_constexpr_equal_outclass = std::bool_constant<constexprequal_test<T>::call()>;
template <class T>
using has_constexpr_equal = or_<has_constexpr_equal_member<T>, has_constexpr_equal_outclass<T>>;

template <class T>
using has_comparatorequal = or_<is_detected_exact<bool, comparatorequal_member_sig, T>, is_detected_exact<bool, comparatorequal_sig, T>>;
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

template <class T, class = void>
struct default_to_neutral : std::false_type
{
};

template <class T>
struct default_to_neutral<T, std::enable_if_t<has_op<T>::value && std::is_default_constructible_v<T> && has_inverse_<T>::value && has_constexpr_equal<T>::value>>
    : std::integral_constant<bool, T() == conserved::op(T(), T().inverse_())>
{
};
template <class T, class = void>
struct is_Abelian : std::false_type
{
};
template <class T>
using abelian_present = decltype(T::is_Abelian);
template <class T>
struct is_Abelian<T, std::enable_if_t<is_detected_v<abelian_present, T>>> : std::integral_constant<bool, T::is_Abelian>
{
};
// the following compile time template constant is true iff the template
// parameter satisfy the constraint for a group that will work with any_quantity
template <class T>
using is_conversed_quantt =
    and_<default_to_neutral<T>, has_op<T>, has_inverse_<T>,
         has_comparatorequal<T>, has_comparatornotequal<T>, is_Abelian<T>>;

template <class T>
constexpr bool is_conserved_quantt_v = is_conversed_quantt<T>::value;

template <class... T>
using all_conserved_quantt = and_<is_conversed_quantt<T>...>;
template <class... T>
constexpr bool all_group_v = all_conserved_quantt<T...>::value;

#if __cplusplus == 202002L
template <class T>
concept a_group = is_conserved_quantt_v<T>;

#endif

} // namespace conserved
} // namespace quantt

#endif /* D241DFD2_9200_4C66_8225_2C3BBD27EDE4 */
