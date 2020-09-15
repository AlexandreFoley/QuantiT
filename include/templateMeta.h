/*
 * File: method_detect.h
 * Project: quantt
 * File Created: Friday, 4th September 2020 1:08:51 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Friday, 4th September 2020 1:08:51 pm
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#ifndef D5434AD3_7D18_4065_A1ED_CD37C3E10D7A
#define D5434AD3_7D18_4065_A1ED_CD37C3E10D7A

#include <tuple>
#include <type_traits>

namespace quantt
{

/*
 * Template metaprograming to detect methods and functions.
 * All of the stuff in here can be replaced with a simple c++20 concept's require statement.
 * Will do that when it officially comes out.
 * The code is lifted from cpp possible implementation about std::experiemental.
 */
namespace
{
struct nonesuch
{

	~nonesuch() = delete;
	nonesuch(nonesuch const &) = delete;
	void operator=(nonesuch const &) = delete;
};

template <class Default, class AlwaysVoid,
          template <class...> class Op, class... Args>
struct detector
{
	using value_t = std::false_type;
	using type = Default;
};

template <class Default, template <class...> class Op, class... Args>
struct detector<Default, ::std::void_t<Op<Args...>>, Op, Args...>
{
	using value_t = std::true_type;
	using type = Op<Args...>;
};

} // namespace

template <template <class...> class Op, class... Args>
using is_detected = typename detector<nonesuch, void, Op, Args...>::value_t;

template <template <class...> class Op, class... Args>
using detected_t = typename detector<nonesuch, void, Op, Args...>::type;

template <class Default, template <class...> class Op, class... Args>
using detected_or = detector<Default, void, Op, Args...>;

template <template <class...> class Op, class... Args>
constexpr bool is_detected_v = is_detected<Op, Args...>::value;
template <class Default, template <class...> class Op, class... Args>
using detected_or_t = typename detected_or<Default, Op, Args...>::type;
template <class Expected, template <class...> class Op, class... Args>
using is_detected_exact = ::std::is_same<Expected, detected_t<Op, Args...>>;
template <class Expected, template <class...> class Op, class... Args>
constexpr bool is_detected_exact_v = is_detected_exact<Expected, Op, Args...>::value;
template <class To, template <class...> class Op, class... Args>
using is_detected_convertible = ::std::is_convertible<detected_t<Op, Args...>, To>;
template <class To, template <class...> class Op, class... Args>
constexpr bool is_detected_convertible_v = is_detected_convertible<To, Op, Args...>::value;

/*
	 * Template metaprograming to apply "logic and" and "logic or" to template parameter pack of 
	 * std::integral_constant<bool,T> or other classes with a value constexpr static member implicitly
	 * convertible to bool (with short circuit).   
	 */
template <typename... Conds>
struct and_ : std::true_type
{
};

template <typename Cond, typename... Conds>
struct and_<Cond, Conds...> : std::conditional<Cond::value, and_<Conds...>, std::false_type>::type
{
};

template <class... Conds>
struct or_ : std::false_type
{
};
template <class cond, class... conds>
struct or_<cond, conds...> : std::conditional_t<cond::value, std::true_type, or_<conds...>>::type
{
};

template <class Tuple1, class F>
constexpr decltype(auto) for_each(Tuple1 &&T1, F &&f)
{
	std::apply([&](auto &&... t1) {
		if constexpr (std::is_same_v<decltype(f(std::get<0>(T1))), void>)
		{                 // we return void..
			(f(t1), ...); // apply the function to each arguments
		}
		else
		{
			return std::tuple(
			    f(t1)...); // apply the function to each agruments and form
			               // a tuple with the result.
		}
	},
	           T1);
} // namespace quantt
template <class Tuple1, class Tuple2, class F>
constexpr decltype(auto) for_each2(Tuple1 &&T1, Tuple2 &&T2, F &&f)
{
	std::apply([&](auto &&... t1) {                               // std::apply require gcc7 or more recent
		return std::apply([&T1, &T2, &f, &t1...](auto &&... t2) { // need explicit capture list here for clang8 and older.
			if constexpr (std::is_same_v<decltype(f(std::get<0>(T1), std::get<0>(T2))),
			                             void>)
			{                     // we return void..
				(f(t1, t2), ...); // apply the function to each pair of arguments
			}
			else
			{
				return std::tuple(f(t1, t2)...); // apply the function to each pair and form
				                                 // a tuple with the result.
			}
		},
		                  T2);
	},
	           T1);
}
} // namespace quantt
#endif /* D5434AD3_7D18_4065_A1ED_CD37C3E10D7A */
