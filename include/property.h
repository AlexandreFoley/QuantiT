/*
 * File: property.h
 * Project: QuanTT
 * File Created: Thursday, 23rd July 2020 11:30:55 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Thursday, 23rd July 2020 11:30:55 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#ifndef A0DBDDD7_5F5B_48D3_A287_E11E64C8B84C
#define A0DBDDD7_5F5B_48D3_A287_E11E64C8B84C
#include <type_traits>
#include <utility>
// with c++20, we can use unique_type = decltype([]{}), to get a completly unique type, even if all the other
// template argument are identical. Doing that mean that every time the template is explicitly written, a new type is
// generated.

/**
 * wrapper for properties, allow direct access to users with checks without an explicit setter and getter.
 * for cheap to copy type, one can use the value itself for the ref_type.
 * Usage of property is incompatible with the keyword auto. if you use auto, you will either get an error.
 * If you use auto& you get a reference to the wrapper type.
 * This is because the auto keyword doesn't do conversions and the magic of this class wrapper is in the fact that it
 * does implicit conversions. unique_type is for the situation where it is desirable to have a different type (setter
 * and getter) for a property with the same owning class and content type. if a completly unique type is necessary the
 * type of an empty lambda []{} can be used for unique_type, because all lambda have a unique type.
 */
template <class content, class owner, class cref_type = const content &, class unique_type = owner> class property final
{
	// @cond
	friend owner;
	// @endcond
	content value;

	property() = default;
	property(cref_type val)
	    : value(val) {} // private constructor, allows owning class to construct without check if wanted.
	template<class... Args,class E = std::enable_if_t<std::is_constructible_v<content, Args...>> >
	property(Args... args):value(std::forward<Args>(args)...) {}

	property(content &&val)
	    : value(std::move(val)) {} // private constructor, allows owning class to construct without check if wanted.
	// basically, the only thing to do with this for non-friend is to look at or copy the content.
	property(property &&) = default;
	property(const property &) = default;
	property &operator=(const property &) = default;
	property &operator=(property &&) = default;
	// redefine if necessary
  public:
	operator cref_type() const noexcept { return value; } //! read access through implicit conversion

	/**
	 * @brief Value or ref assignment operator
	 * Define it to give public write access to value, with any and all checks necessary.
	 * Do not define assignement operators to keep the value private.
	 * @param new_value new value
	 * @return property& *this
	 */
	property &operator=(cref_type new_value);
	/**
	 * @brief move assignment operator
	 * Define it to give public write access to value, with any and all checks necessary.
	 * Do not define assignement operators to keep the value private.
	 * @param new_value new value
	 * @return property& *this
	 */
	property &operator=(content &&new_value);

	/**
	 * @brief Arrow dereferencing operator.
	 * Let you call the members const without going through the conversion operator
	 * first.
	 */
	const content *operator->() const { return &value; }

	~property(){};
};

#if __has_include(<fmt/format.h>)
#include <fmt/format.h>
/**
 * @brief fmt::formatter for property class.
 * 
 * simply reuse the formater of the underlying value type. 
 * 
 */
template <class content, class owner, class cref_type, class unique_type>
struct fmt::formatter<property<content, owner, cref_type, unique_type>> : fmt::formatter<content>
{
};
#endif

#endif /* A0DBDDD7_5F5B_48D3_A287_E11E64C8B84C */
