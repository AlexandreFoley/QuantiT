/*
 * File: numeric.h
 * Project: quantt
 * File Created: Thursday, 27th August 2020 10:39:15 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Thursday, 27th August 2020 10:39:15 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#ifndef BEAD62BB_7BE8_44DB_9116_0DFFA7FAB98C
#define BEAD62BB_7BE8_44DB_9116_0DFFA7FAB98C

#include <torch/torch.h>
#include <limits>
#include <type_traits>

namespace quantt
{
	namespace // anonymous namespace, this stuff can't be  used from outside this file.
	{
		template <class, template <class...> class>
		struct is_instance : public std::false_type
		{
		};

		template <class... T, template <class> class U>
		struct is_instance<U<T...>, U> : public std::true_type
		{
		};

		template <class T, template <class...> class U>
		constexpr bool is_instance_v = is_instance<std::decay_t<T>, U>::value; // std::decay remove the const and ref specification from the type. not removing it would lead to false in situation where we expect true

		template <class T>
		struct num_lim
		{
			static constexpr auto eps() noexcept
			{
				if constexpr (is_instance_v<T, c10::complex>)
				{
					return std::numeric_limits<typename T::value_type>::epsilon();
				}
				else
				{
					return std::numeric_limits<T>::epsilon();
				}
			}

			using real_type = decltype(eps());

			static constexpr torch::ScalarType enum_val()
			{
				return torch::ScalarType::Undefined;
			}
		};
		// note that it return a real value type. if you need to check for epsilon in a complex number context, make sure that the check is performed on the parts or the absolute value.

#define ENUMVAL(act_type, enum)                               \
	template <>                                               \
	constexpr torch::ScalarType num_lim<act_type>::enum_val() \
	{                                                         \
		return torch::ScalarType::enum;                       \
	}

		AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(ENUMVAL)
#undef ENUMVAL
	} // namespace
#define RET_EPS(act_type, Enum_name)     \
	case torch::ScalarType::Enum_name:   \
		return num_lim<act_type>::eps(); \
		break;

	inline torch::Scalar eps(torch::ScalarType intype)
	{
		switch (intype)
		{
			AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(RET_EPS)
			default:
				return 0;
		}
	}
#undef RET_EPS
#define RET_REAL_TYPE(act_type, Enum_name)                        \
	case torch::ScalarType::Enum_name:                            \
		return num_lim<num_lim<act_type>::real_type>::enum_val(); \
		break;

	inline torch::ScalarType real_type(torch::ScalarType intype)
	{
		switch (intype)
		{
			AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(RET_REAL_TYPE)
			default:
				return torch::ScalarType::Undefined;
		}
	}
} // namespace quantt

#endif /* BEAD62BB_7BE8_44DB_9116_0DFFA7FAB98C */
