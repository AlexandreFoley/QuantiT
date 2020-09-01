/*
 * File: TensorGroup.h
 * Project: quantt
 * File Created: Tuesday, 1st September 2020 1:39:16 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Tuesday, 1st September 2020 1:39:17 pm
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#ifndef D56E4C12_98E1_4C9E_B0C4_5B35A5A3CD17
#define D56E4C12_98E1_4C9E_B0C4_5B35A5A3CD17

#include <cstdint>
#include <utility>
#include <memory>

namespace quantt
{
	namespace
	{
		template <class Tuple, class F, std::size_t... I>
		constexpr F for_each_impl(Tuple &&t, F &&f, std::index_sequence<I...>)
		{
			return (void)std::initializer_list<int>{(std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))), 0)...}, f;
		}
		template <class Tuple, class F>
		constexpr F for_each(Tuple &&t, F &&f)
		{
			return for_each_impl(std::forward<Tuple>(t), std::forward<F>(f),
								 std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
		}

		template <class Tuple1, class Tuple2, class F, std::size_t... I>
		F for_each2_impl(Tuple1 &&t1, Tuple2 &&t2, F &&f, std::index_sequence<I...>)
		{
			return (void)std::initializer_list<int>{(std::forward<F>(f)(std::get<I>(std::forward<Tuple1>(t1)), std::get<I>(std::forward<Tuple2>(t2))), 0)...}, f;
		}

		template <class Tuple1, class Tuple2, class F>
		constexpr decltype(auto) for_each2(Tuple1 &&t1, Tuple2 &&t2, F &&f)
		{
			return for_each2_impl(std::forward<Tuple1>(t1), std::forward<Tuple2>(t2), std::forward<F>(f),
								  std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple1>>::value>{});
		}
	} // namespace
	class cgroup_impl
	{
	public:
		virtual cgroup_impl &op(const cgroup_impl &) = 0;
		virtual cgroup_impl &inverse_() = 0;
		virtual std::unique_ptr<cgroup_impl> clone() const = 0;
		virtual bool operator==(const cgroup_impl &) const = 0;
		virtual bool operator!=(const cgroup_impl &) const = 0;
	};

	template <class... groups>
	class conc_cgroup_impl final : public cgroup_impl
	{
		tuple<groups...> val;

	public:
		conc_cgroup_impl(groups... grp) : val(grp...) {}

		std::unique_ptr<conc_cgroup_impl> clone() const
		{
			return std::make_unique(*this);
		}

		conc_cgroup_impl &op(const conc_cgroup_impl &other) override
		{
			for_each2(val, other.val, [](auto &&vl, auto &&ovl) { vl.op(ovl); });
			return *this;
		}
		conc_cgroup_impl &inverse_() override
		{
			for_each(val, [](auto &&vl) { vl.inverse_(); }) return *this;
		}
		bool operator==(const conc_cgroup_impl &other) const override
		{
			return val == other.val;
		}
		bool operator!=(const conc_cgroup_impl &other) const override
		{
			return val != other.val;
		}
	};

	class cgroup final
	{
		std::unique_ptr<cgroup_impl> impl;

	public:
		template <class... Groups>
		cgroup(Groups... groups) : impl(conc_cgroup_impl(std::forward<Groups>(groups...))) {}
		cgroup() = default;
		cgroup(const cgroup &other) : impl(other.impl->clone()) {}
		cgroup(cgroup &&) = default;
		void swap(cgroup &other)
		{
			using std::swap;
			swap(other.impl, impl);
		}
		cgroup &operator=(cgroup other)
		{
			swap(other);
			return *this;
		}
		~cgroup() {}

		cgroup &operator*=(const cgroup &other)
		{
			impl->op(*other.impl);
			return *this;
		}
		friend cgroup operator*(cgroup lhs, const cgroup &rhs)
		{
			return lhs *= rhs;
		}

		cgroup &operator+=(const cgroup &other)
		{
			return (*this) *= other;
		}

		friend cgroup operator+(cgroup lhs, const cgroup &rhs)
		{
			return lhs += rhs;
		}

		cgroup &inverse_()
		{
			impl->inverse_();
			return *this;
		}
		cgroup inverse() const
		{
			cgroup out(*this);
			return out.inverse_();
		}

		bool operator!=(const cgroup &other)
		{
			return (*impl) != (*(other.impl));
		}
		bool operator==(const cgroup &other)
		{
			return (*impl) == (*(other.impl));
		}
	};

	void swap(cgroup &lhs, cgroup &rhs)
	{
		lhs.swap(rhs);
	}

	namespace groups
	{
		template <uint16_t N>
		class Z
		{
			static_assert(N > 0, "only value greater than zero make sense, only greater than 1 are useful.");
			uint16_t val;

		public:
			Z(uint16_t _val)
			noexcept : val(_val)
			{
				val %= N;
			}
			void swap(Z &other) noexcept
			{
				using std::swap;
				swap(other.val, val);
			}
			operator uint16_t() noexcept
			{
				return val;
			}
			Z &operator+=(Z other) noexcept
			{
				val += other.val;
				val -= (val >= N) * N;
				return *this;
			}
			//this function is what is actually used by the group compositor.
			Z &op(Z other) noexcept
			{
				return (*this) += other;
			}
			friend Z op(Z lhs, Z rhs)
			{
				return lhs + rhs;
			}
			Z &operator*=(Z other) noexcept //in group theory we typically talk of a product operator.
			{
				return (*this) += other;
			}
			friend Z operator+(Z lhs, Z rhs) noexcept
			{
				return lhs += rhs;
			}
			friend Z operator*(Z lhs, Z rhs) noexcept
			{
				return lhs *= rhs;
			}
			Z &inverse_() noexcept
			{
				val = N - val;
				return *this;
			}
			Z inverse() const noexcept
			{
				Z out(*this);
				return out.inverse_();
			}

			bool operator==(Z other)
			{
				return val == other.val;
			}
			bool operator!=(Z other)
			{
				return val != other.val;
			}
		};

		class U1
		{
			int16_t val;

		public:
			U1(int16_t _val)
			noexcept : val(_val) {}
			operator uint16_t() noexcept
			{
				return val;
			}
			void swap(U1 &other) noexcept
			{
				using std::swap;
				swap(other.val, val);
			}
			U1 &operator+=(U1 other) noexcept
			{
				val += other.val;
				return *this;
			}
			U1 &operator*=(U1 other) noexcept //in group theory we typically talk of a product operator.
			{
				return (*this) += other;
			}
			friend U1 operator+(U1 lhs, U1 rhs) noexcept
			{
				return lhs += rhs;
			}
			friend U1 operator*(U1 lhs, U1 rhs) noexcept
			{
				return lhs *= rhs;
			}
			U1 &op(U1 other)
			{
				return (*this) += other;
			}
			friend U1 op(U1 lhs, U1 rhs)
			{
				return lhs + rhs;
			}
			U1 &inverse_() noexcept
			{
				val = -val;
				return *this;
			}
			U1 inverse() const noexcept
			{
				U1 out(*this);
				return out.inverse_();
			}
			bool operator==(U1 other)
			{
				return val == other.val;
			}
			bool operator!=(U1 other)
			{
				return val != other.val;
			}
		};

		void swap(U1 &lhs, U1 &rhs) noexcept
		{
			lhs.swap(rhs);
		}
		template <uint16_t N>
		void swap(Z<N> &lhs, Z<N> &rhs) noexcept
		{
			lhs.swap(rhs);
		}

	} // namespace groups

} // namespace quantt

#endif /* D56E4C12_98E1_4C9E_B0C4_5B35A5A3CD17 */
