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
#include <type_traits>
#include "method_detect.h"
#include <ostream>

#include "cond_doctest.h"

namespace quantt
{

	namespace groups
	{
		// custom simple greoup must satisfy the following type traits to work with the composite group type cgoups.
		template <class T>
		using op_sig = decltype(std::declval<T &>().op(std::declval<T &>()));
		template <class T>
		using has_op = is_detected_exact<T &, op_sig, T>;
		// op, inverse_,commute,==,!=
		template <class T>
		using inverse__sig = decltype(std::declval<T &>().inverse_());
		template <class T>
		using has_inverse_ = is_detected_exact<T &, inverse__sig, T>;
		template <class T>
		using commute_sig = decltype(std::declval<T &>().commute(std::declval<T &>()));
		template <class T>
		using has_commute = is_detected_exact<void, commute_sig, T>;
		template <class T>
		using comparatorequal_sig = decltype(std::declval<T &>().operator==(std::declval<T &>()));
		template <class T>
		using has_comparatorequal = is_detected_exact<bool, comparatorequal_sig, T>;
		template <class T>
		using comparatornotequal_sig = decltype(std::declval<T &>().operator!=(std::declval<T &>()));
		template <class T>
		using has_comparatornotequal = is_detected_exact<bool, comparatornotequal_sig, T>;
		// the following compile time template constant is true iff the template parameter satisfy the constraint for a group that will work with cgroup

		template <typename... Conds>
		struct and_ : std::true_type
		{
		};

		template <typename Cond, typename... Conds>
		struct and_<Cond, Conds...> : std::conditional<Cond::value, and_<Conds...>, std::false_type>::type
		{
		};

		template <class T>
		using is_group = and_<has_op<T>, has_inverse_<T>, has_commute<T>, has_comparatorequal<T>, has_comparatornotequal<T>>;

		template <class T>
		constexpr bool is_group_v = is_group<T>::value;

		template <class... T>
		constexpr bool all_group_v = and_<is_group<T>...>::value;

	} // namespace groups
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
		virtual void commute(cgroup_impl &) const = 0;
		virtual std::unique_ptr<cgroup_impl> clone() const = 0;
		virtual bool operator==(const cgroup_impl &) const = 0;
		virtual bool operator!=(const cgroup_impl &) const = 0;
		virtual ~cgroup_impl() {}
	};

	template <class... groups>
	class conc_cgroup_impl final : public cgroup_impl
	{
		std::tuple<groups...> val;

	public:
		//has default constructor and assigment operator as well.
		conc_cgroup_impl(groups... grp) : val(grp...) {}

		std::unique_ptr<cgroup_impl> clone() const override
		{
			return std::make_unique<conc_cgroup_impl>(*this);
		}

		conc_cgroup_impl &op(const conc_cgroup_impl &other)
		{
			for_each2(val, other.val, [](auto &&vl, auto &&ovl) { vl.op(ovl); });
			return *this;
		}
		cgroup_impl &op(const cgroup_impl &other) override
		{
			return op(dynamic_cast<const conc_cgroup_impl &>(other));
		}
		conc_cgroup_impl &inverse_() override
		{
			for_each(val, [](auto &&vl) { vl.inverse_(); });
			return *this;
		}
		void commute(conc_cgroup_impl &other) const
		{
			for_each2(val, other.val, [](auto &&vl, auto &&ovl) { vl.commute(ovl); });
		}
		void commute(cgroup_impl &other) const override
		{
			commute(dynamic_cast<conc_cgroup_impl &>(other));
		}
		bool operator==(const conc_cgroup_impl &other) const
		{
			return val == other.val;
		}
		bool operator==(const cgroup_impl &other) const override
		{
			return operator==(dynamic_cast<const conc_cgroup_impl &>(other));
		}
		bool operator!=(const conc_cgroup_impl &other) const
		{
			return val != other.val;
		}
		bool operator!=(const cgroup_impl &other) const override
		{
			return operator!=(dynamic_cast<const conc_cgroup_impl &>(other));
		}
	};
	class cgroup_ref;
	class cgroup_cref;
	class cgroup;
	class cgroup final
	{
		std::unique_ptr<cgroup_impl> impl;

		cgroup(const cgroup_impl &other) : impl(impl->clone()) {}
		friend cgroup_ref;
		friend cgroup_cref;

	public:
		cgroup(std::unique_ptr<cgroup_impl> &&_impl) : impl(std::move(_impl)) {}

		template <class... Groups, class = std::enable_if_t<groups::all_group_v<Groups...>>>
		cgroup(Groups... groups) : impl(std::make_unique<conc_cgroup_impl<Groups...>>(groups...)) {}

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

		cgroup &operator*=(cgroup_cref other);
		friend cgroup operator*(cgroup lhs, cgroup_cref rhs);

		cgroup &operator+=(cgroup_cref other);

		friend cgroup operator+(cgroup lhs, cgroup_ref rhs);

		cgroup &inverse_()
		{
			impl->inverse_();
			return *this;
		}
		cgroup inverse() const
		{
			return cgroup(*this).inverse_();
		}
		void commute(cgroup_ref other) const;

	};

	void swap(cgroup &lhs, cgroup &rhs)
	{
		lhs.swap(rhs);
	}

	//class to allow easy interopt between cgroup_array and cgroup
	class cgroup_cref
	{
		cgroup_impl *const ref;
		friend cgroup_ref;
		friend cgroup;

	public:
		cgroup_cref() = delete;
		cgroup_cref(const cgroup &other) : ref(other.impl.get()) {}
		cgroup_cref(cgroup_ref &other);
		cgroup_cref(cgroup_cref &other) : ref(other.ref) {}

		operator cgroup() const
		{
			return cgroup(get());
		}

		const cgroup_impl &get() const
		{
			return *ref;
		}

		void commute(cgroup_ref other) const;
	};
	class cgroup_ref final
	{
		cgroup_impl *ref;
		friend cgroup_cref;
		friend cgroup;

	public:
		cgroup_ref() = delete;
		cgroup_ref(const cgroup &other) = delete;
		cgroup_ref(cgroup_impl &other) : ref(&other) {}
		cgroup_ref(cgroup &other) : ref((other.impl.get())) {}

		operator cgroup() const
		{
			return cgroup(*ref);
		}
		operator cgroup_cref() const
		{
			return cgroup_cref(*ref);
		}
		const cgroup_impl &get() const
		{
			return *ref;
		}
		cgroup_impl &get()
		{
			return *ref;
		}

		cgroup_ref &operator*=(cgroup_cref other);
		cgroup_ref &operator+=(cgroup_cref other);
		cgroup_ref &inverse_()
		{
			get().inverse_();
			return *this;
		}
		cgroup inverse() const
		{
			return cgroup(*this).inverse_();
		}
		void commute(cgroup_ref other) const;
	};

	cgroup &cgroup::operator*=(cgroup_cref other)
	{
		impl->op(other.get());
		return *this;
	}
	cgroup operator*(cgroup lhs, cgroup_cref rhs)
	{
		return lhs *= rhs;
	}

	cgroup &cgroup::operator+=(cgroup_cref other)
	{
		return (*this) *= other;
	}

	cgroup operator+(cgroup lhs, cgroup_ref rhs)
	{
		return lhs += rhs;
	}
	void cgroup::commute(cgroup_ref other) const
	{
		impl->commute(other.get());
	}
	bool operator!=(cgroup_cref left,cgroup_cref right)
	{
		return left.get() != (right.get());
	}
	bool operator==(cgroup_cref left, cgroup_cref right) 
	{
		return left.get() == (right.get());
	}

	cgroup_ref &cgroup_ref::operator*=(cgroup_cref other)
	{
		get().op(other.get());
		return *this;
	}

	cgroup_ref &cgroup_ref::operator+=(cgroup_cref other)
	{
		return (*this) *= other;
	}
	void cgroup_ref::commute(cgroup_ref other) const
	{
		get().commute(other.get());
	}


	cgroup_cref::cgroup_cref(cgroup_ref &other) : ref(other.ref) {}
	
	void cgroup_cref::commute(cgroup_ref other) const
	{
		get().commute(other.get());
	}

	/**
	 * Groups tend to have **very** short names in the litterature. 
	 * I want it to be easy to refer to litterature, so we use those short names.
	 * So a namespace will protect us from name clashes.
	 */
	namespace groups
	{

		/**
		 * C_N: the cyclic group with N elements.
		 * Oftentime called Z_N in the literature.
		 * This would create a name clash with the infinite group Z.
		 * 
		 * Note that the implementation limits the length of the cycle to less than 2^16-1
		 */
		template <uint16_t mod>
		class C
		{
		public:
			static constexpr uint16_t N = mod;

		private:
			static_assert(N > 0, "only value greater than zero make sense, only greater than 1 are useful.");
			uint16_t val;

		public:
			C(uint16_t _val)
			noexcept : val(_val)
			{
				val %= N; //only usage of modulo. that thing is expensive.
						  // this can be bad, if the input is greater than N,
						  // it is unclear that the user realize what they're doing.
						  // Perhaps they made a mistake, but the code will happily keep going.
			}
			void swap(C &other) noexcept
			{
				using std::swap;
				swap(other.val, val);
			}
			operator uint16_t() noexcept
			{
				return val;
			}
			C &operator+=(C other) noexcept
			{
				val += other.val;
				val -= (val >= N) * N;
				return *this;
			}
			//this function is what is actually used by the group compositor.
			C &op(C other) noexcept
			{
				return (*this) += other;
			}
			friend C op(C lhs, C rhs)
			{
				return lhs + rhs;
			}
			C &operator*=(C other) noexcept //in group theory we typically talk of a product operator.
			{
				return (*this) += other;
			}
			friend C operator+(C lhs, C rhs) noexcept
			{
				return lhs += rhs;
			}
			friend C operator*(C lhs, C rhs) noexcept
			{
				return lhs *= rhs;
			}
			C &inverse_() noexcept
			{
				val = N - val;
				return *this;
			}
			C inverse() const noexcept
			{
				C out(*this);
				return out.inverse_();
			}

			bool operator==(C other) const noexcept
			{
				return val == other.val;
			}
			bool operator!=(C other) const noexcept
			{
				return val != other.val;
			}

			// compute z such that *this*other = z*(*this), and store the result in other.
			// Cn is Abelian, therefor this function does nothing. Necessary to support non abelian groups.
			void commute(C &other) const {}

			friend std::ostream &operator<<(std::ostream &out, const C &c)
			{
				out << "grp::C<" << C::N << ">(" << c.val << ')';
				return out;
			}
		};

		/**
		 * Abelian group formed by integers under the action of addition.
		 * Useful for particles and spin conservation.
		 * In principle Z has an infinite domain, but here it is limited 
		 * to [-32767,32767] by usage of int16_t in the implementation
		 * 
		 * Note: In the case of particles conservation it is related to the U(1) symmetry 
		 * of the phase of the wavefunction. It is sometime (an abuse of language)
		 * called U(1). U(1) is a continuous group with a finite domain while
		 * N is a discrete group with an infinite domain. They are isomorphic.
		 */
		class Z
		{
			int16_t val;

		public:
			Z(int16_t _val)
			noexcept : val(_val) {}
			operator uint16_t() noexcept
			{
				return val;
			}
			void swap(Z &other) noexcept
			{
				using std::swap;
				swap(other.val, val);
			}
			Z &operator+=(Z other) noexcept
			{
				val += other.val;
				return *this;
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
			// Z& op( other) is the function used by cgroup.
			Z &op(Z other)
			{
				return (*this) += other;
			}
			friend Z op(Z lhs, Z rhs)
			{
				return lhs + rhs;
			}
			Z &inverse_() noexcept
			{
				val = -val;
				return *this;
			}
			Z inverse() const noexcept
			{
				Z out(*this);
				return out.inverse_();
			}
			bool operator==(Z other) const
			{
				return val == other.val;
			}
			bool operator!=(Z other) const
			{
				return val != other.val;
			}

			// compute u such that (*this)*other = u*(*this), and store the result in other.
			// Z is abelian, therefore this function does nothing
			void commute(Z &other) const {}

			friend std::ostream &operator<<(std::ostream &out, const Z &c)
			{
				out << "grp::Z(" << c.val << ')';
				return out;
			}
		};

		void swap(Z &lhs, Z &rhs) noexcept
		{
			lhs.swap(rhs);
		}
		template <uint16_t N>
		void swap(C<N> &lhs, C<N> &rhs) noexcept
		{
			lhs.swap(rhs);
		}

		static_assert(is_group_v<Z>, "Z isn't a group?! something is very wrong");
		static_assert(is_group_v<C<5>>, "C<5> isn't a group?! something is very wrong");

	} // namespace groups

	TEST_CASE("simple groups")
	{
		using namespace groups;
		C<2> c2_1(1);
		C<2> c2_0(0);
		C<2> c2_11 = c2_1 * c2_1;
		CHECK(c2_0 == c2_11);

		C<5> c5_3(3);
		C<5> c5_2(2);
		CHECK(c5_3 != c5_2);
		CHECK(c5_3.inverse() * c5_3 == C<5>(0)); //the product with one's own inverse give the trivial element.
		CHECK(c5_3.inverse() == c5_2);
		CHECK(c5_2.inverse() == c5_3);
		CHECK(c5_2.inverse().inverse() == c5_2);		 //inverse twice gives back the original value
		CHECK(C<5>(c5_2).inverse_().inverse_() == c5_2); //inverse in place twice gives back the original value
		CHECK(c5_2.op(c5_2) == C<5>(4));

		Z Z_1(1);
		Z Z_2(2);
		Z Z_11 = Z_1 * Z_1;
		CHECK(Z_2 == Z_11);

		Z Z_3(3);
		Z Z_m3(-3);
		CHECK(Z_3 != Z_m3);
		CHECK(Z_3.inverse() * Z_3 == Z(0)); //the product with one's own inverse give the trivial element.
		CHECK(Z_3.inverse() == Z_m3);
		CHECK(Z_m3.inverse() == Z_3);
		CHECK(Z_m3.inverse().inverse() == Z_m3);	  //inverse twice gives back the original value
		CHECK(Z(Z_m3).inverse_().inverse_() == Z_m3); //inverse in place twice gives back the original value
		CHECK(Z_3.op(Z_3) == Z(6));
	}
	TEST_CASE("composite groups")
	{
		using namespace groups;
		cgroup A(C<2>(0), Z(3));
		cgroup B(C<2>(1), Z(-1));
		CHECK_NOTHROW(auto c = A + B);
	}

} // namespace quantt

#endif /* D56E4C12_98E1_4C9E_B0C4_5B35A5A3CD17 */
