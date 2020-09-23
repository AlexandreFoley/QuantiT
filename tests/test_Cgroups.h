/*
 * File: test_Cgroups.h
 * Project: quantt
 * File Created: Tuesday, 15th September 2020 1:19:14 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Tuesday, 15th September 2020 1:19:14 pm
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#ifndef C53CF787_82EF_4871_AED3_8ADA2B155742
#define C53CF787_82EF_4871_AED3_8ADA2B155742

#include "composite_group.h"
#include "groups.h"

#include "cond_doctest.h"

TEST_CASE("composite groups")
{
	using namespace quantt;
	using namespace groups;
	cgroup A(C<2>(0), Z(3)); // order matters.
	cgroup B(C<2>(1), Z(-1));
	cgroup A_copy(A);
	cgroup B_copy(B);
	CHECK_NOTHROW(auto c = A + B);
	cgroup D(Z(0), C<2>(1)); // D has a different underlying type.

	// the cast to void silences warnings about unused return values and
	// comparison. We know, it's ok.
	CHECK_THROWS_AS((void)(D == A),
	                const std::bad_cast&); // A and D have different derived
	                                       // type: they're not compatible
	CHECK_THROWS_AS((void)(D != A),
	                const std::bad_cast&); // A and D have different derived
	                                       // type: they're not compatible
	CHECK_THROWS_AS((void)(D * A),
	                const std::bad_cast&); // A and D have different derived
	                                       // type: they're not compatible
	CHECK_THROWS_AS(D *= A,
	                const std::bad_cast&); // A and D have different derived
	                                       // type: they're not compatible
	CHECK_THROWS_AS(D += A,
	                const std::bad_cast&); // A and D have different derived
	                                       // type: they're not compatible
	CHECK_THROWS_AS((void)(D + A),
	                const std::bad_cast&); // A and D have different derived
	                                       // type: they're not compatible
	cgroup_ref A_ref(A);
	cgroup_cref A_cref(A);
	cgroup_cref B_cref(B);
	cgroup_ref B_ref(B);
	// cgroup_ref is a drop-in replacement for a reference to cgroup
	// cgroup_cref is a drop-in-replacement for a reference to a constant
	// cgroup. Those two classes exists to facilitate manipulation of single
	// cgroup located within a special container for this polymorphic type. We
	// make extensive use of the reference type within the tests specifically to
	// verify the correctness of their implementation. We advise avoiding those
	// two reference type whenever possible. const cgroup& and cgroup& are
	// perfectly fine for most purpose.
	CHECK_NOTHROW(A_copy = A_ref);
	CHECK_NOTHROW(A_copy = A_cref);
	// Checking that the ref type doesn't loose track of its target.
	CHECK(A_cref == A_ref);
	CHECK(A_ref == A_cref);
	CHECK(A_ref == A);
	CHECK_NOTHROW(A_ref *= B);
	CHECK(A_ref == A);
	CHECK(A_ref == A_cref);
	CHECK_NOTHROW(A_ref += B);
	CHECK(A_ref == A);
	CHECK(A_ref == A_cref);
	A_copy = A;
	CHECK(B == B_copy); // commute_ act on the the calling object only.
	CHECK(A_copy == A); // this is an abelian group.
	SUBCASE("Cast ambiguity")
	{
		// In this subcase with test different combination of cgroup,cgroup_ref
		// and cgroup_cref to make sure all operation resolve correctly. Because
		// of the different type and the ways in which they are equivalent, a
		// mistake could lead to some operations failing.
		CHECK_NOTHROW(auto _t = A * B);
		CHECK_NOTHROW(auto _t = A_ref * B);
		CHECK_NOTHROW(auto _t = A_cref * B);
		CHECK_NOTHROW(auto _t = A * B_ref);
		CHECK_NOTHROW(auto _t = A_ref * B_ref);
		CHECK_NOTHROW(auto _t = A_cref * B_ref);
		CHECK_NOTHROW(auto _t = A * B_cref);
		CHECK_NOTHROW(auto _t = A_ref * B_cref);
		CHECK_NOTHROW(auto _t = A_cref * B_cref);
		CHECK_NOTHROW(
		    auto _t =
		        A * B_cref.inverse()); // this should use the
		                               // operator*(cgroup_cref&,cgroup&&);
		CHECK_NOTHROW(
		    auto _t = A_ref * B.inverse()); // this should use the
		                                    // operator*(cgroup_cref&,cgroup&&);
		CHECK_NOTHROW(auto _t =
		                  A_cref *
		                  B_ref.inverse()); // this should use the
		                                    // operator*(cgroup_cref&,cgroup&&);
		//+ and * are completly equivalent.
		CHECK_NOTHROW(auto _t = A + B);
		CHECK_NOTHROW(auto _t = A_ref + B);
		CHECK_NOTHROW(auto _t = A_cref + B);
		CHECK_NOTHROW(auto _t = A + B_ref);
		CHECK_NOTHROW(auto _t = A_ref + B_ref);
		CHECK_NOTHROW(auto _t = A_cref + B_ref);
		CHECK_NOTHROW(auto _t = A + B_cref);
		CHECK_NOTHROW(auto _t = A_ref + B_cref);
		CHECK_NOTHROW(auto _t = A_cref + B_cref);
		CHECK_NOTHROW(
		    auto _t =
		        A + B_cref.inverse()); // this should use the
		                               // operator+(cgroup_cref&,cgroup&&);
		CHECK_NOTHROW(
		    auto _t = A_ref + B.inverse()); // this should use the
		                                    // operator+(cgroup_cref&,cgroup&&);
		CHECK_NOTHROW(auto _t =
		                  A_cref +
		                  B_ref.inverse()); // this should use the
		                                    // operator+(cgroup_cref&,cgroup&&);
	}
	auto C = cgroup(B_cref);
	// C == A_ref *B_cref;
	// C == B_cref *A_cref.inverse();
	// C == A_ref *B_cref *A_cref.inverse();		   //A.commute(B) should be
	// an optimisation of the formula on the right.
	CHECK(C == A_ref * B_cref *
	               A_cref.inverse()); // A.commute(B) should be an optimisation
	                                  // of the formula on the right.
	C = A;
	CHECK(C == B.inverse() * A * B); // A.commute_(B) should be an optimization
	                                 // of the formula on the right.
}

#endif /* C53CF787_82EF_4871_AED3_8ADA2B155742 */
