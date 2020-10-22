/*
 * File: doctest_proxy.h
 * Project: quantt
 * File Created: Saturday, 8th August 2020 11:06:52 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Saturday, 8th August 2020 11:06:52 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#ifndef ECD8A80F_40DA_4301_85CD_C29B81A9E3F4
#define ECD8A80F_40DA_4301_85CD_C29B81A9E3F4

#include "doctest_redef.h"

#if defined(DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN) || defined(DOCTEST_CONFIG_IMPLEMENT)
//actually include doctest if we're in a test context.
#include "doctest.h"

#else
// if we're not in a test context,
// define the macro of doctest to do nothing at all. hopefully turn all the test code into unused code without any side effects.
// there's actually many missing macro here. this is the subset i have used so far.
// test case are made a static template: this give them file-local linkage, the contained loose code doesn't violate the single definition rule because of the multiple inclusion points.
#define CONCAT(x, y, z) x##y##z
#define CONCATENATE(x, y, z) CONCAT(x, y, z)
#define DOCUNIQ_IMPL(x, y, z) CONCAT(x, y, z)
#define DOCUNIQ(x) DOCUNIQ_IMPL(x, __COUNTER__, __LINE__)

#define TEST_CASE(...) template <class T> \
static void DOCUNIQ(TEST_CASE_DUMMY_FCT)(const T &a)
#define SUBCASE(...)
#define CHECK_LE(...)
#define CHECK_FALSE(...)
#define CHECK_GE(...)
#define CHECK_GE(...)
#define CHECK_EQ(...)
#define CHECK_NE(...)
#define CHECK_GT(...)
#define CHECK_LT(...)
#define CHECK_LE(...)
#define CHECK(...)
#define CHECK_THROWS_AS(...)
#define CHECK_NOTHROW(...)

#define REQUIRE_LE(...)
#define REQUIRE_FALSE(...)
#define REQUIRE_GE(...)
#define REQUIRE_GE(...)
#define REQUIRE_EQ(...)
#define REQUIRE_NE(...)
#define REQUIRE_GT(...)
#define REQUIRE_LT(...)
#define REQUIRE_LE(...)
#define REQUIRE(...)
#define REQUIRE_THROWS_AS(...)
#define REQUIRE_NOTHROW(...)

#define WARN_LE(...)
#define WARN_GE(...)
#define WARN_GE(...)
#define WARN_EQ(...)
#define WARN_NE(...)
#define WARN_GT(...)
#define WARN_LT(...)
#define WARN_LE(...)
#define WARN(...)
#define WARN_THROWS_AS(...)
#define WARN_NOTHROW(...)
#define WARN_MESSAGE(...)

#endif

#endif /* ECD8A80F_40DA_4301_85CD_C29B81A9E3F4 */
