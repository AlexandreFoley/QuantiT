/*
 * File: TensorGroup_mock.h
 * Project: quantt
 * File Created: Wednesday, 2nd September 2020 11:31:04 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Wednesday, 2nd September 2020 11:31:04 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * All rights reserved
 */

#if __has_include(<gmock/gmock.h>)

#include "TensorGroup.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace quantt
{

	// class mockcgroup_impl : public cgroup_impl
	// {
	// 	MOCK_METHOD(mockcgroup_impl &, op, (const mockcgroup_impl &), (override));
	// 	MOCK_METHOD(mockcgroup_impl &, inverse_, (), (override));
	// 	MOCK_METHOD(std::unique_ptr<mockcgroup_impl>, clone, (), (override));
	// 	MOCK_METHOD(bool, operator==,(const mockcgroup_impl &), (override));
	// 	MOCK_METHOD(bool, operator!=,(const mockcgroup_impl &), (override));
	// };
} // namespace quantt

#endif