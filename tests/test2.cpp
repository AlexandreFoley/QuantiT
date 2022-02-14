/*
 * File: catch_test.cpp
 * Project: QuantiT
 * File Created: Friday, 31st July 2020 11:04:58 am
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * -----
 * Last Modified: Friday, 31st July 2020 11:04:58 am
 * Modified By: Alexandre Foley (Alexandre.foley@usherbrooke.ca>)
 * -----
 * Copyright (c) 2020 Alexandre Foley
 * Licensed under GPL v3
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "DMRG_heisenberg_test.h"

int main(int argc, char **argv)
{
	doctest::Context context;
	at::init_num_threads();
	// custom
	context.addFilter("test-case", "*heisenberg*");

	// defaults
	// context.addFilter("test-case-exclude", "*math*"); // exclude test cases with "math" in their name
	context.setOption("abort-after", 5);   // stop test execution after 5 failed assertions
	context.setOption("order-by", "name"); // sort the test cases by their name
	context.applyCommandLine(argc, argv);

	// overrides
	context.setOption("no-breaks", true); // don't break in the debugger when assertions fail

	int res = context.run(); // run

	if (context.shouldExit()) // important - query flags (and --exit) rely on the user doing this
		return res;           // propagate the result of the tests

	int client_stuff_return_code = 0;
	// your program - if the testing framework is integrated in your production code

	return res + client_stuff_return_code; // the result from doctest is propagated here as well
}
// with the preceding define, doctest autogenerate a main() function that run the test found in the sources included.

// in here the higher level tests, that combine features from multiple .h
// #ifdef NDEBUG
// don't try to run those without optimizations, it's gonna take a long long time instead of a few seconds.
// #endif