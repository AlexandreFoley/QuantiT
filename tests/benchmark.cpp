/*
 * File: benchmark.cpp
 * Project: quantt
 * File Created: Monday, 20th September 2021 12:59:11 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * Copyright (c) 2021 Alexandre Foley
 * All rights reserved
 */

// #define DOCTEST_CONFIG_DISABLE
#define DOCTEST_CONFIG_IMPLEMENT
#define ANKERL_NANOBENCH_IMPLEMENT

#include "blockTensor/btensor.h"
#include "nanobench.h"
#include <random>

template <class T>
void DNO(T &&t)
{
	ankerl::nanobench::doNotOptimizeAway(t);
};

// // the basic case, more option are available without using this macro.
// // use inside main()
#define BENCHMARK(name, function) ankerl::nanobench::Bench().run(name, function)

//TODO: performance test on the trivial group
//TODO: performance test on a two dimensionnal latice? no easy way to generate that right away.

int main()
{
	torch::InferenceMode _GUARD;
	using cval = quantt::quantity<quantt::conserved::Z>;
	auto r = std::mt19937(std::random_device()());
	auto size_generator = std::uniform_int_distribution(10, 40); // size generator
	auto cval_generator = std::uniform_int_distribution(-5, 5); // conserved value generator
	auto sg = [&]() { return size_generator(r); };
	auto cg = [&]() { return cval_generator(r); };
	auto shapeA = 
	    quantt::btensor({{{sg(), cval(cg())}, {sg(), cval(cg())}, {sg(), cval(cg())}, {sg(), cval(cg())}}}, cval(0));
	auto shapeB = quantt::btensor({{{1, cval(cg())},
	                                {1, cval(cg())},
	                                {1, cval(cg())}}},
	                              cval(0));
	auto shapeC = quantt::shape_from(shapeA, shapeB).reshape({}).conj();
	auto shapeX = shape_from(shapeA, shapeB, shapeC);
	auto tX = torch::rand({200,200});
	auto X = quantt::rand_like(shapeX);
	auto Y = X.conj();

	quantt::btensor Z;
	DNO(Z = X.tensordot(Y, {0}, {0}));
	DNO(auto tZ = tX.tensordot(tX,{0},{0}));
	// fmt::print("{}\n",Z.begin());
	//  ankerl::nanobench::Bench().minEpochIterations(2000).run("tensordot",[&]()
	// {
		// DNO(Z = X.tensordot(Y, {0}, {0}));
		// DNO(Z.begin());
	// });
}