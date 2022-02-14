/*
 * File: dmrg_logger.h
 * Project: QuantiT
 * File Created: Friday, 29th October 2021 4:10:27 pm
 * Author: Alexandre Foley (Alexandre.foley@usherbrooke.ca)
 * Copyright (c) 2021 Alexandre Foley
 * Licensed under GPL v3
 */

#ifndef INCLUDE_DMRG_LOGGER_H
#define INCLUDE_DMRG_LOGGER_H

#include "MPT.h"
#include "blockTensor/btensor.h"
#include "dmrg_options.h"
#include <torch/torch.h>

namespace quantit
{
/**
 * Pure virtual base class for logging the state of dmrg.
 * This logger is meant to do macro scale logging: as such it isn't perfomance critical and we can afford the virtual
 * calls. dmrg call init before the first sweep, it_log_all at every iterations and call end_log_all once when the state
 * is converged. the last call to it_log_all receive the same arguments as end_log_all (it's not useful to have both
 * function do something if it_log_all does something at every iterations).
 */
class dmrg_logger
{
  public:
	virtual void log_step(size_t) = 0;
	virtual void log_energy(const torch::Tensor &) = 0;
	virtual void log_energy(const btensor &) = 0;
	virtual void log_bond_dims(const MPS &) = 0;
	virtual void log_bond_dims(const bMPS &) = 0;

	virtual void init(const dmrg_options &) {}

	virtual void it_log_all(size_t step_num, const torch::Tensor &E, const MPS &state) { log_all(step_num, E, state); }
	virtual void it_log_all(size_t step_num, const btensor &E, const bMPS &state) { log_all(step_num, E, state); }
	virtual void end_log_all(size_t step_num, const torch::Tensor &E, const MPS &state) { log_all(step_num, E, state); }
	virtual void end_log_all(size_t step_num, const btensor &E, const bMPS &state) { log_all(step_num, E, state); }

	virtual void log_all(size_t step_num, const torch::Tensor &E, const MPS &state)
	{
		log_step(step_num);
		log_energy(E);
		log_bond_dims(state);
	}
	virtual void log_all(size_t step_num, const btensor &E, const bMPS &state)
	{
		log_step(step_num);
		log_energy(E);
		log_bond_dims(state);
	}

	virtual ~dmrg_logger() {}
};
/**
 * A default logger that does nothing.
 */
class dmrg_default_logger : public dmrg_logger
{
  public:
	void log_step(size_t) override {}
	void log_energy(const torch::Tensor &) override {}
	void log_energy(const btensor &) override {}
	void log_bond_dims(const MPS &) override {}
	void log_bond_dims(const bMPS &) override {}
};

class dmrg_log_simple final : public quantit::dmrg_default_logger
{
  public:
	size_t it_num;
	size_t middle_bond_dim;

	void log_step(size_t it) override { it_num = it; }
	void log_bond_dims(const quantit::bMPS &mps) override { log_bond_impl(mps); }
	void log_bond_dims(const quantit::MPS &mps) override { log_bond_impl(mps); }
	void it_log_all(size_t, const torch::Tensor &, const quantit::MPS &) override {}
	void it_log_all(size_t, const btensor &, const bMPS &) override {}

  private:
	template <class tMPS>
	void log_bond_impl(const tMPS &mps)
	{
		auto pos = mps.size() / 2;
		middle_bond_dim = std::max(mps[pos].sizes()[0], mps[pos].sizes()[2]);
	}
};
class dmrg_log_sweeptime final : public quantit::dmrg_default_logger
{
  public:
	size_t it_num;
	size_t middle_bond_dim;
	std::chrono::steady_clock::time_point then;
	std::vector<double> time_list;
	std::vector<size_t> bond_list;

	void log_step(size_t it) override { it_num = it; }
	void log_energy(const torch::Tensor &) override {}
	void log_energy(const quantit::btensor &) override {}

	void init(const quantit::dmrg_options &opt) override
	{
		then = std::chrono::steady_clock::now();
		time_list = std::vector<double>(opt.maximum_iterations);
		bond_list = std::vector<size_t>(opt.maximum_iterations);
	}

	void log_bond_dims(const quantit::MPS &mps) override
	{
		auto pos = mps.size() / 2;
		middle_bond_dim = std::max(mps[pos].sizes()[0], mps[pos].sizes()[2]);
	}
	void log_bond_dims(const quantit::bMPS &mps) override
	{
		auto pos = mps.size() / 2;
		middle_bond_dim = std::max(mps[pos].sizes()[0], mps[pos].sizes()[2]);
	}
	void it_log_all(size_t it, const quantit::btensor &E0, const quantit::bMPS &mps) override
	{
		auto now = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed_seconds = now - then;
		then = now;
		log_bond_dims(mps);
		bond_list[it] = middle_bond_dim;
		time_list[it] = elapsed_seconds.count();
		log_step(it);
		log_energy(E0);
	}
	void it_log_all(size_t it, const torch::Tensor &E0, const quantit::MPS &mps) override
	{
		auto now = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed_seconds = now - then;
		then = now;
		log_bond_dims(mps);
		bond_list[it] = middle_bond_dim;
		time_list[it] = elapsed_seconds.count();
		log_step(it);
		log_energy(E0);
	}
};

} // namespace quantit

#endif // INCLUDE_DMRG_LOGGER_H
