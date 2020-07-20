#include <torch/torch.h>
#include <iostream>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/format.h>

torch::Tensor generate_tens()
{
	torch::Tensor out;
	return out;
}

struct A
{
	A(int _a) : a(_a) {}
	int a;

	friend std::ostream &operator<<(std::ostream &out, const A &a)
	{
		return out << '{' << a.a << ',' << 'b' << '}';
	}
};

template <>
struct fmt::formatter<torch::Tensor>
{

	constexpr auto parse(parse_context &ctx)
	{
		auto it = ctx.begin(), end = ctx.end();
		++it;
		// Check if reached the end of the range:
		if (it != end && *it != '}')
			throw format_error("invalid format");

		// Return an iterator past the end of the parsed range:
		return it;
	}
	template <typename FormatContext>
	auto format(const torch::Tensor &tens, FormatContext &ctx)
	{
		std::stringstream strstr;
		at::print(strstr, tens, 80);
		std::string tens_string(strstr.str());
		auto end = tens_string.length();
		fmt::print("the end: {}\n", end);
		// find occurences of {} and replace with ()
		// done because fmt use {} to mark format section which can lead to weird interactions...
		auto it = tens_string.find("{", 0);
		while (it < end)
		{
			tens_string[it] = '(';
			++it;
			it = tens_string.find("{", it);
		}
		it = tens_string.find("}", 0);
		while (it < end)
		{
			tens_string[it] = ')';
			++it;
			it = tens_string.find("}", it);
		}

		return format_to(ctx.out(), tens_string);
	}
};

int main()
{

	std::cout << "This is a test\n";
	torch::Device cuda_device(torch::kCPU); // default the cuda device to a gpu. small lie to keep the code working if there isn't one.
	if (torch::cuda::is_available())
	{
		fmt::print("CUDA is available!\n");
		cuda_device = torch::Device(torch::kCUDA);
	}
	torch::Tensor tensor = torch::rand({2, 3});
	std::cout << tensor << '\n';
	std::ostringstream stringstream;
	stringstream << tensor << '\n';
	std::cout << stringstream.str();
	fmt::print("---{}---\n", stringstream.str());
	fmt::print("{}", tensor);
	// tensor = tensor.to(cuda_device);
	// fmt::print("{}", tensor);
	fmt::print("This is a test");
	// fmt::print("{{}}");

	return 0;
}