#include "../include/AVX2Wrapper.hpp"
#include "../include/SSEWrapper.hpp"
#include <vector>
#include <chrono>
#include <iostream>
#include <array>

int main() {
	std::cout << "AVX :" << std::boolalpha << instruction::AVX() << std::endl;
	std::cout << "AVX2:" << instruction::AVX2() << std::endl;
	std::cout << "FMA :" << instruction::FMA() << std::endl;
	std::cout << "SSE4.2" << instruction::SSE4_2() << std::endl;
	std::cout << "SSE4.1" << instruction::SSE4_1() << std::endl;

	type::uint64x2 ine{3,2};
	ine = ine + 9;

	{
		std::cout << "Instrinsics Benchmark" << std::endl;
		vector256_type<float>::vector v1{};
		vector256_type<float>::vector v2{};
		auto j = _mm256_add_ps(v1, v2);

		for (int a = 0; a < 0; a++) {
			auto start = std::chrono::system_clock::now();
			for (int i = 0; i < 100000000; i++) {
				j = _mm256_add_ps(v1, v2);
				v1 = _mm256_sub_ps(j, v2);
				v2 = _mm256_mul_ps(j, v1);
				j = _mm256_div_ps(v1, v2);
				v1 = _mm256_fmadd_ps(v2, j, v1);
			}
			std::cout << (std::chrono::system_clock::now() - start).count() / 1000 << std::endl;
		}
	}
	std::cout << "------------------------" << std::endl;
	{
		std::cout << "Wrapper Benchmark" << std::endl;
		float v[] = { 1.1,-2.2,3.3,4.4,5,6,-7,8 };
		vector256<float> v1;
		v1.load(v);
		v1 = v1.abs().floor();
		vector256<float> v2 = 2.2f;
		vector256<int8_t> d1(1, 4, 5, -4, 1, 4, 5, 4, 1, 4, 5, 4, 1, 4, 5, 4, 1, 4, 5, -4, 1, 4, 5, 4, 1, 4, 5, 4, 1, 4, 5, 4);
		v2.store(v);
		vector256<float> t(10);
		vector256<uint8_t> t1(UINT8_MAX);
		std::cout << vector256<uint8_t>(UINT8_MAX / 2) + vector256<uint8_t>(UINT8_MAX / 2) << std::endl;
		vector256<uint8_t> t2(vector256<uint16_t>(0).concat(vector256<uint16_t>(UINT8_MAX)));
		std::cout << t2.shuffle(vector256<uint16_t>(30).concat(vector256<uint16_t>(3))) << std::endl;
		auto fx = d1[0];
		auto fx2 = d1[1];
		auto fx3 = d1[3];
		int64_t l = 0;
		for (auto e : d1)
			l += e;
		auto m1 = t1.max(t2);
		auto m2 = t1.min(t2);
		std::cout << (t1 < t2).is_all_true();
		auto c = (t1.max(t2) == function::cmp_blend(t1 > t2, t1, t2));
		auto bo = (t == t).is_all_true();
		auto b1 = (~(t == t)).is_all_false();
		auto b2 = function::cmp_blend((t1 > t2), v2, t);
		auto j = v1 + v2;

		//std::cout << t1.alternate(t2) << std::endl;

		for (int a = 0; a < 10; a++) {
			auto start = std::chrono::system_clock::now();
			for (int i = 0; i < 100000000; i++) {
				j = v1 + v2;
				v1 = j - v2;
				v2 = j * v1;
				j = v1 / v2;
				v1 = v2.muladd(j, v1);
			}
			std::cout << (std::chrono::system_clock::now() - start).count() / 1000 << std::endl;
		}
	}
	{
		std::cout << "-------------------------" << std::endl;
		std::cout << "Non SIMD Benchmark" << std::endl;
		float v1[8] = {};
		float v2[8] = {};
		float j[8] = {};

		for (int a = 0; a < 10; a++) {
			auto start = std::chrono::system_clock::now();
			for (int i = 0; i < 100000000; i++) {
				for (int n = 0; n < 8; n++) {
					j[n] = v1[n] + v2[n];
					v1[n] = j[n] - v2[n];
					v2[n] = j[n] * v1[n];
					j[n] = v1[n] / v2[n];
					v1[n] = v2[n] * j[n] + v1[n];
				}
			}
			std::cout << (std::chrono::system_clock::now() - start).count() / 1000 << std::endl;
		}
	}
	return 0;
}
