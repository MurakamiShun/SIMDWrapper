#include "AVXWrapper.hpp"
#include <vector>
#include <chrono>
#include <iostream>
#include <array>

int main() {
	{
		std::cout << "Instrinsics Benchmark" << std::endl;
		AVX_type<float>::vector v1{};
		AVX_type<float>::vector v2{};
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
			std::cout << (std::chrono::system_clock::now() - start).count()/1000 << std::endl;
		}
	}
	std::cout << "------------------------" << std::endl;
	{
		std::cout << "Wrapper Benchmark" << std::endl;
		float v[] = { 1.1,-2.2,3.3,4.4,5,6,-7,8 };
		AVX_vector<float> v1 = v;
		std::cout << "v1[";
		for (auto element : v1) {
			std::cout << element << ",";
		}
		std::cout << "]" << std::endl;
		v1 = v1.abs().floor();
		AVX_vector<float> v2 = 2.2f;
		AVX_vector<double> d1(1, 4, 5, 4);
		v2 >> v;
		AVX_vector<int32_t> t(10);
		AVX_vector<float> t1(1, 3, -4, 6,0,0,0,0);
		AVX_vector<float> t2(2, -4, -5, 6,0,0,0,0);
		auto fx = t1[0];
		auto fx2 = t1[1];
		auto fx3 = t1[3];
		auto m1 = t1.max(t2);
		auto m2 = t1.min(t2);
		auto f = t1 > t2;
		auto c = (t1.max(t2) == function::cmp_blend(t1 > t2, t1, t2));
		auto bo = (t == t).is_all_one();
		auto b1 = (~(t == t)).is_all_zero();
		auto f32 = static_cast<AVX_vector<float>>(t);
		auto j = v1 + v2;

		std::cout << t << std::endl;

		for (int a = 0; a < 10; a++) {
			auto start = std::chrono::system_clock::now();
			for (int i = 0; i < 100000000; i++) {
				j = v1 + v2;
				v1 = j - v2;
				v2 = j * v1;
				j = v1 / v2;
				v1 = v2.muladd(j, v1);
			}
			std::cout << (std::chrono::system_clock::now() - start).count()/1000 << std::endl;
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