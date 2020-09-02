
#include <iomanip>
#include <iostream>
#include <chrono>
#include <array>
#include "../include/SSEWrapper.hpp"
#include "../include/AVX2Wrapper.hpp"
#include "../include/NEONWrapper.hpp"

//#define ENABLE_SIMD

template<typename Type, typename SFINAE = std::enable_if_t<std::disjunction_v<std::is_same<Type, float>, std::is_same<Type, double>>>>
struct mat4x4 {
#if defined(ENABLE_SIMD) && defined(__AVX2__)
	using vector = std::conditional_t< std::is_same_v<Type, double>, type::fp64x4_t, type::fp32x4_t>;
	std::array<vector, 4> elm = {};
#elif defined(ENABLE_SIMD)
	using vector = std::conditional_t< std::is_same_v<Type, double>, type::fp64x2_t, type::fp32x4_t>;
	std::array<vector, std::is_same_v<Type, double> ? 8 : 4> elm = {};
#else
	using vector = std::array<Type, 4>;
	std::array<vector, 4> elm = {};
#endif
	
	mat4x4() = default;
	mat4x4(const decltype(elm)& init) : elm(init){}
	mat4x4(std::initializer_list<std::initializer_list<Type>> init) {
		auto begin = init.begin();
		for(auto y = 0; y < 4; ++y, ++begin){
			auto x_begin = begin->begin();
			for(auto x = 0; x < 4; ++x, ++x_begin){
				elm[(y*4+x)*elm.size()/16][(y*4+x)%(16/elm.size())] = *x_begin;
			}
		}
	}
#ifdef ENABLE_SIMD
	mat4x4 operator*(const mat4x4& mat) const noexcept {
		if constexpr(std::is_same_v<vector, type::fp64x2_t>) {
			mat4x4 result;
			result.elm[0] = (mat.elm[0] * elm[0].dup(0))
				.addmul(mat.elm[2], elm[0].dup(1))
				.addmul(mat.elm[4], elm[1].dup(0))
				.addmul(mat.elm[6], elm[1].dup(1));
			result.elm[1] = (mat.elm[1] * elm[0].dup(0))
				.addmul(mat.elm[3], elm[0].dup(1))
				.addmul(mat.elm[5], elm[1].dup(0))
				.addmul(mat.elm[7], elm[1].dup(1));
			
			result.elm[2] = (mat.elm[0] * elm[2].dup(0))
				.addmul(mat.elm[2], elm[2].dup(1))
				.addmul(mat.elm[4], elm[3].dup(0))
				.addmul(mat.elm[6], elm[3].dup(1));
			result.elm[3] = (mat.elm[1] * elm[2].dup(0))
				.addmul(mat.elm[3], elm[2].dup(1))
				.addmul(mat.elm[5], elm[3].dup(0))
				.addmul(mat.elm[7], elm[3].dup(1));
			
			result.elm[4] = (mat.elm[0] * elm[4].dup(0))
				.addmul(mat.elm[2], elm[4].dup(1))
				.addmul(mat.elm[4], elm[5].dup(0))
				.addmul(mat.elm[6], elm[5].dup(1));
			result.elm[5] = (mat.elm[1] * elm[4].dup(0))
				.addmul(mat.elm[3], elm[4].dup(1))
				.addmul(mat.elm[5], elm[5].dup(0))
				.addmul(mat.elm[7], elm[5].dup(1));
			
			result.elm[6] = (mat.elm[0] * elm[6].dup(0))
				.addmul(mat.elm[2], elm[6].dup(1))
				.addmul(mat.elm[4], elm[7].dup(0))
				.addmul(mat.elm[6], elm[7].dup(1));
			result.elm[7] = (mat.elm[1] * elm[6].dup(0))
				.addmul(mat.elm[3], elm[6].dup(1))
				.addmul(mat.elm[5], elm[7].dup(0))
				.addmul(mat.elm[7], elm[7].dup(1));
			return result;
		}
		else if constexpr(std::is_same_v<vector, type::fp32x4_t>
#ifdef __AVX2__
			|| std::is_same_v<vector, type::fp64x4_t>
#endif
			) {
			mat4x4 result;
			result.elm[0] = (mat.elm[0] * elm[0].dup(0))
				.addmul(mat.elm[1], elm[0].dup(1))
				.addmul(mat.elm[2], elm[0].dup(2))
				.addmul(mat.elm[3], elm[0].dup(3));
			result.elm[1] = (mat.elm[0] * elm[1].dup(0))
				.addmul(mat.elm[1], elm[1].dup(1))
				.addmul(mat.elm[2], elm[1].dup(2))
				.addmul(mat.elm[3], elm[1].dup(3));
			result.elm[2] = (mat.elm[0] * elm[2].dup(0))
				.addmul(mat.elm[1], elm[2].dup(1))
				.addmul(mat.elm[2], elm[2].dup(2))
				.addmul(mat.elm[3], elm[2].dup(3));
			result.elm[3] = (mat.elm[0] * elm[3].dup(0))
				.addmul(mat.elm[1], elm[3].dup(1))
				.addmul(mat.elm[2], elm[3].dup(2))
				.addmul(mat.elm[3], elm[3].dup(3));
			return result;
		}
	}
#else
	mat4x4 operator*(const mat4x4& mat) const noexcept {
		mat4x4 result;
		for(auto y = 0; y < 4; ++y) {
			result.elm[y][0] = elm[y][0] * mat.elm[0][0]
					+ elm[y][1] * mat.elm[1][0]
					+ elm[y][2] * mat.elm[2][0]
					+ elm[y][3] * mat.elm[3][0];

			result.elm[y][1] = elm[y][0] * mat.elm[0][1]
					+ elm[y][1] * mat.elm[1][1]
					+ elm[y][2] * mat.elm[2][1]
					+ elm[y][3] * mat.elm[3][1];

			result.elm[y][2] = elm[y][0] * mat.elm[0][2]
					+ elm[y][1] * mat.elm[1][2]
					+ elm[y][2] * mat.elm[2][2]
					+ elm[y][3] * mat.elm[3][2];

			result.elm[y][3] = elm[y][0] * mat.elm[0][3]
					+ elm[y][1] * mat.elm[1][3]
					+ elm[y][2] * mat.elm[2][3]
					+ elm[y][3] * mat.elm[3][3];
			
		}
		return result;
	}
#endif
};

template<typename T, typename SFINAE = std::enable_if_t<std::disjunction_v<std::is_same<T, float>, std::is_same<T, double>>>>
std::ostream& operator<<(std::ostream& os, const mat4x4<T>& mat) {
	size_t i = 0;
	for(const auto& x : mat.elm){
		for(const auto e : x){
			if (i % 4 == 0) os << "| ";
			os << std::scientific << std::setprecision(2) << e << " ";
			if (i % 4 == 3) os << " |\n";
			++i;
		}
	}
	return os;
}


int main() {
	mat4x4<float> f = {
		{11,12,13,14},
		{21,22,23,24},
		{31,32,33,34},
		{41,42,43,44}
	};
	std::cout << f*f << std::endl;
	{
		std::cout << "fp32" << std::endl;
		float tmp=0;
		mat4x4<float> mat = {{tmp,tmp,tmp,tmp},{tmp,tmp,tmp,tmp},{tmp,tmp,tmp,tmp},{tmp,tmp,tmp,tmp}};
		
		for(auto p = 0; p < 1; ++p){
			auto start = std::chrono::system_clock::now();
			// 448*10^6*50 = 22.4GFLOPS
			for(auto i=0; i < 1000000*50; ++i) {
				// 112 * 4 = 448FLOPS
				mat = mat*mat*mat*mat*mat;
			}

			std::cout << std::fixed << std::setprecision(1) << 448*50 / static_cast<double>(
				std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count()
			) << " GFlops" << std::endl;
		}
		std::cout << mat << std::endl;
	}
	{
		std::cout << "fp64" << std::endl;
		double tmp=0;
		mat4x4<double> mat = {{tmp,tmp,tmp,tmp},{tmp,tmp,tmp,tmp},{tmp,tmp,tmp,tmp},{tmp,tmp,tmp,tmp}};

		for(auto p = 0; p < 1; ++p){
			auto start = std::chrono::system_clock::now();
			// 448*10^6*50 = 22.4GFLOPS
			for(auto i=0; i < 1000000*50; ++i) {
				// 112 * 4 = 448FLOPS
				mat = mat*mat*mat*mat*mat;
			}

			std::cout << std::fixed << std::setprecision(1) << 448*50 / static_cast<double>(
				std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count()
			) << " GFlops" << std::endl;
		}
		std::cout << mat << std::endl;
	}
	return 0;
}