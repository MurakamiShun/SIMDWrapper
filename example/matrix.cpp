
#include <iomanip>
#include <iostream>
#include <chrono>
#include "../include/SSEWrapper.hpp"
#include "../include/AVX2Wrapper.hpp"
#include "../include/NEONWrapper.hpp"

//#define ENABLE_SIMD

template<typename T>
struct mat4x4;

template<>
struct mat4x4<float> {
#ifdef ENABLE_SIMD
	std::array<type::fp32x4_t, 4> elm;
#else
	std::array<std::array<float, 4>, 4> elm;
#endif
	
	mat4x4() = default;
	mat4x4(const decltype(elm)& init) : elm(init){}
	mat4x4(std::initializer_list<std::initializer_list<float>> init) {
		auto begin = init.begin();
		for(auto y = 0; y < 4; ++y, ++begin){
			auto x_begin = begin->begin();
			for(auto x = 0; x < 4; ++x, ++x_begin){
				elm[y][x] = *x_begin;
			}
		}
	}

#ifdef ENABLE_SIMD
	mat4x4 T() const noexcept {
		return function::transpose(elm);
	}
#else
	mat4x4 T() const noexcept {
		return {
			{elm[0][0], elm[1][0], elm[2][0], elm[3][0]},
			{elm[0][1], elm[1][1], elm[2][1], elm[3][1]},
			{elm[0][2], elm[1][2], elm[2][2], elm[3][2]},
			{elm[0][3], elm[1][3], elm[2][3], elm[3][3]}
		};
	}
#endif

#ifdef ENABLE_SIMD
	mat4x4 operator*(const mat4x4& mat) const noexcept {
		mat4x4 result, tmp, matT = mat.T();
		for(auto y = 0; y < 4; ++y) {
			for(auto x = 0; x < 4; ++x){
				tmp.elm[x] = elm[y] * matT.elm[x];
			}
			tmp = tmp.T();
			result.elm[y] = tmp.elm[0] + tmp.elm[1] + tmp.elm[2] + tmp.elm[3];
		}
		return result;
	}
#else
	mat4x4 operator*(mat4x4 mat) const noexcept {
		mat4x4<float> result;
		for(auto y = 0; y < 4; ++y) {
			for(auto x = 0; x < 4; ++x){
				result.elm[y][x] = elm[y][0] * mat.elm[0][x]
					+ elm[y][1] * mat.elm[1][x]
					+ elm[y][2] * mat.elm[2][x]
					+ elm[y][3] * mat.elm[3][x];
			}
		}
		return result;
	}
#endif
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const mat4x4<T>& mat) {
	for(const auto& x : mat.elm){
		os << "| ";
		for(const auto e : x){
			os << std::scientific << std::setprecision(2) << e << " ";
		}
		os << " |\n";
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

	float tmp=0;
	mat4x4<float> mat = {{tmp,tmp,tmp,tmp},{tmp,tmp,tmp,tmp},{tmp,tmp,tmp,tmp},{tmp,tmp,tmp,tmp}};
	
	for(auto p = 0; p < 5; ++p){
		auto start = std::chrono::system_clock::now();
		// 448*10^6 = 448MFLOPS
		for(auto i=0; i < 1000000; ++i) {
			// 112 * 4 = 448FLOPS
			mat = mat*mat*mat*mat;
		}

		std::cout << std::fixed << std::setprecision(1) << 448 / static_cast<double>(
			std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count()
		) << " GFlops" << std::endl;
	}
	std::cout << mat << std::endl;
}