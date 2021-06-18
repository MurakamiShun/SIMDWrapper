#pragma once
#include "common.hpp"
#if defined(__x86_64__) || defined(_M_AMD64) || defined(_M_IX86)
#if defined(__GNUC__)
#include <x86intrin.h>
#include <cpuid.h>
#elif defined(_MSC_VER)
#include <intrin.h>
#endif

#include <vector>
#include <array>
#include <bitset>
#include <sstream>
#include <limits>

namespace SIMDWrapper {
	class instruction {
	public:
		static bool SSE4_1() noexcept { return CPU_ref.SSE4_1; }
		static bool SSE4_2() noexcept { return CPU_ref.SSE4_2; }
		static bool AVX2() noexcept { return CPU_ref.AVX2; }
		static bool AVX() noexcept { return CPU_ref.AVX; }
		static bool FMA() noexcept { return CPU_ref.FMA; }

		static bool SIMD128() noexcept { return CPU_ref.SSE4_2; }
		static bool SIMD256() noexcept { return CPU_ref.AVX2; }
	private:
		struct instruction_set {
			bool SSE4_1 = false;
			bool SSE4_2 = false;
			bool AVX2 = false;
			bool AVX = false;
			bool FMA = false;
			instruction_set() {
				std::vector<std::array<int, 4>> data;
				std::array<int, 4> cpui;
				#if defined(__GNUC__)
				__cpuid(0, cpui[0], cpui[1], cpui[2], cpui[3]);
				#elif defined(_MSC_VER)
				__cpuid(cpui.data(), 0);
				#endif
				
				const int ids = cpui[0];
				for (int i = 0; i < ids; ++i) {
					#if defined(__GNUC__)
					__cpuid_count(i, 0, cpui[0], cpui[1], cpui[2], cpui[3]);
					#elif defined(_MSC_VER)
					__cpuidex(cpui.data(), i, 0);
					#endif
					data.push_back(cpui);
				}
				std::bitset<32> f_1_ECX;
				if (ids >= 1) {
					f_1_ECX = data[1][2];
					SSE4_1 = f_1_ECX[19];
					SSE4_2 = f_1_ECX[20];
					AVX = f_1_ECX[28];
					FMA = f_1_ECX[12];
				}
				std::bitset<32> f_7_EBX;
				if (ids >= 7) {
					f_7_EBX = data[7][1];
					AVX2 = f_7_EBX[5];
				}
			}
		};
		static inline instruction_set CPU_ref;
	};
}

#endif