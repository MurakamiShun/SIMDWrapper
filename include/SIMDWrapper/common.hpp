#pragma once

#include <cstdint>
#include <utility>
#include <type_traits>

namespace SIMDWrapper {
	namespace print_format {
		namespace brancket {
			constexpr auto round = std::make_pair("(", ")");
			constexpr auto square = std::make_pair("[", "]");
			constexpr auto curly = std::make_pair("{", "}");
			constexpr auto pointy = std::make_pair("<", ">");
		}
		namespace delim {
			constexpr auto space = " ";
			constexpr auto comma = ",";
			constexpr auto comma_space = ", ";
			constexpr auto space_comma = " ,";
		}
	}

	template<typename Scalar>
	class vector128;

	namespace type {
		using i8x16_t = vector128<int8_t>;
		using i16x8_t = vector128<int16_t>;
		using i32x4_t = vector128<int32_t>;
		using i64x2_t = vector128<int64_t>;

		using u8x16_t = vector128<uint8_t>;
		using u16x8_t = vector128<uint16_t>;
		using u32x4_t = vector128<uint32_t>;
		using u64x2_t = vector128<uint64_t>;

		using fp32x4_t = vector128<float>;
		using fp64x2_t = vector128<double>;
	}

	template<typename Scalar>
	class vector256;

	namespace type {
		using i8x32_t = vector256<int8_t>;
		using i16x16_t = vector256<int16_t>;
		using i32x8_t = vector256<int32_t>;
		using i64x4_t = vector256<int64_t>;
		
		using u8x32_t = vector256<uint8_t>;
		using u16x16_t = vector256<uint16_t>;
		using u32x8_t = vector256<uint32_t>;
		using u64x4_t = vector256<uint64_t>;
		
		using fp32x8_t = vector256<float>;
		using fp64x4_t = vector256<double>;
	}
}