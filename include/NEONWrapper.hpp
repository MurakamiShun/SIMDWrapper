#pragma once
#ifdef __aarch64__
#if __cplusplus < 201703L
#error C++17 is required.
#else

#include <cstdint>
#include <type_traits>
#include <sstream>
#include <arm_neon.h>
#include <iostream>
#include <array>

#ifdef __linux__
#include <sys/auxv.h>
#include <asm/hwcap.h>
class instruction {
public:
	static bool NEON() noexcept { return CPU_ref.NEON; }
private:
	struct instruction_set {
		bool NEON = false;
		instruction_set() {
			auto hwcaps = getauxval(AT_HWCAP);
			NEON = hwcaps & HWCAP_ASIMD;
		}
	};
	static inline instruction_set CPU_ref;
};
#endif

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
struct vector128_type {
	template<typename T, typename... List>
	using is_any = std::disjunction<std::is_same<T, List>...>;
	
	template<typename T>
	static constexpr auto is_scalar_v = std::is_same<Scalar, T>::value;

	static_assert(is_any<Scalar, float, double>::value || std::is_integral_v<Scalar>, "NEON : Given type is not supported.");

	using scalar = Scalar;
	using vector = typename std::conditional_t< is_scalar_v<double>, float64x2_t,
		typename std::conditional_t< is_scalar_v<float>, float32x4_t,
		typename std::conditional_t< is_scalar_v<int64_t>, int64x2_t,
		typename std::conditional_t< is_scalar_v<uint64_t>, uint64x2_t,
		typename std::conditional_t< is_scalar_v<int32_t>, int32x4_t,
		typename std::conditional_t< is_scalar_v<uint32_t>, uint32x4_t,
		typename std::conditional_t< is_scalar_v<int16_t>, int16x8_t,
		typename std::conditional_t< is_scalar_v<uint16_t>, uint16x8_t,
		typename std::conditional_t< is_scalar_v<int8_t>, int8x16_t,
		typename std::conditional_t< is_scalar_v<uint8_t>, uint8x16_t,
		std::false_type>>>>>>>>>>;

	static constexpr size_t elements_size = 16 / sizeof(scalar);
};

template<typename Scalar>
class vector128 {
private:
	using scalar = typename vector128_type<Scalar>::scalar;
	using vector = typename vector128_type<Scalar>::vector;
	static constexpr size_t elements_size = vector128_type<Scalar>::elements_size;

	template<typename T>
	static constexpr bool is_scalar_v = std::is_same<scalar, T>::value;

	template<typename T>
	static constexpr bool is_scalar_size_v = (sizeof(scalar) == sizeof(T));

	template<typename T>
	static constexpr bool false_v = false;

	template<typename F, typename T, T... Seq>
	static constexpr auto sequence_map(std::integer_sequence<T, Seq...>, F f) {
		return std::integer_sequence<T, f(Seq)...>();
	}

	template<typename E, typename T, T... Seq, size_t size = sizeof...(Seq)>
	static auto expand_sequence(std::integer_sequence<T, Seq...>) {
		return E{Seq...};
	}

	class input_iterator {
	private:
		alignas(16) std::array<scalar, elements_size> tmp = {};
		size_t index;
	public:
		template<size_t N>
		struct Index {};

		input_iterator(const input_iterator& it) noexcept :
			index(it.index),
			tmp(it.tmp) {
		}
		template<size_t N>
		input_iterator(const vector128& arg, Index<N>) noexcept {
			index = N;
			if constexpr (N >= 0 && N < elements_size)
				arg.aligned_store(tmp.data());
		}
		const scalar operator*() const noexcept {
			return tmp[index];
		}
		input_iterator& operator++() noexcept {
			++index;
			return *this;
		}
		bool operator==(const input_iterator& it) const noexcept {
			return (index == it.index);
		}
		bool operator!=(const input_iterator& it) const noexcept {
			return (index != it.index);
		}
	};
public:
	vector v;

	vector128() noexcept {}
	vector128(const scalar arg) noexcept { *this = arg; }
	vector128(const vector arg) noexcept : v(arg) {  }
	vector128(const vector128& arg) noexcept : v(arg.v) {  }
	template<class... Args>
	vector128(const scalar first, const Args... args) noexcept {
		alignas(16) scalar tmp[elements_size] = { first, static_cast<scalar>(args)... };
		aligned_load(tmp);
	}

	input_iterator begin() const noexcept {
		return input_iterator(*this, typename input_iterator::template Index<0>());
	}
	input_iterator end() const noexcept {
		return input_iterator(*this, typename input_iterator::template Index<elements_size>());
	}

	vector128 operator+(const vector128& arg) const noexcept {
		if constexpr(is_scalar_v<double>) return vector128(vaddq_f64(v, arg.v));
		else if constexpr(is_scalar_v<float>) return vector128(vaddq_f32(v, arg.v));
		else if constexpr(is_scalar_v<int64_t>) return vector128(vaddq_s64(v, arg.v));
		else if constexpr(is_scalar_v<uint64_t>) return vector128(vaddq_u64(v, arg.v));
		else if constexpr(is_scalar_v<int32_t>) return vector128(vaddq_s32(v, arg.v));
		else if constexpr(is_scalar_v<uint32_t>) return vector128(vaddq_u32(v, arg.v));
		else if constexpr(is_scalar_v<int16_t>) return vector128(vaddq_s16(v, arg.v));
		else if constexpr(is_scalar_v<uint16_t>) return vector128(vaddq_u16(v, arg.v));
		else if constexpr(is_scalar_v<int8_t>) return vector128(vaddq_s8(v, arg.v));
		else if constexpr(is_scalar_v<uint8_t>) return vector128(vaddq_u8(v, arg.v));
		else static_assert(false_v<scalar>, "NEON : operator+ is not defined in given type.");
	}

	vector128 operator-(const vector128& arg) const noexcept {
		if constexpr(is_scalar_v<double>) return vector128(vsubq_f64(v, arg.v));
		else if constexpr(is_scalar_v<float>) return vector128(vsubq_f32(v, arg.v));
		else if constexpr(is_scalar_v<int64_t>) return vector128(vsubq_s64(v, arg.v));
		else if constexpr(is_scalar_v<uint64_t>) return vector128(vsubq_u64(v, arg.v));
		else if constexpr(is_scalar_v<int32_t>) return vector128(vsubq_s32(v, arg.v));
		else if constexpr(is_scalar_v<uint32_t>) return vector128(vsubq_u32(v, arg.v));
		else if constexpr(is_scalar_v<int16_t>) return vector128(vsubq_s16(v, arg.v));
		else if constexpr(is_scalar_v<uint16_t>) return vector128(vsubq_u16(v, arg.v));
		else if constexpr(is_scalar_v<int8_t>) return vector128(vsubq_s8(v, arg.v));
		else if constexpr(is_scalar_v<uint8_t>) return vector128(vsubq_u8(v, arg.v));
		else static_assert(false_v<scalar>, "NEON : operator- is not defined in given type.");
	}

	vector128 operator*(const vector128& arg) const noexcept {
		if constexpr(is_scalar_v<double>) return vector128(vmulq_f64(v, arg.v));
		else if constexpr(is_scalar_v<float>) return vector128(vmulq_f32(v, arg.v));
		else if constexpr(is_scalar_v<int64_t>) return vector128(vmulq_s64(v, arg.v));
		else if constexpr(is_scalar_v<uint64_t>) return vector128(vmulq_u64(v, arg.v));
		else if constexpr(is_scalar_v<int32_t>) return vector128(vmulq_s32(v, arg.v));
		else if constexpr(is_scalar_v<uint32_t>) return vector128(vmulq_u32(v, arg.v));
		else if constexpr(is_scalar_v<int16_t>) return vector128(vmulq_s16(v, arg.v));
		else if constexpr(is_scalar_v<uint16_t>) return vector128(vmulq_u16(v, arg.v));
		else if constexpr(is_scalar_v<int8_t>) return vector128(vmulq_s8(v, arg.v));
		else if constexpr(is_scalar_v<uint8_t>) return vector128(vmulq_u8(v, arg.v));
		else static_assert(false_v<scalar>, "NEON : operator* is not defined in given type.");
	}

	vector128 operator/(const vector128& arg) const noexcept {
		if constexpr(is_scalar_v<double>) return vector128(vdivq_f64(v, arg.v));
		else if constexpr(is_scalar_v<float>) return vector128(vdivq_f32(v, arg.v));
		else static_assert(false_v<scalar>, "NEON : operator/ is not defined in given type.");
	}

	vector128& operator=(const scalar a) noexcept {
		alignas(16) scalar arg = a;
		if constexpr (is_scalar_v<double>) v = vld1q_dup_f64(&arg);
		else if constexpr(is_scalar_v<float>) v = vld1q_dup_f32(&arg);
		else if constexpr(is_scalar_v<int64_t>) v = vld1q_dup_s64(&arg);
		else if constexpr(is_scalar_v<uint64_t>) v = vld1q_dup_u64(&arg);
		else if constexpr(is_scalar_v<int32_t>) v = vld1q_dup_s32(&arg);
		else if constexpr(is_scalar_v<uint32_t>) v = vld1q_dup_u32(&arg);
		else if constexpr(is_scalar_v<int16_t>) v = vld1q_dup_s16(&arg);
		else if constexpr(is_scalar_v<uint16_t>) v = vld1q_dup_u16(&arg);
		else if constexpr(is_scalar_v<int8_t>) v = vld1q_dup_s8(&arg);
		else if constexpr(is_scalar_v<uint8_t>) v = vld1q_dup_u8(&arg);
		else static_assert(false_v<scalar>, "NEON : operator=(scalar) is not defined in given type.");
		return *this;
	}

	void aligned_load(const scalar* const arg) noexcept {
		if constexpr (is_scalar_v<double>) v = vld1q_f64(arg);
		else if constexpr(is_scalar_v<float>) v = vld1q_f32(arg);
		else if constexpr(is_scalar_v<int64_t>) v = vld1q_s64(arg);
		else if constexpr(is_scalar_v<uint64_t>) v = vld1q_u64(arg);
		else if constexpr(is_scalar_v<int32_t>) v = vld1q_s32(arg);
		else if constexpr(is_scalar_v<uint32_t>) v = vld1q_u32(arg);
		else if constexpr(is_scalar_v<int16_t>) v = vld1q_s16(arg);
		else if constexpr(is_scalar_v<uint16_t>) v = vld1q_u16(arg);
		else if constexpr(is_scalar_v<int8_t>) v = vld1q_s8(arg);
		else if constexpr(is_scalar_v<uint8_t>) v = vld1q_u8(arg);
		else static_assert(false_v<scalar>, "NEON : aligned load is not defined in given type.");
	}

	void aligned_store(scalar* arg) const noexcept {
		if constexpr (is_scalar_v<double>) vst1q_f64(arg, v);
		else if constexpr(is_scalar_v<float>) vst1q_f32(arg, v);
		else if constexpr(is_scalar_v<int64_t>) vst1q_s64(arg, v);
		else if constexpr(is_scalar_v<uint64_t>) vst1q_u64(arg, v);
		else if constexpr(is_scalar_v<int32_t>) vst1q_s32(arg, v);
		else if constexpr(is_scalar_v<uint32_t>) vst1q_u32(arg, v);
		else if constexpr(is_scalar_v<int16_t>) vst1q_s16(arg, v);
		else if constexpr(is_scalar_v<uint16_t>) vst1q_u16(arg, v);
		else if constexpr(is_scalar_v<int8_t>) vst1q_s8(arg, v);
		else if constexpr(is_scalar_v<uint8_t>) vst1q_u8(arg, v);
		else static_assert(false_v<scalar>, "NEON : aligned store is not defined in given type.");
	}

	scalar operator[](const size_t index) const {
		return reinterpret_cast<const scalar*>(&v)[index];
	}
	scalar& operator[](const size_t index) {
		return reinterpret_cast<scalar*>(&v)[index];
	}

	auto operator==(const vector128& arg) const noexcept {
		if constexpr (is_scalar_v<double>) return vector128<uint64_t>(vceqq_f64(v, arg.v));
		else if constexpr(is_scalar_v<float>) return vector128<uint32_t>(vceqq_f32(v, arg.v));
		else if constexpr(is_scalar_v<int64_t>) return vector128<uint64_t>(vceqq_s64(v, arg.v));
		else if constexpr(is_scalar_v<uint64_t>) return vector128<uint64_t>(vceqq_u64(v, arg.v));
		else if constexpr(is_scalar_v<int32_t>) return vector128<uint32_t>(vceqq_s32(v, arg.v));
		else if constexpr(is_scalar_v<uint32_t>) return vector128<uint32_t>(vceqq_u32(v, arg.v));
		else if constexpr(is_scalar_v<int16_t>) return vector128<uint16_t>(vceqq_s16(v, arg.v));
		else if constexpr(is_scalar_v<uint16_t>) return vector128<uint16_t>(vceqq_u16(v, arg.v));
		else if constexpr(is_scalar_v<int8_t>) return vector128<uint8_t>(vceqq_s8(v, arg.v));
		else if constexpr(is_scalar_v<uint8_t>) return vector128<uint8_t>(vceqq_u8(v, arg.v));
		else static_assert(false_v<scalar>, "NEON : operator== is not defined in given type.");
	}

	auto operator!=(const vector128& arg) const noexcept {
		if constexpr (is_scalar_v<double>) return vector128<uint64_t>(vmvnq_u64(vceqq_f64(v, arg.v)));
		else if constexpr(is_scalar_v<float>) return vector128<uint32_t>(vmvnq_u32(vceqq_f32(v, arg.v)));
		else if constexpr(is_scalar_v<int64_t>) return vector128<uint64_t>(vmvnq_u64(vceqq_s64(v, arg.v)));
		else if constexpr(is_scalar_v<uint64_t>) return vector128<uint64_t>(vmvnq_u64(vceqq_u64(v, arg.v)));
		else if constexpr(is_scalar_v<int32_t>) return vector128<uint32_t>(vmvnq_u32(vceqq_s32(v, arg.v)));
		else if constexpr(is_scalar_v<uint32_t>) return vector128<uint32_t>(vmvnq_u32(vceqq_u32(v, arg.v)));
		else if constexpr(is_scalar_v<int16_t>) return vector128<uint16_t>(vmvnq_u16(vceqq_s16(v, arg.v)));
		else if constexpr(is_scalar_v<uint16_t>) return vector128<uint16_t>(vmvnq_u16(vceqq_u16(v, arg.v)));
		else if constexpr(is_scalar_v<int8_t>) return vector128<uint8_t>(vmvnq_u8(vceqq_s8(v, arg.v)));
		else if constexpr(is_scalar_v<uint8_t>) return vector128<uint8_t>(vmvnq_u8(vceqq_u8(v, arg.v)));
		else static_assert(false_v<scalar>, "NEON : operator!= is not defined in given type.");
	}

	auto operator>(const vector128& arg) const noexcept {
		if constexpr (is_scalar_v<double>) return vector128<uint64_t>(vcgtq_f64(v, arg.v));
		else if constexpr(is_scalar_v<float>) return vector128<uint32_t>(vcgtq_f32(v, arg.v));
		else if constexpr(is_scalar_v<int64_t>) return vector128<uint64_t>(vcgtq_s64(v, arg.v));
		else if constexpr(is_scalar_v<uint64_t>) return vector128<uint64_t>(vcgtq_u64(v, arg.v));
		else if constexpr(is_scalar_v<int32_t>) return vector128<uint32_t>(vcgtq_s32(v, arg.v));
		else if constexpr(is_scalar_v<uint32_t>) return vector128<uint32_t>(vcgtq_u32(v, arg.v));
		else if constexpr(is_scalar_v<int16_t>) return vector128<uint16_t>(vcgtq_s16(v, arg.v));
		else if constexpr(is_scalar_v<uint16_t>) return vector128<uint16_t>(vcgtq_u16(v, arg.v));
		else if constexpr(is_scalar_v<int8_t>) return vector128<uint8_t>(vcgtq_s8(v, arg.v));
		else if constexpr(is_scalar_v<uint8_t>) return vector128<uint8_t>(vcgtq_u8(v, arg.v));
		else static_assert(false_v<scalar>, "NEON : operator> is not defined in given type.");
	}
	
	auto operator<(const vector128& arg) const noexcept {
		if constexpr (is_scalar_v<double>) return vector128<uint64_t>(vcltq_f64(v, arg.v));
		else if constexpr(is_scalar_v<float>) return vector128<uint32_t>(vcltq_f32(v, arg.v));
		else if constexpr(is_scalar_v<int64_t>) return vector128<uint64_t>(vcltq_s64(v, arg.v));
		else if constexpr(is_scalar_v<uint64_t>) return vector128<uint64_t>(vcltq_u64(v, arg.v));
		else if constexpr(is_scalar_v<int32_t>) return vector128<uint32_t>(vcltq_s32(v, arg.v));
		else if constexpr(is_scalar_v<uint32_t>) return vector128<uint32_t>(vcltq_u32(v, arg.v));
		else if constexpr(is_scalar_v<int16_t>) return vector128<uint16_t>(vcltq_s16(v, arg.v));
		else if constexpr(is_scalar_v<uint16_t>) return vector128<uint16_t>(vcltq_u16(v, arg.v));
		else if constexpr(is_scalar_v<int8_t>) return vector128<uint8_t>(vcltq_s8(v, arg.v));
		else if constexpr(is_scalar_v<uint8_t>) return vector128<uint8_t>(vcltq_u8(v, arg.v));
		else static_assert(false_v<scalar>, "NEON : operator< is not defined in given type.");
	}

	auto operator>=(const vector128& arg) const noexcept {
		if constexpr (is_scalar_v<double>) return vector128<uint64_t>(vcgeq_f64(v, arg.v));
		else if constexpr(is_scalar_v<float>) return vector128<uint32_t>(vcgeq_f32(v, arg.v));
		else if constexpr(is_scalar_v<int64_t>) return vector128<uint64_t>(vcgeq_s64(v, arg.v));
		else if constexpr(is_scalar_v<uint64_t>) return vector128<uint64_t>(vcgeq_u64(v, arg.v));
		else if constexpr(is_scalar_v<int32_t>) return vector128<uint32_t>(vcgeq_s32(v, arg.v));
		else if constexpr(is_scalar_v<uint32_t>) return vector128<uint32_t>(vcgeq_u32(v, arg.v));
		else if constexpr(is_scalar_v<int16_t>) return vector128<uint16_t>(vcgeq_s16(v, arg.v));
		else if constexpr(is_scalar_v<uint16_t>) return vector128<uint16_t>(vcgeq_u16(v, arg.v));
		else if constexpr(is_scalar_v<int8_t>) return vector128<uint8_t>(vcgeq_s8(v, arg.v));
		else if constexpr(is_scalar_v<uint8_t>) return vector128<uint8_t>(vcgeq_u8(v, arg.v));
		else static_assert(false_v<scalar>, "NEON : operator>= is not defined in given type.");
	}
	
	auto operator<=(const vector128& arg) const noexcept {
		if constexpr (is_scalar_v<double>) return vector128<uint64_t>(vcleq_f64(v, arg.v));
		else if constexpr(is_scalar_v<float>) return vector128<uint32_t>(vcleq_f32(v, arg.v));
		else if constexpr(is_scalar_v<int64_t>) return vector128<uint64_t>(vcleq_s64(v, arg.v));
		else if constexpr(is_scalar_v<uint64_t>) return vector128<uint64_t>(vcleq_u64(v, arg.v));
		else if constexpr(is_scalar_v<int32_t>) return vector128<uint32_t>(vcleq_s32(v, arg.v));
		else if constexpr(is_scalar_v<uint32_t>) return vector128<uint32_t>(vcleq_u32(v, arg.v));
		else if constexpr(is_scalar_v<int16_t>) return vector128<uint16_t>(vcleq_s16(v, arg.v));
		else if constexpr(is_scalar_v<uint16_t>) return vector128<uint16_t>(vcleq_u16(v, arg.v));
		else if constexpr(is_scalar_v<int8_t>) return vector128<uint8_t>(vcleq_s8(v, arg.v));
		else if constexpr(is_scalar_v<uint8_t>) return vector128<uint8_t>(vcleq_u8(v, arg.v));
		else static_assert(false_v<scalar>, "NEON : operator<= is not defined in given type.");
	}

	vector128 operator& (const vector128& arg) const noexcept {
		if constexpr(is_scalar_v<int64_t>) return vector128(vandq_s64(v, arg.v));
		else if constexpr(is_scalar_v<uint64_t>) return vector128(vandq_u64(v, arg.v));
		else if constexpr(is_scalar_v<int32_t>) return vector128(vandq_s32(v, arg.v));
		else if constexpr(is_scalar_v<uint32_t>) return vector128(vandq_u32(v, arg.v));
		else if constexpr(is_scalar_v<int16_t>) return vector128(vandq_s16(v, arg.v));
		else if constexpr(is_scalar_v<uint16_t>) return vector128(vandq_u16(v, arg.v));
		else if constexpr(is_scalar_v<int8_t>) return vector128(vandq_s8(v, arg.v));
		else if constexpr(is_scalar_v<uint8_t>) return vector128(vandq_u8(v, arg.v));
		else static_assert(false_v<scalar>, "NEON : operator& is not defined in given type.");
	}
	
	vector128 operator| (const vector128& arg) const noexcept {
		if constexpr(is_scalar_v<int64_t>) return vector128(vorrq_s64(v, arg.v));
		else if constexpr(is_scalar_v<uint64_t>) return vector128(vorrq_u64(v, arg.v));
		else if constexpr(is_scalar_v<int32_t>) return vector128(vorrq_s32(v, arg.v));
		else if constexpr(is_scalar_v<uint32_t>) return vector128(vorrq_u32(v, arg.v));
		else if constexpr(is_scalar_v<int16_t>) return vector128(vorrq_s16(v, arg.v));
		else if constexpr(is_scalar_v<uint16_t>) return vector128(vorrq_u16(v, arg.v));
		else if constexpr(is_scalar_v<int8_t>) return vector128(vorrq_s8(v, arg.v));
		else if constexpr(is_scalar_v<uint8_t>) return vector128(vorrq_u8(v, arg.v));
		else static_assert(false_v<scalar>, "NEON : operator| is not defined in given type.");
	}

	vector128 operator^ (const vector128& arg) const noexcept {
		if constexpr(is_scalar_v<int64_t>) return vector128(veorq_s64(v, arg.v));
		else if constexpr(is_scalar_v<uint64_t>) return vector128(veorq_u64(v, arg.v));
		else if constexpr(is_scalar_v<int32_t>) return vector128(veorq_s32(v, arg.v));
		else if constexpr(is_scalar_v<uint32_t>) return vector128(veorq_u32(v, arg.v));
		else if constexpr(is_scalar_v<int16_t>) return vector128(veorq_s16(v, arg.v));
		else if constexpr(is_scalar_v<uint16_t>) return vector128(veorq_u16(v, arg.v));
		else if constexpr(is_scalar_v<int8_t>) return vector128(veorq_s8(v, arg.v));
		else if constexpr(is_scalar_v<uint8_t>) return vector128(veorq_u8(v, arg.v));
		else static_assert(false_v<scalar>, "NEON : operator^ is not defined in given type.");
	}
	
	vector128 operator~ () const noexcept {
		if constexpr(is_scalar_v<int64_t>) return vector128(vmvniq_s64(v));
		else if constexpr(is_scalar_v<uint64_t>) return vector128(vmvnq_u64(v));
		else if constexpr(is_scalar_v<int32_t>) return vector128(vmvnq_s32(v));
		else if constexpr(is_scalar_v<uint32_t>) return vector128(vmvnq_u32(v));
		else if constexpr(is_scalar_v<int16_t>) return vector128(vmvnq_s16(v));
		else if constexpr(is_scalar_v<uint16_t>) return vector128(vmvnq_u16(v));
		else if constexpr(is_scalar_v<int8_t>) return vector128(vmvnq_s8(v));
		else if constexpr(is_scalar_v<uint8_t>) return vector128(vmvnq_u8(v));
		else static_assert(false_v<scalar>, "NEON : operator~ is not defined in given type.");
	}

	vector128 operator<<(const vector128& arg) const noexcept {
		if constexpr(is_scalar_v<int64_t>) return vector128(vshlq_s64(v, arg.v));
		else if constexpr(is_scalar_v<uint64_t>) return vector128(vshlq_u64(v, vreinterpretq_s64_u64(arg.v)));
		else if constexpr(is_scalar_v<int32_t>) return vector128(vshlq_s32(v, arg.v));
		else if constexpr(is_scalar_v<uint32_t>) return vector128(vshlq_u32(v, vreinterpretq_s32_u32(arg.v)));
		else if constexpr(is_scalar_v<int16_t>) return vector128(vshlq_s16(v, arg.v));
		else if constexpr(is_scalar_v<uint16_t>) return vector128(vshlq_u16(v, vreinterpretq_s16_u16(arg.v)));
		else if constexpr(is_scalar_v<int8_t>) return vector128(vshlq_s8(v, arg.v));
		else if constexpr(is_scalar_v<uint8_t>) return vector128(vshlq_u8(v, vreinterpretq_s8_u8(arg.v)));
		else static_assert(false_v<scalar>, "NEON : operator<< is not defined in given type.");
	}
	
	vector128 operator<<(const int arg) const noexcept {
	#ifndef __clang__
		if constexpr(is_scalar_v<int64_t>) return vector128(vshlq_n_s64(v, arg));
		else if constexpr(is_scalar_v<uint64_t>) return vector128(vshlq_n_u64(v, arg));
		else if constexpr(is_scalar_v<int32_t>) return vector128(vshlq_n_s32(v, arg));
		else if constexpr(is_scalar_v<uint32_t>) return vector128(vshlq_n_u32(v, arg));
		else if constexpr(is_scalar_v<int16_t>) return vector128(vshlq_n_s16(v, arg));
		else if constexpr(is_scalar_v<uint16_t>) return vector128(vshlq_n_u16(v, arg));
		else if constexpr(is_scalar_v<int8_t>) return vector128(vshlq_n_s8(v, arg));
		else if constexpr(is_scalar_v<uint8_t>) return vector128(vshlq_n_u8(v, arg));
		else static_assert(false_v<scalar>, "NEON : operator<< is not defined in given type.");
	#else
		alignas(16) scalar a = arg;
		if constexpr(is_scalar_v<int64_t>) return vector128(vshlq_n_s64(v, vld1q_dup_s64(&a)));
		else if constexpr(is_scalar_v<uint64_t>) return vector128(vshlq_n_u64(v, vld1q_dup_u64(&a)));
		else if constexpr(is_scalar_v<int32_t>) return vector128(vshlq_n_s32(v, vld1q_dup_s32(&a)));
		else if constexpr(is_scalar_v<uint32_t>) return vector128(vshlq_n_u32(v, vld1q_dup_u32(&a)));
		else if constexpr(is_scalar_v<int16_t>) return vector128(vshlq_n_s16(v, vld1q_dup_s16(&a)));
		else if constexpr(is_scalar_v<uint16_t>) return vector128(vshlq_n_u16(v, vld1q_dup_u16(&a)));
		else if constexpr(is_scalar_v<int8_t>) return vector128(vshlq_n_s8(v, vld1q_dup_s8(&a)));
		else if constexpr(is_scalar_v<uint8_t>) return vector128(vshlq_n_u8(v, vld1q_dup_u8(&a)));
		else static_assert(false_v<scalar>, "NEON : operator<< is not defined in given type.");
	#endif
	}
	vector128 operator>>(const vector128& arg) const noexcept {
		if constexpr(is_scalar_v<int64_t>) return vector128(vshlq_s64(v, vnegq_s64(arg.v)));
		else if constexpr(is_scalar_v<uint64_t>) return vector128(vshlq_u64(v, vnegq_s64(vreinterpretq_s64_u64(arg.v))));
		else if constexpr(is_scalar_v<int32_t>) return vector128(vshlq_s32(v, vnegq_s32(arg.v)));
		else if constexpr(is_scalar_v<uint32_t>) return vector128(vshlq_u32(v, vnegq_s32(vreinterpretq_s32_u32(arg.v))));
		else if constexpr(is_scalar_v<int16_t>) return vector128(vshlq_s16(v, vnegq_s16(arg.v)));
		else if constexpr(is_scalar_v<uint16_t>) return vector128(vshlq_u16(v, vnegq_s16(vreinterpretq_s32_u32(arg.v))));
		else if constexpr(is_scalar_v<int8_t>) return vector128(vshlq_s8(v, vnegq_s8(arg.v)));
		else if constexpr(is_scalar_v<uint8_t>) return vector128(vshlq_u8(v, vnegq_s8(vreinterpretq_s32_u32(arg.v))));
		else static_assert(false_v<scalar>, "NEON : operator>> is not defined in given type.");
	}
	vector128 operator>>(const int arg) const noexcept {
	#ifndef __clang__
		if constexpr(is_scalar_v<int64_t>) return vector128(vshrq_n_s64(v, arg));
		else if constexpr(is_scalar_v<uint64_t>) return vector128(vshrq_n_u64(v, arg));
		else if constexpr(is_scalar_v<int32_t>) return vector128(vshrq_n_s32(v, arg));
		else if constexpr(is_scalar_v<uint32_t>) return vector128(vshrq_n_u32(v, arg));
		else if constexpr(is_scalar_v<int16_t>) return vector128(vshrq_n_s16(v, arg));
		else if constexpr(is_scalar_v<uint16_t>) return vector128(vshrq_n_u16(v, arg));
		else if constexpr(is_scalar_v<int8_t>) return vector128(vshrq_n_s8(v, arg));
		else if constexpr(is_scalar_v<uint8_t>) return vector128(vshrq_n_u8(v, arg));
		else static_assert(false_v<scalar>, "NEON : operator>> is not defined in given type.");
	#else
		alignas(16) scalar a = arg;
		if constexpr(is_scalar_v<int64_t>) return vector128(vshrq_n_s64(v, vld1q_dup_s64(&a)));
		else if constexpr(is_scalar_v<uint64_t>) return vector128(vshrq_n_u64(v, vld1q_dup_u64(&a)));
		else if constexpr(is_scalar_v<int32_t>) return vector128(vshrq_n_s32(v, vld1q_dup_s32(&a)));
		else if constexpr(is_scalar_v<uint32_t>) return vector128(vshrq_n_u32(v, vld1q_dup_u32(&a)));
		else if constexpr(is_scalar_v<int16_t>) return vector128(vshrq_n_s16(v, vld1q_dup_s16(&a)));
		else if constexpr(is_scalar_v<uint16_t>) return vector128(vshrq_n_u16(v, vld1q_dup_u16(&a)));
		else if constexpr(is_scalar_v<int8_t>) return vector128(vshrq_n_s8(v, vld1q_dup_s8(&a)));
		else if constexpr(is_scalar_v<uint8_t>) return vector128(vshrq_n_u8(v, vld1q_dup_u8(&a)));
		else static_assert(false_v<scalar>, "NEON : operator>> is not defined in given type.");
	#endif
	}
	vector128 rcp() const noexcept {
		if constexpr (is_scalar_v<double>) return vector128(vrecpeq_f64(v));
		else if constexpr(is_scalar_v<float>) return vector128(vrecpeq_f32(v));
		else static_assert(false_v<scalar>, "NEON : rcp is not defined in given type.");
	}

	vector128 sqrt() const noexcept {
		if constexpr (is_scalar_v<double>) return vector128(vsqrtq_f64(v));
		else if constexpr(is_scalar_v<float>) return vector128(vsqrtq_f32(v));
		else static_assert(false_v<scalar>, "NEON : sqrt is not defined in given type.");
	}

	vector128 rsqrt() const noexcept {
		if constexpr (is_scalar_v<double>) return vector128(vrsqrteq_f64(v));
		else if constexpr(is_scalar_v<float>) return vector128(vrsqrteq_f32(v));
		else static_assert(false_v<scalar>, "NEON : rsqrt is not defined in given type.");
	}

	vector128 abs() const noexcept {
		if constexpr (is_scalar_v<double>) return vector128(vabsq_f64(v));
		else if constexpr(is_scalar_v<float>) return vector128(vabsq_f32(v));
		else if constexpr(is_scalar_v<int64_t>) return vector128(vqabsq_s64(v));
		else if constexpr(is_scalar_v<int32_t>) return vector128(vqabsq_n_s32(v));
		else if constexpr(is_scalar_v<int16_t>) return vector128(vqabsq_n_s16(v));
		else if constexpr(is_scalar_v<int8_t>) return vector128(vqabsq_n_s8(v));
		else static_assert(false_v<scalar>, "NEON : abs is not defined in given type.");
	}
	vector128 ceil() const noexcept {
		if constexpr (is_scalar_v<double>) return vector128(vrndpq_f32(v));
		else if constexpr (is_scalar_v<float>) return vector128(vrndpq_f64(v));
		else static_assert(false_v<Scalar>, "NEON : ceil is not defined in given type.");
	}
	vector128 floor() const noexcept {
		if constexpr (is_scalar_v<double>) return vector128(vrndmq_f32(v));
		else if constexpr (is_scalar_v<float>) return vector128(vrndmq_f64(v));
		else static_assert(false_v<scalar>, "NEON : floor is not defined in given type.");
	}
	vector128 round() const noexcept {
		if constexpr (is_scalar_v<double>) return vector128(vrndaq_f32(v));
		else if constexpr (is_scalar_v<float>) return vector128(vrndaq_f64(v));
		else static_assert(false_v<scalar>, "NEON : round is not defined in given type.");
	}
	// this + a * b 
	vector128 addmul(const vector128& a, const vector128& b) const noexcept {
		if constexpr (is_scalar_v<double>) return vector128(vfmaq_f64(v, a.v, b.v));
		else if constexpr (is_scalar_v<float>) return vector128(vfmaq_f32(v, a.v, b.v));
		else static_assert(false_v<scalar>, "NEON : addmul is not defined in given type.");
	}
	// this - a * b 
	vector128 submul(const vector128& a, const vector128& b) const noexcept {
		if constexpr (is_scalar_v<double>) return vector128(vfmsq_f64(v, a.v, b.v));
		else if constexpr (is_scalar_v<float>) return vector128(vfmsq_f32(v, a.v, b.v));
		else static_assert(false_v<scalar>, "NEON : submul is not defined in given type.");
	}
	// this * a + b
	vector128 muladd(const vector128& a, const vector128& b) const noexcept {
		if constexpr (is_scalar_v<double>) return vector128(vfmaq_f64(b.v, v, a.v));
		else if constexpr (is_scalar_v<float>) return vector128(vfmaq_f32(b.v, v, a.v));
		else static_assert(false_v<scalar>, "NEON : muladd is not defined in given type.");
	}
	// this* a -b
	vector128 mulsub(const vector128& a, const vector128& b) const noexcept {
		if constexpr (is_scalar_v<double>) return vector128(vfmaq_f64(vnegq_f64(b.v), v, a.v));
		else if constexpr (is_scalar_v<float>) return vector128(vfmaq_f32(vnegq_f32(b.v), v, a.v));
		else static_assert(false_v<scalar>, "NEON : mulsub is not defined in given type.");
	}
	// -(this * a) + b
	vector128 nmuladd(const vector128& a, const vector128& b) const noexcept {
		if constexpr (is_scalar_v<double>) return vector128(vfmsq_f64(b.v, v, a.v));
		else if constexpr (is_scalar_v<float>) return vector128(vfmsq_f32(b.v, v, a.v));
		else static_assert(false_v<scalar>, "NEON : nmuladd is not defined in given type.");
	}
	// -(this* a) -b
	vector128 nmulsub(const vector128& a, const vector128& b) const noexcept {
		if constexpr (is_scalar_v<double>) return vector128(vnegq_f64(vfmaq_f64(b.v, v, a.v)));
		else if constexpr (is_scalar_v<float>) return vector128(vnegq_f32(vfmaq_f32(b.v, v, a.v)));
		else static_assert(false_v<scalar>, "NEON : nmulsub is not defined in given type.");
	}

	vector128 max(const vector128& arg) const noexcept {
		if constexpr (is_scalar_v<double>) return vector128(vmaxq_f64(v, arg.v));
		else if constexpr(is_scalar_v<float>) return vector128(vmaxq_f32(v, arg.v));
		else if constexpr(is_scalar_v<int64_t>) return vector128(vmaxq_s64(v, arg.v));
		else if constexpr(is_scalar_v<uint64_t>) return vector128(vmaxq_u64(v, arg.v));
		else if constexpr(is_scalar_v<int32_t>) return vector128(vmaxq_s32(v, arg.v));
		else if constexpr(is_scalar_v<uint32_t>) return vector128(vmaxq_u32(v, arg.v));
		else if constexpr(is_scalar_v<int16_t>) return vector128(vmaxq_s16(v, arg.v));
		else if constexpr(is_scalar_v<uint16_t>) return vector128(vmaxq_u16(v, arg.v));
		else if constexpr(is_scalar_v<int8_t>) return vector128(vmaxq_s8(v, arg.v));
		else if constexpr(is_scalar_v<uint8_t>) return vector128(vmaxq_u8(v, arg.v));
		else static_assert(false_v<scalar>, "NEON : max is not defined in given type.");
	}

	vector128 min(const vector128& arg) const noexcept {
		if constexpr (is_scalar_v<double>) return vector128(vminq_f64(v, arg.v));
		else if constexpr(is_scalar_v<float>) return vector128(vminq_f32(v, arg.v));
		else if constexpr(is_scalar_v<int64_t>) return vector128(vminq_s64(v, arg.v));
		else if constexpr(is_scalar_v<uint64_t>) return vector128(vminq_u64(v, arg.v));
		else if constexpr(is_scalar_v<int32_t>) return vector128(vminq_s32(v, arg.v));
		else if constexpr(is_scalar_v<uint32_t>) return vector128(vminq_u32(v, arg.v));
		else if constexpr(is_scalar_v<int16_t>) return vector128(vminq_s16(v, arg.v));
		else if constexpr(is_scalar_v<uint16_t>) return vector128(vminq_u16(v, arg.v));
		else if constexpr(is_scalar_v<int8_t>) return vector128(vminq_s8(v, arg.v));
		else if constexpr(is_scalar_v<uint8_t>) return vector128(vminq_u8(v, arg.v));
		else static_assert(false_v<scalar>, "NEON : min is not defined in given type.");
	}

	scalar sum() const noexcept {
		if constexpr (is_scalar_v<double>) return vector128(vaddvq_f64(v));
		else if constexpr(is_scalar_v<float>) return vector128(vaddvq_f32(v));
		else if constexpr(is_scalar_v<int64_t>) return vector128(vaddvq_s64(v));
		else if constexpr(is_scalar_v<uint64_t>) return vector128(vaddvq_u64(v));
		else if constexpr(is_scalar_v<int32_t>) return vector128(vaddvq_s32(v));
		else if constexpr(is_scalar_v<uint32_t>) return vector128(vaddvq_u32(v));
		else if constexpr(is_scalar_v<int16_t>) return vector128(vaddvq_s16(v));
		else if constexpr(is_scalar_v<uint16_t>) return vector128(vaddvq_u16(v));
		else if constexpr(is_scalar_v<int8_t>) return vector128(vaddvq_s8(v));
		else if constexpr(is_scalar_v<uint8_t>) return vector128(vaddvq_u8(v));
		else static_assert(false_v<scalar>, "NEON : sum is not defined in given type.");
	}
	// duplicate a lane
	vector128 dup(const size_t idx) const noexcept {
	#ifndef __clang__
		if constexpr (is_scalar_v<double>) return vector128(vdupq_laneq_f64(v, idx));
		else if constexpr (is_scalar_v<float>) return vector128(vdupq_laneq_f32(v, idx));
		else if constexpr (is_scalar_v<int64_t>) return vector128(vdupq_laneq_s64(v, idx));
		else if constexpr (is_scalar_v<uint64_t>) return vector128(vdupq_laneq_u64(v, idx));
		else if constexpr (is_scalar_v<int32_t>) return vector128(vdupq_laneq_s32(v, idx));
		else if constexpr (is_scalar_v<uint32_t>) return vector128(vdupq_laneq_u32(v, idx));
		else if constexpr (is_scalar_v<int16_t>) return vector128(vdupq_laneq_s16(v, idx));
		else if constexpr (is_scalar_v<uint16_t>) return vector128(vdupq_laneq_u16(v, idx));
		else if constexpr (is_scalar_v<int8_t>) return vector128(vdupq_laneq_s8(v, idx));
		else if constexpr (is_scalar_v<uint8_t>) return vector128(vdupq_laneq_u8(v, idx));
		else static_assert(false_v<scalar>, "NEON : duplicate is not defined in given type.");
	#else
		if constexpr (is_scalar_v<double>)
			switch(idx) {
				case 0: return vector128(vdupq_laneq_f64(v, 0));
				case 1: return vector128(vdupq_laneq_f64(v, 1));
				default: return vector128();
			}
		else if constexpr (is_scalar_v<float>)
			switch(idx) {
				case 0: return vector128(vdupq_laneq_f32(v, 0));
				case 1: return vector128(vdupq_laneq_f32(v, 1));
				case 2: return vector128(vdupq_laneq_f32(v, 2));
				case 3: return vector128(vdupq_laneq_f32(v, 3));
				default: return vector128();
			}
		else if constexpr (is_scalar_v<int64_t>)
			switch(idx) {
				case 0: return vector128(vdupq_laneq_s64(v, 0));
				case 1: return vector128(vdupq_laneq_s64(v, 1));
				default: return vector128();
			}
		else if constexpr (is_scalar_v<uint64_t>)
			switch(idx) {
				case 0: return vector128(vdupq_laneq_u64(v, 0));
				case 1: return vector128(vdupq_laneq_u64(v, 1));
				default: return vector128();
			}
		else if constexpr (is_scalar_v<int32_t>)
			switch(idx) {
				case 0: return vector128(vdupq_laneq_s32(v, 0));
				case 1: return vector128(vdupq_laneq_s32(v, 1));
				case 2: return vector128(vdupq_laneq_s32(v, 2));
				case 3: return vector128(vdupq_laneq_s32(v, 3));
				default: return vector128();
			}
		else if constexpr (is_scalar_v<uint32_t>)
			switch(idx) {
				case 0: return vector128(vdupq_laneq_u32(v, 0));
				case 1: return vector128(vdupq_laneq_u32(v, 1));
				case 2: return vector128(vdupq_laneq_u32(v, 2));
				case 3: return vector128(vdupq_laneq_u32(v, 3));
				default: return vector128();
			}
		else if constexpr (is_scalar_v<int16_t>)
			switch(idx) {
				case 0: return vector128(vdupq_laneq_s16(v, 0));
				case 1: return vector128(vdupq_laneq_s16(v, 1));
				case 2: return vector128(vdupq_laneq_s16(v, 2));
				case 3: return vector128(vdupq_laneq_s16(v, 3));
				case 4: return vector128(vdupq_laneq_s16(v, 4));
				case 5: return vector128(vdupq_laneq_s16(v, 5));
				case 6: return vector128(vdupq_laneq_s16(v, 6));
				case 7: return vector128(vdupq_laneq_s16(v, 7));
				default: return vector128();
			}
		else if constexpr (is_scalar_v<uint16_t>)
			switch(idx) {
				case 0: return vector128(vdupq_laneq_u16(v, 0));
				case 1: return vector128(vdupq_laneq_u16(v, 1));
				case 2: return vector128(vdupq_laneq_u16(v, 2));
				case 3: return vector128(vdupq_laneq_u16(v, 3));
				case 4: return vector128(vdupq_laneq_u16(v, 4));
				case 5: return vector128(vdupq_laneq_u16(v, 5));
				case 6: return vector128(vdupq_laneq_u16(v, 6));
				case 7: return vector128(vdupq_laneq_u16(v, 7));
				default: return vector128();
			}
		else if constexpr (is_scalar_v<int8_t>)
			switch(idx) {
				case 0: return vector128(vdupq_laneq_s8(v, 0));
				case 1: return vector128(vdupq_laneq_s8(v, 1));
				case 2: return vector128(vdupq_laneq_s8(v, 2));
				case 3: return vector128(vdupq_laneq_s8(v, 3));
				case 4: return vector128(vdupq_laneq_s8(v, 4));
				case 5: return vector128(vdupq_laneq_s8(v, 5));
				case 6: return vector128(vdupq_laneq_s8(v, 6));
				case 7: return vector128(vdupq_laneq_s8(v, 7));
				case 8: return vector128(vdupq_laneq_s8(v, 8));
				case 9: return vector128(vdupq_laneq_s8(v, 9));
				case 10: return vector128(vdupq_laneq_s8(v, 10));
				case 11: return vector128(vdupq_laneq_s8(v, 11));
				case 12: return vector128(vdupq_laneq_s8(v, 12));
				case 13: return vector128(vdupq_laneq_s8(v, 13));
				case 14: return vector128(vdupq_laneq_s8(v, 14));
				case 15: return vector128(vdupq_laneq_s8(v, 15));
				default: return vector128();
			}
		else if constexpr (is_scalar_v<uint8_t>)
			switch(idx) {
				case 0: return vector128(vdupq_laneq_u8(v, 0));
				case 1: return vector128(vdupq_laneq_u8(v, 1));
				case 2: return vector128(vdupq_laneq_u8(v, 2));
				case 3: return vector128(vdupq_laneq_u8(v, 3));
				case 4: return vector128(vdupq_laneq_u8(v, 4));
				case 5: return vector128(vdupq_laneq_u8(v, 5));
				case 6: return vector128(vdupq_laneq_u8(v, 6));
				case 7: return vector128(vdupq_laneq_u8(v, 7));
				case 8: return vector128(vdupq_laneq_u8(v, 8));
				case 9: return vector128(vdupq_laneq_u8(v, 9));
				case 10: return vector128(vdupq_laneq_u8(v, 10));
				case 11: return vector128(vdupq_laneq_u8(v, 11));
				case 12: return vector128(vdupq_laneq_u8(v, 12));
				case 13: return vector128(vdupq_laneq_u8(v, 13));
				case 14: return vector128(vdupq_laneq_u8(v, 14));
				case 15: return vector128(vdupq_laneq_u8(v, 15));
				default: return vector128();
			}
		else static_assert(false_v<scalar>, "NEON : duplicate is not defined in given type.");
	#endif
	}

	// reinterpret cast (data will not change)
	template<typename Cvt>
	vector128<Cvt> reinterpret() const noexcept {
		using cvt_vector = typename vector128_type<Cvt>::vector;
		return vector128<Cvt>(*reinterpret_cast<const cvt_vector*>(&v));
	}

	template<typename ArgScalar>
	vector128 shuffle(const vector128<ArgScalar>& idx) const noexcept {
		static_assert(is_scalar_size_v<ArgScalar>, "NEON : wrong mask is given to shuufle.");
		
		if constexpr(!is_scalar_size_v<int8_t>){
			constexpr uint8_t stride = sizeof(scalar) / sizeof(uint8_t);
			const auto offsets = expand_sequence<vector128<uint8_t>>(sequence_map(
					std::make_integer_sequence<uint8_t, 16>(),
					[](auto n){ return n % stride; }
			));
			const auto copy_idx = expand_sequence<vector128<uint8_t>>(sequence_map(
					std::make_integer_sequence<uint8_t, 16>(),
					[](auto n){ return (n / stride) * stride; }
			));
			const vector128<uint8_t> strides = stride;
			const auto actually_idx = vaddq_u8(
				vqtbl1q_u8(
					vmulq_u8(
						*reinterpret_cast<const uint8x16_t*>(&idx.v),
						strides.v
					),
					copy_idx.v
				),
				offsets.v
			);
			return vector128<int8_t>(vqtbl1q_s8(
						*reinterpret_cast<const int8x16_t*>(&v),
						actually_idx
					)).reinterpret<scalar>();
		}
		else {
			return vector128<int8_t>(vqtbl1q_s8(
						*reinterpret_cast<const int8x16_t*>(&v),
						*reinterpret_cast<const uint8x16_t*>(&idx)
					)).reinterpret<scalar>();
		}
	}

	std::string to_str(const std::pair<std::string_view, std::string_view> brancket = print_format::brancket::square, std::string_view delim = print_format::delim::space) const {
		std::ostringstream ss;
		alignas(16) scalar elements[elements_size];
		aligned_store(elements);
		ss << brancket.first;
		for (size_t i = 0; i < elements_size; ++i) {
			ss << (i ? delim : "");
			ss << ((std::is_integral_v<scalar> && is_scalar_size_v<int8_t>) ? static_cast<int>(elements[i]) : elements[i]);
		}
		ss << brancket.second;
		return ss.str();
	}
};

template<typename Scalar>
std::ostream& operator<<(std::ostream& os, const vector128<Scalar>& v) {
	os << v.to_str();
	return os;
}

namespace function {
	std::array<vector128<float>, 4> transpose(const std::array<vector128<float>, 4>& arg) {
		auto tmp = vld4q_f32(reinterpret_cast<const float*>(arg.data()));
		return {
			vector128<float>(tmp.val[0]), vector128<float>(tmp.val[1]),
			vector128<float>(tmp.val[2]), vector128<float>(tmp.val[3])
		};
	}
	std::array<vector128<double>, 2> transpose(const std::array<vector128<double>, 2>& arg) {
		auto tmp = vld2q_f64(reinterpret_cast<const double*>(arg.data()));
		return {
			vector128<double>(tmp.val[0]),
			vector128<double>(tmp.val[1])
		};
	}
}

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

#endif
#endif