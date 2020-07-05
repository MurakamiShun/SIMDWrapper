#pragma once
#ifdef __x86_64

#include <cstdint>
#include <vector>
#include <array>
#include <type_traits>
#include <tuple>
#include <bitset>
#include <sstream>

#if defined(__GNUC__)
#include <x86intrin.h>
#include <cpuid.h>
#elif defined(_MSC_VER)
#include <intrin.h>
#endif

class Instruction {
public:
	static bool SSE4_1() noexcept { return CPU_ref.SSE4_1; }
	static bool SSE4_2() noexcept { return CPU_ref.SSE4_2; }
	static bool AVX2() noexcept { return CPU_ref.AVX2; }
	static bool AVX() noexcept { return CPU_ref.AVX; }
	static bool FMA() noexcept { return CPU_ref.FMA; }
private:
	struct InstructionSet {
		bool SSE4_1 = false;
		bool SSE4_2 = false;
		bool AVX2 = false;
		bool AVX = false;
		bool FMA = false;
		InstructionSet() {
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
	static inline InstructionSet CPU_ref;
};

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

	static_assert(is_any<Scalar, float, double>::value || std::is_integral_v<Scalar>, "SSE4.2 : Given type is not supported.");

	struct m128_wrapper{ using type = __m128; };
	struct m128i_wrapper{ using type = __m128i; };
	struct m128d_wrapper{ using type = __m128d; };
	struct false_type{ using type = std::false_type; };

	using scalar = Scalar;
	using vector = typename std::conditional_t< std::is_same_v<Scalar, double>, m128d_wrapper,
		typename std::conditional_t< std::is_same_v<Scalar, float>, m128_wrapper,
		typename std::conditional_t< std::is_integral_v<Scalar>, m128i_wrapper,
		false_type>>>::type;

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

	template<class... Args, size_t... I, size_t N = sizeof...(Args)>
	void init_by_reversed_argments(std::index_sequence<I...>, scalar last, Args&&... args) noexcept {
		constexpr bool is_right_args = ((N + 1) == elements_size);
		auto args_tuple = std::make_tuple(std::forward<Args>(args)...);

		if constexpr (is_scalar_v<double>) {
			static_assert(is_right_args, "SSE4.2 : wrong number of arguments (expected 4).");
			v = _mm_set_pd(std::get<N - 1 - I>(args_tuple)..., last);
		}
		else if constexpr (is_scalar_v<float>) {
			static_assert(is_right_args, "SSE4.2 : wrong number of arguments (expected 8).");
			v = _mm_set_ps(std::get<N - 1 - I>(args_tuple)..., last);
		}
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (is_scalar_size_v<int8_t>) {
				static_assert(is_right_args, "SSE4.2 : wrong number of arguments (expected 32).");
				v = _mm_set_epi8(std::get<N - 1 - I>(args_tuple)..., last);
			}
			else if constexpr (is_scalar_size_v<int16_t>) {
				static_assert(is_right_args, "SSE4.2 : wrong number of arguments (expected 16).");
				v = _mm_set_epi16(std::get<N - 1 - I>(args_tuple)..., last);
			}
			else if constexpr (is_scalar_size_v<int32_t>) {
				static_assert(is_right_args, "SSE4.2 : wrong number of arguments (expected 8).");
				v = _mm_set_epi32(std::get<N - 1 - I>(args_tuple)..., last);
			}
			else if constexpr (is_scalar_size_v<int64_t>) {
				static_assert(is_right_args, "SSE4.2 : wrong number of arguments (expected 4).");
				v = _mm_set_epi64x(std::get<N - 1 - I>(args_tuple)..., last);
			}
			else
				static_assert(false_v<Scalar>, "SSE4.2 : initializer is not defined in given type.");
		}
		else
			static_assert(false_v<Scalar>, "SSE4.2 : initializer is not defined in given type.");
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

	vector128() noexcept : v() {}
	vector128(const scalar arg) noexcept { *this = arg; }
	vector128(const vector arg) noexcept : v(arg) {  }
	template<class... Args, typename Indices = std::make_index_sequence<sizeof...(Args)>>
	vector128(scalar first, Args... args) noexcept {
		init_by_reversed_argments(Indices(), first, std::forward<Args>(args)...);
	}
	vector128(const vector128& arg) noexcept : v(arg.v) {  }

	input_iterator begin() const noexcept {
		return input_iterator(*this, typename input_iterator::template Index<0>());
	}
	input_iterator end() const noexcept {
		return input_iterator(*this, typename input_iterator::template Index<elements_size>());
	}

	vector128 operator+(const vector128& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector128(_mm_add_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector128(_mm_add_ps(v, arg.v));
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (is_scalar_size_v<int8_t>)
				return vector128(_mm_add_epi8(v, arg.v));
			else if constexpr (is_scalar_size_v<int16_t>)
				return vector128(_mm_add_epi16(v, arg.v));
			else if constexpr (is_scalar_size_v<int32_t>)
				return vector128(_mm_add_epi32(v, arg.v));
			else if constexpr (is_scalar_size_v<int64_t>)
				return vector128(_mm_add_epi64(v, arg.v));
			else
				static_assert(false_v<Scalar>, "SSE4.2 : operator+ is not defined in given type.");
		}
		else
			static_assert(false_v<Scalar>, "SSE4.2 : operator+ is not defined in given type.");
	}
	vector128 operator-(const vector128& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector128(_mm_sub_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector128(_mm_sub_ps(v, arg.v));
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (is_scalar_size_v<int8_t>)
				return vector128(_mm_sub_epi8(v, arg.v));
			else if constexpr (is_scalar_size_v<int16_t>)
				return vector128(_mm_sub_epi16(v, arg.v));
			else if constexpr (is_scalar_size_v<int32_t>)
				return vector128(_mm_sub_epi32(v, arg.v));
			else if constexpr (is_scalar_size_v<int64_t>)
				return vector128(_mm_sub_epi64(v, arg.v));
			else
				static_assert(false_v<Scalar>, "SSE4.2 : operator- is not defined in given type.");
		}
		else
			static_assert(false_v<Scalar>, "SSE4.2 : operator- is not defined in given type.");
	}
	auto operator*(const vector128& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector128(_mm_mul_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector128(_mm_mul_ps(v, arg.v));
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (std::is_signed_v<scalar>) {
				if constexpr (is_scalar_size_v<int32_t>)
					return vector128<int64_t>(_mm_mul_epi32(v, arg.v));
				else
					static_assert(false_v<Scalar>, "SSE4.2 : operator* is not defined in given type.");
			}
			else {
				if constexpr (is_scalar_size_v<int32_t>)
					return vector128<uint64_t>(_mm_mul_epu32(v, arg.v));
				else
					static_assert(false_v<Scalar>, "SSE4.2 : operator* is not defined in given type.");
			}
		}
		else
			static_assert(false_v<Scalar>, "SSE4.2 : operator* is not defined in given type.");
	}
	vector128 operator/(const vector128& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector128(_mm_div_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector128(_mm_div_ps(v, arg.v));
		else
			static_assert(false_v<Scalar>, "SSE4.2 : operator/ is not defined in given type.");
	}
	vector128& operator=(const scalar arg) noexcept {
		if constexpr (is_scalar_v<double>)
			v = _mm_set1_pd(arg);
		else if constexpr (is_scalar_v<float>)
			v = _mm_set1_ps(arg);
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (is_scalar_size_v<int8_t>)
				v = _mm_set1_epi8(arg);
			else if constexpr (is_scalar_size_v<int16_t>)
				v = _mm_set1_epi16(arg);
			else if constexpr (is_scalar_size_v<int32_t>)
				v = _mm_set1_epi32(arg);
			else if constexpr (is_scalar_size_v<int64_t>)
				v = _mm_set1_epi64x(arg);
			else
				static_assert(false_v<Scalar>, "SSE4.2 : operator=(scalar) is not defined in given type.");
		}
		else
			static_assert(false_v<Scalar>, "SSE4.2 : operator=(scalar) is not defined in given type.");
		return *this;
	}
	vector128& load(const scalar* const arg) noexcept {
		if constexpr (is_scalar_v<double>)
			v = _mm_loadu_pd(arg);
		else if constexpr (is_scalar_v<float>)
			v = _mm_loadu_ps(arg);
		else if constexpr (std::is_integral_v<scalar>)
			v = _mm_loadu_si128(reinterpret_cast<const vector*>(arg));
		else
			static_assert(false_v<Scalar>, "SSE4.2 : load(pointer) is not defined in given type.");
		return *this;
	}
	vector128& aligned_load(const scalar* const arg) noexcept {
		if constexpr (is_scalar_v<double>)
			v = _mm_load_pd(arg);
		else if constexpr (is_scalar_v<float>)
			v = _mm_load_ps(arg);
		else if constexpr (std::is_integral_v<scalar>)
			v = _mm_load_si128(arg);
		else
			static_assert(false_v<Scalar>, "SSE4.2 : load(pointer) is not defined in given type.");
		return *this;
	}
	void store(scalar* arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			_mm_storeu_pd(arg, v);
		else if constexpr (is_scalar_v<float>)
			_mm_storeu_ps(arg, v);
		else if constexpr (std::is_integral_v<scalar>)
			_mm_storeu_si128(reinterpret_cast<vector*>(arg), v);
		else
			static_assert(false_v<Scalar>, "SSE4.2 : store(pointer) is not defined in given type.");
	}
	void aligned_store(scalar* arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			_mm_store_pd(arg, v);
		else if constexpr (is_scalar_v<float>)
			_mm_store_ps(arg, v);
		else if constexpr (std::is_integral_v<scalar>)
			_mm_store_si128(reinterpret_cast<vector*>(arg), v);
		else
			static_assert(false_v<Scalar>, "SSE4.2 : store(pointer) is not defined in given type.");
	}
	scalar operator[](const size_t index) const noexcept {
		return reinterpret_cast<const scalar*>(&v)[index];
	}
	vector128 operator==(const vector128& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector128(_mm_cmpeq_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector128(_mm_cmpeq_ps(v, arg.v));
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (is_scalar_size_v<int8_t>)
				return vector128(_mm_cmpeq_epi8(v, arg.v));
			else if constexpr (is_scalar_size_v<int16_t>)
				return vector128(_mm_cmpeq_epi16(v, arg.v));
			else if constexpr (is_scalar_size_v<int32_t>)
				return vector128(_mm_cmpeq_epi32(v, arg.v));
			else if constexpr (is_scalar_size_v<int64_t>)
				return vector128(_mm_cmpeq_epi64(v, arg.v));
			else
				static_assert(false_v<Scalar>, "SSE4.2 : operator== is not defined in given type.");
		}
		else
			static_assert(false_v<Scalar>, "SSE4.2 : operator== is not defined in given type.");
	}
	vector128 operator>(const vector128& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector128(_mm_cmpgt_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector128(_mm_cmpgt_ps(v, arg.v));
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (std::is_signed_v<scalar>) {
				if constexpr (is_scalar_size_v<int8_t>)
					return vector128(_mm_cmpgt_epi8(v, arg.v));
				else if constexpr (is_scalar_size_v<int16_t>)
					return vector128(_mm_cmpgt_epi16(v, arg.v));
				else if constexpr (is_scalar_size_v<int32_t>)
					return vector128(_mm_cmpgt_epi32(v, arg.v));
				else if constexpr (is_scalar_size_v<int64_t>)
					return vector128(_mm_cmpgt_epi64(v, arg.v));
				else
					static_assert(false_v<Scalar>, "SSE4.2 : operator> is not defined in given type.");
			}
			else {
				if constexpr (is_scalar_size_v<int8_t>)
					return vector128(_mm_cmpgt_epi8(
						_mm_xor_si128(v, _mm_set1_epi8(INT8_MIN)),
						_mm_xor_si128(arg.v, _mm_set1_epi8(INT8_MIN))
					));
				else if constexpr (is_scalar_size_v<int16_t>)
					return vector128(_mm_cmpgt_epi16(
						_mm_xor_si128(v, _mm_set1_epi16(INT16_MIN)),
						_mm_xor_si128(arg.v, _mm_set1_epi16(INT16_MIN))
					));
				else if constexpr (is_scalar_size_v<int32_t>)
					return vector128(_mm_cmpgt_epi32(
						_mm_xor_si128(v, _mm_set1_epi32(INT32_MIN)),
						_mm_xor_si128(arg.v, _mm_set1_epi32(INT32_MIN))
					));
				else if constexpr (is_scalar_size_v<int64_t>)
					return vector128(_mm_cmpgt_epi64(
						_mm_xor_si128(v, _mm_set1_epi64x(INT64_MIN)),
						_mm_xor_si128(arg.v, _mm_set1_epi64x(INT64_MIN))
					));
				else
					static_assert(false_v<Scalar>, "SSE4.2 : operator> is not defined in given type.");
			}
		}
		else
			static_assert(false_v<Scalar>, "SSE4.2 : operator> is not defined in given type.");
	}
	vector128 operator<(const vector128& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector128(_mm_cmpgt_pd(arg.v, v));
		else if constexpr (is_scalar_v<float>)
			return vector128(_mm_cmpgt_ps(arg.v, v));
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (std::is_signed_v<scalar>) {
				if constexpr (is_scalar_size_v<int8_t>)
					return vector128(_mm_cmpgt_epi8(arg.v, v));
				else if constexpr (is_scalar_size_v<int16_t>)
					return vector128(_mm_cmpgt_epi16(arg.v, v));
				else if constexpr (is_scalar_size_v<int32_t>)
					return vector128(_mm_cmpgt_epi32(arg.v, v));
				else if constexpr (is_scalar_size_v<int64_t>)
					return vector128(_mm_cmpgt_epi64(arg.v, v));
				else
					static_assert(false_v<Scalar>, "SSE4.2 : operator< is not defined in given type.");
			}
			else {
				if constexpr (is_scalar_size_v<int8_t>)
					return vector128(_mm_cmpgt_epi8(
						_mm_xor_si128(arg.v, _mm_set1_epi8(INT8_MIN)),
						_mm_xor_si128(v, _mm_set1_epi8(INT8_MIN))
					));
				else if constexpr (is_scalar_size_v<int16_t>)
					return vector128(_mm_cmpgt_epi16(
						_mm_xor_si128(arg.v, _mm_set1_epi16(INT16_MIN)),
						_mm_xor_si128(v, _mm_set1_epi16(INT16_MIN))
					));
				else if constexpr (is_scalar_size_v<int32_t>)
					return vector128(_mm_cmpgt_epi32(
						_mm_xor_si128(arg.v, _mm_set1_epi32(INT32_MIN)),
						_mm_xor_si128(v, _mm_set1_epi32(INT32_MIN))
					));
				else if constexpr (is_scalar_size_v<int64_t>)
					return vector128(_mm_cmpgt_epi64(
						_mm_xor_si128(arg.v, _mm_set1_epi64x(INT64_MIN)),
						_mm_xor_si128(v, _mm_set1_epi64x(INT64_MIN))
					));
				else
					static_assert(false_v<Scalar>, "SSE4.2 : operator< is not defined in given type.");
			}
		}
		else
			static_assert(false_v<Scalar>, "SSE4.2 : operator< is not defined in given type.");
	}

	bool is_all_zero() const noexcept {
		if constexpr (is_scalar_v<double>)
			return bool(_mm_testz_si128(
				_mm_castpd_si128(v),
				_mm_cmpeq_epi64(_mm_castpd_si128(v), _mm_castpd_si128(v))
			));
		else if constexpr (is_scalar_v<float>)
			return bool(_mm_testz_si128(
				_mm_castps_si128(v),
				_mm_cmpeq_epi64(_mm_castps_si128(v), _mm_castps_si128(v))
			));
		else if constexpr (std::is_integral_v<scalar>)
			return bool(_mm_testz_si128(v, _mm_cmpeq_epi64(v, v)));
		else
			static_assert(false_v<Scalar>, "SSE4.2 : is_all_zero is not defined in given type.");
	}
	bool is_all_one() const noexcept {
		if constexpr (is_scalar_v<double>)
			return bool(_mm_testc_si128(
				_mm_castpd_si128(v),
				_mm_cmpeq_epi64(_mm_castpd_si128(v), _mm_castpd_si128(v))
			));
		else if constexpr (is_scalar_v<float>)
			return bool(_mm_testc_si128(
				_mm_castps_si128(v),
				_mm_cmpeq_epi64(_mm_castps_si128(v), _mm_castps_si128(v))
			));
		else if constexpr (std::is_integral_v<scalar>)
			return bool(_mm_testc_si128(v, _mm_cmpeq_epi64(v, v)));
		else
			static_assert(false_v<Scalar>, "SSE4.2 : is_all_one is not defined in given type.");
	}
	vector128 operator& (const vector128& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector128(_mm_and_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector128(_mm_and_ps(v, arg.v));
		else if constexpr (std::is_integral_v<scalar>)
			return vector128(_mm_and_si128(v, arg.v));
		else
			static_assert(false_v<Scalar>, "SSE4.2 : and is not defined in given type.");
	}
	vector128 nand(const vector128& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector128(_mm_andnot_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector128(_mm_andnot_ps(v, arg.v));
		else if constexpr (std::is_integral_v<scalar>)
			return vector128(_mm_andnot_si128(v, arg.v));
		else
			static_assert(false_v<Scalar>, "SSE4.2 : nand is not defined in given type.");
	}
	vector128 operator~() const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector128(_mm_andnot_pd(v, v));
		else if constexpr (is_scalar_v<float>)
			return vector128(_mm_andnot_ps(v, v));
		else if constexpr (std::is_integral_v<scalar>)
			return vector128(_mm_andnot_si128(v, v));
		else
			static_assert(false_v<Scalar>, "SSE4.2 : not is not defined in given type.");
	}
	vector128 operator| (const vector128& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector128(_mm_or_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector128(_mm_or_ps(v, arg.v));
		else if constexpr (std::is_integral_v<scalar>)
			return vector128(_mm_or_si128(v, arg.v));
		else
			static_assert(false_v<Scalar>, "SSE4.2 : or is not defined in given type.");
	}
	vector128 operator^ (const vector128& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector128(_mm_xor_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector128(_mm_xor_ps(v, arg.v));
		else if constexpr (std::is_integral_v<scalar>)
			return vector128(_mm_xor_si128(v, arg.v));
		else
			static_assert(false_v<Scalar>, "SSE4.2 : xor is not defined in given type.");
	}
	vector128 operator>>(const int n) const noexcept {
		if constexpr (std::is_integral_v<scalar>) {
			if constexpr (is_scalar_size_v<int16_t>)
				return vector128(_mm_srl_epi16(
					v,
					_mm_set1_epi64x(n)
				));
			else if constexpr (is_scalar_size_v<int32_t>)
				return vector128(_mm_srlv_epi32(v, _mm_set1_epi32(n)));
			else if constexpr (is_scalar_size_v<int64_t>)
				return vector128(_mm_srlv_epi64(v, _mm_set1_epi64x(n)));
			else
				static_assert(false_v<Scalar>, "SSE4.2 : operator>> is not defined in given type.");
		}
		else
			static_assert(false_v<Scalar>, "SSE4.2 : operator>> is not defined in given type.");
	}
	vector128 operator>>(const vector128& arg) const noexcept {
		if constexpr (std::is_integral_v<scalar>) {
			if constexpr (is_scalar_size_v<int32_t>)
				return vector128(_mm_srlv_epi32(v, arg.v));
			else if constexpr (is_scalar_size_v<int64_t>)
				return vector128(_mm_srlv_epi64(v, arg.v));
			else
				static_assert(false_v<Scalar>, "SSE4.2 : operator>>(vector128) is not defined in given type.");
		}
		else
			static_assert(false_v<Scalar>, "SSE4.2 : operator>>(vector128) is not defined in given type.");
	}
	vector128 operator<<(const int n) const noexcept {
		if constexpr (std::is_integral_v<scalar>) {
			if constexpr (is_scalar_size_v<int16_t>)
				return vector128(_mm_sll_epi16(
					v,
					_mm_set1_epi64x(n)
				));
			else if constexpr (is_scalar_size_v<int32_t>)
				return vector128(_mm_sllv_epi32(v, _mm_set1_epi32(n)));
			else if constexpr (is_scalar_size_v<int64_t>)
				return vector128(_mm_sllv_epi64(v, _mm_set1_epi64x(n)));
			else
				static_assert(false_v<Scalar>, "SSE4.2 : operator<< is not defined in given type.");
		}
		else
			static_assert(false_v<Scalar>, "SSE4.2 : operator<< is not defined in given type.");
	}
	vector128 operator<<(const vector128& arg) const noexcept {
		if constexpr (std::is_integral_v<scalar>) {
			if constexpr (is_scalar_size_v<int32_t>)
				return vector128(_mm_sllv_epi32(v, arg.v));
			else if constexpr (is_scalar_size_v<int64_t>)
				return vector128(_mm_sllv_epi64(v, arg.v));
			else
				static_assert(false_v<Scalar>, "SSE4.2 : operator<<(vector128) is not defined in given type.");
		}
		else
			static_assert(false_v<Scalar>, "SSE4.2 : operator<<(vector128) is not defined in given type.");
	}
	// Reciprocal approximation < 1.5*2^12
	vector128 rcp() const noexcept {
		if constexpr (is_scalar_v<float>)
			return vector128(_mm_rcp_ps(v));
		else
			static_assert(false_v<Scalar>, "SSE4.2 : rcp is not defined in given type.");
	}
	// this * (1 / arg)
	vector128 fast_div(const vector128& arg) const noexcept {
		if constexpr (is_scalar_v<float>)
			return vector128(_mm_mul_ps(v, _mm_rcp_ps(arg.v)));
		else
			static_assert(false_v<Scalar>, "SSE4.2 : fast_div is not defined in given type.");
	}
	vector128 sqrt() const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector128(_mm_sqrt_pd(v));
		else if constexpr (is_scalar_v<float>)
			return vector128(_mm_sqrt_ps(v));
		else
			static_assert(false_v<Scalar>, "SSE4.2 : sqrt is not defined in given type.");
	}
	// 1 / sqrt()
	vector128 rsqrt() const noexcept {
		if constexpr (is_scalar_v<float>)
			return vector128(_mm_rsqrt_ps(v));
		else
			static_assert(false_v<Scalar>, "SSE4.2 : rsqrt is not defined in given type.");
	}
	vector128 abs() const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector128(_mm_andnot_pd(_mm_set1_pd(-0.0), v));
		else if constexpr (is_scalar_v<float>)
			return vector128(_mm_andnot_ps(_mm_set1_ps(-0.0f), v));
		else if constexpr (std::is_integral_v<scalar>&& std::is_signed_v<scalar>) {
			if constexpr (is_scalar_size_v<int8_t>)
				return vector128(_mm_abs_epi8(v));
			else if constexpr (is_scalar_size_v<int16_t>)
				return vector128(_mm_abs_epi16(v));
			else if constexpr (is_scalar_size_v<int32_t>)
				return vector128(_mm_abs_epi32(v));
			else if constexpr (is_scalar_size_v<int64_t>) {
				vector mask = _mm_cmpgt_epi64(_mm_setzero_si128(), v);
				return vector128(_mm_sub_epi64(_mm_xor_si128(v, mask), mask));
			}
			else
				static_assert(false_v<Scalar>, "SSE4.2 : abs is not defined in given type.");
		}
		else
			static_assert(false_v<Scalar>, "SSE4.2 : abs is not defined in given type.");
	}
	vector128 max(const vector128& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector128(_mm_max_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector128(_mm_max_ps(v, arg.v));
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (std::is_signed_v<scalar>) {
				if constexpr (is_scalar_size_v<int8_t>)
					return vector128(_mm_max_epi8(v, arg.v));
				else if constexpr (is_scalar_size_v<int16_t>)
					return vector128(_mm_max_epi16(v, arg.v));
				else if constexpr (is_scalar_size_v<int32_t>)
					return vector128(_mm_max_epi32(v, arg.v));
				else if constexpr (is_scalar_size_v<int64_t>)
					return vector128(_mm_blendv_epi8(arg.v, v, _mm_cmpgt_epi64(v, arg.v)));
				else
					static_assert(false_v<Scalar>, "SSE4.2 : max is not defined in given type.");
			}
			else {
				if constexpr (is_scalar_size_v<int8_t>)
					return vector128(_mm_max_epu8(v, arg.v));
				else if constexpr (is_scalar_size_v<int16_t>)
					return vector128(_mm_max_epu16(v, arg.v));
				else if constexpr (is_scalar_size_v<int32_t>)
					return vector128(_mm_max_epu32(v, arg.v));
				else if constexpr (is_scalar_size_v<int64_t>)
					return vector128(_mm_blendv_epi8(
						arg.v,
						v,
						_mm_cmpgt_epi64(
							_mm_xor_si128(v, _mm_set1_epi64x(INT64_MIN)),
							_mm_xor_si128(arg.v, _mm_set1_epi64x(INT64_MIN))
						)
					));
				else
					static_assert(false_v<Scalar>, "SSE4.2 : max is not defined in given type.");
			}
		}
		else
			static_assert(false_v<Scalar>, "SSE4.2 : max is not defined in given type.");
	}
	vector128 min(const vector128& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector128(_mm_min_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector128(_mm_min_ps(v, arg.v));
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (std::is_signed_v<scalar>) {
				if constexpr (is_scalar_size_v<int8_t>)
					return vector128(_mm_min_epi8(v, arg.v));
				else if constexpr (is_scalar_size_v<int16_t>)
					return vector128(_mm_min_epi16(v, arg.v));
				else if constexpr (is_scalar_size_v<int32_t>)
					return vector128(_mm_min_epi32(v, arg.v));
				else if constexpr (is_scalar_size_v<int64_t>)
					return vector128(_mm_blendv_epi8(v, arg.v, _mm_cmpgt_epi64(v, arg.v)));
				else
					static_assert(false_v<Scalar>, "SSE4.2 : min is not defined in given type.");
			}
			if constexpr (std::is_unsigned_v<scalar>) {
				if constexpr (is_scalar_size_v<int8_t>)
					return vector128(_mm_min_epu8(v, arg.v));
				else if constexpr (is_scalar_size_v<int16_t>)
					return vector128(_mm_min_epu16(v, arg.v));
				else if constexpr (is_scalar_size_v<int32_t>)
					return vector128(_mm_min_epu32(v, arg.v));
				else if constexpr (is_scalar_size_v<int64_t>)
					return vector128(_mm_blendv_epi8(
						v,
						arg.v,
						_mm_cmpgt_epi64(
							_mm_xor_si128(v, _mm_set1_epi64x(INT64_MIN)),
							_mm_xor_si128(arg.v, _mm_set1_epi64x(INT64_MIN))
						)
					));
				else
					static_assert(false_v<Scalar>, "SSE4.2 : min is not defined in given type.");
			}
		}
		else
			static_assert(false_v<Scalar>, "SSE4.2 : min is not defined in given type.");
	}
	vector128 ceil() const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector128(_mm_ceil_pd(v));
		else if constexpr (is_scalar_v<float>)
			return vector128(_mm_ceil_ps(v));
		else
			static_assert(false_v<Scalar>, "SSE4.2 : ceil is not defined in given type.");
	}
	vector128 floor() const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector128(_mm_floor_pd(v));
		else if constexpr (is_scalar_v<float>)
			return vector128(_mm_floor_ps(v));
		else
			static_assert(false_v<Scalar>, "SSE4.2 : floor is not defined in given type.");
	}
	// { this[0] + this[1], arg[0] + arg[1], this[2] + this[3], ... }
	vector128 hadd(const vector128& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector128(_mm_hadd_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector128(_mm_hadd_ps(v, arg.v));
		else if constexpr (std::is_integral_v<scalar>&& std::is_signed_v<scalar>) {
			if constexpr (is_scalar_size_v<int16_t>)
				return vector128(_mm_hadd_epi16(v, arg.v));
			else if constexpr (is_scalar_size_v<int32_t>)
				return vector128(_mm_hadd_epi32(v, arg.v));
			else
				static_assert(false_v<Scalar>, "SSE4.2 : hadd is not defined in given type.");
		}
		else
			static_assert(false_v<Scalar>, "SSE4.2 : hadd is not defined in given type.");
	}
	// (mask) ? this : a
	template<typename MaskScalar>
	vector128 cmp_blend(const vector128& a, const vector128<MaskScalar>& mask) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector128(_mm_blendv_pd(a.v, v, *reinterpret_cast<const __m256d*>(&(mask.v))));
		else if constexpr (is_scalar_v<float>)
			return vector128(_mm_blendv_ps(a.v, v, *reinterpret_cast<const __m256*>(&(mask.v))));
		else if constexpr (std::is_integral_v<scalar>)
			return vector128(_mm_blendv_epi8(a.v, v, *reinterpret_cast<const __m256i*>(&(mask.v))));
		else
			static_assert(false_v<Scalar>, "SSE4.2 : cmp_blend is not defined in given type.");
	}
	template<typename Cvt>
	explicit operator vector128<Cvt>() const noexcept {
		if constexpr (is_scalar_v<float>&& std::is_same_v<Cvt, int32_t>)
			return vector128<Cvt>(_mm_cvtps_epi32(v));
		else if constexpr (is_scalar_v<int32_t>&& std::is_same_v<Cvt, float>)
			return vector128<Cvt>(_mm_cvtepi32_ps(v));
		else
			static_assert(false_v<Scalar>, "SSE4.2 : type casting is not defined in given type.");
	}
	// reinterpret cast (data will not change)
	template<typename Cvt>
	vector128<Cvt> reinterpret() const noexcept {
		using cvt_vector = typename vector128_type<Cvt>::vector;
		return vector128<Cvt>(*reinterpret_cast<const cvt_vector*>(&v));
	}

	std::string to_str(std::string_view delim = print_format::delim::space, const std::pair<std::string_view, std::string_view> brancket = print_format::brancket::square) const {
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
	// max(a, b)
	template<typename Scalar>
	vector128<Scalar> max(const vector128<Scalar>& a, const vector128<Scalar>& b) {
		return a.max(b);
	}
	// min(a, b)
	template<typename Scalar>
	vector128<Scalar> min(const vector128<Scalar>& a, const vector128<Scalar>& b) {
		return a.min(b);
	}
	// (==) ? a : b
	template<typename MaskScalar, typename Scalar>
	vector128<Scalar> cmp_blend(const vector128<MaskScalar>& mask, const vector128<Scalar>& a, const vector128<Scalar>& b) {
		return a.cmp_blend(b, mask);
	}
	// { a[0]+a[1], b[0]+b[1], a[2]+a[3], b[2]+b[3], ...}
	template<typename Scalar>
	vector128<Scalar> hadd(const vector128<Scalar>& a, const vector128<Scalar>& b) {
		return a.hadd(b);
	}
	// reinterpret cast (data will not change)
	template<typename Cvt, typename Scalar>
	vector128<Cvt> reinterpret(const vector128<Scalar>& arg) {
		return arg.template reinterpret<Cvt>();
	}
}


namespace type {
	using int8x16 = vector128<int8_t>;
	using int16x8 = vector128<int16_t>;
	using int32x4 = vector128<int32_t>;
	using int64x2 = vector128<int64_t>;

	using uint8x16 = vector128<uint8_t>;
	using uint16x8 = vector128<uint16_t>;
	using uint32x4 = vector128<uint32_t>;
	using uint64x2 = vector128<uint64_t>;

	using floatx4 = vector128<float>;
	using doublex2 = vector128<double>;
}

#endif