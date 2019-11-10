#pragma once

#include <immintrin.h>
#include <type_traits>
#include <cstdint>
#include <tuple>
#include <array>

template<typename Scalar>
struct AVX_type {
	using scalar = Scalar;
	using vector = typename std::conditional< std::is_same<Scalar, double>::value, __m256d,
				typename std::conditional< std::is_same<Scalar, float>::value, __m256,
				typename std::conditional< std::is_integral<Scalar>::value, __m256i,
				std::false_type>::type>::type>::type;

	static_assert(!std::is_same<vector, std::false_type>::value, "AVX2 : Given type is not supported.");
};

template<typename Scalar>
class AVX_vector {
private:
	using scalar = typename AVX_type<Scalar>::scalar;
	using vector = typename AVX_type<Scalar>::vector;
	static constexpr size_t elements_size = 32 / sizeof(scalar);

	template<class... Args, size_t... I, size_t N = sizeof...(Args)>
	void init_by_reversed_argments(std::index_sequence<I...>, scalar last, Args&&... args) {
		if constexpr (std::is_same<scalar, double>::value) {
			static_assert(N+1 == 4, "AVX2 : wrong number of arguments (expected 4).");
			v = _mm256_set_pd(std::get<N - 1 - I>(std::make_tuple(std::forward<Args>(args)...))..., last);
		}
		else if constexpr (std::is_same<scalar, float>::value) {
			static_assert(N+1 == 8, "AVX2 : wrong number of arguments (expected 8).");
			v = _mm256_set_ps(std::get<N - 1 - I>(std::make_tuple(std::forward<Args>(args)...))..., last);
		}
		else if constexpr (std::is_integral<scalar>::value) {
			if constexpr (sizeof(scalar) == sizeof(int8_t)) {
				static_assert(N+1 == 32, "AVX2 : wrong number of arguments (expected 32).");
				v = _mm256_set_epi8(std::get<N - 1 - I>(std::make_tuple(std::forward<Args>(args)...))..., last);
			}
			else if constexpr (sizeof(scalar) == sizeof(int16_t)) {
				static_assert(N+1 == 16, "AVX2 : wrong number of arguments (expected 16).");
				v = _mm256_set_epi16(std::get<N - 1 - I>(std::make_tuple(std::forward<Args>(args)...))..., last);
			}
			else if constexpr (sizeof(scalar) == sizeof(int32_t)) {
				static_assert(N+1 == 8, "AVX2 : wrong number of arguments (expected 8).");
				v = _mm256_set_epi32(std::get<N - 1 - I>(std::make_tuple(std::forward<Args>(args)...))..., last);
			}
			else if constexpr (sizeof(scalar) == sizeof(int64_t)) {
				static_assert(N+1 == 4, "AVX2 : wrong number of arguments (expected 4).");
				v = _mm256_set_epi64x(std::get<N - 1 - I>(std::make_tuple(std::forward<Args>(args)...))..., last);
			}
			else
				static_assert(false, "AVX2 : initializer is not defined in given type.");
		}
		else
			static_assert(false, "AVX2 : initializer is not defined in given type.");
	}

	class input_iterator {
	private:
		std::array<scalar, elements_size> tmp;
		size_t index;
	public:
		template<size_t N>
		struct Index {};

		input_iterator(const input_iterator& it) : 
			index(it.index), 
			tmp(it.tmp){
		}
		template<size_t N>
		input_iterator(const AVX_vector& arg, Index<N>) {
			index = N;
			if constexpr (N == 0)
				arg >> tmp.data();
			else
				tmp = std::array<scalar, elements_size>();
		}
		const scalar operator*() const {
			return tmp[index];
		}
		input_iterator& operator++() {
			index++;
			return *this;
		}
		bool operator==(const input_iterator& it) const {
			return (index == it.index);
		}
		bool operator!=(const input_iterator& it) const {
			return (index != it.index);
		}
	};
public:
	vector v;

	AVX_vector() : v() {}
	AVX_vector(const scalar arg) { *this = arg; }
	AVX_vector(const scalar* arg) { *this = arg; }
	AVX_vector(const vector arg) : v(arg) {  }
	template<class... Args, typename Indices = std::make_index_sequence<sizeof...(Args)>>
	AVX_vector(scalar first, Args... args){
		init_by_reversed_argments(Indices(), first, std::forward<Args>(args)...);
	}

	input_iterator begin() const {
		return input_iterator(*this, input_iterator::template Index<0>());
	}
	input_iterator end() const {
		return input_iterator(*this, input_iterator::template Index<elements_size>());
	}

	AVX_vector operator+(const AVX_vector& arg) const {
		if constexpr (std::is_same<scalar, double>::value)
			return AVX_vector(_mm256_add_pd(v, arg.v));
		else if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_add_ps(v, arg.v));
		else if constexpr (std::is_integral<scalar>::value) {
			if constexpr (sizeof(scalar) == sizeof(int8_t))
				return AVX_vector(_mm256_add_epi8(v, arg.v));
			else if constexpr (sizeof(scalar) == sizeof(int16_t))
				return AVX_vector(_mm256_add_epi16(v, arg.v));
			else if constexpr (sizeof(scalar) == sizeof(int32_t))
				return AVX_vector(_mm256_add_epi32(v, arg.v));
			else if constexpr (sizeof(scalar) == sizeof(int64_t))
				return AVX_vector(_mm256_add_epi64(v, arg.v));
			else
				static_assert(false, "AVX2 : operator+ is not defined in given type.");
		}
		else
			static_assert(false, "AVX2 : operator+ is not defined in given type.");
	}
	AVX_vector operator-(const AVX_vector& arg) const {
		if constexpr (std::is_same<scalar, double>::value)
			return AVX_vector(_mm256_sub_pd(v, arg.v));
		else if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_sub_ps(v, arg.v));
		else if constexpr (std::is_integral<scalar>::value) {
			if constexpr (sizeof(scalar) == sizeof(int8_t))
				return AVX_vector(_mm256_sub_epi8(v, arg.v));
			else if constexpr (sizeof(scalar) == sizeof(int16_t))
				return AVX_vector(_mm256_sub_epi16(v, arg.v));
			else if constexpr (sizeof(scalar) == sizeof(int32_t))
				return AVX_vector(_mm256_sub_epi32(v, arg.v));
			else if constexpr (sizeof(scalar) == sizeof(int64_t))
				return AVX_vector(_mm256_sub_epi64(v, arg.v));
			else
				static_assert(false, "AVX2 : operator- is not defined in given type.");
		}
		else
			static_assert(false, "AVX2 : operator- is not defined in given type.");
	}
	AVX_vector operator*(const AVX_vector& arg) const {
		if constexpr (std::is_same<scalar, double>::value)
			return AVX_vector(_mm256_mul_pd(v, arg.v));
		else if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_mul_ps(v, arg.v));
		else if constexpr (std::is_integral<scalar>::value) {
			if constexpr (std::is_signed<scalar>::value) {
				if constexpr (sizeof(scalar) == sizeof(int64_t))
					return AVX_vector(_mm256_mul_epi32(v, arg.v));
				else if constexpr (sizeof(scalar) == sizeof(int32_t))
					return AVX_vector(_mm256_mullo_epi32(v, arg.v));
				else if constexpr (sizeof(scalar) == sizeof(int16_t))
					return AVX_vector(_mm256_mullo_epi16(v, arg.v));
				else
					static_assert(false, "AVX2 : operator* is not defined in given type.");
			}
			else {
				if constexpr (sizeof(scalar) == sizeof(int64_t))
					return AVX_vector(_mm256_mul_epu32(v, arg.v));
				else
					static_assert(false, "AVX2 : operator* is not defined in given type.");
			}
		}
		else
			static_assert(false, "AVX2 : operator* is not defined in given type.");
	}
	AVX_vector operator/(const AVX_vector& arg) const {
		if constexpr (std::is_same<scalar, double>::value)
			return AVX_vector(_mm256_div_pd(v, arg.v));
		else if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_div_ps(v, arg.v));
		else
			static_assert(false, "AVX2 : operator/ is not defined in given type.");
	}
	AVX_vector& operator=(const scalar arg) {
		if constexpr (std::is_same<scalar, double>::value)
			v = _mm256_set1_pd(arg);
		else if constexpr (std::is_same<scalar, float>::value)
			v = _mm256_set1_ps(arg);
		else if constexpr (std::is_integral<scalar>::value) {
			if constexpr (sizeof(scalar) == sizeof(int8_t))
				v = _mm256_set1_epi8(arg);
			else if constexpr (sizeof(scalar) == sizeof(int16_t))
				v = _mm256_set1_epi16(arg);
			else if constexpr (sizeof(scalar) == sizeof(int32_t))
				v = _mm256_set1_epi32(arg);
			else if constexpr (sizeof(scalar) == sizeof(int64_t))
				v = _mm256_set1_epi64x(arg);
			else
				static_assert(false, "AVX2 : operator=(scalar) is not defined in given type.");
		}
		else
			static_assert(false, "AVX2 : operator=(scalar) is not defined in given type.");
		return *this;
	}
	AVX_vector& operator=(const scalar* const arg) {
		if constexpr (std::is_same<scalar, double>::value)
			v = _mm256_loadu_pd(arg);
		else if constexpr (std::is_same<scalar, float>::value)
			v = _mm256_loadu_ps(arg);
		else if constexpr (std::is_integral<scalar>::value)
			v = _mm256_loadu_si256(arg);
		// v = _mm256_lddqu_si256(arg);
		else
			static_assert(false, "AVX2 : operator=(pointer) is not defined in given type.");
		return *this;
	}
	AVX_vector& operator<<(const scalar* const arg) {
		return operator=(arg);
	}
	scalar* operator>>(scalar* arg) const {
		if constexpr (std::is_same<scalar, double>::value)
			_mm256_storeu_pd(arg, v);
		else if constexpr (std::is_same<scalar, float>::value)
			_mm256_storeu_ps(arg, v);
		else if constexpr (std::is_integral<scalar>::value)
			_mm256_storeu_si256(arg, v);
		else
			static_assert(false, "AVX2 : operator>>(pointer) is not defined in given type.");
		return arg;
	}
	scalar operator[](const int index) const {
		if constexpr (std::is_same<scalar, double>::value) {
			double tmp[4];
			_mm256_store_pd(tmp, v);
			return tmp[index];
		}
		else if constexpr (std::is_same<scalar, float>::value)
			return _mm256_cvtss_f32(_mm256_permutevar8x32_ps(v, _mm256_set1_epi32(index)));
		else if constexpr (std::is_integral<scalar>::value) {
			if constexpr (sizeof(scalar) == sizeof(int8_t))
				return _mm256_extract_epi8(v, index);
			else if constexpr (sizeof(scalar) == sizeof(int16_t))
				return _mm256_extract_epi16(v, index);
			else if constexpr (sizeof(scalar) == sizeof(int32_t))
				return _mm256_extract_epi32(v, index);
			else if constexpr (sizeof(scalar) == sizeof(int64_t))
				return _mm256_extract_epi64(v, index);
			else
				static_assert(false, "AVX2 : operator[] is not defined in given type.");
		}
		else
			static_assert(false, "AVX2 : operator[] is not defined in given type.");

	}
	AVX_vector operator==(const AVX_vector& arg) const {
		if constexpr (std::is_same<scalar, double>::value)
			return AVX_vector(_mm256_cmp_pd(v, arg.v, _CMP_EQ_OQ));
		else if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_cmp_ps(v, arg.v, _CMP_EQ_OQ));
		else if constexpr (std::is_integral<scalar>::value) {
			if constexpr (sizeof(scalar) == sizeof(int8_t))
				return AVX_vector(_mm256_cmpeq_epi8(v, arg.v));
			else if constexpr (sizeof(scalar) == sizeof(int16_t))
				return AVX_vector(_mm256_cmpeq_epi16(v, arg.v));
			else if constexpr (sizeof(scalar) == sizeof(int32_t))
				return AVX_vector(_mm256_cmpeq_epi32(v, arg.v));
			else if constexpr (sizeof(scalar) == sizeof(int64_t))
				return AVX_vector(_mm256_cmpeq_epi64(v, arg.v));
			else
				static_assert(false, "AVX2 : operator== is not defined in given type.");
		}
		else
			static_assert(false, "AVX2 : operator== is not defined in given type.");
	}
	bool is_all_zero() const {
		if constexpr (std::is_integral<scalar>::value)
			return static_cast<bool>(_mm256_testz_si256(v, _mm256_cmpeq_epi64(v,v)));
		else
			static_assert(false, "AVX2 : is_all_zero is not defined in given type.");
	}
	bool is_all_one() const {
		if constexpr (std::is_integral<scalar>::value)
			return static_cast<bool>(_mm256_testc_si256(v, _mm256_cmpeq_epi64(v, v)));
		else
			static_assert(false, "AVX2 : is_all_one is not defined in given type.");
	}
	AVX_vector operator>(const AVX_vector& arg) const {
		if constexpr (std::is_same<scalar, double>::value)
			return AVX_vector(_mm256_cmp_pd(v, arg.v, _CMP_GT_OQ));
		else if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_cmp_ps(v, arg.v, _CMP_GT_OQ));
		else if constexpr (std::is_integral<scalar>::value) {
			if constexpr (sizeof(scalar) == sizeof(int8_t))
				return AVX_vector(_mm256_cmpgt_epi8(v, arg.v));
			else if constexpr (sizeof(scalar) == sizeof(int16_t))
				return AVX_vector(_mm256_cmpgt_epi16(v, arg.v));
			else if constexpr (sizeof(scalar) == sizeof(int32_t))
				return AVX_vector(_mm256_cmpgt_epi32(v, arg.v));
			else if constexpr (sizeof(scalar) == sizeof(int64_t))
				return AVX_vector(_mm256_cmpgt_epi64(v, arg.v));
			else
				static_assert(false, "AVX2 : operator> is not defined in given type.");
		}
		else
			static_assert(false, "AVX2 : operator> is not defined in given type.");
	}
	AVX_vector operator<(const AVX_vector& arg) const {
		if constexpr (std::is_same<scalar, double>::value)
			return AVX_vector(_mm256_cmp_pd(arg.v, v, _CMP_GT_OQ));
		else if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_cmp_ps(arg.v, v, _CMP_GT_OQ));
		else if constexpr (std::is_integral<scalar>::value) {
			if constexpr (sizeof(scalar) == sizeof(int8_t))
				return AVX_vector(_mm256_cmpgt_epi8(arg.v, v));
			else if constexpr (sizeof(scalar) == sizeof(int16_t))
				return AVX_vector(_mm256_cmpgt_epi16(arg.v, v));
			else if constexpr (sizeof(scalar) == sizeof(int32_t))
				return AVX_vector(_mm256_cmpgt_epi32(arg.v, v));
			else if constexpr (sizeof(scalar) == sizeof(int64_t))
				return AVX_vector(_mm256_cmpgt_epi64(arg.v, v));
			else
				static_assert(false, "AVX2 : operator< is not defined in given type.");
		}
		else
			static_assert(false, "AVX2 : operator< is not defined in given type.");
	}
	AVX_vector operator& (const AVX_vector& arg) const {
		if constexpr (std::is_same<scalar, double>::value)
			return AVX_vector(_mm256_and_pd(v, arg.v));
		else if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_and_ps(v, arg.v));
		else if constexpr (std::is_integral<scalar>::value)
			return AVX_vector(_mm256_and_si256(v, arg.v));
		else
			static_assert(false, "AVX2 : and is not defined in given type.");
	}
	AVX_vector nand(const AVX_vector& arg) const {
		if constexpr (std::is_same<scalar, double>::value)
			return AVX_vector(_mm256_andnot_pd(v, arg.v));
		else if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_andnot_ps(v, arg.v));
		else if constexpr (std::is_integral<scalar>::value)
			return AVX_vector(_mm256_andnot_si256(v, arg.v));
		else
			static_assert(false, "AVX2 : nand is not defined in given type.");
	}
	AVX_vector operator~() const {
		if constexpr (std::is_same<scalar, double>::value)
			return AVX_vector(_mm256_andnot_pd(v, v));
		else if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_andnot_ps(v, v));
		else if constexpr (std::is_integral<scalar>::value)
			return AVX_vector(_mm256_andnot_si256(v, v));
		else
			static_assert(false, "AVX2 : not is not defined in given type.");
	}
	AVX_vector operator| (const AVX_vector& arg) const {
		if constexpr (std::is_same<scalar, double>::value)
			return AVX_vector(_mm256_or_pd(v, arg.v));
		else if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_or_ps(v, arg.v));
		else if constexpr (std::is_integral<scalar>::value)
			return AVX_vector(_mm256_or_si256(v, arg.v));
		else
			static_assert(false, "AVX2 : or is not defined in given type.");
	}
	AVX_vector operator^ (const AVX_vector& arg) const {
		if constexpr (std::is_same<scalar, double>::value)
			return AVX_vector(_mm256_xor_pd(v, arg.v));
		else if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_xor_ps(v, arg.v));
		else if constexpr (std::is_integral<scalar>::value)
			return AVX_vector(_mm256_xor_si256(v, arg.v));
		else
			static_assert(false, "AVX2 : xor is not defined in given type.");
	}
	// Reciprocal
	AVX_vector rcp() const {
		if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_rcp_ps(v));
		else
			static_assert(false, "AVX2 : rcp is not defined in given type.");
	}
	// this * (1 / arg)
	AVX_vector fast_div(const AVX_vector& arg) const {
		if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_mul_ps(v, _mm256_rcp_ps(arg.v)));
		else
			static_assert(false, "AVX2 : fast_div is not defined in given type.");
	}
	AVX_vector abs() const {
		if constexpr (std::is_same<scalar, double>::value)
			return AVX_vector(_mm256_andnot_pd(_mm256_set1_pd(-0.0), v));
		else if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_andnot_ps(_mm256_set1_ps(-0.0f), v));
		else if constexpr (std::is_integral<scalar>::value&& std::is_signed<scalar>::value) {
			if constexpr (sizeof(scalar) == sizeof(int8_t))
				return AVX_vector(_mm256_abs_epi8(v));
			else if constexpr (sizeof(scalar) == sizeof(int16_t))
				return AVX_vector(_mm256_abs_epi16(v));
			else if constexpr (sizeof(scalar) == sizeof(int32_t))
				return AVX_vector(_mm256_abs_epi32(v));
			else if constexpr (sizeof(scalar) == sizeof(int64_t)) {
				vector mask = _mm256_cmpgt_epi64(_mm256_setzero_si256(), v);
				return AVX_vector(_mm256_sub_epi64(_mm256_xor_si256(v, mask), mask));
			}
			else
				static_assert(false, "AVX2 : abs is not defined in given type.");
		}
		else
			static_assert(false, "AVX2 : abs is not defined in given type.");
	}
	AVX_vector sqrt() const {
		if constexpr (std::is_same<scalar, double>::value)
			return AVX_vector(_mm256_sqrt_pd(v));
		else if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_sqrt_ps(v));
		else
			static_assert(false, "AVX2 : sqrt is not defined in given type.");
	}
	// 1 / sqrt()
	AVX_vector rsqrt() const {
		if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_rsqrt_ps(v));
		else
			static_assert(false, "AVX2 : rsqrt is not defined in given type.");
	}
	AVX_vector max(const AVX_vector& arg) const {
		if constexpr (std::is_same<scalar, double>::value)
			return AVX_vector(_mm256_max_pd(v, arg.v));
		else if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_max_ps(v, arg.v));
		else if constexpr (std::is_integral<scalar>::value) {
			if constexpr (std::is_signed<scalar>::value) {
				if constexpr (sizeof(scalar) == sizeof(int8_t))
					return AVX_vector(_mm256_max_epi8(v, arg.v));
				else if constexpr (sizeof(scalar) == sizeof(int16_t))
					return AVX_vector(_mm256_max_epi16(v, arg.v));
				else if constexpr (sizeof(scalar) == sizeof(int32_t))
					return AVX_vector(_mm256_max_epi32(v, arg.v));
				else if constexpr (sizeof(scalar) == sizeof(int64_t))
					return AVX_vector(_mm256_blendv_epi8(arg.v, v, _mm256_cmpgt_epi64(v, arg.v)));
				else
					static_assert(false, "AVX2 : max is not defined in given type.");
			}
			else {
				if constexpr (sizeof(scalar) == sizeof(int8_t))
					return AVX_vector(_mm256_max_epu8(v, arg.v));
				else if constexpr (sizeof(scalar) == sizeof(int16_t))
					return AVX_vector(_mm256_max_epu16(v, arg.v));
				else if constexpr (sizeof(scalar) == sizeof(int32_t))
					return AVX_vector(_mm256_max_epu32(v, arg.v));
				else
					static_assert(false, "AVX2 : max is not defined in given type.");
			}
		}
		else
			static_assert(false, "AVX2 : max is not defined in given type.");
	}
	AVX_vector min(const AVX_vector& arg) const {
		if constexpr (std::is_same<scalar, double>::value)
			return AVX_vector(_mm256_min_pd(v, arg.v));
		else if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_min_ps(v, arg.v));
		else if constexpr (std::is_integral<scalar>::value) {
			if constexpr (std::is_signed<scalar>::value) {
				if constexpr (sizeof(scalar) == sizeof(int8_t))
					return AVX_vector(_mm256_min_epi8(v, arg.v));
				else if constexpr (sizeof(scalar) == sizeof(int16_t))
					return AVX_vector(_mm256_min_epi16(v, arg.v));
				else if constexpr (sizeof(scalar) == sizeof(int32_t))
					return AVX_vector(_mm256_min_epi32(v, arg.v));
				else if constexpr (sizeof(scalar) == sizeof(int64_t))
					return AVX_vector(_mm256_blendv_epi8(v, arg.v, _mm256_cmpgt_epi64(v, arg.v)));
				else
					static_assert(false, "AVX2 : min is not defined in given type.");
			}
			if constexpr (std::is_unsigned<scalar>::value) {
				if constexpr (sizeof(scalar) == sizeof(int8_t))
					return AVX_vector(_mm256_min_epu8(v, arg.v));
				else if constexpr (sizeof(scalar) == sizeof(int16_t))
					return AVX_vector(_mm256_min_epu16(v, arg.v));
				else if constexpr (sizeof(scalar) == sizeof(int32_t))
					return AVX_vector(_mm256_min_epu32(v, arg.v));
				else
					static_assert(false, "AVX2 : min is not defined in given type.");
			}
		}
		else
			static_assert(false, "AVX2 : min is not defined in given type.");
	}
	AVX_vector ceil() const {
		if constexpr (std::is_same<scalar, double>::value)
			return AVX_vector(_mm256_ceil_pd(v));
		else if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_ceil_ps(v));
		else
			static_assert(false, "AVX2 : ceil is not defined in given type.");
	}
	AVX_vector floor() const {
		if constexpr (std::is_same<scalar, double>::value)
			return AVX_vector(_mm256_floor_pd(v));
		else if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_floor_ps(v));
		else
			static_assert(false, "AVX2 : floor is not defined in given type.");
	}
	// this * a + b
	AVX_vector muladd(const AVX_vector& a, const AVX_vector& b) const {
		if constexpr (std::is_same<scalar, double>::value)
			return AVX_vector(_mm256_fmadd_pd(v, a.v, b.v));
		else if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_fmadd_ps(v, a.v, b.v));
		else
			static_assert(false, "AVX2 : mulladd is not defined in given type.");
	}
	// this * a - b
	AVX_vector mulsub(const AVX_vector& a, const AVX_vector& b) const {
		if constexpr (std::is_same<scalar, double>::value)
			return AVX_vector(_mm256_fmsub_pd(v, a.v, b.v));
		else if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_fmsub_ps(v, a.v, b.v));
		else
			static_assert(false, "AVX2 : mullsub is not defined in given type.");
	}
	// { this[0] + this[1], arg[0] + arg[1], this[2] + this[3], ... }
	AVX_vector hadd(const AVX_vector& arg) const{
		if constexpr (std::is_same<scalar, double>::value)
			return AVX_vector(_mm256_hadd_pd(v, arg.v));
		else if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_hadd_ps(v, arg.v));
		else if constexpr (std::is_integral<scalar>::value && std::is_signed<scalar>::value) {
			if constexpr (sizeof(scalar) == sizeof(int16_t))
				return AVX_vector(_mm256_hadd_epi16(v. arg.v));
			else if constexpr (sizeof(scalar) == sizeof(int32_t))
				return AVX_vector(_mm256_hadd_epi32(v, arg.v));
			else
				static_assert(false, "AVX2 : hadd is not defined in given type.");
		}
		else
			static_assert(false, "AVX2 : hadd is not defined in given type.");
	}
	// (mask) ? this : a
	AVX_vector cmp_blend(const AVX_vector& a, const AVX_vector& mask) const {
		if constexpr (std::is_same<scalar, double>::value)
			return AVX_vector(_mm256_blendv_pd(a.v, v, mask.v));
		else if constexpr (std::is_same<scalar, float>::value)
			return AVX_vector(_mm256_blendv_ps(a.v, v, mask.v));
		else if constexpr (std::is_integral<scalar>::value)
			return AVX_vector(_mm256_blendv_epi8(a.v, v, mask.v));
		else
			static_assert(false, "AVX2 : cmp_blend is not defined in given type.");
	}
	template<typename cvt>
	explicit operator AVX_vector<cvt>() const {
		if constexpr (std::is_same<scalar, float>::value&& std::is_same<cvt, int32_t>::value)
			return AVX_vector<cvt>(_mm256_cvtps_epi32(v));
		else if constexpr (std::is_same<scalar, int32_t>::value&& std::is_same<cvt, float>::value)
			return AVX_vector<cvt>(_mm256_cvtepi32_ps(v));
		else
			static_assert(false, "AVX2 : type casting is not defined in given type.");
	}
};

namespace function {
	// max(a, b)
	template<typename Scalar>
	AVX_vector<Scalar> max(const AVX_vector<Scalar>& a, const AVX_vector<Scalar>& b) {
		return a.max(b);
	}
	// min(a, b)
	template<typename Scalar>
	AVX_vector<Scalar> min(const AVX_vector<Scalar>& a, const AVX_vector<Scalar>& b) {
		return a.min(b);
	}
	// (==) ? a : b
	template<typename Scalar>
	AVX_vector<Scalar> cmp_blend(const AVX_vector<Scalar>& mask, const AVX_vector<Scalar>& a, const AVX_vector<Scalar>& b) {
		return a.cmp_blend(b, mask);
	}
	// a * b + c
	template<typename Scalar>
	AVX_vector<Scalar> muladd(const AVX_vector<Scalar>& a, const AVX_vector<Scalar>& b, const AVX_vector<Scalar>& c) {
		return a.muladd(b, c);
	}
	// a * b - c
	template<typename Scalar>
	AVX_vector<Scalar> mulsub(const AVX_vector<Scalar>& a, const AVX_vector<Scalar>& b, const AVX_vector<Scalar>& c) {
		return a.mulsub(b, c);
	}
	// { a[0]+a[1], b[0]+b[1], a[2]+a[3], b[2]+b[3], ...}
	template<typename Scalar>
	AVX_vector<Scalar> hadd(const AVX_vector<Scalar>& a, const AVX_vector<Scalar>& b) {
		return a.hadd(b);
	}
}