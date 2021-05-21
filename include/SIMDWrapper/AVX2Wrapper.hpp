#pragma once
#if defined(__AVX2__) && (defined(__x86_64__) || defined(_M_AMD64) || defined(_M_IX86))
#if __cplusplus < 201703L
#error C++17 is required.
#else
#include "SSEWrapper.hpp"

#define ENABLED_SIMD256

template<typename Scalar>
struct vector256_type {
	template<typename T, typename... List>
	using is_any = std::disjunction<std::is_same<T, List>...>;

	static_assert(is_any<Scalar, float, double>::value || std::is_integral_v<Scalar>, "AVX2 : Given type is not supported.");

	struct m256_wrapper{ using type = __m256; };
	struct m256i_wrapper{ using type = __m256i; };
	struct m256d_wrapper{ using type = __m256d; };
	struct false_type{ using type = std::false_type; };

	using scalar = Scalar;
	using vector = typename std::conditional_t<std::is_same_v<Scalar, double>, m256d_wrapper,
			typename std::conditional_t< std::is_same_v<Scalar, float>, m256_wrapper,
			typename std::conditional_t< std::is_integral_v<Scalar>, m256i_wrapper,
			false_type>>>::type;

	static constexpr size_t elements_size = 32 / sizeof(Scalar);
};

template<typename Scalar>
class vector256 {
private:
	using scalar = typename vector256_type<Scalar>::scalar;
	using vector = typename vector256_type<Scalar>::vector;
	static constexpr size_t elements_size = vector256_type<Scalar>::elements_size;

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
			static_assert(is_right_args, "AVX2 : wrong number of arguments (expected 4).");
			v = _mm256_set_pd(std::get<N - 1 - I>(args_tuple)..., last);
		}
		else if constexpr (is_scalar_v<float>) {
			static_assert(is_right_args, "AVX2 : wrong number of arguments (expected 8).");
			v = _mm256_set_ps(std::get<N - 1 - I>(args_tuple)..., last);
		}
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (is_scalar_size_v<int8_t>) {
				static_assert(is_right_args, "AVX2 : wrong number of arguments (expected 32).");
				v = _mm256_set_epi8(std::get<N - 1 - I>(args_tuple)..., last);
			}
			else if constexpr (is_scalar_size_v<int16_t>) {
				static_assert(is_right_args, "AVX2 : wrong number of arguments (expected 16).");
				v = _mm256_set_epi16(std::get<N - 1 - I>(args_tuple)..., last);
			}
			else if constexpr (is_scalar_size_v<int32_t>) {
				static_assert(is_right_args, "AVX2 : wrong number of arguments (expected 8).");
				v = _mm256_set_epi32(std::get<N - 1 - I>(args_tuple)..., last);
			}
			else if constexpr (is_scalar_size_v<int64_t>) {
				static_assert(is_right_args, "AVX2 : wrong number of arguments (expected 4).");
				v = _mm256_set_epi64x(std::get<N - 1 - I>(args_tuple)..., last);
			}
			else
				static_assert(false_v<Scalar>, "AVX2 : initializer is not defined in given type.");
		}
		else
			static_assert(false_v<Scalar>, "AVX2 : initializer is not defined in given type.");
	}

	class input_iterator {
	private:
		std::array<scalar, elements_size> tmp = {};
		size_t index;
	public:
		template<size_t N>
		struct Index {};

		input_iterator(const input_iterator& it) noexcept :
			index(it.index),
			tmp(it.tmp) {
		}
		template<size_t N>
		input_iterator(const vector256& arg, Index<N>) noexcept {
			index = N;
			if constexpr (N >= 0 && N < elements_size)
				arg.aligned_store(tmp.data());
		}
		const scalar operator*() const noexcept{
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
	static constexpr scalar truthy = [](){
		if constexpr (is_scalar_v<double>)
			return -std::numeric_limits<double>::quiet_NaN();
		else if constexpr (is_scalar_v<float>)
			return -std::numeric_limits<float>::quiet_NaN();
		else if constexpr (std::is_integral_v<scalar>)
			return static_cast<scalar>(-1);
		else
			static_assert(false_v<Scalar>, "vector256 is not defined in given type.");
	}();
	static constexpr scalar falsy = [](){
		if constexpr (is_scalar_v<double>)
			return 0.0;
		else if constexpr (is_scalar_v<float>)
			return 0.0f;
		else if constexpr (std::is_integral_v<scalar>)
			return 0;
		else
			static_assert(false_v<Scalar>, "vector256 is not defined in given type.");
	}();

	vector v;

	vector256() noexcept : v() {}
	vector256(const scalar arg) noexcept { *this = arg; }
	vector256(const vector arg) noexcept : v(arg) {  }
	template<class... Args, typename Indices = std::make_index_sequence<sizeof...(Args)>>
	vector256(scalar first, Args... args) noexcept {
		init_by_reversed_argments(Indices(), first, std::forward<Args>(args)...);
	}
	vector256(const vector256& arg) noexcept : v(arg.v) {  }

	input_iterator begin() const noexcept {
		return input_iterator(*this, typename input_iterator::template Index<0>());
	}
	input_iterator end() const noexcept {
		return input_iterator(*this, typename input_iterator::template Index<elements_size>());
	}

	vector256 operator+(const vector256& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_add_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_add_ps(v, arg.v));
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (is_scalar_size_v<int8_t>)
				return vector256(_mm256_add_epi8(v, arg.v));
			else if constexpr (is_scalar_size_v<int16_t>)
				return vector256(_mm256_add_epi16(v, arg.v));
			else if constexpr (is_scalar_size_v<int32_t>)
				return vector256(_mm256_add_epi32(v, arg.v));
			else if constexpr (is_scalar_size_v<int64_t>)
				return vector256(_mm256_add_epi64(v, arg.v));
			else
				static_assert(false_v<Scalar>, "AVX2 : operator+ is not defined in given type.");
		}
		else
			static_assert(false_v<Scalar>, "AVX2 : operator+ is not defined in given type.");
	}
	vector256 operator-(const vector256& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_sub_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_sub_ps(v, arg.v));
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (is_scalar_size_v<int8_t>)
				return vector256(_mm256_sub_epi8(v, arg.v));
			else if constexpr (is_scalar_size_v<int16_t>)
				return vector256(_mm256_sub_epi16(v, arg.v));
			else if constexpr (is_scalar_size_v<int32_t>)
				return vector256(_mm256_sub_epi32(v, arg.v));
			else if constexpr (is_scalar_size_v<int64_t>)
				return vector256(_mm256_sub_epi64(v, arg.v));
			else
				static_assert(false_v<Scalar>, "AVX2 : operator- is not defined in given type.");
		}
		else
			static_assert(false_v<Scalar>, "AVX2 : operator- is not defined in given type.");
	}
	auto operator*(const vector256& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_mul_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_mul_ps(v, arg.v));
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (std::is_signed_v<scalar>) {
				if constexpr (is_scalar_size_v<int32_t>)
					return vector256<int64_t>(_mm256_mul_epi32(v, arg.v));
				else
					static_assert(false_v<Scalar>, "AVX2 : operator* is not defined in given type.");
			}
			else {
				if constexpr (is_scalar_size_v<int32_t>)
					return vector256<uint64_t>(_mm256_mul_epu32(v, arg.v));
				else
					static_assert(false_v<Scalar>, "AVX2 : operator* is not defined in given type.");
			}
		}
		else
			static_assert(false_v<Scalar>, "AVX2 : operator* is not defined in given type.");
	}
	vector256 operator/(const vector256& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_div_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_div_ps(v, arg.v));
		else
			static_assert(false_v<Scalar>, "AVX2 : operator/ is not defined in given type.");
	}
	vector256& operator=(const scalar arg) noexcept {
		if constexpr (is_scalar_v<double>)
			v = _mm256_set1_pd(arg);
		else if constexpr (is_scalar_v<float>)
			v = _mm256_set1_ps(arg);
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (is_scalar_size_v<int8_t>)
				v = _mm256_set1_epi8(arg);
			else if constexpr (is_scalar_size_v<int16_t>)
				v = _mm256_set1_epi16(arg);
			else if constexpr (is_scalar_size_v<int32_t>)
				v = _mm256_set1_epi32(arg);
			else if constexpr (is_scalar_size_v<int64_t>)
				v = _mm256_set1_epi64x(arg);
			else
				static_assert(false_v<Scalar>, "AVX2 : operator=(scalar) is not defined in given type.");
		}
		else
			static_assert(false_v<Scalar>, "AVX2 : operator=(scalar) is not defined in given type.");
		return *this;
	}
	vector256& load(const scalar* const arg) noexcept {
		if constexpr (is_scalar_v<double>)
			v = _mm256_loadu_pd(arg);
		else if constexpr (is_scalar_v<float>)
			v = _mm256_loadu_ps(arg);
		else if constexpr (std::is_integral_v<scalar>)
			v = _mm256_loadu_si256(reinterpret_cast<const vector*>(arg));
		else
			static_assert(false_v<Scalar>, "AVX2 : load(pointer) is not defined in given type.");
		return *this;
	}
	vector256& aligned_load(const scalar* const arg) noexcept {
		if constexpr (is_scalar_v<double>)
			v = _mm256_load_pd(arg);
		else if constexpr (is_scalar_v<float>)
			v = _mm256_load_ps(arg);
		else if constexpr (std::is_integral_v<scalar>)
			v = _mm256_load_si256(arg);
		else
			static_assert(false_v<Scalar>, "AVX2 : load(pointer) is not defined in given type.");
		return *this;
	}
	void store(scalar* arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			_mm256_storeu_pd(arg, v);
		else if constexpr (is_scalar_v<float>)
			_mm256_storeu_ps(arg, v);
		else if constexpr (std::is_integral_v<scalar>)
			_mm256_storeu_si256(reinterpret_cast<vector*>(arg), v);
		else
			static_assert(false_v<Scalar>, "AVX2 : store(pointer) is not defined in given type.");
	}
	void aligned_store(scalar* arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			_mm256_store_pd(arg, v);
		else if constexpr (is_scalar_v<float>)
			_mm256_store_ps(arg, v);
		else if constexpr (std::is_integral_v<scalar>)
			_mm256_store_si256(reinterpret_cast<vector*>(arg), v);
		else
			static_assert(false_v<Scalar>, "AVX2 : store(pointer) is not defined in given type.");
	}
	scalar operator[](const size_t index) const {
		return reinterpret_cast<const scalar*>(&v)[index];	
	}
	scalar& operator[](const size_t index) {
		return reinterpret_cast<scalar*>(&v)[index];
	}
	vector256 operator==(const vector256& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_cmp_pd(v, arg.v, _CMP_EQ_OQ));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_cmp_ps(v, arg.v, _CMP_EQ_OQ));
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (is_scalar_size_v<int8_t>)
				return vector256(_mm256_cmpeq_epi8(v, arg.v));
			else if constexpr (is_scalar_size_v<int16_t>)
				return vector256(_mm256_cmpeq_epi16(v, arg.v));
			else if constexpr (is_scalar_size_v<int32_t>)
				return vector256(_mm256_cmpeq_epi32(v, arg.v));
			else if constexpr (is_scalar_size_v<int64_t>)
				return vector256(_mm256_cmpeq_epi64(v, arg.v));
			else
				static_assert(false_v<Scalar>, "AVX2 : operator== is not defined in given type.");
		}
		else
			static_assert(false_v<Scalar>, "AVX2 : operator== is not defined in given type.");
	}
	vector256 operator!=(const vector256& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_cmp_pd(v, arg.v, _CMP_NEQ_OQ));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_cmp_ps(v, arg.v, _CMP_NEQ_OQ));
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (is_scalar_size_v<int8_t>)
				return vector256(_mm256_xor_si256(
					_mm256_cmpeq_epi8(v, arg.v),
					_mm256_set1_epi8(-1)
				));
			else if constexpr (is_scalar_size_v<int16_t>)
				return vector256(_mm256_xor_si256(
					_mm256_cmpeq_epi16(v, arg.v),
					_mm256_set1_epi16(-1)
				));
			else if constexpr (is_scalar_size_v<int32_t>)
				return vector256(_mm256_xor_si256(
					_mm256_cmpeq_epi32(v, arg.v),
					_mm256_set1_epi32(-1)
				));
			else if constexpr (is_scalar_size_v<int64_t>)
				return vector256(_mm256_xor_si256(
					_mm256_cmpeq_epi64(v, arg.v),
					_mm256_set1_epi64x(-1)
				));
			else
				static_assert(false_v<Scalar>, "AVX2 : operator!= is not defined in given type.");
		}
		else
			static_assert(false_v<Scalar>, "AVX2 : operator!= is not defined in given type.");
	}
	vector256 operator>(const vector256& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_cmp_pd(v, arg.v, _CMP_GT_OQ));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_cmp_ps(v, arg.v, _CMP_GT_OQ));
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (std::is_signed_v<scalar>) {
				if constexpr (is_scalar_size_v<int8_t>)
					return vector256(_mm256_cmpgt_epi8(v, arg.v));
				else if constexpr (is_scalar_size_v<int16_t>)
					return vector256(_mm256_cmpgt_epi16(v, arg.v));
				else if constexpr (is_scalar_size_v<int32_t>)
					return vector256(_mm256_cmpgt_epi32(v, arg.v));
				else if constexpr (is_scalar_size_v<int64_t>)
					return vector256(_mm256_cmpgt_epi64(v, arg.v));
				else
					static_assert(false_v<Scalar>, "AVX2 : operator> is not defined in given type.");
			}
			else {
				if constexpr (is_scalar_size_v<int8_t>)
					return vector256(_mm256_cmpgt_epi8(
						_mm256_xor_si256(v, _mm256_set1_epi8(INT8_MIN)),
						_mm256_xor_si256(arg.v, _mm256_set1_epi8(INT8_MIN))
					));
				else if constexpr (is_scalar_size_v<int16_t>)
					return vector256(_mm256_cmpgt_epi16(
						_mm256_xor_si256(v, _mm256_set1_epi16(INT16_MIN)),
						_mm256_xor_si256(arg.v, _mm256_set1_epi16(INT16_MIN))
					));
				else if constexpr (is_scalar_size_v<int32_t>)
					return vector256(_mm256_cmpgt_epi32(
						_mm256_xor_si256(v, _mm256_set1_epi32(INT32_MIN)),
						_mm256_xor_si256(arg.v, _mm256_set1_epi32(INT32_MIN))
					));
				else if constexpr (is_scalar_size_v<int64_t>)
					return vector256(_mm256_cmpgt_epi64(
						_mm256_xor_si256(v, _mm256_set1_epi64x(INT64_MIN)),
						_mm256_xor_si256(arg.v, _mm256_set1_epi64x(INT64_MIN))
					));
				else
					static_assert(false_v<Scalar>, "AVX2 : operator> is not defined in given type.");
			}
		}
		else
			static_assert(false_v<Scalar>, "AVX2 : operator> is not defined in given type.");
	}
	vector256 operator<(const vector256& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_cmp_pd(arg.v, v, _CMP_GT_OQ));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_cmp_ps(arg.v, v, _CMP_GT_OQ));
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (std::is_signed_v<scalar>) {
				if constexpr (is_scalar_size_v<int8_t>)
					return vector256(_mm256_cmpgt_epi8(arg.v, v));
				else if constexpr (is_scalar_size_v<int16_t>)
					return vector256(_mm256_cmpgt_epi16(arg.v, v));
				else if constexpr (is_scalar_size_v<int32_t>)
					return vector256(_mm256_cmpgt_epi32(arg.v, v));
				else if constexpr (is_scalar_size_v<int64_t>)
					return vector256(_mm256_cmpgt_epi64(arg.v, v));
				else
					static_assert(false_v<Scalar>, "AVX2 : operator< is not defined in given type.");
			}
			else {
				if constexpr (is_scalar_size_v<int8_t>)
					return vector256(_mm256_cmpgt_epi8(
						_mm256_xor_si256(arg.v, _mm256_set1_epi8(INT8_MIN)),
						_mm256_xor_si256(v, _mm256_set1_epi8(INT8_MIN))
					));
				else if constexpr (is_scalar_size_v<int16_t>)
					return vector256(_mm256_cmpgt_epi16(
						_mm256_xor_si256(arg.v, _mm256_set1_epi16(INT16_MIN)),
						_mm256_xor_si256(v, _mm256_set1_epi16(INT16_MIN))
					));
				else if constexpr (is_scalar_size_v<int32_t>)
					return vector256(_mm256_cmpgt_epi32(
						_mm256_xor_si256(arg.v, _mm256_set1_epi32(INT32_MIN)),
						_mm256_xor_si256(v, _mm256_set1_epi32(INT32_MIN))
					));
				else if constexpr (is_scalar_size_v<int64_t>)
					return vector256(_mm256_cmpgt_epi64(
						_mm256_xor_si256(arg.v, _mm256_set1_epi64x(INT64_MIN)),
						_mm256_xor_si256(v, _mm256_set1_epi64x(INT64_MIN))
					));
				else
					static_assert(false_v<Scalar>, "AVX2 : operator< is not defined in given type.");
			}
		}
		else
			static_assert(false_v<Scalar>, "AVX2 : operator< is not defined in given type.");
	}
	vector256 operator>=(const vector256& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_cmp_pd(v, arg.v, _CMP_GE_OQ));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_cmp_ps(v, arg.v, _CMP_GE_OQ));
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (std::is_signed_v<scalar>) {
				if constexpr (is_scalar_size_v<int8_t>)
					return vector256(_mm256_xor_si256(
						_mm256_cmpgt_epi8(arg.v, v),
						_mm256_set1_epi8(-1)
					));
				else if constexpr (is_scalar_size_v<int16_t>)
					return vector256(_mm256_xor_si256(
						_mm256_cmpgt_epi16(arg.v, v),
						_mm256_set1_epi16(-1)
					));
				else if constexpr (is_scalar_size_v<int32_t>)
					return vector256(_mm256_xor_si256(
						_mm256_cmpgt_epi32(arg.v, v),
						_mm256_set1_epi32(-1)
					));
				else if constexpr (is_scalar_size_v<int64_t>)
					return vector256(_mm256_xor_si256(
						_mm256_cmpgt_epi64(arg.v, v),
						_mm256_set1_epi64x(-1)
					));
				else
					static_assert(false_v<Scalar>, "AVX2 : operator>= is not defined in given type.");
			}
			else {
				if constexpr (is_scalar_size_v<int8_t>)
					return vector256(_mm256_xor_si256(
						_mm256_cmpgt_epi8(
							_mm256_xor_si256(arg.v, _mm256_set1_epi8(INT8_MIN)),
							_mm256_xor_si256(v, _mm256_set1_epi8(INT8_MIN))
						),
						_mm256_set1_epi8(-1)
					));
				else if constexpr (is_scalar_size_v<int16_t>)
					return vector256(_mm256_xor_si256(
						_mm256_cmpgt_epi16(
							_mm256_xor_si256(arg.v, _mm256_set1_epi16(INT16_MIN)),
							_mm256_xor_si256(v, _mm256_set1_epi16(INT16_MIN))
						),
						_mm256_set1_epi16(-1)
					));
				else if constexpr (is_scalar_size_v<int32_t>)
					return vector256(_mm256_xor_si256(
						_mm256_cmpgt_epi32(
							_mm256_xor_si256(arg.v, _mm256_set1_epi32(INT32_MIN)),
							_mm256_xor_si256(v, _mm256_set1_epi32(INT32_MIN))
						),
						_mm256_set1_epi32(-1)
					));
				else if constexpr (is_scalar_size_v<int64_t>)
					return vector256(_mm256_xor_si256(
						_mm256_cmpgt_epi64(
							_mm256_xor_si256(arg.v, _mm256_set1_epi64x(INT64_MIN)),
							_mm256_xor_si256(v, _mm256_set1_epi64x(INT64_MIN))
						),
						_mm256_set1_epi64x(-1)
					));
				else
					static_assert(false_v<Scalar>, "AVX2 : operator>= is not defined in given type.");
			}
		}
		else
			static_assert(false_v<Scalar>, "AVX2 : operator>= is not defined in given type.");
	}
	vector256 operator<=(const vector256& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_cmp_pd(arg.v, v, _CMP_GE_OQ));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_cmp_ps(arg.v, v, _CMP_GE_OQ));
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (std::is_signed_v<scalar>) {
				if constexpr (is_scalar_size_v<int8_t>)
					return vector256(_mm256_xor_si256(
						_mm256_cmpgt_epi8(v, arg.v),
						_mm256_set1_epi8(-1)
					));
				else if constexpr (is_scalar_size_v<int16_t>)
					return vector256(_mm256_xor_si256(
						_mm256_cmpgt_epi16(v, arg.v),
						_mm256_set1_epi16(-1)
					));
				else if constexpr (is_scalar_size_v<int32_t>)
					return vector256(_mm256_xor_si256(
						_mm256_cmpgt_epi32(v, arg.v),
						_mm256_set1_epi32(-1)
					));
				else if constexpr (is_scalar_size_v<int64_t>)
					return vector256(_mm256_xor_si256(
						_mm256_cmpgt_epi64(v, arg.v),
						_mm256_set1_epi64x(-1)
					));
				else
					static_assert(false_v<Scalar>, "AVX2 : operator<= is not defined in given type.");
			}
			else {
				if constexpr (is_scalar_size_v<int8_t>)
					return vector256(_mm256_xor_si256(
						_mm256_cmpgt_epi8(
							_mm256_xor_si256(v, _mm256_set1_epi8(INT8_MIN)),
							_mm256_xor_si256(arg.v, _mm256_set1_epi8(INT8_MIN))
						),
						_mm256_set1_epi8(-1)
					));
				else if constexpr (is_scalar_size_v<int16_t>)
					return vector256(_mm256_xor_si256(
						_mm256_cmpgt_epi16(
							_mm256_xor_si256(v, _mm256_set1_epi16(INT16_MIN)),
							_mm256_xor_si256(arg.v, _mm256_set1_epi16(INT16_MIN))
						),
						_mm256_set1_epi16(-1)
					));
				else if constexpr (is_scalar_size_v<int32_t>)
					return vector256(_mm256_xor_si256(
						_mm256_cmpgt_epi32(
							_mm256_xor_si256(v, _mm256_set1_epi32(INT32_MIN)),
							_mm256_xor_si256(arg.v, _mm256_set1_epi32(INT32_MIN))
						),
						_mm256_set1_epi32(-1)
					));
				else if constexpr (is_scalar_size_v<int64_t>)
					return vector256(_mm256_xor_si256(
						_mm256_cmpgt_epi64(
							_mm256_xor_si256(v, _mm256_set1_epi64x(INT64_MIN)),
							_mm256_xor_si256(arg.v, _mm256_set1_epi64x(INT64_MIN))
						),
						_mm256_set1_epi64x(-1)
					));
				else
					static_assert(false_v<Scalar>, "AVX2 : operator<= is not defined in given type.");
			}
		}
		else
			static_assert(false_v<Scalar>, "AVX2 : operator<= is not defined in given type.");
	}
	vector256 operator&&(const vector256& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_and_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_and_ps(v, arg.v));
		else if constexpr (std::is_integral_v<scalar>)
			return vector256(_mm256_and_si256(v, arg.v));
		else
			static_assert(false_v<Scalar>, "AVX2 : operator&& is not defined in given type.");
	}
	vector256 operator||(const vector256& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_or_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_or_ps(v, arg.v));
		else if constexpr (std::is_integral_v<scalar>)
			return vector256(_mm256_or_si256(v, arg.v));
		else
			static_assert(false_v<Scalar>, "AVX2 : operator|| is not defined in given type.");
	}
	vector256 operator!() const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_xor_pd(v, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_xor_ps(v, _mm256_castsi256_ps(_mm256_set1_epi64x(-1))));
		else if constexpr (std::is_integral_v<scalar>)
			return vector256(_mm256_xor_si256(v, _mm256_set1_epi64x(-1)));
		else
			static_assert(false_v<Scalar>, "AVX2 : operator! is not defined in given type.");
	}
	bool is_all_false() const noexcept {
		if constexpr (is_scalar_v<double>)
			return bool(_mm256_testz_si256(
				_mm256_castpd_si256(v),
				_mm256_set1_epi64x(-1)
			));
		else if constexpr (is_scalar_v<float>)
			return bool(_mm256_testz_si256(
				_mm256_castps_si256(v),
				_mm256_set1_epi64x(-1)
			));
		else if constexpr (std::is_integral_v<scalar>)
			return bool(_mm256_testz_si256(v, _mm256_set1_epi64x(-1)));
		else
			static_assert(false_v<Scalar>, "AVX2 : is_all_zero is not defined in given type.");
	}
	bool is_all_true() const noexcept {
		if constexpr (is_scalar_v<double>)
			return bool(_mm256_testc_si256(
				_mm256_castpd_si256(v),
				_mm256_set1_epi64x(-1)
			));
		else if constexpr (is_scalar_v<float>)
			return bool(_mm256_testc_si256(
				_mm256_castps_si256(v),
				_mm256_set1_epi64x(-1)
			));
		else if constexpr (std::is_integral_v<scalar>)
			return bool(_mm256_testc_si256(v, _mm256_set1_epi64x(-1)));
		else
			static_assert(false_v<Scalar>, "AVX2 : is_all_one is not defined in given type.");
	}
	vector256 operator& (const vector256& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_and_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_and_ps(v, arg.v));
		else if constexpr (std::is_integral_v<scalar>)
			return vector256(_mm256_and_si256(v, arg.v));
		else
			static_assert(false_v<Scalar>, "AVX2 : and is not defined in given type.");
	}
	vector256 operator~() const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_xor_pd(v, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_xor_ps(v, _mm256_castsi256_ps(_mm256_set1_epi64x(-1))));
		else if constexpr (std::is_integral_v<scalar>)
			return vector256(_mm256_xor_si256(v, _mm256_set1_epi64x(-1)));
		else
			static_assert(false_v<Scalar>, "AVX2 : not is not defined in given type.");
	}
	vector256 operator| (const vector256& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_or_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_or_ps(v, arg.v));
		else if constexpr (std::is_integral_v<scalar>)
			return vector256(_mm256_or_si256(v, arg.v));
		else
			static_assert(false_v<Scalar>, "AVX2 : or is not defined in given type.");
	}
	vector256 operator^ (const vector256& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_xor_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_xor_ps(v, arg.v));
		else if constexpr (std::is_integral_v<scalar>)
			return vector256(_mm256_xor_si256(v, arg.v));
		else
			static_assert(false_v<Scalar>, "AVX2 : xor is not defined in given type.");
	}
	vector256 operator>>(const int n) const noexcept {
		if constexpr (std::is_integral_v<scalar>) {
			if constexpr (is_scalar_size_v<int16_t>)
				return vector256(_mm256_srl_epi16(
					v,
					_mm256_castsi256_si128(_mm256_set1_epi64x(n))
				));
			else if constexpr (is_scalar_size_v<int32_t>)
				return vector256(_mm256_srlv_epi32(v, _mm256_set1_epi32(n)));
			else if constexpr (is_scalar_size_v<int64_t>)
				return vector256(_mm256_srlv_epi64(v, _mm256_set1_epi64x(n)));
			else
				static_assert(false_v<Scalar>, "AVX2 : operator>> is not defined in given type.");
		}
		else
			static_assert(false_v<Scalar>, "AVX2 : operator>> is not defined in given type.");
	}
	vector256 operator>>(const vector256& arg) const noexcept {
		if constexpr (std::is_integral_v<scalar>) {
			if constexpr (is_scalar_size_v<int32_t>)
				return vector256(_mm256_srlv_epi32(v, arg.v));
			else if constexpr (is_scalar_size_v<int64_t>)
				return vector256(_mm256_srlv_epi64(v, arg.v));
			else
				static_assert(false_v<Scalar>, "AVX2 : operator>>(vector256) is not defined in given type.");
		}
		else
			static_assert(false_v<Scalar>, "AVX2 : operator>>(vector256) is not defined in given type.");
	}
	vector256 operator<<(const int n) const noexcept {
		if constexpr (std::is_integral_v<scalar>) {
			if constexpr (is_scalar_size_v<int16_t>)
				return vector256(_mm256_sll_epi16(
					v,
					_mm256_castsi256_si128(_mm256_set1_epi64x(n))
				));
			else if constexpr (is_scalar_size_v<int32_t>)
				return vector256(_mm256_sllv_epi32(v, _mm256_set1_epi32(n)));
			else if constexpr (is_scalar_size_v<int64_t>)
				return vector256(_mm256_sllv_epi64(v, _mm256_set1_epi64x(n)));
			else
				static_assert(false_v<Scalar>, "AVX2 : operator<< is not defined in given type.");
		}
		else
			static_assert(false_v<Scalar>, "AVX2 : operator<< is not defined in given type.");
	}
	vector256 operator<<(const vector256& arg) const noexcept {
		if constexpr (std::is_integral_v<scalar>) {
			if constexpr (is_scalar_size_v<int32_t>)
				return vector256(_mm256_sllv_epi32(v, arg.v));
			else if constexpr (is_scalar_size_v<int64_t>)
				return vector256(_mm256_sllv_epi64(v, arg.v));
			else
				static_assert(false_v<Scalar>, "AVX2 : operator<<(vector256) is not defined in given type.");
		}
		else
			static_assert(false_v<Scalar>, "AVX2 : operator<<(vector256) is not defined in given type.");
	}
	// Reciprocal approximation < 1.5*2^12
	vector256 rcp() const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_cvtps_pd(
				_mm_rcp_ps(_mm256_cvtpd_ps(v))
			));
		else if constexpr (is_scalar_v<float>){
			return vector256(_mm256_rcp_ps(v));
		}
		else
			static_assert(false_v<Scalar>, "AVX2 : rcp is not defined in given type.");
	}
	// this * (1 / arg)
	vector256 fast_div(const vector256& arg) const noexcept {
		if constexpr (is_scalar_v<float>)
			return vector256(_mm256_mul_ps(v, _mm256_rcp_ps(arg.v)));
		else
			static_assert(false_v<Scalar>, "AVX2 : fast_div is not defined in given type.");
	}
	vector256 abs() const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_andnot_pd(_mm256_set1_pd(-0.0), v));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_andnot_ps(_mm256_set1_ps(-0.0f), v));
		else if constexpr (std::is_integral_v<scalar>&& std::is_signed_v<scalar>) {
			if constexpr (is_scalar_size_v<int8_t>)
				return vector256(_mm256_abs_epi8(v));
			else if constexpr (is_scalar_size_v<int16_t>)
				return vector256(_mm256_abs_epi16(v));
			else if constexpr (is_scalar_size_v<int32_t>)
				return vector256(_mm256_abs_epi32(v));
			else if constexpr (is_scalar_size_v<int64_t>) {
				vector mask = _mm256_cmpgt_epi64(_mm256_setzero_si256(), v);
				return vector256(_mm256_sub_epi64(_mm256_xor_si256(v, mask), mask));
			}
			else
				static_assert(false_v<Scalar>, "AVX2 : abs is not defined in given type.");
		}
		else
			static_assert(false_v<Scalar>, "AVX2 : abs is not defined in given type.");
	}
	vector256 sqrt() const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_sqrt_pd(v));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_sqrt_ps(v));
		else
			static_assert(false_v<Scalar>, "AVX2 : sqrt is not defined in given type.");
	}
	// 1 / sqrt()
	vector256 rsqrt() const noexcept {
		if constexpr (is_scalar_v<float>)
			return vector256(_mm256_rsqrt_ps(v));
		else
			static_assert(false_v<Scalar>, "AVX2 : rsqrt is not defined in given type.");
	}
	vector256 max(const vector256& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_max_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_max_ps(v, arg.v));
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (std::is_signed_v<scalar>) {
				if constexpr (is_scalar_size_v<int8_t>)
					return vector256(_mm256_max_epi8(v, arg.v));
				else if constexpr (is_scalar_size_v<int16_t>)
					return vector256(_mm256_max_epi16(v, arg.v));
				else if constexpr (is_scalar_size_v<int32_t>)
					return vector256(_mm256_max_epi32(v, arg.v));
				else if constexpr (is_scalar_size_v<int64_t>)
					return vector256(_mm256_blendv_epi8(arg.v, v, _mm256_cmpgt_epi64(v, arg.v)));
				else
					static_assert(false_v<Scalar>, "AVX2 : max is not defined in given type.");
			}
			else {
				if constexpr (is_scalar_size_v<int8_t>)
					return vector256(_mm256_max_epu8(v, arg.v));
				else if constexpr (is_scalar_size_v<int16_t>)
					return vector256(_mm256_max_epu16(v, arg.v));
				else if constexpr (is_scalar_size_v<int32_t>)
					return vector256(_mm256_max_epu32(v, arg.v));
				else if constexpr (is_scalar_size_v<int64_t>)
					return vector256(_mm256_blendv_epi8(
						arg.v,
						v,
						_mm256_cmpgt_epi64(
							_mm256_xor_si256(v, _mm256_set1_epi64x(INT64_MIN)),
							_mm256_xor_si256(arg.v, _mm256_set1_epi64x(INT64_MIN))
						)
					));
				else
					static_assert(false_v<Scalar>, "AVX2 : max is not defined in given type.");
			}
		}
		else
			static_assert(false_v<Scalar>, "AVX2 : max is not defined in given type.");
	}
	vector256 min(const vector256& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_min_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_min_ps(v, arg.v));
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (std::is_signed_v<scalar>) {
				if constexpr (is_scalar_size_v<int8_t>)
					return vector256(_mm256_min_epi8(v, arg.v));
				else if constexpr (is_scalar_size_v<int16_t>)
					return vector256(_mm256_min_epi16(v, arg.v));
				else if constexpr (is_scalar_size_v<int32_t>)
					return vector256(_mm256_min_epi32(v, arg.v));
				else if constexpr (is_scalar_size_v<int64_t>)
					return vector256(_mm256_blendv_epi8(v, arg.v, _mm256_cmpgt_epi64(v, arg.v)));
				else
					static_assert(false_v<Scalar>, "AVX2 : min is not defined in given type.");
			}
			if constexpr (std::is_unsigned_v<scalar>) {
				if constexpr (is_scalar_size_v<int8_t>)
					return vector256(_mm256_min_epu8(v, arg.v));
				else if constexpr (is_scalar_size_v<int16_t>)
					return vector256(_mm256_min_epu16(v, arg.v));
				else if constexpr (is_scalar_size_v<int32_t>)
					return vector256(_mm256_min_epu32(v, arg.v));
				else if constexpr (is_scalar_size_v<int64_t>)
					return vector256(_mm256_blendv_epi8(
						v,
						arg.v,
						_mm256_cmpgt_epi64(
							_mm256_xor_si256(v, _mm256_set1_epi64x(INT64_MIN)),
							_mm256_xor_si256(arg.v, _mm256_set1_epi64x(INT64_MIN))
						)
					));
				else
					static_assert(false_v<Scalar>, "AVX2 : min is not defined in given type.");
			}
		}
		else
			static_assert(false_v<Scalar>, "AVX2 : min is not defined in given type.");
	}
	vector256 ceil() const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_ceil_pd(v));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_ceil_ps(v));
		else
			static_assert(false_v<Scalar>, "AVX2 : ceil is not defined in given type.");
	}
	vector256 floor() const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_floor_pd(v));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_floor_ps(v));
		else
			static_assert(false_v<Scalar>, "AVX2 : floor is not defined in given type.");
	}
	vector256 round() const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_round_pd(v, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
		else
			static_assert(false_v<Scalar>, "AVX2 : round is not defined in given type.");
	}
	// this * a + b
	vector256 muladd(const vector256& a, const vector256& b) const noexcept {
	#ifdef __FMA__
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_fmadd_pd(v, a.v, b.v));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_fmadd_ps(v, a.v, b.v));
		else
			static_assert(false_v<Scalar>, "FMA : muladd is not defined in given type.");
	#else
		return (*this) * a + b;
	#endif
	}
	// this + a * b
	vector256 addmul(const vector256& a, const vector256& b) const noexcept {
	#ifdef __FMA__
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_fmadd_pd(a.v, b.v, v));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_fmadd_ps(a.v, b.v, v));
		else
			static_assert(false_v<Scalar>, "FMA : addmul is not defined in given type.");
	#else
		return (*this) + a * b;
	#endif
	}
	// -(this * a) + b
	vector256 nmuladd(const vector256& a, const vector256& b) const noexcept {
	#ifdef __FMA__
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_fnmadd_pd(v, a.v, b.v));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_fnmadd_ps(v, a.v, b.v));
		else
			static_assert(false_v<Scalar>, "FMA : nmuladd is not defined in given type.");
	#else
		return b - (*this) * a;
	#endif
	}
	// this - a * b
	vector256 submul(const vector256& a, const vector256& b) const noexcept {
	#ifdef __FMA__
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_fnmadd_pd(a.v, b.v, v));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_fnmadd_ps(a.v, b.v, v));
		else
			static_assert(false_v<Scalar>, "FMA : submul is not defined in given type.");
	#else
		return (*this) - a + b;
	#endif
	}
	// this * a - b
	vector256 mulsub(const vector256& a, const vector256& b) const noexcept {
	#ifdef __FMA__
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_fmsub_pd(v, a.v, b.v));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_fmsub_ps(v, a.v, b.v));
		else
			static_assert(false_v<Scalar>, "FMA : mulsub is not defined in given type.");
	#else
		return (*this) * a - b;
	#endif
	}
	// -(this * a) - b
	vector256 nmulsub(const vector256& a, const vector256& b) const noexcept {
	#ifdef __FMA__
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_fnmsub_pd(v, a.v, b.v));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_fnmsub_ps(v, a.v, b.v));
		else
			static_assert(false_v<Scalar>, "FMA : nmullsub is not defined in given type.");
	#else
		return -(*this) * a - b;
	#endif
	}
	// { this[0] + this[1], arg[0] + arg[1], this[2] + this[3], ... }
	vector256 hadd(const vector256& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_hadd_pd(v, arg.v));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_hadd_ps(v, arg.v));
		else if constexpr (std::is_integral_v<scalar>&& std::is_signed_v<scalar>) {
			if constexpr (is_scalar_size_v<int16_t>)
				return vector256(_mm256_hadd_epi16(v, arg.v));
			else if constexpr (is_scalar_size_v<int32_t>)
				return vector256(_mm256_hadd_epi32(v, arg.v));
			else
				static_assert(false_v<Scalar>, "AVX2 : hadd is not defined in given type.");
		}
		else
			static_assert(false_v<Scalar>, "AVX2 : hadd is not defined in given type.");
	}
	// duplicate a lane
	vector256 dup(const size_t idx) const noexcept {
	#ifndef __clang__
		if constexpr (is_scalar_v<double>){
			auto tmp = _mm256_extractf128_pd(v, idx>>1);
			tmp = _mm_shuffle_pd(tmp, tmp, idx&1);
			return vector256(_mm256_setr_m128d(tmp, tmp));	
			// permute is low throughput
			//return vector256(_mm256_permute4x64_pd(v, load_mask));
		}
		else if constexpr (is_scalar_v<float>){
			auto tmp = _mm256_extractf128_ps(v, idx>>2);
			tmp = _mm_shuffle_ps(tmp, tmp, idx&3);
			return vector256(_mm256_setr_m128(tmp, tmp));
		}
		else if constexpr (is_scalar_size_v<int32_t>){
			auto tmp = _mm256_extractf128_si256(v, idx>>2);
			tmp = _mm_shuffle_epi32(tmp, idx&3);
			return vector256(_mm256_setr_m128i(tmp, tmp));
		}
		else return vector256((*this)[idx]);
	#else
		typename vector128_type<scalar>::vector tmp;
		if constexpr (is_scalar_v<double>){
			switch(idx) {
				case 0:
					tmp = _mm256_extractf128_pd(v, 0);
					tmp = _mm_shuffle_pd(tmp, tmp, 0);
					return vector256(_mm256_setr_m128d(tmp, tmp));
				case 1:
					tmp = _mm256_extractf128_pd(v, 0);
					tmp = _mm_shuffle_pd(tmp, tmp, 1);
					return vector256(_mm256_setr_m128d(tmp, tmp));
				case 2:
					tmp = _mm256_extractf128_pd(v, 1);
					tmp = _mm_shuffle_pd(tmp, tmp, 0);
					return vector256(_mm256_setr_m128d(tmp, tmp));
				case 3:
					tmp = _mm256_extractf128_pd(v, 1);
					tmp = _mm_shuffle_pd(tmp, tmp, 1);
					return vector256(_mm256_setr_m128d(tmp, tmp));
				default: return vector256();
			}
		}
		else if constexpr (is_scalar_v<float>){
			switch(idx) {
				case 0:
					tmp = _mm256_extractf128_pd(v, 0);
					tmp = _mm_shuffle_pd(tmp, tmp, 0);
					return vector256(_mm256_setr_m128d(tmp, tmp));
				case 1:
					tmp = _mm256_extractf128_pd(v, 0);
					tmp = _mm_shuffle_pd(tmp, tmp, 1);
					return vector256(_mm256_setr_m128d(tmp, tmp));
				case 2:
					tmp = _mm256_extractf128_pd(v, 0);
					tmp = _mm_shuffle_pd(tmp, tmp, 2);
					return vector256(_mm256_setr_m128d(tmp, tmp));
				case 3:
					tmp = _mm256_extractf128_pd(v, 0);
					tmp = _mm_shuffle_pd(tmp, tmp, 3);
					return vector256(_mm256_setr_m128d(tmp, tmp));
				case 4:
					tmp = _mm256_extractf128_pd(v, 1);
					tmp = _mm_shuffle_pd(tmp, tmp, 0);
					return vector256(_mm256_setr_m128d(tmp, tmp));
				case 5:
					tmp = _mm256_extractf128_pd(v, 1);
					tmp = _mm_shuffle_pd(tmp, tmp, 1);
					return vector256(_mm256_setr_m128d(tmp, tmp));
				case 6:
					tmp = _mm256_extractf128_pd(v, 1);
					tmp = _mm_shuffle_pd(tmp, tmp, 2);
					return vector256(_mm256_setr_m128d(tmp, tmp));
				case 7:
					tmp = _mm256_extractf128_pd(v, 1);
					tmp = _mm_shuffle_pd(tmp, tmp, 3);
					return vector256(_mm256_setr_m128d(tmp, tmp));
				default: return vector256();
			}
		}
		else if constexpr (is_scalar_size_v<int32_t>){
			switch(idx) {
				case 0:
					tmp = _mm256_extractf128_si256(v, 0);
					tmp = _mm_shuffle_epi32(tmp, 0);
					return vector256(_mm256_setr_m128i(tmp, tmp));
				case 1:
					tmp = _mm256_extractf128_si256(v, 0);
					tmp = _mm_shuffle_epi32(tmp, 1);
					return vector256(_mm256_setr_m128i(tmp, tmp));
				case 2:
					tmp = _mm256_extractf128_si256(v, 0);
					tmp = _mm_shuffle_epi32(tmp, 2);
					return vector256(_mm256_setr_m128i(tmp, tmp));
				case 3:
					tmp = _mm256_extractf128_si256(v, 0);
					tmp = _mm_shuffle_epi32(tmp, 3);
					return vector256(_mm256_setr_m128i(tmp, tmp));
				case 4:
					tmp = _mm256_extractf128_si256(v, 1);
					tmp = _mm_shuffle_epi32(tmp, 0);
					return vector256(_mm256_setr_m128i(tmp, tmp));
				case 5:
					tmp = _mm256_extractf128_si256(v, 1);
					tmp = _mm_shuffle_epi32(tmp, 1);
					return vector256(_mm256_setr_m128i(tmp, tmp));
				case 6:
					tmp = _mm256_extractf128_si256(v, 1);
					tmp = _mm_shuffle_epi32(tmp, 2);
					return vector256(_mm256_setr_m128i(tmp, tmp));
				case 7:
					tmp = _mm256_extractf128_si256(v, 1);
					tmp = _mm_shuffle_epi32(tmp, 3);
					return vector256(_mm256_setr_m128i(tmp, tmp));
				default: return vector256();
			}
		}
		else return vector256((*this)[idx]);
	#endif
	}
	// (mask) ? this : a
	template<typename MaskScalar>
	vector256 cmp_blend(const vector256& a, const vector256<MaskScalar>& mask) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_blendv_pd(a.v, v, *reinterpret_cast<const __m256d*>(&(mask.v))));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_blendv_ps(a.v, v, *reinterpret_cast<const __m256*>(&(mask.v))));
		else if constexpr (std::is_integral_v<scalar>)
			return vector256(_mm256_blendv_epi8(a.v, v, *reinterpret_cast<const __m256i*>(&(mask.v))));
		else
			static_assert(false_v<Scalar>, "AVX2 : cmp_blend is not defined in given type.");
	}
	template<typename Cvt>
	explicit operator vector256<Cvt>() const noexcept {
		if constexpr (is_scalar_v<float>&& std::is_same_v<Cvt, int32_t>)
			return vector256<Cvt>(_mm256_cvtps_epi32(v));
		else if constexpr (is_scalar_v<int32_t>&& std::is_same_v<Cvt, float>)
			return vector256<Cvt>(_mm256_cvtepi32_ps(v));
		else
			static_assert(false_v<Scalar>, "AVX2 : type casting is not defined in given type.");
	}
	// reinterpret cast (data will not change)
	template<typename Cvt>
	vector256<Cvt> reinterpret() const noexcept {
		using cvt_vector = typename vector256_type<Cvt>::vector;
		return vector256<Cvt>(*reinterpret_cast<const cvt_vector*>(&v));
	}
	// FP64x4x2 -> FP32x8, { a[0], a[1], .... b[n-1], b[n] }
	auto concat(const vector256& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256<float>(_mm256_set_m128(_mm256_cvtpd_ps(arg.v), _mm256_cvtpd_ps(v)));
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (std::is_signed_v<scalar>) {
				if constexpr (is_scalar_size_v<int16_t>)
					return vector256<int8_t>(_mm256_permute4x64_epi64(
						_mm256_packs_epi16(v, arg.v),
						216
					));
				else if constexpr (is_scalar_size_v<int32_t>)
					return vector256<int16_t>(_mm256_permute4x64_epi64(
						_mm256_packs_epi32(v, arg.v),
						216
					));
				else if constexpr (is_scalar_size_v<int64_t>)
					return vector256<int32_t>(_mm256_sub_epi32(
						_mm256_permutevar8x32_epi32(
							_mm256_or_si256(
								_mm256_and_si256(
									_mm256_add_epi64(v, _mm256_set1_epi64x(INT32_MAX / 2)),
									_mm256_set1_epi64x(UINT32_MAX)
								),
								_mm256_slli_epi64(
									_mm256_add_epi64(arg.v, _mm256_set1_epi64x(INT32_MAX / 2))
									, 32
								)
							),
							_mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0)
						),
						_mm256_set1_epi32(INT32_MAX / 2)
					));
				else
					static_assert(false_v<Scalar>, "AVX2 : concat is not defined in given type.");
			}
			else {
				if constexpr (is_scalar_size_v<int16_t>)
					return vector256<uint8_t>(_mm256_permute4x64_epi64(
						_mm256_add_epi8(
							_mm256_packs_epi16(
								_mm256_sub_epi16(v, _mm256_set1_epi16(UINT8_MAX / 2u + 1)),
								_mm256_sub_epi16(arg.v, _mm256_set1_epi16(UINT8_MAX / 2u + 1))
							),
							_mm256_set1_epi8(UINT8_MAX / 2u + 1)
						),
						216
					));
				else if constexpr (is_scalar_size_v<int32_t>)
					return vector256<uint16_t>(_mm256_permute4x64_epi64(
						_mm256_add_epi16(
							_mm256_packs_epi32(
								_mm256_sub_epi32(v, _mm256_set1_epi32(UINT16_MAX / 2 + 1)),
								_mm256_sub_epi32(arg.v, _mm256_set1_epi32(UINT16_MAX / 2 + 1))
							),
							_mm256_set1_epi16(UINT16_MAX / 2u + 1)
						),
						216
					));
				else if constexpr (is_scalar_size_v<int64_t>)
					return vector256<uint32_t>(_mm256_permutevar8x32_epi32(
						_mm256_or_si256(
							_mm256_and_si256(v, _mm256_set1_epi64x(UINT32_MAX)),
							_mm256_slli_epi64(arg.v, 32)
						),
						_mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0)
					));
				else
					static_assert(false_v<Scalar>, "AVX2 : concat is not defined in given type.");
			}
		}
		else
			static_assert(false_v<Scalar>, "AVX2 : concat is not defined in given type.");
	}
	// FP64x4x2 -> FP32x8, { a[0], b[0], .... a[n], b[n] }
	auto alternate(const vector256& arg) const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256<float>(_mm256_permutevar8x32_ps(
				_mm256_set_m128(
					_mm256_cvtpd_ps(arg.v),
					_mm256_cvtpd_ps(v)
				),
				_mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0)
			));
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (std::is_signed_v<scalar>) {
				if constexpr (is_scalar_size_v<int16_t>)
					return vector256<int8_t>(_mm256_or_si256(
						_mm256_or_si256(
							_mm256_and_si256(v, _mm256_set1_epi16(UINT8_MAX >> 1)),
							_mm256_srli_epi16(v, 8)
						),
						_mm256_or_si256(
							_mm256_srli_epi16(_mm256_slli_epi16(arg.v, 9), 1),
							_mm256_and_si256(arg.v, _mm256_set1_epi16(1U << 15))
						)
					));
				else if constexpr (is_scalar_size_v<int32_t>)
					return vector256<int16_t>(_mm256_or_si256(
						_mm256_or_si256(
							_mm256_and_si256(v, _mm256_set1_epi32(UINT16_MAX >> 1)),
							_mm256_srli_epi32(v, 16)
						),
						_mm256_or_si256(
							_mm256_srli_epi32(_mm256_slli_epi32(arg.v, 17), 1),
							_mm256_and_si256(arg.v, _mm256_set1_epi32(1UL << 31))
						)
					));
				else if constexpr (is_scalar_size_v<int64_t>)
					return vector256<int32_t>(_mm256_or_si256(
						_mm256_or_si256(
							_mm256_and_si256(v, _mm256_set1_epi64x(UINT32_MAX >> 1)),
							_mm256_srli_epi64(v, 32)
						),
						_mm256_or_si256(
							_mm256_srli_epi64(_mm256_slli_epi64(arg.v, 33), 1),
							_mm256_and_si256(arg.v, _mm256_set1_epi64x(1ULL << 63))
						)
					));
				else
					static_assert(false_v<Scalar>, "AVX2 : alternate is not defined in given type.");
			}
			else {
				if constexpr (is_scalar_size_v<int16_t>)
					return vector256<uint8_t>(_mm256_or_si256(
						_mm256_and_si256(v, _mm256_set1_epi16(UINT8_MAX)),
						_mm256_slli_epi16(arg.v, 8)
					));
				else if constexpr (is_scalar_size_v<int32_t>)
					return vector256<uint16_t>(_mm256_or_si256(
						_mm256_and_si256(v, _mm256_set1_epi32(UINT16_MAX)),
						_mm256_slli_epi32(arg.v, 16)
					));
				else if constexpr (is_scalar_size_v<int64_t>)
					return vector256<uint32_t>(_mm256_or_si256(
						_mm256_and_si256(v, _mm256_set1_epi64x(UINT32_MAX)),
						_mm256_slli_epi64(arg.v, 32)
					));
				else
					static_assert(false_v<Scalar>, "AVX2 : alternate is not defined in given type.");
			}
		}
		else
			static_assert(false_v<Scalar>, "AVX2 : alternate is not defined in given type.");
	}
	template<typename ArgScalar>
	vector256 shuffle(vector256<ArgScalar> arg) const noexcept {
		static_assert(is_scalar_size_v<ArgScalar>, "AVX2 : wrong mask is given to shuufle.");

		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_castps_pd(_mm256_permutevar8x32_ps(
				_mm256_castpd_ps(v),
				_mm256_add_epi32(
					_mm256_slli_epi32(
						_mm256_castps_si256(
							_mm256_moveldup_ps(_mm256_castsi256_ps(arg.v))
						),
						1
					),
					_mm256_setr_epi32(0, 1, 0, 1, 0, 1, 0, 1)
				)
			)));
		else if constexpr (is_scalar_v<float>)
			return vector256(_mm256_permutevar8x32_ps(v, arg.v));
		else if constexpr (std::is_integral_v<scalar>) {
			if constexpr (is_scalar_size_v<int8_t>)
				return vector256(_mm256_or_si256(
					// lower
					_mm256_and_si256(
						_mm256_shuffle_epi8(_mm256_permute2f128_si256(v, v, 0), arg.v),
						_mm256_cmpgt_epi8(_mm256_set1_epi8(4), arg.v)
					),
					// upper
					_mm256_and_si256(
						_mm256_shuffle_epi8(_mm256_permute2f128_si256(v, v, 17), arg.v),
						_mm256_cmpgt_epi8(arg.v, _mm256_set1_epi8(4))
					)
				));
			else if constexpr (is_scalar_size_v<int16_t>)
				return vector256(_mm256_or_si256(
					// lower 
					_mm256_and_si256(
						_mm256_srlv_epi32(
							_mm256_permutevar8x32_epi32(
								v,
								_mm256_srai_epi32(arg.v, 1)
							),
							_mm256_slli_epi32(
								_mm256_and_si256(arg.v, _mm256_set1_epi32(1)),
								4
							)
						),
						_mm256_set1_epi32(UINT16_MAX)
					),
					// upper
					_mm256_and_si256(
						_mm256_sllv_epi32(
							_mm256_permutevar8x32_epi32(
								v,
								_mm256_srai_epi32(arg.v, 17)
							),
							_mm256_slli_epi32(
								_mm256_andnot_si256(_mm256_srai_epi32(arg.v, 16), _mm256_set1_epi32(1)),
								4
							)
						),
						_mm256_set1_epi32(~UINT16_MAX)
					)
				));
			else if constexpr (is_scalar_size_v<int32_t>)
				return vector256(_mm256_permutevar8x32_epi32(v, arg.v));
			else if constexpr (is_scalar_size_v<int64_t>)
				return vector256(_mm256_permutevar8x32_epi32(
					v,
					_mm256_add_epi32(
						_mm256_slli_epi32(
							_mm256_castps_si256(
								_mm256_moveldup_ps(_mm256_castsi256_ps(arg.v))
							),
							1
						),
						_mm256_setr_epi32(0, 1, 0, 1, 0, 1, 0, 1)
					)
				));
			else
				static_assert(false_v<Scalar>, "AVX2 : shuffle is not defined in given type.");
		}
		else
			static_assert(false_v<Scalar>, "AVX2 : shuffle is not defined in given type.");
	}
	template<typename... Args>
	vector256 shuffle(Args... args) const noexcept {
		if constexpr (is_scalar_v<double>)
			return shuffle(vector256<uint64_t>(args...));
		else if constexpr (is_scalar_v<float>)
			return shuffle(vector256<uint32_t>(args...));
		else if constexpr (std::is_integral_v<scalar>)
			return shuffle(vector256(args...));
		else
			static_assert(false_v<Scalar>, "AVX2 : shuffle is not defined in given type.");
	}
	vector256 swap_hilo() const noexcept {
		if constexpr (is_scalar_v<double>)
			return vector256(_mm256_setr_m128d(
				_mm256_extractf128_pd(v, 1),
				_mm256_extractf128_pd(v, 0)
			));
		else if constexpr (is_scalar_v<float>){
			return vector256(_mm256_setr_m128(
				_mm256_extractf128_ps(v, 1),
				_mm256_extractf128_ps(v, 0)
			));
			//return vector256(_mm256_permute2f128_ps(v, v, 0b100));
		}
		else if constexpr (std::is_integral_v<scalar>)
			return vector256(_mm256_setr_m128i(
				_mm256_extractf128_si256(v, 1),
				_mm256_extractf128_si256(v, 0)
			));
		else
			static_assert(false_v<Scalar>, "AVX2 : swap_hi_lo is not defined in given type.");
	
	}
	std::string to_str(const std::pair<std::string_view, std::string_view> brancket = print_format::brancket::square, std::string_view delim = print_format::delim::space) const {
		std::ostringstream ss;
		alignas(32) scalar elements[elements_size];
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
std::ostream& operator<<(std::ostream& os, const vector256<Scalar>& v) {
	os << v.to_str();
	return os;
}

namespace function {
	// max(a, b)
	template<typename Scalar>
	vector256<Scalar> max(const vector256<Scalar>& a, const vector256<Scalar>& b) noexcept {
		return a.max(b);
	}
	// min(a, b)
	template<typename Scalar>
	vector256<Scalar> min(const vector256<Scalar>& a, const vector256<Scalar>& b) noexcept {
		return a.min(b);
	}
	// (==) ? a : b
	template<typename MaskScalar, typename Scalar>
	vector256<Scalar> cmp_blend(const vector256<MaskScalar>& mask, const vector256<Scalar>& a, const vector256<Scalar>& b) noexcept {
		return a.cmp_blend(b, mask);
	}
	// a * b + c
	template<typename Scalar>
	vector256<Scalar> muladd(const vector256<Scalar>& a, const vector256<Scalar>& b, const vector256<Scalar>& c) noexcept {
		return a.muladd(b, c);
	}
	// -(a * b) + c
	template<typename Scalar>
	vector256<Scalar> nmuladd(const vector256<Scalar>& a, const vector256<Scalar>& b, const vector256<Scalar>& c) noexcept {
		return a.nmuladd(b, c);
	}
	// a * b - c
	template<typename Scalar>
	vector256<Scalar> mulsub(const vector256<Scalar>& a, const vector256<Scalar>& b, const vector256<Scalar>& c) noexcept {
		return a.mulsub(b, c);
	}
	// -(a * b) - c
	template<typename Scalar>
	vector256<Scalar> nmulsub(const vector256<Scalar>& a, const vector256<Scalar>& b, const vector256<Scalar>& c) noexcept {
		return a.nmulsub(b, c);
	}
	// { a[0]+a[1], b[0]+b[1], a[2]+a[3], b[2]+b[3], ...}
	template<typename Scalar>
	vector256<Scalar> hadd(const vector256<Scalar>& a, const vector256<Scalar>& b) noexcept {
		return a.hadd(b);
	}
	// reinterpret cast (data will not change)
	template<typename Cvt, typename Scalar>
	vector256<Cvt> reinterpret(const vector256<Scalar>& arg) noexcept {
		return arg.template reinterpret<Cvt>();
	}
	// FP64x4x2 -> FP32x8, { a[0], a[1], .... b[n-1], b[n] }
	template<typename Scalar>
	auto concat(const vector256<Scalar>& a, const vector256<Scalar>& b) noexcept {
		return a.concat(b);
	}
	template<typename Scalar>
	vector256<Scalar> concat(const vector128<Scalar>& lo, const vector128<Scalar>& hi) noexcept {
		if constexpr(std::is_same_v<Scalar, double>) return vector256<double>(_mm256_setr_m128d(lo.v, hi.v));
		else if constexpr(std::is_same_v<Scalar, float>) return vector256<float>(_mm256_setr_m128(lo.v, hi.v));
		else return vector256<Scalar>(_mm256_setr_m128i(lo.v, hi.v));
	}
	// FP64x4x2 -> FP32x8, { a[0], b[0], .... a[n], b[n] }
	template<typename Scalar>
	auto alternate(const vector256<Scalar>& a, const vector256<Scalar>& b) noexcept {
		return a.alternate(b);
	}
	std::array<vector256<double>, 4> transpose(const std::array<vector256<double>, 4>& arg) noexcept {
		vector256_type<double>::vector tmp[4] = {
			_mm256_unpacklo_pd(arg[0].v, arg[1].v),
			_mm256_unpackhi_pd(arg[0].v, arg[1].v),
			_mm256_unpacklo_pd(arg[2].v, arg[3].v),
			_mm256_unpackhi_pd(arg[2].v, arg[3].v),
		};
		return {
			_mm256_setr_m128d(
				_mm256_extractf128_pd(tmp[0], 0),
				_mm256_extractf128_pd(tmp[3], 0)
			),
			_mm256_setr_m128d(
				_mm256_extractf128_pd(tmp[1], 0),
				_mm256_extractf128_pd(tmp[2], 0)
			),
			_mm256_setr_m128d(
				_mm256_extractf128_pd(tmp[0], 1),
				_mm256_extractf128_pd(tmp[3], 1)
			),
			_mm256_setr_m128d(
				_mm256_extractf128_pd(tmp[0], 1),
				_mm256_extractf128_pd(tmp[3], 1)
			)
		};
	}
}

#endif
#endif