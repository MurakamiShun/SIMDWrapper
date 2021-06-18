#pragma once
#if __cplusplus < 201703L
#error C++17 is required.
#else

#include "SIMDWrapper/SSEWrapper.hpp"
#include "SIMDWrapper/AVX2Wrapper.hpp"
#include "SIMDWrapper/NEONWrapper.hpp"

namespace SIMDWrapper {
	constexpr inline bool enabled_simd128 = 
	#if defined(ENABLED_SIMD128)
	true;
	#else
	false;
	#endif

	constexpr inline bool enabled_simd256 = 
	#if defined(ENABLED_SIMD256)
	true;
	#else
	false;
	#endif
}

#endif