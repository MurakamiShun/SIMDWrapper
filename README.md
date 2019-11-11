# AVX2Wrapper
AVX2Wrapper is a C++17 header only library for AVX/AVX2/FMA .
# Usage
- code
```c++
#include "AVX2Wrapper.hpp"
#include <iostream>
int main() {
	std::cout << "AVX :" << std::boolalpha << Instruction::AVX() << "\n"
		<< "AVX2:" << Instruction::AVX2() << "\n"
		<< "FMA :" << Instruction::FMA() << std::endl;
	
	vector256<double> d1(3.14, -3.14, 1.73, 1.41);
	vector256<double> d2(8.10, 1.91, 3.30, - 3.33);
	
	vector256<float> f1 = d1.concat(d2.abs()*2);
	vector256<int32_t> source(0, 1, 2, 3, 4, 5, 6, 7);
	
	auto f_source = static_cast<vector256<float>>(source);
	
	std::cout << function::cmp_blend(f1 > f_source, f_source, f1+3) << std::endl;
	
	return 0;
}
```
- result
```
AVX :true
AVX2:true
FMA :true
[0 -0.14 4.73 4.41 4 6.82 6 9.66]
```

# License
This software is released under the MIT License, see [LICENSE](https://github.com/MurakamiShun/AVX2Wrapper/blob/master/LICENSE).