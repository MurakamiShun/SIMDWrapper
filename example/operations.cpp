#include "../AVX2Wrapper.hpp"
#include <iostream>

int mai() {
	
	vector256<float> f1(4.5f);
	vector256<float> f2(0,1,2,3,4,5,6,7);
	f2 = f1.abs()
		+ f2 - f1 * f2 / f2
		.muladd(f1, f2).mulsub(f2, f1)
		.rcp()
		.fast_div(2.5f)
		.rsqrt()
		.sqrt()
		.shuffle(7,6,5,4,3,2,1,0)
		.hadd(f2)
		.floor()
		.ceil()
		.max(f2).min(f1)
		.nand(f1) & f2 | f1 ^ (~f2)
		.cmp_blend(f1 < f2 == f2 > f1, f2);

	std::cout << f1
		<< f2.to_str()
		<< static_cast<vector256<int32_t>>(f1)
			.to_str(print_format::delim::comma, print_format::brancket::curly)
		<< std::endl;

	int64_t data[] = { 6,7,8,9 };
	vector256<int64_t> t1;
	vector256<int64_t> t2(0, -1, 2, -3);
	
	t1.load(data);
	t2.store(data);
	
	int32_t element = t1.concat(t2)[1];
	element = t1.alternate(t2)[7];
	
	bool be_true = (t1 > t2).is_all_one();
	bool be_false = (t1 == t2).is_all_zero();
	
	t2 = t1 >> 1;
	t2 = t1 << vector256<uint32_t>(3).reinterpret<int64_t>();
	
	for (auto e : t1) {
		std::cout << e << ", ";
	}

	return 0;
}