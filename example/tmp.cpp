#include <iostream>
#include "../include/AVX2Wrapper.hpp"
#include <chrono>

int main(){

    vector256<uint64_t> a(-14,652,1,-1);
    vector256<uint64_t> b(-1, 1, 1, 1);

    std::cout << a << std::endl;
    std::cout << b << std::endl;

    vector256<int64_t> s(0);

    auto start = std::chrono::system_clock::now();

    for(auto i=0; i < 1000000; ++i){
        s = s + (a<=b).reinterpret<int64_t>();
        s = s + (a>=b).reinterpret<int64_t>();
    }
  
    std::cout << (std::chrono::system_clock::now() - start).count() << std::endl;

    std::cout << (a<=b) << std::endl;

    return 0;
}