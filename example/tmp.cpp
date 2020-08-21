#include <iostream>
#include "../include/AVX2Wrapper.hpp"
#include <chrono>
#include <iomanip>

constexpr double f(double x, double y_n){
    return y_n*(2-x*y_n);
}

constexpr double f(double x){
    return f(x, 1.0f/float(x));
}

int main(){

    vector128<double> a(31);
    vector128<double> b(7);

    std::cout << vector128<double>(-0.1).floor() << std::endl;

    std::cout << a << std::endl;
    std::cout << std::fixed << std::setprecision(15) << b.rcp()[1] << std::endl;

    vector128<double> s(0);

    auto start = std::chrono::system_clock::now();

    for(auto i=0; i < 1000000; ++i){
        s = b.rcp().rcp().rcp().rcp().rcp().rcp().rcp();

        //s = vector128<double>(1)/1/1/1/1/1/1/b;
        s = a.rcp().rcp().rcp().rcp().rcp().rcp().rcp();
        //s = vector128<double>(1)/1/1/1/1/1/1/a;
    }
  
    std::cout << (std::chrono::system_clock::now() - start).count() << std::endl;

    std::cout << (a<=b) << std::endl;

    std::cout <<std::fixed << std::setprecision(15) << (vector128<float>(7.0).rcp())[0] << std::endl;

    std::cout << std::fixed << std::setprecision(15) << 1.0f/7.0f << std::endl;
    std::cout << std::fixed << std::setprecision(15) << 1.0/7.0 << std::endl;

    std::cout << std::fixed << std::setprecision(15) << f(7.0) << std::endl;

    return 0;
}