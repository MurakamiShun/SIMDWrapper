#include <iostream>
#include <array>

#include "../include/NEONWrapper.hpp"

int main(){
    vector128<int8_t> idx = 0;
    vector128<int8_t> n = 1;

    std::cout << n.shuffle(idx) << std::endl;

    return 0;
}