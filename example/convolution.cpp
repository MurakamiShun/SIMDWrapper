#include <iostream>
#include <array>
#include <memory>
#include <numeric>
#include <algorithm>

template<typename DTYPE, typename DIM, typename OPTION = void>
struct tensor;

template<size_t... Args>
struct dim{
    constexpr static auto as_array() noexcept { return std::array<size_t, sizeof...(Args)>{Args...}; }
    size_t operator[](const size_t N) const noexcept { return as_array()[N]; }
    constexpr static size_t get(const size_t N) noexcept { return as_array()[N]; }
    constexpr static size_t size() noexcept { return (Args * ...); }
};

template<size_t NDIM>
struct dynamic{
    std::array<size_t, NDIM> dim;
    auto as_array() const noexcept { return dim; }
    size_t operator[](const size_t N) const noexcept { return dim[N]; }
    size_t get(const size_t N) const noexcept { return dim[N]; }
    size_t size() const noexcept { return std::reduce(dim.begin(), dim.end(), 1, [](auto a, auto b){ return a * b; }); }
};

template<typename DTYPE, typename DIM>
struct tensor_common {
    using dim_type = DIM;
    using dtype = DTYPE;
    
    template<typename... COORDS>
    const dtype& operator()(COORDS... coords) const;
    template<typename... COORDS>
    dtype& operator()(COORDS... coords);
};

template<typename DTYPE, size_t... DIMS>
struct tensor_default_common : tensor_common<DTYPE, dim<DIMS...>>{
    std::unique_ptr<DTYPE[]> data;
    
    constexpr static dim<DIMS...> dim;

    const enum class memory_layout {
        row_major, col_major
    } mem_layout = memory_layout::row_major;
    const std::array<size_t, sizeof...(DIMS)> offset;

    static auto make_offset(memory_layout layout){
        auto make_constant_offset = [](memory_layout layout)constexpr{
            std::array<size_t, sizeof...(DIMS)> a = { DIMS... };
            auto reverse = [](auto& a) constexpr {
                for(size_t i = 0; i < a.size() / 2; ++i){
                    auto tmp = a[i];
                    a[i] = a[a.size() - i - 1];
                    a[a.size() - i - 1] = tmp;
                }
            };
            //row-major (constexpr std::reverse is not C++17)
            if(layout == memory_layout::row_major) reverse(a);
            decltype(a) offset;
            offset[0] = 1;
            for(size_t i = 1; i < offset.size(); ++i){
                offset[i] = offset[i - 1] * a[i - 1];
            }
            if(layout == memory_layout::row_major) reverse(offset);
            return offset;
        };
        constexpr static auto row_majored_offset = make_constant_offset(memory_layout::row_major);
        constexpr static auto col_majored_offset = make_constant_offset(memory_layout::col_major);
        if(layout == memory_layout::row_major) return row_majored_offset;
        else return col_majored_offset;
    };

    tensor_default_common():
        data(new DTYPE[(DIMS * ...)]),
        offset(make_offset(memory_layout::row_major)){

    }
    
    template<typename... COORDS>
    size_t coordinate_to_index(COORDS... coords) {
        static_assert(sizeof...(COORDS) == sizeof...(DIMS), "coordinates are not enough.");
        const auto a = { static_cast<size_t>(coords)... };
        return std::inner_product(a.begin(), a.end(), offset.begin(), 0);
    }
    
    template<typename... COORDS>
    const DTYPE& operator()(COORDS... coords) const { return data[coordinate_to_index(coords...)]; };
    template<typename... COORDS>
    DTYPE& operator()(COORDS... coords) { return data[coordinate_to_index(coords...)]; };
};

template<typename DTYPE, size_t... DIMS>
struct tensor<DTYPE, dim<DIMS...>> : tensor_default_common<DTYPE, DIMS...> {
    
};

template<typename DTYPE, size_t NDIM>
struct tensor_dynamic_common : tensor_common<DTYPE, dynamic<NDIM>>{
    std::unique_ptr<DTYPE[]> data;
    
    dynamic<NDIM> dim;
    
    tensor_dynamic_common(){}
    template<typename... DIMS>
    tensor_dynamic_common(DIMS... dims) : data(new DTYPE[(dims * ...)]), dim({dims...}){}
    
    template<typename... COORDS>
    size_t coordinate_to_index(COORDS... coords) {
        static_assert(sizeof...(COORDS) == NDIM, "coordinates are not enough.");
        const auto a = dim.as_array(), b = {static_cast<size_t>(coords)...};
        return std::inner_product(a.begin(), a.end(), b.begin(), 0);
    }
    
    template<typename... COORDS>
    const DTYPE& operator()(COORDS... coords) const { return data[coordinate_to_index(coords...)]; };
    template<typename... COORDS>
    DTYPE& operator()(COORDS... coords) { return data[coordinate_to_index(coords...)]; };
};

template<typename DTYPE, size_t NDIM>
struct tensor<DTYPE, dynamic<NDIM>> : tensor_dynamic_common<DTYPE, NDIM> {
    template<typename... DIMS>
    tensor(DIMS... dims) : tensor_dynamic_common<DTYPE,NDIM>(static_cast<size_t>(dims)...){}
};

/*
tensor conv2d(const tensor& input, const tensor& weight, const tensor& bias) {
    const auto i_dim = input.dim.as_array();
    const auto w_dim = weight.dim.as_array();
    tensor output = zeros(w_dim[2],i_dim[1]-(w_dim[1]-1),i_dim[0]-(w_dim[0]-1));
    for(auto c_out=0; c < output.dim[2]; ++c_out){
        for(auto y=0; y < output.dim[1]; ++y){
            for(auto x=0; x < output.dim[0]; ++x){
                for(auto c_in=0; c_in < w_dim[2]; ++c_in){
                    for(auto k_h=0; k_h < w_dim[1]; ++k_h){
                        for(auto k_w=0; k_w < w_dim[0]; ++k_w){
                            output(c_out, y, x) += input(c_in, y+k_h, x+k_w) * weight(c_out, c_in, k_h, k_w);
                        }
                    }
                }
                output(c_out, y, x) += bias(c_out);
            }
        }
    }
    return output;
}
*/

int main(){
    tensor<float, dim<2,23,4>> a;
    tensor<double, dynamic<3>> b(4,23,3);
    std::cout << a.dim.size() << std::endl;
    a(0,1,2) = 10;
    std::cout << a(0,1,2) << std::endl;
    std::cout << a.coordinate_to_index(0,1,2) << std::endl;
    std::cout << a.dim[0] << ","<< a.dim[1] << ","<< a.dim[2] << "," << std::endl;
    std::cout << a.offset[0] << ","<< a.offset[1] << ","<< a.offset[2] << "," << std::endl;
    
    std::cout << b.dim.size() << std::endl;
    return 0;
}