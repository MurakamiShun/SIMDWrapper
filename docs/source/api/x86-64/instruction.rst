###########
instrcution
###########

.. cpp:class:: instruction

    This class provides checking functions for extended instruction sets of x86-64 architecture.

    Example

    .. code-block:: cpp

        #include <iostream>
        #include <SSEWrapper.hpp>

        int main() {
            std::cout << SIMDW::instruction::SSE4_2() << '\n'
                      << SIMDW::instruction::AVX2()   << '\n'
                      << SIMDW::instruction::FMA()    << std::endl;
        }

    .. cpp:function:: static bool SSE4_1() noexcept
      
       Returns a bool indicationg if SSE4.1 is currently available.
   
    .. cpp:function:: static bool SSE4_2() noexcept

       Returns a bool indicationg if SSE4.2 is currently available.

    .. cpp:function:: static bool AVX() noexcept

       Returns a bool indicationg if AVX is currently available.

    .. cpp:function:: static bool AVX2() noexcept

       Returns a bool indicationg if AVX2 is currently available.

    .. cpp:function:: static bool FMA() noexcept

       Returns a bool indicationg if FMA is currently available.
