###########
Instrcution
###########

.. cpp:class:: Instruction

    This class provides checking functions for extended instruction sets of x86-64 architecture.

    Example

    .. code-block:: cpp

        #include <iostream>
        #include <SSEWrapper.hpp>

        int main() {
            std::cout << SIMDW::Instruction::SSE4_2() << '\n'
                      << SIMDW::Instruction::AVX2()   << '\n'
                      << SIMDW::Instruction::FMA()    << std::endl;
        }

    .. cpp:function:: static bool SSE4_1() const noexcept
      
       Returns a bool indicationg if SSE4.1 is currently available.
   
    .. cpp:function:: static bool SSE4_2() const noexcept

       Returns a bool indicationg if SSE4.2 is currently available.

    .. cpp:function:: static bool AVX() const noexcept

       Returns a bool indicationg if AVX is currently available.

    .. cpp:function:: static bool AVX2() const noexcept

       Returns a bool indicationg if AVX2 is currently available.

    .. cpp:function:: static bool FMA() const noexcept

       Returns a bool indicationg if FMA is currently available.