=======================
member function details
=======================

Arithmetic operations
=====================

.. _vector128_operator+:
.. cpp:function:: vector128 operator+(const vector128& input) const noexcept
    
    return a new vector with input added to ``this`` at each element.
    
    .. math::
        {\rm out}[ i ] = {\rm this}[ i ] + {\rm input}[ i ]

.. _vector128_operator-:
.. cpp:function:: vector128 operator-(const vector128& input) const noexcept
    
    return a new vector calculated input minus ``this`` at each element.
    
    .. math::
        {\rm out}[i] = {\rm this}[i] - {\rm input}[i]

.. _vector128_operator*:
.. cpp:function:: vector128 operator*(const vector128& input) const noexcept
    
    return a new vector with ``this`` multipled by input at each element.
    
    .. math::
        {\rm out}[i] = {\rm this}[i] \times {\rm input}[i]

    .. warning::
        * This operation is valid only double, float, int32_t and uint32_t.

.. _vector128_operator/:
.. cpp:function:: vector128 operator/(const vector128& input) const noexcept
    
    return a new vector with ``this`` divied by input at each element.
    
    .. math::
        {\rm out}[i] = \frac{{\rm this}[i]}{{\rm input}[i]}

    .. warning::
        * This operation is valid only double and float.

.. _vector128_rcp:
.. cpp:function:: vector128 rcp() const noexcept

    Computes element-wise approximate reciprocals of float value

    .. math::
        {\rm out}[i] = \frac{1}{{\rm this}[i]}

    .. warning::
        * This operation is valid only float and  double.
        * This approximation is less than :math:`1.5 \times 2^{-12}`

.. _vector128_fast_div:
.. cpp:function:: vector128 fast_div(const vector128& input) const noexcept

    Computes element-wise approximate division faster than ``operator/``
    
    .. math::
        {\rm out}[i] = {\rm this}[i] \times \frac{1}{{\rm input}[i]}
    
    .. warning::
        * This operation is valid only float
        * Result will be approximate value.

.. _vector128_sqrt:
.. cpp:function:: vector128 sqrt() const noexcept

    Computes element-wise square root.
    
    .. math::
        {\rm out}[i] = \sqrt{{\rm this}[i]}
    
    .. warning::
        * This operation is valid only float and double.

.. _vector128_rsqrt:
.. cpp:function:: vector128 rsqrt() const noexcept

    Computes element-wise approximate reciprocal square root.
    
    .. math::
        {\rm out}[i] = \frac{1}{\sqrt{{\rm this}[i]}}
    
    .. warning::
        * This operation is valid only float.

.. _vector128_abs:
.. cpp:function:: vector128 abs() const noexcept
    
    Computed the element-wise absolute values of ``this``.
    
    .. math::
        {\rm out}[i] = |{\rm this}[i]|

.. _vector128_hadd:
.. cpp:function:: vector128 hadd(const vector128& input) const noexcept

    Computes horizontally add adjacent pairs in ``this`` and input.

    .. math::
        \begin{gathered}
            n = \frac{128}{\rm element\ bit\ width} \\\\
            {\rm out}[i] = \left\{
                \begin{array}{l}\begin{gathered}
                    {\rm this}[i \times 2] &+& {\rm this}[i \times 2+1] & (i < \frac{n}{2}) \\
                    {\rm input}[(i-\frac{n}{2}) \times 2] &+& {\rm input}[(i-\frac{n}{2}) \times 2+1] & (i \ge \frac{n}{2})
                \end{gathered}\end{array}
            \right.
        \end{gathered}
    
    .. list-table:: hadd example (16bit elements)
        :header-rows: 1
        :widths: 10 15
        
        * - output index
          - element value
        * - 0
          - this[0] + this[1]
        * - 1
          - this[2] + this[3]
        * - 2
          - this[4] + this[5]
        * - 3
          - this[6] + this[7]
        * - 4
          - input[0] + input[1]
        * - 5
          - input[2] + input[3]
        * - 6
          - input[4] + input[5]
        * - 7
          - input[6] + input[7]

Comparison operations
=====================

Comparison operations make bit mask. value will be :math:`\tilde 0` (all bits are 1) when conditional expression is true.

.. list-table::
    :header-rows: 1

    * - true
      - false
    * - :math:`\tilde 0`
      - :math:`0`

.. _vector128_operator==:
.. cpp:function:: vector128 operator==(const vector128& input) const noexcept

    Check whether ``this`` and input are equal at each element and make bitmask.

    .. math::
        {\rm out}[i] = \left\{
            \begin{array}{r}
                \tilde 0 & ({\rm this}[i] = {\rm input}[i]) \\
                0 & ({\rm this}[i] \ne {\rm input}[i])
            \end{array}
        \right.

.. _vector128_operator!=:
.. cpp:function:: vector128 operator!=(const vector128& input) const noexcept

    Check whether ``this`` and input are not equal at each element and make bitmask.

    .. math::
        {\rm out}[i] = \left\{
            \begin{array}{r}
                \tilde 0 & ({\rm this}[i] \ne {\rm input}[i]) \\
                0 & ({\rm this}[i] = {\rm input}[i])
            \end{array}
        \right.

.. _vector128_operator\<=:
.. cpp:function:: vector128 operator<=(const vector128& input) const noexcept

    Check whether ``this`` is less than or equal input at each element and make bitmask.

    .. math::
        {\rm out}[i] = \left\{
            \begin{array}{r}
                \tilde 0 & ({\rm this}[i] \le {\rm input}[i]) \\
                0 & ({\rm this}[i] > {\rm input}[i])
            \end{array}
        \right.

.. _vector128_operator\>=:
.. cpp:function:: vector128 operator>=(const vector128& input) const noexcept

    Check whether ``this`` is greater than or equal input at each element and make bitmask.

    .. math::
        {\rm out}[i] = \left\{
            \begin{array}{r}
                \tilde 0 & ({\rm this}[i] \ge {\rm input}[i]) \\
                0 & ({\rm this}[i] < {\rm input}[i])
            \end{array}
        \right.

.. _vector128_operator\<:
.. cpp:function:: vector128 operator<(const vector128& input) const noexcept

    Check whether ``this`` is less than input at each element and make bitmask.

    .. math::
        {\rm out}[i] = \left\{
            \begin{array}{r}
                \tilde 0 & ({\rm this}[i] < {\rm input}[i]) \\
                0 & ({\rm this}[i] \ge {\rm input}[i])
            \end{array}
        \right.

.. _vector128_operator\>:
.. cpp:function:: vector128 operator>(const vector128& input) const noexcept

    Check whether ``this`` is greater than input at each element and make bitmask.

    .. math::
        {\rm out}[i] = \left\{
            \begin{array}{r}
                \tilde 0 & ({\rm this}[i] > {\rm input}[i]) \\
                0 & ({\rm this}[i] \le {\rm input}[i])
            \end{array}
        \right.

Boolean operations
==================

.. _vector128_operator&&:
.. cpp:function:: vector128 operator&&(const vector128& input) const noexcept

    Calculates :ref:`logical AND<vector128_operator&>` of this and input.

.. _vector128_operator||:
.. cpp:function:: vector128 operator||(const vector128& input) const noexcept

    Calculates :ref:`logical OR<vector128_operator|>` of this and input.

.. _vector128_operator!:
.. cpp:function:: vector128 operator!() const noexcept

    Calculates :ref:`logical NOT<vector128_operator~>` of this.

.. _vector128_is_all_true:
.. cpp:function:: bool is_all_true() const noexcept

    Check whether all elements of ``this`` are true.

    .. math::
        {\rm out} = \prod \left\{
            \begin{array}{l}
                1 &({\rm this}[i] = \tilde 0) \\
                0 &({\rm this}[i] = 0)
            \end{array}
        \right.

.. _vector128_is_all_false:
.. cpp:function:: bool is_all_false() const noexcept

    Check whether all elements of ``this`` are false.

    .. math::
        {\rm out} = \prod \left\{
            \begin{array}{l}
                1 &({\rm this}[i] = 0) \\
                0 &({\rm this}[i] = \tilde 0)
            \end{array}
        \right.


Binary operations
=================

.. _vector128_operator&:
.. cpp:function:: vector128 operator&(const vector128& input) const noexcept

    Calculates logical AND of this and input.

    .. math::
        {\rm out} = {\rm this} \land {\rm input}

.. _vector128_operator|:
.. cpp:function:: vector128 operator|(const vector128& input) const noexcept

    Calculates logical OR of this and input.

    .. math::
        {\rm out} = {\rm this} \lor {\rm input}

.. _vector128_operator^:
.. cpp:function:: vector128 operator^(const vector128& input) const noexcept

    Calculates logical XOR of this and input.

    .. math::
        {\rm out} = {\rm this} \oplus {\rm input}

.. _vector128_operator~:
.. cpp:function:: vector128 operator~() const noexcept

    Calculates logical NOT of this and input.

    .. math::
        {\rm out} = \widetilde{\rm this}

.. _vector128_operator\>\>:
.. cpp:function:: vector128 operator>>(const vector128&) const noexcept

    Shifts ``this`` right by input byte at each element.

    .. math::
        {\rm out}[i] =  \left\lfloor
            \frac{{\rm this}[i]}{2^{{\rm input}[i]}}
        \right\rfloor

.. _vector128_operator\<\<:
.. cpp:function:: vector128 operator<<(const vector128&) const noexcept

    Shifts ``this`` left by input byte at each element.

    .. math::
        {\rm out}[i] =  \left\lfloor
            {\rm this}[i] \times 2^{{\rm input}[i]}
        \right\rfloor

Cast operations
===============

.. _vector128_static_cast:
.. cpp:function:: template<typename Cvt> \
                explicit operator vector128<Cvt>() const noexcept

    Convert ``this`` to Cvt at each element.

    .. math::
        {\rm out}[i] = {\rm Convert\_To\_Cvt}({\rm this}[i])

.. _vector128_reinterpret:
.. cpp:function:: template<typename Cvt> \
                vector128<Cvt> reinterpret() const noexcept

    Reinterpret cast ``this`` to Cvt at each element. Data will not change.

Other operations
================

.. _vector128_max:
.. cpp:function:: vector128 max(const vector128& input) const noexcept

    Compare ``this`` and input and Return maximum value at each element.

    .. math::
        {\rm out}[i] = \left\{
            \begin{array}{l}
                {\rm this}[i] & ({\rm this}[i] > {\rm input}[i]) \\
                {\rm input}[i] & ({\rm this}[i] \le {\rm input}[i])
            \end{array}
        \right.

.. _vector128_min:
.. cpp:function:: vector128 min(const vector128& input) const noexcept

    Compare ``this`` and input and Return minimum value at each element.

    .. math::
        {\rm out}[i] = \left\{
            \begin{array}{l}
                {\rm this}[i] & ({\rm this}[i] < {\rm input}[i]) \\
                {\rm input}[i] & ({\rm this}[i] \ge {\rm input}[i])
            \end{array}
        \right.

.. _vector128_cmp_blend:
.. cpp:function:: vector128 cmp_blend(const vector128& condition, const vector128& input) const noexcept

    If condition is true, return ``this``. In other case, return input. This operation will apply at each element.

    .. math::
        {\rm out}[i] = \left\{
            \begin{array}{l}
                {\rm this}[i] & ({\rm condition}[i] = \tilde 0) \\
                {\rm input}[i] & ({\rm condition}[i] = 0)
            \end{array}
        \right.

.. _vector128_ceil:
.. cpp:function:: vector128 ceil() const noexcept

    Round ``this`` up to an integer value at each element.

    .. math::
        {\rm out}[i] = \lceil {\rm this}[i] \rceil

.. _vector128_floor:
.. cpp:function:: vector128 floor() const noexcept

    Round ``this`` down to an integer value at each element.

    .. math::
        {\rm out}[i] = \lfloor {\rm this}[i] \rfloor

.. _vector128_to_str:
.. cpp:function:: std::string to_str(const std::pair<std::string_view, std::string_view> brancket, std::string_view delim) const noexcept

    Convert all elements to a string.

    .. code-block:: cpp

        #include <string>
        #include <SSEWrapper.hpp>

        int main() {
            vector128<int32_t> v{ -1, 2, -3, 4 };
            std::string str = v.to_str(); // [-1 2 -3 4]

            return 0;
        }

    .. list-table:: branckets of vector
        :header-rows: 1

        * - brancket name
          - brancket charctor
          - note
        * - ``print_format::brancket::round``
          - ``(`` ``)``
          - 
        * - ``print_format::brancket::square``
          - ``[`` ``]``
          - default
        * - ``print_format::brancket::curly``
          - ``{`` ``}``
          - 
        * - ``print_format::brancket::pointy``
          - ``<`` ``>``
          - 

    .. list-table:: delims of elements
        :header-rows: 1

        * - delim name
          - delim charctor
          - note
        * - ``print_format::delim::space``
          - ``\s``
          - default
        * - ``print_format::delim::comma``
          - ``,``
          - 
        * - ``print_format::delim::comma_space``
          - ``,\s``
          - 
        * - ``print_format::delim::space_comma``
          - ``\s,``
          - 

.. _vector128_operator\[\]:
.. cpp:function:: scalar operator[](const size_t index) const

    Return a element at index.

    .. math::
        {\rm out} = {\rm this}[{\rm index}]
