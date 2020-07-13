===========================
vector128 member operations
===========================

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
        {\rm out}[i] = {\rm this}[i] * {\rm input}[i]

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
        This operation is valid only float.

.. _vector128_fast_div:
.. cpp:function:: vector128 fast_div(const vector128& input) const noexcept

    Computes element-wise approximate division faster than ``operator/``
    
    .. math::
        {\rm out}[i] = {\rm this}[i] * \frac{1}{{\rm input}[i]}
    
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
        {\rm out}[i] = {\rm abs}({\rm this}[i])

.. _vector128_hadd:
.. cpp:function:: vector128 hadd(const vector128& input) const noexcept

    Computes horizontally add adjacent pairs in ``this`` and input.

    .. math::
        \begin{gathered}
            n = \frac{128}{\rm element\ bit\ width} \\\\
            {\rm out}[i] = \left\{
                \begin{array}{l}\begin{gathered}
                    {\rm this}[i*2] &+& {\rm this}[i*2+1] & (i < \frac{n}{2}) \\
                    {\rm input}[(i-\frac{n}{2})*2] &+& {\rm input}[(i-\frac{n}{2})*2+1] & (i \ge \frac{n}{2})
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

Comparison operations make bit mask. value will be ``~0`` (all bits are 1) when conditional expression is true.

+------+-------+
| true | false |
+======+=======+
|  ~0  |   0   |
+------+-------+

.. _vector128_operator==:
.. cpp:function:: vector128 operator==(const vector128& input) const noexcept

    Check whether ``this`` and input are equal at each element and make bitmask.

    .. math::
        {\rm out}[i] = \left\{
            \begin{array}{r}
                \lnot 0 & ({\rm this}[i] = {\rm input}[i]) \\
                0 & ({\rm this}[i] \ne {\rm input}[i])
            \end{array}
        \right.

.. _vector128_operator!=:
.. cpp:function:: vector128 operator!=(const vector128& input) const noexcept

    Check whether ``this`` and input are not equal at each element and make bitmask.

    .. math::
        {\rm out}[i] = \left\{
            \begin{array}{r}
                \lnot 0 & ({\rm this}[i] \ne {\rm input}[i]) \\
                0 & ({\rm this}[i] = {\rm input}[i])
            \end{array}
        \right.

.. _vector128_operator\<=:
.. cpp:function:: vector128 operator<=(const vector128& input) const noexcept

    Check whether ``this`` is less than or equal input at each element and make bitmask.

    .. math::
        {\rm out}[i] = \left\{
            \begin{array}{r}
                \lnot 0 & ({\rm this}[i] \le {\rm input}[i]) \\
                0 & ({\rm this}[i] > {\rm input}[i])
            \end{array}
        \right.

.. _vector128_operator\>=:
.. cpp:function:: vector128 operator>=(const vector128& input) const noexcept

    Check whether ``this`` is greater than or equal input at each element and make bitmask.

    .. math::
        {\rm out}[i] = \left\{
            \begin{array}{r}
                \lnot 0 & ({\rm this}[i] \ge {\rm input}[i]) \\
                0 & ({\rm this}[i] < {\rm input}[i])
            \end{array}
        \right.

.. _vector128_operator\<:
.. cpp:function:: vector128 operator<(const vector128& input) const noexcept

    Check whether ``this`` is less than input at each element and make bitmask.

    .. math::
        {\rm out}[i] = \left\{
            \begin{array}{r}
                \lnot 0 & ({\rm this}[i] < {\rm input}[i]) \\
                0 & ({\rm this}[i] \ge {\rm input}[i])
            \end{array}
        \right.

.. _vector128_operator\>:
.. cpp:function:: vector128 operator>(const vector128& input) const noexcept

    Check whether ``this`` is greater than input at each element and make bitmask.

    .. math::
        {\rm out}[i] = \left\{
            \begin{array}{r}
                \lnot 0 & ({\rm this}[i] > {\rm input}[i]) \\
                0 & ({\rm this}[i] \le {\rm input}[i])
            \end{array}
        \right.

Binary operations
=================

.. _vector128_operator&:
.. cpp:function:: vector128 operator&(const vector128&) const noexcept

.. _vector128_operator|:
.. cpp:function:: vector128 operator|(const vector128&) const noexcept

.. _vector128_operator^:
.. cpp:function:: vector128 operator^(const vector128&) const noexcept

.. _vector128_operator~:
.. cpp:function:: vector128 operator~(const vector128&) const noexcept

.. _vector128_operator\>\>:
.. cpp:function:: vector128 operator>>(const vector128&) const noexcept

.. _vector128_operator\<\<:
.. cpp:function:: vector128 operator<<(const vector128&) const noexcept

Cast operations
===============

.. _vector128_static_cast:
.. cpp:function:: template<typename Cvt> \
                explicit operator vector128<Cvt>() const noexcept


.. _vector128_reinterpret:
.. cpp:function:: template<typename Cvt> \
                vector128<Cvt> reinterpret() const noexcept


Other operations
================

.. _vector128_max:
.. cpp:function:: vector128 max(const vector128&) const noexcept

.. _vector128_min:
.. cpp:function:: vector128 min(const vector128&) const noexcept

.. _vector128_cmp_blend:
.. cpp:function:: vector128 cmp_blend(const vector128&, const vector128&) const noexcept

.. _vector128_ceil:
.. cpp:function:: vector128 ceil() const noexcept

.. _vector128_floor:
.. cpp:function:: vector128 floor() const noexcept

.. _vector128_to_str:
.. cpp:function:: vector128 to_str() const noexcept