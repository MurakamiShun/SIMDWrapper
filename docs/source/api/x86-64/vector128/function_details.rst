================
function details
================

.. _vector128_max_function:
.. cpp:function:: vector128 max(const vector128& a, const vector128& b)

  Compare and Return maximum value at each element.

    .. math::
        {\rm out}[i] = \left\{
            \begin{array}{l}
                {\rm a}[i] & ({\rm a}[i] > {\rm b}[i]) \\
                {\rm b}[i] & ({\rm a}[i] \le {\rm b}[i])
            \end{array}
        \right.

.. _vector128_min_function:
.. cpp:function:: vector128 min(const vector128& a, const vector128& b)

  Compare and Return minimum value at each element.

    .. math::
        {\rm out}[i] = \left\{
            \begin{array}{l}
                {\rm a}[i] & ({\rm a}[i] < {\rm b}[i]) \\
                {\rm b}[i] & ({\rm a}[i] \ge {\rm b}[i])
            \end{array}
        \right.

.. _vector128_cmp_blend_function:
.. cpp:function:: vector128 cmp_blend(const vector128& condition, const vector128& a, const vector128& b)

    If condition is true, return a. In other case, return b. This operation will apply at each element.

    .. math::
        {\rm out}[i] = \left\{
            \begin{array}{l}
                {\rm a}[i] & ({\rm condition}[i] = \tilde 0) \\
                {\rm b}[i] & ({\rm condition}[i] = 0)
            \end{array}
        \right.

.. _vector128_hadd_function:
.. cpp:function:: vector128 hadd(const vector128& a, const vector128& b)

    Computes horizontally add adjacent pairs.
    
    .. math::
        \begin{gathered}
            n = \frac{128}{\rm element\ bit\ width} \\\\
            {\rm out}[i] = \left\{
                \begin{array}{l}\begin{gathered}
                    {\rm a}[i \times 2] &+& {\rm a}[i \times 2+1] & (i < \frac{n}{2}) \\
                    {\rm b}[(i-\frac{n}{2}) \times 2] &+& {\rm b}[(i-\frac{n}{2}) \times 2+1] & (i \ge \frac{n}{2})
                \end{gathered}\end{array}
            \right.
        \end{gathered}

    .. list-table:: hadd example (16bit elements)
        :header-rows: 1
        :widths: 10 15
        
        * - output index
          - element value
        * - 0
          - a[0] + a[1]
        * - 1
          - a[2] + a[3]
        * - 2
          - a[4] + a[5]
        * - 3
          - a[6] + a[7]
        * - 4
          - b[0] + b[1]
        * - 5
          - b[2] + b[3]
        * - 6
          - b[4] + b[5]
        * - 7
          - b[6] + b[7]

.. _vector128_reinterpret_function:
.. cpp:function:: template<typename Cvt> \
                vector128<Cvt> reinterpret(const vector128& a)

    Reinterpret cast to Cvt at each element. Data will not change.