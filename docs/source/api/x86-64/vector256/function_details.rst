================
function details
================

.. _vector256_max_function:
.. cpp:function:: vector256 max(const vector256& a, const vector256& b)

  Compare and Return maximum value at each element.

    .. math::
        {\rm out}[i] = \left\{
            \begin{array}{l}
                {\rm a}[i] & ({\rm a}[i] > {\rm b}[i]) \\
                {\rm b}[i] & ({\rm a}[i] \le {\rm b}[i])
            \end{array}
        \right.

.. _vector256_min_function:
.. cpp:function:: vector256 min(const vector256& a, const vector256& b)

  Compare and Return minimum value at each element.

    .. math::
        {\rm out}[i] = \left\{
            \begin{array}{l}
                {\rm a}[i] & ({\rm a}[i] < {\rm b}[i]) \\
                {\rm b}[i] & ({\rm a}[i] \ge {\rm b}[i])
            \end{array}
        \right.

.. _vector256_cmp_blend_function:
.. cpp:function:: vector256 cmp_blend(const vector256& condition, const vector256& a, const vector256& b)

    If condition is true, return a. In other case, return b. This operation will apply at each element.

    .. math::
        {\rm out}[i] = \left\{
            \begin{array}{l}
                {\rm a}[i] & ({\rm condition}[i] = \tilde 0) \\
                {\rm b}[i] & ({\rm condition}[i] = 0)
            \end{array}
        \right.

.. _vector256_hadd_function:
.. cpp:function:: vector256 hadd(const vector256& a, const vector256& b)

    Computes horizontally add adjacent pairs.

    .. list-table:: hadd example (32bit elements)
        :header-rows: 1
        :widths: 10 15
        
        * - output index
          - element value
        * - 0
          - a[0] + a[1]
        * - 1
          - a[2] + a[3]
        * - 2
          - b[0] + b[1]
        * - 3
          - b[2] + b[3]
        * - 4
          - a[4] + a[5]
        * - 5
          - a[6] + a[7]
        * - 6
          - b[4] + b[5]
        * - 7
          - b[6] + b[7]

.. _vector256_muladd_function:
.. cpp:function:: vector256 muladd(const vector256& a, const vector256& b, const vector256& c) const noexcept

    Computes element-wise multiplication and additions.

    .. math::
      {\rm out}[i] = {\rm a}[i] * {\rm b}[i] + {\rm c}[i]

.. _vector256_nmuladd_function:
.. cpp:function:: vector256 nmuladd(const vector256& a, const vector256& b, const vector256& c) const noexcept

    Computes element-wise negative multiplication and additions.

    .. math::
      {\rm out}[i] = -({\rm a}[i] * {\rm b}[i]) + {\rm c}[i]

.. _vector256_mulsub_function:
.. cpp:function:: vector256 mulsub(const vector256& a, const vector256& b, const vector256& c) const noexcept

    Computes element-wise multiplication and subtractions.

    .. math::
      {\rm out}[i] = {\rm a}[i] * {\rm b}[i] - {\rm c}[i]

.. _vector256_nmulsub_function:
.. cpp:function:: vector256 nmulsub(const vector256& a, const vector256& b, const vector256& c) const noexcept

    Computes element-wise negative multiplication and subtractions.

    .. math::
      {\rm out}[i] = -({\rm a}[i] * {\rm b}[i]) - {\rm c}[i]

.. _vector256_reinterpret_function:
.. cpp:function:: template<typename Cvt> \
                vector256<Cvt> reinterpret(const vector256& a)

    Reinterpret cast to Cvt at each element. Data will not change.