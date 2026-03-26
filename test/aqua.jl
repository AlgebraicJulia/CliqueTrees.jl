using Aqua
using CliqueTrees
using CliqueTrees.Multifrontal.Differential: mul_frule_impl, mul_rrule_impl, mul_rrule

Aqua.test_all(CliqueTrees; ambiguities = (exclude = [mul_frule_impl, mul_rrule_impl, mul_rrule],))
