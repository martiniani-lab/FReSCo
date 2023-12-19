#ifndef __FRESCO_TEST_UTILS_HPP__
#define __FRESCO_TEST_UTILS_HPP__

#include <iostream>
#include <stdexcept>
#include <gtest/gtest.h>
#include <cmath>
#include <memory>

// Inspired by https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
// Adapted from http://stackoverflow.com/questions/13940316/floating-point-comparison-revisited
template<typename T>
::testing::AssertionResult almostEqual(const T a, const T b, int maxULPsDiff=4)
{
    typedef std::numeric_limits<T> limits;

    // Handle NaN.
    if (std::isnan(a) || std::isnan(b))
        return ::testing::AssertionFailure() << a << " != " << b;

    // Handle infinity
    if( !std::isfinite(a) || !std::isfinite(b) ) {
        if (a == b) {
            return ::testing::AssertionSuccess();
        } else {
            return ::testing::AssertionFailure() << a << " != " << b;
        }
    }

    // Handle very small and exactly equal values.
    if (std::abs(a-b) <= maxULPsDiff * limits::epsilon())
        return ::testing::AssertionSuccess();

    // frexp() does the wrong thing for zero.  But if we get this far
    // and either number is zero, then the other is too big, so just
    // handle that now.
    if (a == 0 || b == 0)
        return ::testing::AssertionFailure() << a << " != " << b;

    // Break the numbers into significand and exponent, sorting them by
    // exponent.
    int min_exp, max_exp;
    T min_frac = std::frexp(a, &min_exp);
    T max_frac = std::frexp(b, &max_exp);
    if (min_exp > max_exp)
    {
        std::swap(min_frac, max_frac);
        std::swap(min_exp, max_exp);
    }

    // Convert the smaller to the scale of the larger by adjusting its
    // significand.
    const T scaled_min_frac = std::ldexp(min_frac, min_exp-max_exp);

    // Since the significands are now in the same scale, and the larger
    // is in the range [0.5, 1), 1 ulp is just epsilon/2.
    if(std::abs(max_frac - scaled_min_frac) <= maxULPsDiff * limits::epsilon() * 0.5)
        return ::testing::AssertionSuccess();

    double ulp_diff = std::abs(max_frac - scaled_min_frac) / (limits::epsilon() * 0.5);
    return ::testing::AssertionFailure() << a << " != " << b << " (" << ulp_diff << " ULPs difference)";
}

#endif
