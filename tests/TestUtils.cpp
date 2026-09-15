// Deterministic unit tests for src/Utils.h (header-inline helpers).
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "Utils.h"

using NEAT::Real;

namespace {

    int g_failures = 0;

#define CHECK(cond)                                                                         \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::cerr << "FAILED " << __FILE__ << ":" << __LINE__ << ": " << #cond << "\n"; \
            ++g_failures;                                                                   \
        }                                                                                   \
    } while (0)

    bool near(Real a, Real b, Real eps = 1e-9) { return std::fabs(a - b) <= eps; }

}  // namespace

int TestUtils(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    using namespace NEAT;

    // GetMaxMin
    {
        std::vector<Real> v{3.0, -1.5, 7.25, 0.0};
        Real mn = 0.0, mx = 0.0;
        getMaxMin(v, mn, mx);
        CHECK(near(mn, -1.5));
        CHECK(near(mx, 7.25));
    }

    // Regression: all-negative input must yield the most-negative value as max,
    // not the smallest positive Real (numeric_limits::min() seeding bug).
    {
        std::vector<Real> v{-5.0, -3.0, -9.0, -0.25};
        Real mn = 0.0, mx = 0.0;
        getMaxMin(v, mn, mx);
        CHECK(near(mn, -9.0));
        CHECK(near(mx, -0.25));
    }

    // itos / ftos round-trip basics
    {
        CHECK(intToString(0) == std::string("0"));
        CHECK(intToString(-42) == std::string("-42"));
        CHECK(intToString(12345) == std::string("12345"));
        // ftos must at least parse back to the same value
        const Real x = 3.25;
        CHECK(near(std::stof(floatToString(x)), x, 1e-6));
    }

    // Clamp Real / float / int
    {
        Real d = -5.0;
        clamp(d, -1.0, 1.0);
        CHECK(near(d, -1.0));
        d = 5.0;
        clamp(d, -1.0, 1.0);
        CHECK(near(d, 1.0));
        d = 0.25;
        clamp(d, -1.0, 1.0);
        CHECK(near(d, 0.25));

        float f = -2.0f;
        clamp(f, 0.0f, 1.0f);
        CHECK(f == 0.0f);
        f = 0.5f;
        clamp(f, 0.0f, 1.0f);
        CHECK(f == 0.5f);

        int i = -3;
        clamp(i, 0, 10);
        CHECK(i == 0);
        i = 99;
        clamp(i, 0, 10);
        CHECK(i == 10);
        i = 4;
        clamp(i, 0, 10);
        CHECK(i == 4);
    }

    // Rounded / RoundUnderOffset
    {
        CHECK(rounded(1.2) == 1);
        CHECK(rounded(1.5) == 2);
        CHECK(rounded(2.49) == 2);
        CHECK(roundUnderOffset(1.2, 0.5) == 1);
        CHECK(roundUnderOffset(1.7, 0.5) == 2);
        CHECK(roundUnderOffset(1.2, 0.1) == 2);
    }

    // Scale Real / float: [0..4] -> [-12..12], 2 maps to 0
    {
        Real a = 2.0;
        scale(a, 0.0, 4.0, -12.0, 12.0);
        CHECK(near(a, 0.0));
        a = 0.0;
        scale(a, 0.0, 4.0, -12.0, 12.0);
        CHECK(near(a, -12.0));
        a = 4.0;
        scale(a, 0.0, 4.0, -12.0, 12.0);
        CHECK(near(a, 12.0));

        float b = 2.0f;
        scale(b, 0.0, 4.0, -12.0, 12.0);
        CHECK(std::fabs(b - 0.0f) < 1e-5f);
    }

    // Scale with a degenerate source range must not throw; result is
    // inf/nan by construction (division by zero). Just document it.
    {
        Real a = 1.0;
        scale(a, 1.0, 1.0, 0.0, 1.0);
        CHECK(std::isinf(a) || std::isnan(a));
    }

    // Abs
    {
        CHECK(near(abs(2.5), 2.5));
        CHECK(near(abs(-2.5), 2.5));
        CHECK(near(abs(0.0), 0.0));
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestUtils with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestUtils\n";
    return 0;
}
