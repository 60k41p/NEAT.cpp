// Deterministic unit tests for src/Utils.h (header-inline helpers).
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "Utils.h"

namespace {

    int g_failures = 0;

#define CHECK(cond)                                                                         \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::cerr << "FAILED " << __FILE__ << ":" << __LINE__ << ": " << #cond << "\n"; \
            ++g_failures;                                                                   \
        }                                                                                   \
    } while (0)

    bool Near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) <= eps; }

}  // namespace

int TestUtils(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    // GetMaxMin
    {
        std::vector<double> v{3.0, -1.5, 7.25, 0.0};
        double mn = 0.0, mx = 0.0;
        GetMaxMin(v, mn, mx);
        CHECK(Near(mn, -1.5));
        CHECK(Near(mx, 7.25));
    }

    // Regression: all-negative input must yield the most-negative value as max,
    // not the smallest positive double (numeric_limits::min() seeding bug).
    {
        std::vector<double> v{-5.0, -3.0, -9.0, -0.25};
        double mn = 0.0, mx = 0.0;
        GetMaxMin(v, mn, mx);
        CHECK(Near(mn, -9.0));
        CHECK(Near(mx, -0.25));
    }

    // itos / ftos round-trip basics
    {
        CHECK(itos(0) == std::string("0"));
        CHECK(itos(-42) == std::string("-42"));
        CHECK(itos(12345) == std::string("12345"));
        // ftos must at least parse back to the same value
        const double x = 3.25;
        CHECK(Near(std::stod(ftos(x)), x, 1e-9));
    }

    // Clamp double / float / int
    {
        double d = -5.0;
        Clamp(d, -1.0, 1.0);
        CHECK(Near(d, -1.0));
        d = 5.0;
        Clamp(d, -1.0, 1.0);
        CHECK(Near(d, 1.0));
        d = 0.25;
        Clamp(d, -1.0, 1.0);
        CHECK(Near(d, 0.25));

        float f = -2.0f;
        Clamp(f, 0.0f, 1.0f);
        CHECK(f == 0.0f);
        f = 0.5f;
        Clamp(f, 0.0f, 1.0f);
        CHECK(f == 0.5f);

        int i = -3;
        Clamp(i, 0, 10);
        CHECK(i == 0);
        i = 99;
        Clamp(i, 0, 10);
        CHECK(i == 10);
        i = 4;
        Clamp(i, 0, 10);
        CHECK(i == 4);
    }

    // Rounded / RoundUnderOffset
    {
        CHECK(Rounded(1.2) == 1);
        CHECK(Rounded(1.5) == 2);
        CHECK(Rounded(2.49) == 2);
        CHECK(RoundUnderOffset(1.2, 0.5) == 1);
        CHECK(RoundUnderOffset(1.7, 0.5) == 2);
        CHECK(RoundUnderOffset(1.2, 0.1) == 2);
    }

    // Scale double / float: [0..4] -> [-12..12], 2 maps to 0
    {
        double a = 2.0;
        Scale(a, 0.0, 4.0, -12.0, 12.0);
        CHECK(Near(a, 0.0));
        a = 0.0;
        Scale(a, 0.0, 4.0, -12.0, 12.0);
        CHECK(Near(a, -12.0));
        a = 4.0;
        Scale(a, 0.0, 4.0, -12.0, 12.0);
        CHECK(Near(a, 12.0));

        float b = 2.0f;
        Scale(b, 0.0, 4.0, -12.0, 12.0);
        CHECK(std::fabs(b - 0.0f) < 1e-5f);
    }

    // Scale with a degenerate source range must not throw; result is
    // inf/nan by construction (division by zero). Just document it.
    {
        double a = 1.0;
        Scale(a, 1.0, 1.0, 0.0, 1.0);
        CHECK(std::isinf(a) || std::isnan(a));
    }

    // Abs
    {
        CHECK(Near(Abs(2.5), 2.5));
        CHECK(Near(Abs(-2.5), 2.5));
        CHECK(Near(Abs(0.0), 0.0));
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestUtils with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestUtils\n";
    return 0;
}
