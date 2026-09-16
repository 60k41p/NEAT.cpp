// Deterministic unit tests for src/Utils.h (header-inline helpers).
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "Utils.h"
using NEAT::Real;
using namespace NEAT;

namespace {

    int g_failures = 0;

#define CHECK(cond)                                                                         \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::cerr << "FAILED " << __FILE__ << ":" << __LINE__ << ": " << #cond << "\n"; \
            ++g_failures;                                                                   \
        }                                                                                   \
    } while (0)

    bool Near(Real a, Real b, Real eps = 1e-9) { return std::fabs(a - b) <= eps; }

}  // namespace

int TestUtils(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    // GetMaxMin
    {
        std::vector<Real> v{3.0, -1.5, 7.25, 0.0};
        Real mn = 0.0, mx = 0.0;
        GetMaxMin(v, mn, mx);
        CHECK(Near(mn, -1.5));
        CHECK(Near(mx, 7.25));
    }

    // Regression: all-negative input must yield the most-negative value as max,
    // not the smallest positive Real (numeric_limits::min() seeding bug).
    {
        std::vector<Real> v{-5.0, -3.0, -9.0, -0.25};
        Real mn = 0.0, mx = 0.0;
        GetMaxMin(v, mn, mx);
        CHECK(Near(mn, -9.0));
        CHECK(Near(mx, -0.25));
    }

    // Empty input yields 0,0 instead of garbage extremes.
    {
        std::vector<Real> v;
        Real mn = 123.0, mx = 456.0;
        GetMaxMin(v, mn, mx);
        CHECK(Near(mn, 0.0));
        CHECK(Near(mx, 0.0));
        // Empty vector scale is a no-op.
        Scale(v, 0.0, 1.0);
        CHECK(v.empty());
    }

    // Vector scale honors the requested target range.
    {
        std::vector<Real> v{0.0, 5.0, 10.0};
        Scale(v, -1.0, 1.0);
        CHECK(Near(v[0], -1.0));
        CHECK(Near(v[1], 0.0));
        CHECK(Near(v[2], 1.0));
    }

    // itos / ftos round-trip basics
    {
        CHECK(itos(0) == std::string("0"));
        CHECK(itos(-42) == std::string("-42"));
        CHECK(itos(12345) == std::string("12345"));
        // ftos must at least parse back to the same value
        const Real x = 3.25;
        CHECK(Near(std::stod(ftos(x)), x, 1e-9));
    }

    // Clamp Real / int
    {
        Real d = -5.0;
        Clamp(d, -1.0, 1.0);
        CHECK(Near(d, -1.0));
        d = 5.0;
        Clamp(d, -1.0, 1.0);
        CHECK(Near(d, 1.0));
        d = 0.25;
        Clamp(d, -1.0, 1.0);
        CHECK(Near(d, 0.25));

        Real f = -2.0;
        Clamp(f, 0.0, 1.0);
        CHECK(f == 0.0);
        f = 0.5;
        Clamp(f, 0.0, 1.0);
        CHECK(f == 0.5);

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

    // Rounded (lround: halves away from zero, incl. negatives)
    {
        CHECK(Rounded(1.2) == 1);
        CHECK(Rounded(1.5) == 2);
        CHECK(Rounded(2.49) == 2);
        CHECK(Rounded(-1.6) == -2);
        CHECK(Rounded(-1.5) == -2);
        CHECK(Rounded(-1.2) == -1);
    }

    // Scale Real: [0..4] -> [-12..12], 2 maps to 0
    {
        Real a = 2.0;
        Scale(a, 0.0, 4.0, -12.0, 12.0);
        CHECK(Near(a, 0.0));
        a = 0.0;
        Scale(a, 0.0, 4.0, -12.0, 12.0);
        CHECK(Near(a, -12.0));
        a = 4.0;
        Scale(a, 0.0, 4.0, -12.0, 12.0);
        CHECK(Near(a, 12.0));

        Real b = 2.0;
        Scale(b, 0.0, 4.0, -12.0, 12.0);
        CHECK(std::fabs(b - 0.0) < 1e-9);
    }

    // Scale with a degenerate source range snaps to the target midpoint
    // (no division by zero); degenerate target range snaps to the target.
    {
        Real a = 1.0;
        Scale(a, 1.0, 1.0, 0.0, 1.0);
        CHECK(Near(a, 0.5));
        a = 7.0;
        Scale(a, 0.0, 4.0, 3.0, 3.0);
        CHECK(Near(a, 3.0));
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestUtils with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestUtils\n";
    return 0;
}
