// Tests for NEAT::Real (src/Types.h): the double-precision scalar used for direct
// parity with the MultiNEAT2 reference (weights, activations, fitness, distances, parameters).
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <type_traits>
#include <vector>

#include "Genes.h"
#include "Random.h"
#include "Traits.h"
#include "Types.h"
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

    bool near(Real a, Real b, Real eps = 1e-6) { return std::fabs(a - b) <= eps; }

}  // namespace

int TestTypes(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    using namespace NEAT;

    static_assert(std::is_same<Real, double>::value, "Real must be double precision for reference parity");

    // Memory contract: 8-byte scalars match the reference implementation.
    {
        CHECK(sizeof(Real) == 8);
        CHECK(sizeof(double) == 8);
        const std::vector<Real> v(1000, 0.5);
        CHECK(v.size() * sizeof(Real) == 8000);
        CHECK(std::numeric_limits<Real>::max_digits10 == 17);
    }

    // ftos round-trips exactly, including values distinguishing float
    // from double precision (0.1 and 1/3 are not exact in binary).
    {
        const Real values[] = {0.1, 1.0 / 3.0, -8.0, 1e-6, 1e6, 3.25, 0.0};
        for (Real x : values) {
            CHECK(std::stod(ftos(x)) == x);
        }
    }

    // RNG draws are Real-typed and honor their ranges.
    {
        RNG rng;
        rng.Seed(42);
        for (int i = 0; i < 64; ++i) {
            const Real u = rng.RandFloat();
            CHECK(u >= 0.0 && u <= 1.0);
            const Real s = rng.RandFloatSigned();
            CHECK(s >= -1.0 && s <= 1.0);
            const Real g = rng.RandGaussSigned();
            CHECK(g >= -1.0 && g <= 1.0);
        }
    }

    // Double trait values stay in the Real variant alternative and average in
    // double precision (MateTraits must not narrow to float).
    {
        Gene a, b;
        Trait ta, tb;
        ta.value = Real(1.0);
        tb.value = Real(3.0);
        a.m_Traits["v"] = ta;
        b.m_Traits["v"] = tb;
        CHECK(std::holds_alternative<Real>(a.m_Traits["v"].value));
        RNG rng;
        rng.Seed(5);
        bool sawAverage = false;
        for (int i = 0; i < 64; ++i) {
            Gene c = a;
            c.MateTraits(b.m_Traits, rng);
            const Real v = std::get<Real>(c.m_Traits["v"].value);
            CHECK(v == Real(1.0) || v == Real(3.0) || near(v, Real(2.0)));
            sawAverage = sawAverage || near(v, Real(2.0));
        }
        CHECK(sawAverage);  // averaging branch was exercised, not just pick-one
    }

    // Backward compatibility: legacy files written with double precision
    // (%3.20f / %3.18f / %3.8f) still parse into Real fields.
    {
        std::istringstream in("0.12345678901234567890123");
        Real x = 0.0;
        in >> x;
        CHECK(near(x, Real(0.12345678901234567890123), 1e-12));
        CHECK(x == static_cast<Real>(0.12345678901234567890123));
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestTypes with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestTypes\n";
    return 0;
}
