// Tests for NEAT::Real (src/Types.h): the single-precision scalar used for all
// library numerics (weights, activations, fitness, distances, parameters).
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

    static_assert(std::is_same<Real, float>::value, "Real must be single precision");

    // Memory contract: 4-byte scalars halve genome/population bandwidth vs double.
    {
        CHECK(sizeof(Real) == 4);
        CHECK(sizeof(float) == 4);
        const std::vector<Real> v(1000, 0.5f);
        CHECK(v.size() * sizeof(Real) == 4000);
        CHECK(std::numeric_limits<Real>::max_digits10 == 9);
    }

    // floatToString round-trips exactly, including values distinguishing float
    // from double precision (0.1 and 1/3 are not exact in binary).
    {
        const Real values[] = {0.1f, 1.0f / 3.0f, -8.0f, 1e-6f, 1e6f, 3.25f, 0.0f};
        for (Real x : values) {
            CHECK(std::stof(floatToString(x)) == x);
        }
    }

    // RNG draws are Real-typed and honor their ranges.
    {
        RNG rng;
        rng.seed(42);
        for (int i = 0; i < 64; ++i) {
            const Real u = rng.randFloat();
            CHECK(u >= 0.0f && u <= 1.0f);
            const Real s = rng.randFloatSigned();
            CHECK(s >= -1.0f && s <= 1.0f);
            const Real g = rng.randGaussSigned();
            CHECK(g >= -1.0f && g <= 1.0f);
        }
    }

    // Float trait values stay in the Real variant alternative and average in
    // single precision (mateTraits must not promote back to double).
    {
        Gene a, b;
        Trait ta, tb;
        ta.value = Real(1.0f);
        tb.value = Real(3.0f);
        a.traits_["v"] = ta;
        b.traits_["v"] = tb;
        CHECK(std::holds_alternative<Real>(a.traits_["v"].value));
        RNG rng;
        rng.seed(5);
        bool sawAverage = false;
        for (int i = 0; i < 64; ++i) {
            Gene c = a;
            c.mateTraits(b.traits_, rng);
            const Real v = std::get<Real>(c.traits_["v"].value);
            CHECK(v == Real(1.0f) || v == Real(3.0f) || near(v, Real(2.0f)));
            sawAverage = sawAverage || near(v, Real(2.0f));
        }
        CHECK(sawAverage);  // averaging branch was exercised, not just pick-one
    }

    // Backward compatibility: legacy files written with double precision
    // (%3.20f / %3.18f / %3.8f) still parse into Real fields.
    {
        std::istringstream in("0.12345678901234567890123");
        Real x = 0.0f;
        in >> x;
        CHECK(near(x, Real(0.12345678901234567890123), 1e-7f));
        CHECK(x == static_cast<Real>(0.12345678901234567890123));
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestTypes with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestTypes\n";
    return 0;
}
