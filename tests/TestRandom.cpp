// Tests for NEAT::RNG (src/Random.h/.cpp). mt19937 is deterministic per the
// C++ standard, so same seed => same sequence on all platforms.
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Random.h"

namespace {

    int g_failures = 0;

#define CHECK(cond)                                                                         \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::cerr << "FAILED " << __FILE__ << ":" << __LINE__ << ": " << #cond << "\n"; \
            ++g_failures;                                                                   \
        }                                                                                   \
    } while (0)

}  // namespace

int TestRandom(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    using NEAT::RNG;

    // Same seed => identical sequences (determinism contract for evolution tests).
    {
        RNG a, b;
        a.Seed(42);
        b.Seed(42);
        for (int i = 0; i < 16; ++i) {
            CHECK(a.RandFloat() == b.RandFloat());
        }
    }
    {
        RNG a, b;
        a.Seed(1234);
        b.Seed(1234);
        for (int i = 0; i < 16; ++i) {
            CHECK(a.RandInt(-100, 100) == b.RandInt(-100, 100));
        }
    }

    // Different seeds very likely diverge (guard against a no-op Seed()).
    {
        RNG a, b;
        a.Seed(1);
        b.Seed(2);
        bool any_diff = false;
        for (int i = 0; i < 8; ++i) {
            if (a.RandFloat() != b.RandFloat()) {
                any_diff = true;
                break;
            }
        }
        CHECK(any_diff);
    }

    // Range contracts over many draws.
    {
        RNG rng;
        rng.Seed(7);
        for (int i = 0; i < 1000; ++i) {
            const double u = rng.RandFloat();
            CHECK(u >= 0.0 && u <= 1.0);
            const double s = rng.RandFloatSigned();
            CHECK(s >= -1.0 && s <= 1.0);
            const double g = rng.RandGaussSigned();
            CHECK(g >= -1.0 && g <= 1.0);
            const int pn = rng.RandPosNeg();
            CHECK(pn == 1 || pn == -1);
            const int ri = rng.RandInt(3, 9);
            CHECK(ri >= 3 && ri <= 9);
        }
        // Degenerate integer range always returns the bound.
        for (int i = 0; i < 16; ++i) {
            CHECK(rng.RandInt(5, 5) == 5);
        }
    }

    // Roulette: valid index, respects zero weights, deterministic per seed.
    {
        RNG a, b;
        a.Seed(99);
        b.Seed(99);
        std::vector<double> probs{0.2, 0.5, 0.3};
        for (int i = 0; i < 32; ++i) {
            const int ia = a.Roulette(probs);
            const int ib = b.Roulette(probs);
            CHECK(ia == ib);
            CHECK(ia >= 0 && ia < 3);
        }
    }
    {
        RNG rng;
        rng.Seed(11);
        std::vector<double> forced{0.0, 0.0, 1.0};
        for (int i = 0; i < 20; ++i) {
            CHECK(rng.Roulette(forced) == 2);
        }
    }

    // TimeSeed must not crash (value itself is intentionally non-deterministic).
    {
        RNG rng;
        rng.TimeSeed();
        (void)rng.RandFloat();
    }

    // RandInt validates its range; Roulette validates its input.
    {
        RNG rng;
        rng.Seed(5);
        bool threw = false;
        try {
            (void)rng.RandInt(9, 3);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
        threw = false;
        try {
            std::vector<double> empty;
            (void)rng.Roulette(empty);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
        threw = false;
        try {
            std::vector<double> negative{0.5, -0.1};
            (void)rng.Roulette(negative);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
        // All-zero weights fall back to a uniform valid index.
        std::vector<double> zeros{0.0, 0.0, 0.0};
        for (int i = 0; i < 16; ++i) {
            const int idx = rng.Roulette(zeros);
            CHECK(idx >= 0 && idx < 3);
        }
    }

    // RandNormal / RandCauchy validation and determinism.
    {
        RNG a, b;
        a.Seed(77);
        b.Seed(77);
        CHECK(a.RandNormal(0.0, 1.0) == b.RandNormal(0.0, 1.0));
        CHECK(a.RandCauchy(0.0, 1.0) == b.RandCauchy(0.0, 1.0));
        bool threw = false;
        try {
            (void)a.RandNormal(0.0, 0.0);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
        threw = false;
        try {
            (void)a.RandCauchy(0.0, -1.0);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
    }

    // Serialize / Deserialize round-trips the engine state.
    {
        RNG a, b;
        a.Seed(2026);
        (void)a.RandFloat();
        const std::string state = a.Serialize();
        b.Deserialize(state);
        for (int i = 0; i < 8; ++i) {
            CHECK(a.RandFloat() == b.RandFloat());
        }
        bool threw = false;
        try {
            b.Deserialize("not-a-valid-state!!!");
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestRandom with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestRandom\n";
    return 0;
}
