// Micro/macro benchmarks for NEAT.cpp. Standalone executable; prints one line
// per benchmark: <name> <cpu_ms> (<ops> ops, <ns_per_op> ns/op).
//
// Build:  cmake -S . -B build/bench -DCMAKE_BUILD_TYPE=Release -DNEATCPP_ENABLE_BENCHMARKS=ON
// Run:    ./build/bench/benchmarks/NEATcppBench
//
// Seeded and fixed-iteration so runs are comparable across changes;
// compare output against RESULTS.md baselines.
//
// Timing uses calling-thread CPU time (not wall time), so preemption, sleep
// and blocked file I/O do not count towards the reported totals.
#if defined(_WIN32)
#include <windows.h>
#endif
#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/thread_info.h>
#endif
#if !defined(_WIN32) && !defined(__APPLE__)
#include <time.h>
#endif
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <string>
#include <vector>

#include "Genome.h"
#include "Innovation.h"
#include "NeuralNetwork.h"
#include "Parameters.h"
#include "Population.h"
#include "Random.h"

using namespace NEAT;

// Calling-thread CPU clock satisfying the Cpp17Clock requirements. Measures
// user + system time consumed by the calling thread only; time the thread is
// descheduled or blocked (e.g. on disk I/O) is excluded.
class ThreadCpuClock {
   public:
    using rep = long long;
    using period = std::nano;
    using duration = std::chrono::duration<rep, period>;
    using time_point = std::chrono::time_point<ThreadCpuClock>;
    static constexpr bool is_steady = true;

    static time_point now() noexcept { return time_point(duration(static_cast<rep>(threadCpuNanos()))); }

   private:
    static std::uint64_t threadCpuNanos() noexcept {
#if defined(_WIN32)
        FILETIME creationTime, exitTime, kernelTime, userTime;
        if (GetThreadTimes(GetCurrentThread(), &creationTime, &exitTime, &kernelTime, &userTime) == 0) {
            return 0;
        }
        ULARGE_INTEGER kernel, user;
        kernel.LowPart = kernelTime.dwLowDateTime;
        kernel.HighPart = kernelTime.dwHighDateTime;
        user.LowPart = userTime.dwLowDateTime;
        user.HighPart = userTime.dwHighDateTime;
        // FILETIME ticks are 100ns intervals.
        return (kernel.QuadPart + user.QuadPart) * 100ULL;
#elif defined(__APPLE__)
        thread_basic_info_data_t info;
        mach_msg_type_number_t count = THREAD_BASIC_INFO_COUNT;
        const thread_port_t port = mach_thread_self();
        const kern_return_t result = thread_info(port, THREAD_BASIC_INFO, reinterpret_cast<thread_info_t>(&info), &count);
        mach_port_deallocate(mach_task_self(), port);
        if (result != KERN_SUCCESS) {
            return 0;
        }
        const std::uint64_t user =
            static_cast<std::uint64_t>(info.user_time.seconds) * 1000000000ULL + static_cast<std::uint64_t>(info.user_time.microseconds) * 1000ULL;
        const std::uint64_t system =
            static_cast<std::uint64_t>(info.system_time.seconds) * 1000000000ULL + static_cast<std::uint64_t>(info.system_time.microseconds) * 1000ULL;
        return user + system;
#else
#if defined(CLOCK_THREAD_CPUTIME_ID)
        struct timespec ts;
        if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0) {
            return 0;
        }
        return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<std::uint64_t>(ts.tv_nsec);
#else
        const std::clock_t ticks = std::clock();
        if (ticks == static_cast<std::clock_t>(-1)) {
            return 0;
        }
        return static_cast<std::uint64_t>(ticks) * 1000000000ULL / static_cast<std::uint64_t>(CLOCKS_PER_SEC);
#endif
#endif
    }
};

using Clock = ThreadCpuClock;

namespace {

    struct Result {
        const char *name;
        double total_ms;
        unsigned long ops;
    };
    std::vector<Result> g_results;

    void report(const char *name, double total_ms, unsigned long ops) {
        const double ns_per_op = (total_ms * 1e6) / static_cast<double>(ops);
        std::printf("%-28s %10.2f ms total  %8lu ops  %10.1f ns/op\n", name, total_ms, ops, ns_per_op);
        g_results.push_back({name, total_ms, ops});
    }

    Parameters DefaultParams() {
        Parameters p;
        p.Reset();
        return p;
    }

    Genome SeedGenome(int num_inputs, int num_outputs) {
        Parameters p = DefaultParams();
        GenomeInitStruct init;
        init.NumInputs = num_inputs;
        init.NumOutputs = num_outputs;
        init.SeedType = PERCEPTRON;
        return Genome(p, init);
    }

    // Grow a large genome: num_inputs inputs + num_outputs outputs + many hidden
    // neurons/links, representative of a complexified NEAT topology.
    Genome LargeGenome(int num_inputs, int num_outputs, int target_hidden, unsigned seed) {
        Parameters p = DefaultParams();
        RNG rng;
        rng.Seed(seed);
        InnovationDatabase innovs;
        innovs.Init(1, 1000);
        Genome g = SeedGenome(num_inputs, num_outputs);
        while (g.NumNeurons() - num_inputs - num_outputs < target_hidden) {
            if (!g.Mutate_AddNeuron(innovs, p, rng)) {
                break;
            }
        }
        for (int i = 0; i < target_hidden * 4; ++i) {
            g.Mutate_AddLink(innovs, p, rng);
        }
        return g;
    }

    void bench_phenotype_build_small() {
        Genome g = SeedGenome(10, 3);
        const unsigned ops = 200000;
        NeuralNetwork net;
        auto t0 = Clock::now();
        for (unsigned i = 0; i < ops; ++i) {
            g.BuildPhenotype(net);
        }
        auto t1 = Clock::now();
        report("BuildPhenotype small", std::chrono::duration<double, std::milli>(t1 - t0).count(), ops);
    }

    void bench_phenotype_build_large() {
        Genome g = LargeGenome(10, 3, 200, 11);
        const unsigned ops = 2000;
        NeuralNetwork net;
        auto t0 = Clock::now();
        for (unsigned i = 0; i < ops; ++i) {
            g.BuildPhenotype(net);
        }
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        report("BuildPhenotype large", ms, ops);
    }

    void bench_activate() {
        Genome g = LargeGenome(10, 3, 200, 12);
        NeuralNetwork net;
        g.BuildPhenotype(net);
        std::vector<Real> in(net.m_num_inputs, 0.5f);
        const unsigned ops = 20000;
        auto t0 = Clock::now();
        for (unsigned i = 0; i < ops; ++i) {
            net.Flush();
            net.Input(in);
            net.Activate();
        }
        auto t1 = Clock::now();
        report("Activate large net", std::chrono::duration<double, std::milli>(t1 - t0).count(), ops);
    }

    void bench_compatibility_distance() {
        Genome a = LargeGenome(10, 3, 200, 13);
        Genome b = LargeGenome(10, 3, 200, 14);
        Parameters p = DefaultParams();
        const unsigned ops = 20000;
        volatile Real sink = 0.0f;
        auto t0 = Clock::now();
        for (unsigned i = 0; i < ops; ++i) {
            sink += a.CompatibilityDistance(b, p);
        }
        auto t1 = Clock::now();
        report("CompatibilityDistance", std::chrono::duration<double, std::milli>(t1 - t0).count(), ops);
    }

    void bench_mutate_genome() {
        Parameters p = DefaultParams();
        RNG rng;
        rng.Seed(15);
        InnovationDatabase innovs;
        innovs.Init(1, 1000);
        Genome g = LargeGenome(10, 3, 200, 16);
        const unsigned ops = 50000;
        auto t0 = Clock::now();
        for (unsigned i = 0; i < ops; ++i) {
            Genome copy(g);
            copy.Mutate_LinkWeights(p, rng);
            copy.Mutate_AddLink(innovs, p, rng);
        }
        auto t1 = Clock::now();
        report("Copy+mutate genome", std::chrono::duration<double, std::milli>(t1 - t0).count(), ops);
    }

    void bench_epoch() {
        // One full Epoch of a small XOR-style population (the dominant per-generation cost).
        Parameters p;
        p.PopulationSize = 100;
        p.DynamicCompatibility = true;
        p.CompatTreshold = 2.0;
        p.CrossoverRate = 0.0;
        p.SurvivalRate = 0.2;
        Genome g = SeedGenome(3, 1);
        Population pop(g, p, true, 1.0, 21);
        const unsigned ops = 100;
        auto t0 = Clock::now();
        for (unsigned i = 0; i < ops; ++i) {
            for (unsigned j = 0; j < pop.NumGenomes(); ++j) {
                Genome &gg = pop.AccessGenomeByIndex(static_cast<int>(j));
                gg.SetFitness(1.0f + static_cast<Real>(gg.NumLinks()));
                gg.SetEvaluated();
            }
            pop.Epoch();
        }
        auto t1 = Clock::now();
        report("Epoch pop100", std::chrono::duration<double, std::milli>(t1 - t0).count(), ops);
    }

    void bench_xor_solve() {
        // The reference XOR recipe from upstream MultiNEAT: generation count and
        // thread CPU time until best fitness > 15.0 (seeded, deterministic).
        Parameters p;
        p.PopulationSize = 100;
        p.DynamicCompatibility = true;
        p.NormalizeGenomeSize = true;
        p.WeightDiffCoeff = 0.1;
        p.CompatTreshold = 2.0;
        p.YoungAgeTreshold = 15;
        p.SpeciesMaxStagnation = 15;
        p.OldAgeTreshold = 35;
        p.MinSpecies = 2;
        p.MaxSpecies = 10;
        p.RouletteWheelSelection = false;
        p.RecurrentProb = 0.0;
        p.OverallMutationRate = 1.0;
        p.ArchiveEnforcement = false;
        p.MutateWeightsProb = 0.05;
        p.WeightMutationMaxPower = 0.5;
        p.WeightReplacementMaxPower = 8.0;
        p.MutateWeightsSevereProb = 0.0;
        p.WeightMutationRate = 0.25;
        p.WeightReplacementRate = 0.9;
        p.MaxWeight = 8.0;
        p.MutateAddNeuronProb = 0.001;
        p.MutateAddLinkProb = 0.3;
        p.MutateRemLinkProb = 0.0;
        p.MinActivationA = 4.9;
        p.MaxActivationA = 4.9;
        p.ActivationFunction_SignedSigmoid_Prob = 0.0;
        p.ActivationFunction_UnsignedSigmoid_Prob = 1.0;
        p.ActivationFunction_Tanh_Prob = 0.0;
        p.ActivationFunction_SignedStep_Prob = 0.0;
        p.CrossoverRate = 0.0;
        p.MultipointCrossoverRate = 0.0;
        p.SurvivalRate = 0.2;
        p.MutateNeuronTraitsProb = 0.0;
        p.MutateLinkTraitsProb = 0.0;
        p.AllowLoops = true;
        p.AllowClones = true;

        Parameters q;
        q.Reset();
        GenomeInitStruct init;
        init.NumInputs = 3;
        init.NumOutputs = 1;
        init.SeedType = PERCEPTRON;
        Population pop(Genome(q, init), p, true, 1.0, 1);

        const Real xi[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        const Real xt[4] = {0, 1, 1, 0};

        auto t0 = Clock::now();
        unsigned gen = 0;
        for (; gen < 300; ++gen) {
            for (unsigned i = 0; i < pop.NumGenomes(); ++i) {
                Genome &gg = pop.AccessGenomeByIndex(static_cast<int>(i));
                NeuralNetwork net;
                gg.BuildPhenotype(net);
                Real err = 0.0f;
                for (int pat = 0; pat < 4; ++pat) {
                    net.Flush();
                    std::vector<Real> in = {xi[pat][0], xi[pat][1], 1.0f};
                    net.Input(in);
                    net.Activate();
                    net.Activate();
                    err += std::fabs(net.Output()[0] - xt[pat]);
                }
                const Real rem = 4.0f - err;
                gg.SetFitness(rem * rem);
                gg.SetEvaluated();
            }
            if (pop.GetBestFitnessEver() > 15.0) {
                break;
            }
            pop.Epoch();
        }
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("(xor solved at generation %u)\n", gen);
        report("XOR solve seed1", ms, 1);
    }

    void bench_genome_save_load() {
        Genome g = LargeGenome(10, 3, 200, 17);
        const auto path = std::string("/tmp/neatcpp_bench_genome.txt");
        const unsigned ops = 100;
        auto t0 = Clock::now();
        for (unsigned i = 0; i < ops; ++i) {
            g.Save(path.c_str());
            Genome g2(path.c_str());
        }
        auto t1 = Clock::now();
        report("Genome save+load", std::chrono::duration<double, std::milli>(t1 - t0).count(), ops);
        std::remove(path.c_str());
    }

    void bench_population_save_load() {
        Parameters p;
        p.PopulationSize = 100;
        Genome g = SeedGenome(10, 3);
        Population pop(g, p, true, 1.0, 22);
        const auto path = std::string("/tmp/neatcpp_bench_pop.txt");
        const unsigned ops = 20;
        auto t0 = Clock::now();
        for (unsigned i = 0; i < ops; ++i) {
            pop.Save(path.c_str());
            Population pop2(path);
        }
        auto t1 = Clock::now();
        report("Population save+load", std::chrono::duration<double, std::milli>(t1 - t0).count(), ops);
        std::remove(path.c_str());
    }

}  // namespace

int main() {
    std::printf("NEAT.cpp benchmarks\n");
    bench_phenotype_build_small();
    bench_phenotype_build_large();
    bench_activate();
    bench_compatibility_distance();
    bench_mutate_genome();
    bench_epoch();
    bench_xor_solve();
    bench_genome_save_load();
    bench_population_save_load();
    return 0;
}
