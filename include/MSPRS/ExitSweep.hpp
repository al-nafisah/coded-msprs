// Measures the equalizer's EXIT characteristic over an Eb/N0 grid.
//
// Es = 1 and m = 2 bits per symbol at Rc = 1/2, so Es/N0 = Eb/N0, the same
// convention the BER sweep uses.
//
// Results depend on the thread count: a-priori draws are seeded per thread and
// the IA grid is dealt out round-robin, so reproducing a run needs the same
// --threads.
#ifndef MSPRS_EXIT_SWEEP_HPP
#define MSPRS_EXIT_SWEEP_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "MSPRS/Exit.hpp"
#include "MSPRS/Modem_MSPRS.hpp"
#include "MSPRS/Taps.hpp"

namespace msprs
{

struct ExitConfig
{
    int    bits     = 4998;
    int    n_ia     = 100;
    int    n_trials = 50;
    int    threads  = 1;
    bool   fresh_rx = false;   // redraw the channel per trial
    double ia_scale = 0.999, ia_lo = 1e-3, ia_hi = 1.0;
    double ebn0_min = 0.0, ebn0_max = 10.01, ebn0_step = 0.2;
};

struct ExitPoint
{
    double eb_no_db, ia, ie_avg, ie_hist, ie_mag, ia_measured;
};

inline std::vector<ExitPoint> exit_sweep(const ExitConfig& cfg, const Taps& taps)
{
    const int K   = cfg.bits;
    const int Ns  = Modem_MSPRS<>::size_mod(K, taps.L0);
    const int nth = std::max(1, cfg.threads);

    std::vector<double> IA((size_t)cfg.n_ia);
    for (int i = 0; i < cfg.n_ia; i++)
        IA[(size_t)i] = cfg.ia_scale *
            (cfg.ia_lo + (cfg.ia_hi - cfg.ia_lo) * (double)i / (double)(cfg.n_ia - 1));

    std::vector<std::unique_ptr<Modem_MSPRS<>>> mdm;
    for (int t = 0; t < nth; t++) mdm.emplace_back(new Modem_MSPRS<>(K, taps));

    std::vector<ExitPoint> out;
    for (double ebn0 = cfg.ebn0_min; ebn0 < cfg.ebn0_max; ebn0 += cfg.ebn0_step)
    {
        const float sigma = (float)std::sqrt(1.0 / (2.0 * std::pow(10.0, ebn0 / 10.0)));
        const std::vector<float> CP = { sigma };

        std::mt19937_64 gen(0xE117u ^ (uint64_t)std::lround(ebn0 * 1000.0));
        std::vector<int> bits((size_t)K);
        for (int i = 0; i < K; i++) bits[(size_t)i] = (int)(gen() & 1ull);
        std::vector<float> sym((size_t)Ns), rx((size_t)Ns);
        mdm[0]->modulate(bits, sym);
        std::normal_distribution<double> nd(0.0, 1.0);
        for (int i = 0; i < Ns; i++) rx[(size_t)i] = sym[(size_t)i] + (float)(sigma * nd(gen));

        std::vector<std::array<double, 4>> res((size_t)cfg.n_ia);
        std::vector<std::thread> pool;
        for (int t = 0; t < nth; t++)
            pool.emplace_back([&, t] {
                std::mt19937_64 rng(0xA5A5u + 7919ull * (uint64_t)t
                                    + 104729ull * (uint64_t)std::lround(ebn0 * 1000.0));
                std::vector<double> la((size_t)K), ext((size_t)K);
                std::vector<float>  laf((size_t)K), extf((size_t)K);
                std::vector<int>    lb((size_t)K);
                std::vector<float>  lsym((size_t)Ns), lrx((size_t)Ns);
                std::normal_distribution<double> lnd(0.0, 1.0);

                for (int k = t; k < cfg.n_ia; k += nth)
                {
                    const double sa = i_inv(IA[(size_t)k]);
                    double a = 0, h = 0, m = 0, iam = 0;
                    for (int tr = 0; tr < cfg.n_trials; tr++)
                    {
                        // One channel realisation per Eb/N0 leaves its own noise
                        // in the estimate; averaging over a-priori draws cannot
                        // remove it because every trial sees the same rx.
                        const std::vector<int>*   pb = &bits;
                        const std::vector<float>* pr = &rx;
                        if (cfg.fresh_rx)
                        {
                            for (int i = 0; i < K; i++) lb[(size_t)i] = (int)(rng() & 1ull);
                            mdm[(size_t)t]->modulate(lb, lsym);
                            for (int i = 0; i < Ns; i++)
                                lrx[(size_t)i] = lsym[(size_t)i] + (float)(sigma * lnd(rng));
                            pb = &lb; pr = &lrx;
                        }
                        gen_llrs(*pb, sa, rng, la);
                        if (tr == 0) iam = mi_avg(la, *pb);
                        for (int i = 0; i < K; i++) laf[(size_t)i] = (float)(-la[(size_t)i]);
                        mdm[(size_t)t]->tdemodulate(CP, *pr, laf, extf);
                        for (int i = 0; i < K; i++) ext[(size_t)i] = -(double)extf[(size_t)i];
                        a += mi_avg(ext, *pb);
                        h += mi_hist(ext, *pb);
                        m += mi_mag(ext);
                    }
                    res[(size_t)k] = { a / cfg.n_trials, h / cfg.n_trials, m / cfg.n_trials, iam };
                }
            });
        for (auto& th : pool) th.join();

        for (int k = 0; k < cfg.n_ia; k++)
            out.push_back({ ebn0, IA[(size_t)k], res[(size_t)k][0], res[(size_t)k][1],
                            res[(size_t)k][2], res[(size_t)k][3] });
    }
    return out;
}

} // namespace msprs

#endif
