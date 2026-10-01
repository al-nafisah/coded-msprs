// Single-thread decoding time of the turbo receiver, per BCJR algorithm.
//
// One coded frame is drawn at the given Eb/N0 and run through one loop
// iteration, so the equaliser and the outer decoder are then timed on the
// inputs they see mid-loop. A frame costs iters + 1 passes of each.
#ifndef MSPRS_TIMING_HPP
#define MSPRS_TIMING_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

#include <aff3ct.hpp>

#include "MSPRS/Bcjr.hpp"
#include "MSPRS/Modem_MSPRS.hpp"
#include "MSPRS/NSC.hpp"
#include "MSPRS/Taps.hpp"

namespace msprs
{

struct PassTime
{
    double equaliser_us, decoder_us;
};

//! Median over `rounds` of the mean time of `reps` calls, after a warm-up.
template<class F>
inline double median_us(F&& f, const int reps = 20, const int rounds = 9)
{
    for (int i = 0; i < 3; i++) f();
    std::vector<double> t;
    for (int r = 0; r < rounds; r++)
    {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; i++) f();
        t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

inline PassTime time_passes(const Bcjr algo, const Taps& taps, const NSC_Trellis& tr, const int K,
                            const int itl_seed, const double ebn0)
{
    const int N  = tr.codeword_length(K);
    const int Ns = Modem_MSPRS<>::size_mod(N, taps.L0);

    Encoder_NSC<>      enc(K, tr);
    Decoder_NSC_SISO<> dec(K, tr);
    Modem_MSPRS<>      modem(N, taps);
    dec.set_bcjr(algo);
    modem.set_bcjr(algo);
    aff3ct::tools::Interleaver_core_random<> core(N, itl_seed, false);
    aff3ct::module::Interleaver<int>         itl_b(core);
    aff3ct::module::Interleaver<float>       itl_l(core);

    std::mt19937_64 gen(1);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<int>   u((size_t)K), cw((size_t)N), cw_i((size_t)N), hard((size_t)K);
    std::vector<float> sym((size_t)Ns), rx((size_t)Ns);
    std::vector<float> La_i((size_t)N, 0.f), Le_i((size_t)N), Le_n((size_t)N), De_n((size_t)N);
    for (auto& b : u) b = (int)(gen() & 1ull);
    enc.encode(u, cw);
    itl_b.interleave(cw, cw_i);
    modem.modulate(cw_i, sym);
    // Coded at Rc = 1/2 with m = 2, so Es/N0 = Eb/N0, as in the BER sweep.
    const std::vector<float> CP = { (float)std::sqrt(1.0 / (2.0 * std::pow(10.0, ebn0 / 10.0))) };
    for (int i = 0; i < Ns; i++) rx[(size_t)i] = sym[(size_t)i] + CP[0] * (float)nd(gen);

    modem.tdemodulate(CP, rx, La_i, Le_i);
    itl_l.deinterleave(Le_i, Le_n);
    dec.decode_both(Le_n.data(), De_n.data(), hard.data());
    itl_l.interleave(De_n, La_i);

    PassTime p;
    p.equaliser_us = median_us([&] { modem.tdemodulate(CP, rx, La_i, Le_i); });
    p.decoder_us   = median_us([&] { dec.decode_both(Le_n.data(), De_n.data(), hard.data()); });
    return p;
}

} // namespace msprs

#endif
