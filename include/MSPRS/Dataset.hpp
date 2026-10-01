// What the learned inner SISO equaliser sees, and the data it is trained on.
//
// Each sample is one coded bit: a window of received symbols around the bit's
// trellis step, a window of a priori LLRs with the bit's own entry zeroed, the
// noise level, and the sub-stream the bit rides. The label is the transmitted
// bit, and the network's output is trained as the extrinsic LLR. The exact
// BCJR extrinsic is kept only to compare against; the network never learns to
// copy it.
//
// LLRs are in the modem's convention, ln P(b=0)/P(b=1).
#ifndef MSPRS_DATASET_HPP
#define MSPRS_DATASET_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "MSPRS/Exit.hpp"
#include "MSPRS/Modem_MSPRS.hpp"
#include "MSPRS/Taps.hpp"

namespace msprs
{

//! Trellis step of one bit, and whether it drives the FIR stream x0.
struct Site
{
    int  step;
    bool fir;
};

//! Every bit's site, in the order the modulator consumes them: (b0, b1) per
//! step while x0 is free, then b1 alone while h0 is flushed.
inline std::vector<Site> bit_sites(const int N, const Taps& taps)
{
    const Modem_MSPRS<> m(N, taps);
    std::vector<Site>   s;
    s.reserve((size_t)N);
    for (int t = 0; t < Modem_MSPRS<>::size_mod(N, taps.L0); t++)
    {
        if (m.b0_free(t)) s.push_back({ t, true });
        if (m.b1_free(t)) s.push_back({ t, false });
    }
    if ((int)s.size() != N)
        throw std::runtime_error("bit_sites: trellis holds " + std::to_string(s.size()) +
                                 " bits, expected " + std::to_string(N));
    return s;
}

//! The network's input for one bit. The dataset writer and the receiver both
//! build it here, so training and inference cannot drift apart.
struct Window
{
    int               half_sym = 4;   // received symbols each side of the bit's step
    int               half_bit = 8;   // a priori LLRs each side of the bit
    // A ReLU network extrapolates wildly beyond what it was trained on, and a
    // converging turbo loop feeds back ever larger LLRs. Past this a bit is
    // settled anyway, so the input stops growing here.
    float             la_clip  = 20.0f;
    std::vector<Site> sites;

    Window(const int N, const Taps& taps) : sites(bit_sites(N, taps)) {}

    int dim() const { return (2 * half_sym + 1) + (2 * half_bit + 1) + 2; }

    void fill(const int i, const std::vector<float>& rx, const std::vector<float>& la,
              const float sigma, float* x) const
    {
        const int   N  = (int)sites.size();
        const int   Ns = (int)rx.size();
        const Site& st = sites[(size_t)i];

        int k = 0;
        for (int s = -half_sym; s <= half_sym; s++)
        {
            const int t = st.step + s;
            x[k++] = (t >= 0 && t < Ns) ? rx[(size_t)t] : 0.0f;
        }
        for (int b = -half_bit; b <= half_bit; b++)
        {
            const int j = i + b;
            x[k++] = (b == 0 || j < 0 || j >= N) ? 0.0f : std::clamp(la[(size_t)j], -la_clip, la_clip);
        }
        x[k++] = sigma;   // the receiver knows its noise level
        // Without the sub-stream the network cannot tell which relationship it
        // is inverting: the FIR one with memory, or the single tap.
        x[k++] = st.fir ? 1.0f : -1.0f;
    }
};

struct DatasetConfig
{
    int    bits     = 10000;   // coded bits per frame, as in the turbo loop
    int    frames   = 200;
    double ebn0_min = 2.0;     // Eb/N0 of the coded system, drawn per frame
    double ebn0_max = 7.0;
    double ia_max   = 0.999;   // a priori MI drawn up to here; a converged loop sits near 1
    int    stride   = 7;       // take every stride-th bit, to decorrelate samples
    int    seed     = 1;
};

// Writes, under `dir`: X.bin (n x dim float32 features), b.bin (n uint8 bits)
// and y.bin (n float32, the BCJR extrinsic, for comparison only).
inline long long write_dataset(const DatasetConfig& cfg, const Taps& taps, const std::string& dir)
{
    const int    N  = cfg.bits;
    const int    Ns = Modem_MSPRS<>::size_mod(N, taps.L0);
    const Window win(N, taps);
    const int    D  = win.dim();

    Modem_MSPRS<> modem(N, taps);
    std::mt19937_64 gen((uint64_t)cfg.seed);
    std::uniform_real_distribution<double> snr(cfg.ebn0_min, cfg.ebn0_max);
    std::uniform_real_distribution<double> ia_pick(0.0, cfg.ia_max);
    std::normal_distribution<double> nd(0.0, 1.0);

    std::ofstream xf(dir + "/X.bin", std::ios::binary), bf(dir + "/b.bin", std::ios::binary);
    std::ofstream yf(dir + "/y.bin", std::ios::binary);
    if (!xf || !bf || !yf) throw std::runtime_error("cannot open dataset output in '" + dir + "'");

    std::vector<int>   bits((size_t)N);
    std::vector<float> sym((size_t)Ns), rx((size_t)Ns);
    std::vector<float> la((size_t)N), ext((size_t)N), x((size_t)D);

    long long written = 0;
    for (int f = 0; f < cfg.frames; f++)
    {
        for (int i = 0; i < N; i++) bits[(size_t)i] = (int)(gen() & 1ull);
        modem.modulate(bits, sym);

        const float sigma = (float)noise_sigma(snr(gen), 0.5);
        for (int i = 0; i < Ns; i++) rx[(size_t)i] = sym[(size_t)i] + (float)(sigma * nd(gen));

        // Consistent Gaussian a priori at a random mutual information, so the
        // network sees the whole range the turbo loop walks through.
        const double sa = i_inv(ia_pick(gen));
        for (int i = 0; i < N; i++)
            la[(size_t)i] = (float)(0.5 * sa * sa * (1.0 - 2.0 * bits[(size_t)i]) + sa * nd(gen));

        modem.tdemodulate(std::vector<float>{ sigma }, rx, la, ext);

        // The offset moves with the frame, so the frame edges are sampled too.
        for (int i = f % cfg.stride; i < N; i += cfg.stride)
        {
            win.fill(i, rx, la, sigma, x.data());
            const uint8_t b = (uint8_t)bits[(size_t)i];
            xf.write(reinterpret_cast<const char*>(x.data()), (std::streamsize)(D * sizeof(float)));
            bf.write(reinterpret_cast<const char*>(&b), 1);
            yf.write(reinterpret_cast<const char*>(&ext[(size_t)i]), sizeof(float));
            written++;
        }
    }
    return written;
}

} // namespace msprs

#endif
