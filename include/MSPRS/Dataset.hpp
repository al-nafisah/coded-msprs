// Training data for a learned inner SISO equaliser.
//
// Each sample is one coded bit: a window of received symbols around it, a
// window of a priori LLRs with the bit's own entry zeroed, and the extrinsic
// LLR the exact BCJR produced. Training on that pair teaches the network to
// imitate the BCJR without the trellis.
//
// LLRs are in the modem's convention, ln P(b=0)/P(b=1).
#ifndef MSPRS_DATASET_HPP
#define MSPRS_DATASET_HPP

#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "MSPRS/Exit.hpp"
#include "MSPRS/Modem_MSPRS.hpp"
#include "MSPRS/Taps.hpp"

namespace msprs
{

struct DatasetConfig
{
    int    bits      = 4998;
    int    frames    = 200;
    int    half_sym  = 4;     // received symbols each side
    int    half_bit  = 8;     // a priori LLRs each side
    double ebn0_min  = 2.0;   // a sample's Eb/N0 is drawn in this range
    double ebn0_max  = 7.0;
    int    stride    = 7;     // take every stride-th bit, to decorrelate samples
    int    seed      = 1;
};

inline int dataset_features(const DatasetConfig& c)
{
    return (2 * c.half_sym + 1) + (2 * c.half_bit + 1) + 2;  // + sigma, sub-stream
}

// Writes X.bin (n x D float32) and y.bin (n float32).
inline long long write_dataset(const DatasetConfig& cfg, const Taps& taps,
                               const std::string& x_path, const std::string& y_path)
{
    const int N  = cfg.bits;
    const int Ns = Modem_MSPRS<>::size_mod(N, taps.L0);
    const int D  = dataset_features(cfg);

    Modem_MSPRS<> modem(N, taps);
    std::mt19937_64 gen((uint64_t)cfg.seed);
    std::uniform_real_distribution<double> snr(cfg.ebn0_min, cfg.ebn0_max);
    std::uniform_real_distribution<double> ia_pick(0.0, 0.95);
    std::normal_distribution<double> nd(0.0, 1.0);

    std::ofstream xf(x_path, std::ios::binary), yf(y_path, std::ios::binary);
    if (!xf || !yf) throw std::runtime_error("cannot open dataset output");

    std::vector<int>   bits((size_t)N);
    std::vector<float> sym((size_t)Ns), rx((size_t)Ns);
    std::vector<float> la((size_t)N), ext((size_t)N);
    std::vector<float> feat((size_t)D);

    long long written = 0;
    for (int f = 0; f < cfg.frames; f++)
    {
        for (int i = 0; i < N; i++) bits[(size_t)i] = (int)(gen() & 1ull);
        modem.modulate(bits, sym);

        const double ebn0  = snr(gen);
        const double sigma = std::sqrt(1.0 / (2.0 * std::pow(10.0, ebn0 / 10.0)));
        for (int i = 0; i < Ns; i++) rx[(size_t)i] = sym[(size_t)i] + (float)(sigma * nd(gen));

        // Consistent-channel a priori at a random mutual information, so the
        // network sees the whole range the turbo loop will walk through.
        const double sa = i_inv(ia_pick(gen));
        for (int i = 0; i < N; i++)
        {
            const double mean = 0.5 * sa * sa * (1.0 - 2.0 * bits[(size_t)i]);
            la[(size_t)i] = (float)(mean + sa * nd(gen));
        }

        const std::vector<float> CP = { (float)sigma };
        modem.tdemodulate(CP, rx, la, ext);

        for (int i = cfg.half_bit; i < N - cfg.half_bit; i += cfg.stride)
        {
            const int c = i / 2;   // two bits per symbol
            int k = 0;
            for (int s = -cfg.half_sym; s <= cfg.half_sym; s++)
            {
                const int idx = c + s;
                feat[(size_t)k++] = (idx >= 0 && idx < Ns) ? rx[(size_t)idx] : 0.0f;
            }
            for (int b = -cfg.half_bit; b <= cfg.half_bit; b++)
                feat[(size_t)k++] = (b == 0) ? 0.0f : la[(size_t)(i + b)];
            feat[(size_t)k++] = (float)sigma;   // the receiver knows its noise level
            // Even bits ride the FIR stream, odd bits the memoryless one. Without
            // this the network cannot tell which relationship it is inverting.
            feat[(size_t)k++] = (i % 2 == 0) ? 1.0f : -1.0f;

            xf.write(reinterpret_cast<const char*>(feat.data()), (std::streamsize)D * sizeof(float));
            yf.write(reinterpret_cast<const char*>(&ext[(size_t)i]), sizeof(float));
            written++;
        }
    }
    return written;
}

} // namespace msprs

#endif
