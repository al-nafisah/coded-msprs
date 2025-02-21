// Oversampled received waveform, for eye diagrams.
//
// Modulate, root-raised-cosine shape, add noise at a commanded Eb/N0, matched
// filter. The sample rate is a display choice: with unit-energy shaping and
// matched filtering the noise variance at the symbol instants does not depend
// on it, so the eye closes by the same amount at any oversampling.
#ifndef MSPRS_EYE_HPP
#define MSPRS_EYE_HPP

#include <cmath>
#include <random>
#include <vector>

#include "MSPRS/Modem_FTN.hpp"
#include "MSPRS/Modem_MSPRS.hpp"
#include "MSPRS/Taps.hpp"

namespace msprs
{

struct EyeConfig
{
    int    symbols  = 3000;
    int    sps      = 32;
    double rolloff  = 0.35;
    int    span     = 12;
    double eb_no_db = 15.11;
    int    seed     = 20260918;
};

struct Eye
{
    std::vector<double> samples;
    int                 sps;
    int                 delay;   // index of the first symbol instant
};

inline Eye eye_waveform(const EyeConfig& cfg, const Taps& taps)
{
    const int nbits = 2 * cfg.symbols;
    std::mt19937_64 gen((uint64_t)cfg.seed);

    std::vector<int> bits((size_t)nbits);
    for (int i = 0; i < nbits; i++) bits[(size_t)i] = (int)(gen() & 1ull);

    Modem_MSPRS<> m(nbits, taps);
    std::vector<float> sym((size_t)Modem_MSPRS<>::size_mod(nbits, taps.L0));
    m.modulate(bits, sym);

    double es = 0.0;
    for (float v : sym) es += (double)v * v;
    es /= (double)sym.size();
    const double scale = 1.0 / std::sqrt(es);

    auto h = rrc_taps(cfg.rolloff, cfg.sps, cfg.span);
    double e = 0.0;
    for (double v : h) e += v * v;
    for (double& v : h) v /= std::sqrt(e);

    // upsample and shape
    std::vector<double> up(sym.size() * (size_t)cfg.sps, 0.0);
    for (size_t i = 0; i < sym.size(); i++) up[i * (size_t)cfg.sps] = (double)sym[i] * scale;

    auto convolve = [](const std::vector<double>& x, const std::vector<double>& k)
    {
        std::vector<double> y(x.size() + k.size() - 1, 0.0);
        for (size_t i = 0; i < x.size(); i++)
            if (x[i] != 0.0)
                for (size_t j = 0; j < k.size(); j++) y[i + j] += x[i] * k[j];
        return y;
    };

    auto tx = convolve(up, h);

    // Rate 2 carries two bits per symbol and this run is uncoded, so Eb = Es/2
    // with Es = 1, the same convention the BER sweep uses.
    const double eb = 0.5;
    const double nv = eb / (2.0 * std::pow(10.0, cfg.eb_no_db / 10.0));
    std::normal_distribution<double> nd(0.0, std::sqrt(nv));
    for (double& v : tx) v += nd(gen);

    Eye out;
    out.samples = convolve(tx, h);
    out.sps     = cfg.sps;
    out.delay   = (int)h.size() - 1;   // both filters are causal
    return out;
}

} // namespace msprs

#endif
