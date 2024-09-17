// EXIT machinery: J^-1, Gaussian a-priori LLRs, three MI estimators.
//
// The three estimators are kept because they fail differently. Averaging and
// histogram need the true bits; the magnitude form does not and is
// sign-agnostic, so a sign error shows as IE_mag agreeing while the other two
// collapse.
//
// These take the sign L = ln P(b=1)/P(b=0), opposite to the modems, so the
// sweep negates the extrinsic before measuring. Getting that wrong does not
// crash, it silently produces an EXIT curve that falls instead of rises.
#ifndef MSPRS_EXIT_HPP
#define MSPRS_EXIT_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace msprs
{

//! Scalar J^-1(I_A). ten Brink 2001, eq. 13-14.
inline double i_inv(const double ia)
{
    if (ia <= 0.3646) return 1.09542 * ia * ia + 0.214217 * ia + 2.33737 * std::sqrt(ia);
    double arg = 0.386013 * (1.0 - ia);
    if (arg < 1e-300) arg = 1e-300;
    return -0.706692 * std::log(arg) + 1.75017 * ia;
}

//! Gaussian a-priori LLRs of the given sigma_a, in the PYTHON sign.
template<class RNG>
inline void gen_llrs(const std::vector<int>& bits, const double sigma_a, RNG& rng, std::vector<double>& out)
{
    std::normal_distribution<double> n(0.0, 1.0);
    const double                     half_var = sigma_a * sigma_a * 0.5;
    out.resize(bits.size());
    for (size_t i = 0; i < bits.size(); i++)
        out[i] = (2.0 * bits[i] - 1.0) * half_var + sigma_a * n(rng);
}

//! Averaging estimator. ten Brink 2001, eq. 14. Needs the true bits.
inline double mi_avg(const std::vector<double>& llr, const std::vector<int>& bits)
{
    double total = 0.0;
    long   count = 0;
    for (size_t i = 0; i < llr.size(); i++)
    {
        const double s = (2 * bits[i] - 1) * llr[i];
        const double h = std::log2(1.0 + std::exp(-s));
        if (std::isfinite(h)) { total += h; count++; }
    }
    if (count == 0) return 0.0;
    return std::min(1.0, std::max(0.0, 1.0 - total / (double)count));
}

//! Histogram estimator, Scott's bandwidth. ten Brink 2001, eq. 15.
inline double mi_hist(const std::vector<double>& llr, const std::vector<int>& bits)
{
    long n0 = 0, n1 = 0;
    for (int b : bits) (b == 0 ? n0 : n1)++;
    if (n0 == 0 || n1 == 0) return 0.0;

    double s0 = 0, s1 = 0, q0 = 0, q1 = 0, lo = INFINITY, hi = -INFINITY;
    long   cnt = 0;
    for (size_t i = 0; i < llr.size(); i++)
    {
        const double v = llr[i];
        if (!std::isfinite(v)) continue;
        cnt++;
        lo = std::min(lo, v);
        hi = std::max(hi, v);
        if (bits[i] == 0) { s0 += v; q0 += v * v; }
        else              { s1 += v; q1 += v * v; }
    }
    if (cnt < 2) return 0.0;

    const double m0 = s0 / n0, m1 = s1 / n1;
    const double std0 = std::sqrt(std::max(q0 / n0 - m0 * m0, 0.0));
    const double std1 = std::sqrt(std::max(q1 / n1 - m1 * m1, 0.0));
    const double sp   = 0.5 * (std0 + std1);
    if (sp < 1e-10) return std::abs(m0 - m1) < 1e-10 ? 0.0 : 1.0;

    const double bw = 3.49 * sp * std::pow((double)cnt, -1.0 / 3.0);
    if (bw == 0.0) return 0.0;

    const long lo_bin = (long)std::floor(lo / bw) - 1;
    const long n_bins = (long)std::ceil(hi / bw) + 2 - lo_bin;

    std::vector<long> h0((size_t)n_bins, 0), h1((size_t)n_bins, 0);
    for (size_t i = 0; i < llr.size(); i++)
    {
        if (!std::isfinite(llr[i])) continue;
        const long idx = (long)std::floor(llr[i] / bw) - lo_bin;
        if (idx >= 0 && idx < n_bins) (bits[i] == 0 ? h0 : h1)[(size_t)idx]++;
    }

    double ie = 0.0;
    for (long b = 0; b < n_bins; b++)
    {
        const double p0b = (double)h0[(size_t)b] / n0;
        const double p1b = (double)h1[(size_t)b] / n1;
        const double pt  = p0b + p1b;
        if (pt <= 0.0) continue;
        if (p0b > 0.0) ie += 0.5 * p0b * std::log2(2.0 * p0b / pt);
        if (p1b > 0.0) ie += 0.5 * p1b * std::log2(2.0 * p1b / pt);
    }
    return std::min(1.0, std::max(0.0, ie));
}

//! Magnitude estimator. Hagenauer 2004, eq. 9. Sign-agnostic, no true bits.
inline double mi_mag(const std::vector<double>& llr)
{
    double total = 0.0;
    long   count = 0;
    for (double v : llr)
    {
        if (!std::isfinite(v)) continue;
        const double a   = std::abs(v);
        const double lae = (a > 0.0) ? a + std::log1p(std::exp(-a)) : std::log(2.0); // logaddexp(0, a)
        const double sig = 1.0 / (1.0 + std::exp(-a));
        const double h   = (lae - a * sig) / std::log(2.0);
        if (std::isfinite(h)) { total += h; count++; }
    }
    if (count == 0) return 0.0;
    return std::min(1.0, std::max(0.0, 1.0 - total / (double)count));
}

} // namespace msprs

#endif
