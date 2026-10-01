// The three BCJR algorithms, as the arithmetic the recursions run in.
//
// MAP works on probabilities and is exact; it is the reference. log-MAP works
// on their logarithms, where adding two probabilities is max(a, b) plus the
// Jacobian correction log(1 + e^-|a-b|); it reads that correction from a small
// table, as practical decoders do, which is what makes it cheaper than MAP.
// max-log-MAP drops the correction and keeps the max. Each is a policy with the
// same operations, so a trellis is written once and compiled three times; the
// choice is made once per frame, not per edge.
#ifndef MSPRS_BCJR_HPP
#define MSPRS_BCJR_HPP

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace msprs
{

enum class Bcjr { map, log_map, max_log_map };

inline Bcjr bcjr_from(const std::string& s)
{
    if (s == "map") return Bcjr::map;
    if (s == "log-map") return Bcjr::log_map;
    if (s == "max-log-map") return Bcjr::max_log_map;
    throw std::invalid_argument("unknown BCJR '" + s + "', expected map, log-map or max-log-map");
}

inline std::string to_string(const Bcjr a)
{
    switch (a)
    {
        case Bcjr::map: return "map";
        case Bcjr::log_map: return "log-map";
        case Bcjr::max_log_map: return "max-log-map";
    }
    return "?";
}

namespace bcjr
{
constexpr double NEG_INF = -std::numeric_limits<double>::infinity();

//! log(1 + e^-d) on [0, 4) in steps of 1/8, sampled mid-step; beyond 4 it is
//! below 0.02 and taken as 0. The largest error is about 0.03.
inline const std::array<double, 32> jacobian_table = [] {
    std::array<double, 32> t{};
    for (size_t i = 0; i < t.size(); i++) t[i] = std::log1p(std::exp(-((double)i + 0.5) / 8.0));
    return t;
}();

//! Logarithms, with the Jacobian correction read from a table.
struct LogMap
{
    static double zero() { return NEG_INF; }
    static double one() { return 0.0; }
    static double mul(const double a, const double b) { return a + b; }
    static double div(const double a, const double b) { return a - b; }
    static double from_log(const double x) { return x; }
    static double llr(const double p0, const double p1) { return p0 - p1; }

    //! An infinite or NaN difference (either side -inf) fails d < 4 and
    //! returns the max, so -inf needs no branch of its own.
    static double add(const double a, const double b)
    {
        const double m = a > b ? a : b;
        const double d = std::abs(a - b);
        return d < 4.0 ? m + jacobian_table[(size_t)(d * 8.0)] : m;
    }

    //! log P(bit) up to a term both bits share, which the per-step
    //! normalisation removes.
    static double apriori(const double llr, const int bit) { return bit == 0 ? 0.5 * llr : -0.5 * llr; }
};

//! Logarithms, with max in place of the Jacobian logarithm.
struct MaxLogMap
{
    static double zero() { return NEG_INF; }
    static double one() { return 0.0; }
    static double mul(const double a, const double b) { return a + b; }
    static double div(const double a, const double b) { return a - b; }
    static double from_log(const double x) { return x; }
    static double llr(const double p0, const double p1) { return p0 - p1; }
    static double add(const double a, const double b) { return a > b ? a : b; }

    //! log P(bit) up to a term both bits share, which the per-step
    //! normalisation removes.
    static double apriori(const double llr, const int bit) { return bit == 0 ? 0.5 * llr : -0.5 * llr; }
};

//! Probabilities.
struct Map
{
    static double zero() { return 0.0; }
    static double one() { return 1.0; }
    static double mul(const double a, const double b) { return a * b; }
    static double div(const double a, const double b) { return a / b; }
    static double add(const double a, const double b) { return a + b; }
    static double from_log(const double x) { return std::exp(x); }
    static double llr(const double p0, const double p1) { return std::log(p0) - std::log(p1); }

    static double apriori(const double llr, const int bit)
    {
        const double l = (bit == 0) ? llr : -llr;
        return 1.0 / (1.0 + std::exp(-l));
    }
};
} // namespace bcjr

} // namespace msprs

#endif
