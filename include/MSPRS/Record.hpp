// Writes one JSON record per (scheme, Eb/N0) under results/ber.
//
// A zero-error point is stored as an upper bound, never as ber = 0, so a
// figure can draw it as a bound rather than silently dropping it.
#ifndef MSPRS_RECORD_HPP
#define MSPRS_RECORD_HPP

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <nlohmann/json.hpp>

#include "MSPRS/Sweep.hpp"

namespace msprs
{

namespace detail
{
// Regularised incomplete beta, continued-fraction form.
inline double betacf(double a, double b, double x)
{
    const int    MAXIT = 200;
    const double EPS = 3.0e-14, FPMIN = 1.0e-300;
    const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
    double c = 1.0, d = 1.0 - qab * x / qap;
    if (std::abs(d) < FPMIN) d = FPMIN;
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m <= MAXIT; m++)
    {
        const int m2 = 2 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d; if (std::abs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c; if (std::abs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d; if (std::abs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c; if (std::abs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::abs(del - 1.0) < EPS) break;
    }
    return h;
}

inline double betai(double a, double b, double x)
{
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    const double bt = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b)
                               + a * std::log(x) + b * std::log1p(-x));
    return (x < (a + 1.0) / (a + b + 2.0)) ? bt * betacf(a, b, x) / a
                                           : 1.0 - bt * betacf(b, a, 1.0 - x) / b;
}

// Beta quantile by bisection. The interval is monotone in x, so 200 halvings
// reach double precision from [0,1].
inline double beta_inv(double p, double a, double b)
{
    double lo = 0.0, hi = 1.0;
    for (int i = 0; i < 200; i++)
    {
        const double mid = 0.5 * (lo + hi);
        (betai(a, b, mid) < p) ? lo = mid : hi = mid;
    }
    return 0.5 * (lo + hi);
}
} // namespace detail

struct Interval { double lo, hi; };

inline Interval clopper_pearson(long long k, long long n, double alpha = 0.05)
{
    if (n <= 0) return { 0.0, 1.0 };
    const double lo = (k == 0) ? 0.0 : detail::beta_inv(alpha / 2.0, (double)k, (double)(n - k + 1));
    const double hi = (k == n) ? 1.0 : detail::beta_inv(1.0 - alpha / 2.0, (double)(k + 1), (double)(n - k));
    return { lo, hi };
}

inline void mkdir_p(const std::string& path)
{
    std::string acc;
    for (size_t i = 0; i < path.size(); i++)
    {
        acc += path[i];
        if (path[i] == '/' || i + 1 == path.size())
            ::mkdir(acc.c_str(), 0755);
    }
}

// `stamp` is the timestamp written into the record; empty means leave it out.
inline std::string write_point(const std::string& results_dir, const std::string& scheme,
                               const Point& pt, const nlohmann::json& config,
                               const std::string& implementation, const std::string& stamp)
{
    const auto ci = clopper_pearson(pt.errors, pt.bits);
    const bool bound = (pt.errors == 0);

    nlohmann::json r;
    r["eb_no_db"]  = pt.eb_no_db;
    r["ber"]       = bound ? ci.hi : (double)pt.errors / (double)pt.bits;
    r["ers_cnt"]   = pt.errors;
    r["bits_cnt"]  = pt.bits;
    r["n_frames"]  = pt.frames;
    r["ber_lo95"]  = ci.lo;
    r["ber_ub95"]  = ci.hi;
    r["is_upper_bound"] = bound;
    r["duration_s"] = pt.seconds;
    if (!pt.per_iter.empty()) r["ers_per_iter"] = pt.per_iter;
    r["config"]    = config;
    r["implementation"] = implementation;
    if (!stamp.empty()) r["timestamp"] = stamp;

    const std::string dir = results_dir + "/" + scheme;
    mkdir_p(dir);
    std::ostringstream name;
    name << dir << "/snr_" << std::fixed << std::setprecision(1) << pt.eb_no_db << "dB.json";

    std::ofstream out(name.str());
    if (!out) throw std::runtime_error("cannot write '" + name.str() + "'");
    out << r.dump(2) << "\n";
    return name.str();
}

} // namespace msprs

#endif
