// The turbo receiver in integer arithmetic, bit-true to what the FPGA computes.
//
// Both SISOs run max-log-MAP on costs, -log P up to a constant per step, so
// smaller is better and every metric is an integer. Max-log is invariant to a
// common scale of the channel metric and the a priori, so the equaliser's
// metric is (y - s)^2 on the received grid with no 1/(2 sigma^2) and no noise
// estimate: the scale is set once, by the amplitude the front end normalises
// the symbols to. In floating point this is exact; only the fixed grid feels
// the noise level, because one LSB is a larger LLR at high SNR.
//
// Words and their rules:
//   y      received symbol, y_bits signed, y_amp LSBs per unit of amplitude
//          (Es = 1); the constellation is rounded to the same grid
//   bm     branch metric, (y - s)^2 >> bm_shift rounded, saturated to bm_bits
//          unsigned
//   sm     state metric, the distance above the best state at that step
//          (subtract-min normalisation), saturated to sm_bits unsigned
//   msg    the LLRs exchanged, symmetric signed codes of eq_bits / dec_bits,
//          one step = 2^eq_shift / 2^dec_shift metric LSBs, rounded half away
//          from zero, after an optional extrinsic scale of eq_scale/4 and
//          dec_scale/4
//   dsm    decoder state metric, dsm_bits unsigned, in steps of the
//          equaliser's message
//
// A bit's a priori enters as the cost b * La of deciding b = 1, which differs
// from -log P(b) by a constant per step that the normalisation removes.
// Unreachable states start at the top of the state metric range.
#ifndef MSPRS_FIXED_HPP
#define MSPRS_FIXED_HPP

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "MSPRS/Bounds.hpp"
#include "MSPRS/NSC.hpp"
#include "MSPRS/Params.hpp"
#include "MSPRS/Taps.hpp"

namespace msprs
{

struct FixedFormat
{
    int    y_bits    = 8;
    double y_amp     = 26.1;
    int    bm_shift  = 5;
    int    bm_bits   = 7;
    int    sm_bits   = 7;
    int    eq_bits   = 4;
    int    eq_shift  = 3;
    int    dec_bits  = 4;
    int    dec_shift = 3;
    int    dsm_bits  = 5;
    int    eq_scale  = 4;
    int    dec_scale = 4;

    //! "key=value,key=value" over the fields above.
    void set(const std::string& spec)
    {
        std::stringstream ss(spec);
        for (std::string kv; std::getline(ss, kv, ',');)
        {
            const auto eq = kv.find('=');
            if (eq == std::string::npos) throw std::invalid_argument("--fx wants key=value, got '" + kv + "'");
            const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
            if      (k == "y_bits")    y_bits    = std::stoi(v);
            else if (k == "y_amp")     y_amp     = std::stod(v);
            else if (k == "bm_shift")  bm_shift  = std::stoi(v);
            else if (k == "bm_bits")   bm_bits   = std::stoi(v);
            else if (k == "sm_bits")   sm_bits   = std::stoi(v);
            else if (k == "eq_bits")   eq_bits   = std::stoi(v);
            else if (k == "eq_shift")  eq_shift  = std::stoi(v);
            else if (k == "dec_bits")  dec_bits  = std::stoi(v);
            else if (k == "dec_shift") dec_shift = std::stoi(v);
            else if (k == "dsm_bits")  dsm_bits  = std::stoi(v);
            else if (k == "eq_scale")  eq_scale  = std::stoi(v);
            else if (k == "dec_scale") dec_scale = std::stoi(v);
            else throw std::invalid_argument("unknown fixed-point field '" + k + "'");
        }
        check();
    }

    void check() const
    {
        auto in = [](int v, int lo, int hi) { return v >= lo && v <= hi; };
        if (!in(y_bits, 3, 16) || !in(bm_bits, 2, 16) || !in(sm_bits, 2, 24) || !in(dsm_bits, 2, 24) ||
            !in(eq_bits, 2, 12) || !in(dec_bits, 2, 12) || !in(bm_shift, 0, 16) || !in(eq_shift, 0, 12) ||
            !in(dec_shift, 0, 12) || !in(eq_scale, 1, 4) || !in(dec_scale, 1, 4) || !(y_amp > 0.0))
            throw std::invalid_argument("fixed-point format out of range");
    }

    nlohmann::json json() const
    {
        return { {"y_bits", y_bits}, {"y_amp", y_amp}, {"bm_shift", bm_shift}, {"bm_bits", bm_bits},
                 {"sm_bits", sm_bits}, {"eq_bits", eq_bits}, {"eq_shift", eq_shift}, {"dec_bits", dec_bits},
                 {"dec_shift", dec_shift}, {"dsm_bits", dsm_bits}, {"eq_scale", eq_scale},
                 {"dec_scale", dec_scale} };
    }
};

//! The chosen formats, from the "fixed_point" section of config.json.
inline FixedFormat load_fixed_format(const std::string& path = Params::default_path())
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open '" + path + "'");
    nlohmann::json j;
    in >> j;
    const auto& t = j.at("fixed_point").at("turbo");
    FixedFormat f;
    f.y_bits = t.at("y_bits"); f.y_amp = t.at("y_amp"); f.bm_shift = t.at("bm_shift");
    f.bm_bits = t.at("bm_bits"); f.sm_bits = t.at("sm_bits"); f.eq_bits = t.at("eq_bits");
    f.eq_shift = t.at("eq_shift"); f.dec_bits = t.at("dec_bits"); f.dec_shift = t.at("dec_shift");
    f.dsm_bits = t.at("dsm_bits"); f.eq_scale = t.at("eq_scale"); f.dec_scale = t.at("dec_scale");
    f.check();
    return f;
}

namespace fx
{
//! Largest code of a symmetric signed word; -top() .. top() are its values.
inline int top(const int bits) { return (1 << (bits - 1)) - 1; }

inline int sat(const long v, const int bits)
{
    const long t = top(bits);
    return (int)std::max(-t, std::min(t, v));
}

//! v / 2^sh rounded half away from zero; a negative sh shifts left.
inline long round_shift(const long v, const int sh)
{
    if (sh <= 0) return v * (1L << -sh);
    const long h = 1L << (sh - 1);
    return v >= 0 ? (v + h) >> sh : -((-v + h) >> sh);
}

//! An extrinsic value to a message: scale by q/4, round to the message step
//! 2^sh, saturate to `bits`.
inline int message(const long ext, const int q, const int sh, const int bits)
{
    return sat(round_shift(ext * q, sh + 2), bits);
}
} // namespace fx

//! The MS-PRS equaliser: y words and decoder messages in, equaliser messages out.
class FixedEqualiser
{
  public:
    FixedEqualiser(const int N, const Taps& taps, const FixedFormat& f)
      : f(f), tr(taps), L((N + taps.L0) / 2), M(1 << (taps.L0 - 1)),
        n0((N + taps.L0) / 2 - taps.L0 + 1), even_L0(taps.L0 % 2 == 0),
        sym(tr.out.size()), D((size_t)(((N + taps.L0) / 2) + 1) * (1 << (taps.L0 - 1))),
        E(D.size()), cost((size_t)((N + taps.L0) / 2) * (1 << (taps.L0 - 1)) * 4)
    {
        for (size_t i = 0; i < sym.size(); i++) sym[i] = (int)std::lround(f.y_amp * tr.out[i]);
    }

    int symbols() const { return L; }
    const std::vector<int>& constellation() const { return sym; }

    //! A received sample on the y grid, as the front end delivers it.
    int quantise(const double r) const { return fx::sat(std::lround(f.y_amp * r), f.y_bits); }

    //! One pass. `La` and `Le` are N messages in transmission order. `trace`,
    //! when given, receives the state metrics forward then backward, step by
    //! step, M per step.
    void run(const int* y, const int* La, int* Le, std::vector<int>* trace = nullptr)
    {
        const long bm_max = (1L << f.bm_bits) - 1;
        const int  sm_max = (1 << f.sm_bits) - 1;

        int k = 0;
        for (int t = 0; t < L; t++)
        {
            const bool f0 = t < n0, f1 = !(even_L0 && t == L - 1);
            const long a0 = f0 ? (long)La[k++] * (1L << f.dec_shift) : 0;
            const long a1 = f1 ? (long)La[k++] * (1L << f.dec_shift) : 0;
            // Final step of an even-L0 trellis: the float receiver drops the
            // sample, and so does this one.
            const bool drop = even_L0 && t == L - 1;
            for (int s = 0; s < M; s++)
                for (int b0 = 0; b0 < 2; b0++)
                    for (int b1 = 0; b1 < 2; b1++)
                    {
                        const size_t e = ((size_t)s * 2 + b0) * 2 + b1;
                        if ((b0 && !f0) || (b1 && !f1)) { cost[(size_t)t * M * 4 + e] = NONE; continue; }
                        const long d  = (long)y[t] - sym[e];
                        const long bm = drop ? 0 : std::min(bm_max, fx::round_shift(d * d, f.bm_shift));
                        cost[(size_t)t * M * 4 + e] = (int)(bm + b0 * a0 + b1 * a1);
                    }
        }

        std::fill(D.begin(), D.begin() + M, sm_max);
        D[0] = 0;
        for (int t = 0; t < L; t++)
        {
            int* nx = &D[(size_t)(t + 1) * M];
            std::fill(nx, nx + M, BIG);
            for (int s = 0; s < M; s++)
                for (int b0 = 0; b0 < 2; b0++)
                {
                    const int c = best_b1(t, s, b0);
                    if (c == NONE) continue;
                    int& v = nx[tr.next[(size_t)s * 2 + b0]];
                    v = std::min(v, D[(size_t)t * M + s] + c);
                }
            normalise(nx, sm_max);
        }

        std::fill(E.begin() + (size_t)L * M, E.end(), sm_max);
        E[(size_t)L * M] = 0;
        for (int t = L - 1; t >= 0; t--)
        {
            int* cur = &E[(size_t)t * M];
            for (int s = 0; s < M; s++)
            {
                cur[s] = BIG;
                for (int b0 = 0; b0 < 2; b0++)
                {
                    const int c = best_b1(t, s, b0);
                    if (c == NONE) continue;
                    cur[s] = std::min(cur[s], c + E[(size_t)(t + 1) * M + tr.next[(size_t)s * 2 + b0]]);
                }
            }
            normalise(cur, sm_max);
        }

        if (trace)
        {
            trace->assign(D.begin(), D.end());
            trace->insert(trace->end(), E.begin(), E.end());
        }

        k = 0;
        for (int t = 0; t < L; t++)
        {
            long p[2][2] = { { BIG, BIG }, { BIG, BIG } };
            for (int s = 0; s < M; s++)
                for (int b0 = 0; b0 < 2; b0++)
                {
                    const long ab = (long)D[(size_t)t * M + s] + E[(size_t)(t + 1) * M + tr.next[(size_t)s * 2 + b0]];
                    for (int b1 = 0; b1 < 2; b1++)
                    {
                        const int c = cost[(size_t)t * M * 4 + ((size_t)s * 2 + b0) * 2 + b1];
                        if (c != NONE) p[b0][b1] = std::min(p[b0][b1], ab + c);
                    }
                }
            const bool f0 = t < n0, f1 = !(even_L0 && t == L - 1);
            if (f0)
            {
                const long llr = std::min(p[1][0], p[1][1]) - std::min(p[0][0], p[0][1]);
                Le[k] = fx::message(llr - (long)La[k] * (1L << f.dec_shift), f.eq_scale, f.eq_shift, f.eq_bits);
                k++;
            }
            if (f1)
            {
                const long llr = std::min(p[0][1], p[1][1]) - std::min(p[0][0], p[1][0]);
                Le[k] = fx::message(llr - (long)La[k] * (1L << f.dec_shift), f.eq_scale, f.eq_shift, f.eq_bits);
                k++;
            }
        }
    }

  private:
    static constexpr int NONE = 1 << 30;   // an edge the trellis does not have
    static constexpr int BIG  = 1 << 29;   // above any reachable sum

    int best_b1(const int t, const int s, const int b0) const
    {
        const int* c = &cost[(size_t)t * M * 4 + ((size_t)s * 2 + b0) * 2];
        if (c[0] == NONE) return NONE;   // b0 forced to 0 here
        return std::min(c[0], c[1]);     // a missing b1 = 1 edge is NONE, never the min
    }

    void normalise(int* m, const int top) const
    {
        const int lo = *std::min_element(m, m + M);
        for (int s = 0; s < M; s++) m[s] = std::min(top, m[s] - lo);
    }

    const FixedFormat f;
    const PairTrellis tr;
    const int         L, M, n0;
    const bool        even_L0;
    std::vector<int>  sym, D, E, cost;
};

//! The (5,7) decoder on the coded bits: equaliser messages in (deinterleaved),
//! decoder messages and information-bit decisions out.
class FixedDecoder
{
  public:
    FixedDecoder(const int K, const NSC_Trellis& tr, const FixedFormat& f)
      : f(f), tr(tr), K(K), L(K + tr.memory), M(tr.n_states), n(tr.n_out),
        D((size_t)(K + tr.memory + 1) * tr.n_states), E(D.size()),
        cost((size_t)(K + tr.memory) * tr.n_states * 2) {}

    void run(const int* Lin, int* Lext, int* u_hat, std::vector<int>* trace = nullptr)
    {
        const int sm_max = (1 << f.dsm_bits) - 1;

        for (int t = 0; t < L; t++)
            for (int s = 0; s < M; s++)
                for (int u = 0; u < 2; u++)
                {
                    int c = 0;
                    for (int j = 0; j < n; j++) c += tr.out_bits[(size_t)s * 2 + u][j] * Lin[n * t + j];
                    cost[((size_t)t * M + s) * 2 + u] = (u == 1 && t >= K) ? NONE : c;   // zero tail
                }

        std::fill(D.begin(), D.begin() + M, sm_max);
        D[0] = 0;
        for (int t = 0; t < L; t++)
        {
            int* nx = &D[(size_t)(t + 1) * M];
            std::fill(nx, nx + M, BIG);
            for (int s = 0; s < M; s++)
                for (int u = 0; u < 2; u++)
                {
                    const int c = cost[((size_t)t * M + s) * 2 + u];
                    if (c == NONE) continue;
                    int& v = nx[tr.nxt[(size_t)s * 2 + u]];
                    v = std::min(v, D[(size_t)t * M + s] + c);
                }
            normalise(nx, sm_max);
        }

        std::fill(E.begin() + (size_t)L * M, E.end(), sm_max);
        E[(size_t)L * M] = 0;
        for (int t = L - 1; t >= 0; t--)
        {
            int* cur = &E[(size_t)t * M];
            for (int s = 0; s < M; s++)
            {
                cur[s] = BIG;
                for (int u = 0; u < 2; u++)
                {
                    const int c = cost[((size_t)t * M + s) * 2 + u];
                    if (c == NONE) continue;
                    cur[s] = std::min(cur[s], c + E[(size_t)(t + 1) * M + tr.nxt[(size_t)s * 2 + u]]);
                }
            }
            normalise(cur, sm_max);
        }

        if (trace)
        {
            trace->assign(D.begin(), D.end());
            trace->insert(trace->end(), E.begin(), E.end());
        }

        const int sh = f.dec_shift - f.eq_shift;
        std::vector<long> p0((size_t)n), p1((size_t)n);
        for (int t = 0; t < L; t++)
        {
            std::fill(p0.begin(), p0.end(), (long)BIG);
            std::fill(p1.begin(), p1.end(), (long)BIG);
            long u0 = BIG, u1 = BIG;
            for (int s = 0; s < M; s++)
                for (int u = 0; u < 2; u++)
                {
                    const int c = cost[((size_t)t * M + s) * 2 + u];
                    if (c == NONE) continue;
                    const long v = (long)D[(size_t)t * M + s] + c + E[(size_t)(t + 1) * M + tr.nxt[(size_t)s * 2 + u]];
                    (u ? u1 : u0) = std::min(u ? u1 : u0, v);
                    for (int j = 0; j < n; j++)
                    {
                        long& p = tr.out_bits[(size_t)s * 2 + u][j] ? p1[(size_t)j] : p0[(size_t)j];
                        p = std::min(p, v);
                    }
                }
            for (int j = 0; j < n; j++)
                Lext[n * t + j] = fx::message(p1[(size_t)j] - p0[(size_t)j] - Lin[n * t + j], f.dec_scale, sh, f.dec_bits);
            if (t < K) u_hat[t] = (u1 < u0) ? 1 : 0;
        }
    }

  private:
    static constexpr int NONE = 1 << 30;
    static constexpr int BIG  = 1 << 29;

    void normalise(int* m, const int top) const
    {
        const int lo = *std::min_element(m, m + M);
        for (int s = 0; s < M; s++) m[s] = std::min(top, m[s] - lo);
    }

    const FixedFormat f;
    const NSC_Trellis tr;
    const int         K, L, M, n;
    std::vector<int>  D, E, cost;
};

//! What one pass sends across the loop, and the state metrics of both SISOs:
//! the vectors the HDL is tested against.
struct PassTrace
{
    std::vector<int> la;       //!< decoder messages into the equaliser, transmission order
    std::vector<int> le;       //!< equaliser messages out, transmission order
    std::vector<int> le_n;     //!< the same deinterleaved, into the decoder
    std::vector<int> ld_n;     //!< decoder messages out, code order
    std::vector<int> eq_sm;    //!< equaliser state metrics, forward then backward
    std::vector<int> dec_sm;   //!< decoder state metrics, forward then backward
};

//! The whole integer receiver: equaliser, deinterleaver, decoder, interleaver.
class FixedTurbo
{
  public:
    FixedTurbo(const int K, const NSC_Trellis& tr, const Taps& taps, const FixedFormat& f,
               const std::vector<uint32_t>& pi)
      : eq((int)pi.size(), taps, f), dec(K, tr, f), pi(pi), la(pi.size(), 0), le(pi.size()),
        le_n(pi.size()), ld_n(pi.size())
    {
        if ((int)pi.size() != tr.codeword_length(K)) throw std::runtime_error("interleaver does not match the code");
    }

    FixedEqualiser&       equaliser() { return eq; }
    const FixedEqualiser& equaliser() const { return eq; }

    //! The first pass has no a priori.
    void start() { std::fill(la.begin(), la.end(), 0); }

    //! One equaliser and one decoder pass; u_hat gets the K decisions.
    void pass(const int* y, int* u_hat, PassTrace* t = nullptr)
    {
        if (t) t->la = la;
        eq.run(y, la.data(), le.data(), t ? &t->eq_sm : nullptr);
        for (size_t i = 0; i < pi.size(); i++) le_n[pi[i]] = le[i];
        dec.run(le_n.data(), ld_n.data(), u_hat, t ? &t->dec_sm : nullptr);
        for (size_t i = 0; i < pi.size(); i++) la[i] = ld_n[pi[i]];
        if (t) { t->le = le; t->le_n = le_n; t->ld_n = ld_n; }
    }

  private:
    FixedEqualiser        eq;
    FixedDecoder          dec;
    std::vector<uint32_t> pi;
    std::vector<int>      la, le, le_n, ld_n;
};

} // namespace msprs

#endif
