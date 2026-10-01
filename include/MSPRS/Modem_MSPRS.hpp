// Multi-Stream Partial Response Signaling as an AFF3CT modem.
//
// Rate 2: even bits drive x0 through the FIR h0 of length L0, odd bits drive
// x1 through the scalar h1.
//
//     s[k] = sum_j h0[j] x0[k-j] + h1 x1[k],   ||h0||^2 + h1^2 = 1
//
// Only x0 carries memory, so the trellis has 2^(L0-1) states and each branch
// label appears twice, once per sign of h1. Detection is BCJR, exposed as
// tdemodulate so the turbo loop can feed a-priori LLRs back in.
//
// MS-PRS runs at the Nyquist rate, so with an orthonormal pulse the
// matched-filter output is r[k] = s[k] + n[k] with white noise and the
// symbol-rate model is exact.
//
// Conventions: bit to symbol is x = 1 - 2b, LLR is ln P(b=0)/P(b=1), and
// CP[0] is sigma, not sigma^2. The branch metric divides by 2 sigma^2;
// dropping the 2 doubles every extrinsic LLR and wrecks turbo convergence
// while still looking plausible.
#ifndef MSPRS_MODEM_MSPRS_HPP
#define MSPRS_MODEM_MSPRS_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <aff3ct.hpp>

#include "MSPRS/Bcjr.hpp"
#include "MSPRS/Taps.hpp"

namespace msprs
{

namespace detail
{
constexpr double NEG_INF = bcjr::NEG_INF;

// The FTN reference modem still runs its own exact log-domain BCJR on these.

//! log(e^a + e^b), stable.
inline double log_add(const double a, const double b)
{
    if (a == NEG_INF) return b;
    if (b == NEG_INF) return a;
    const double d = std::abs(a - b);
    const double m = a > b ? a : b;
    return d > 20.0 ? m : m + std::log1p(std::exp(-d));
}

//! log P(bit) given an LLR in AFF3CT sign (positive -> bit 0).
inline double log_p(const double llr, const int bit)
{
    const double l = (bit == 0) ? llr : -llr;
    if (l > 10.0) return -1e-6;
    if (l < -10.0) return l;
    return l - std::log1p(std::exp(l));
}
} // namespace detail

template<typename B = int, typename R = float, typename Q = R>
class Modem_MSPRS : public aff3ct::module::Modem<B, R, Q>
{
  public:
    //! Symbols produced by `N` coded bits: ceil((N + L0 - 1) / 2).
    //! The extra L0-1 bit positions flush the h0 memory; they round up to whole
    //! symbols, so this is NOT N/2 and assuming it is costs you a symbol.
    static int size_mod(const int N, const int L0) { return (N + L0) / 2; }

    Modem_MSPRS(const int N, const Taps& taps)
      : aff3ct::module::Modem<B, R, Q>(N, size_mod(N, taps.L0))
      , L0(taps.L0)
      , h0(taps.h0)
      , h1(taps.h1)
      , L(size_mod(N, taps.L0))
      , M(1 << (taps.L0 - 1))
      , n0(size_mod(N, taps.L0) - taps.L0 + 1)
      , even_L0(taps.L0 % 2 == 0)
    {
        if (L0 < 2) throw std::runtime_error("MS-PRS: L0 must be >= 2");
        const std::string name = "Modem_MSPRS";
        this->set_name(name);
        this->set_short_name(name);

        // Branch labels: the h0 contribution of (state, new bit). State bits are
        // newest-first, i.e. bit (L0-2) of `s` is x0[k-1] and bit 0 is x0[k-L0+1].
        sym0.resize((size_t)M * 2);
        nxt.resize((size_t)M * 2);
        for (int s = 0; s < M; s++)
            for (int b = 0; b < 2; b++)
            {
                double acc = h0[0] * (1.0 - 2.0 * b);
                for (int m = 0; m < L0 - 1; m++)
                {
                    const int bit = (s >> (L0 - 2 - m)) & 1;
                    acc += h0[m + 1] * (1.0 - 2.0 * bit);
                }
                sym0[(size_t)s * 2 + b] = acc;
                nxt[(size_t)s * 2 + b]  = (s >> 1) | (b << (L0 - 2));
            }

        // Full constellation point per (state, b0, b1). Trellis-static, so
        // hoisting it out of the per-symbol gamma loop costs 4M doubles and
        // saves an add per edge per symbol per turbo iteration.
        symf.resize((size_t)M * 4);
        for (int s = 0; s < M; s++)
            for (int b0 = 0; b0 < 2; b0++)
                for (int b1 = 0; b1 < 2; b1++)
                    symf[((size_t)s * 2 + b0) * 2 + b1] = sym0[(size_t)s * 2 + b0] + (b1 == 0 ? h1 : -h1);

        gamma.resize((size_t)L * M * 4);
        alpha.resize((size_t)(L + 1) * M);
        beta.resize((size_t)(L + 1) * M);
    }

    virtual ~Modem_MSPRS() = default;

    Modem_MSPRS<B, R, Q>* clone() const override { return new Modem_MSPRS<B, R, Q>(*this); }

    //! MAP (the default), log-MAP or max-log-MAP.
    void set_bcjr(const Bcjr a) { algo = a; }
    Bcjr get_bcjr() const { return algo; }

    //! True when trellis step `t` still admits a free x0 bit.
    bool b0_free(const int t) const { return t < n0; }
    //! False only for the final step of an even-L0 trellis, whose x1 is known.
    bool b1_free(const int t) const { return !(even_L0 && t == L - 1); }

  protected:
    void _modulate(const B* X_N1, R* X_N2, const size_t /*frame_id*/) override
    {
        // Bits are consumed strictly in order: (b0, b1) per step while x0 is
        // free, then b1 alone once the FIR is being flushed. Outside its range
        // x0 takes the boundary bit 0, which is what puts the trellis in state 0
        // at both ends.
        std::vector<double> x0(n0);
        std::vector<double> x1(L, 1.0); // boundary / known tail symbol: bit 0 -> +1

        int k = 0;
        for (int t = 0; t < L; t++)
        {
            if (b0_free(t)) x0[t] = 1.0 - 2.0 * (double)X_N1[k++];
            if (b1_free(t)) x1[t] = 1.0 - 2.0 * (double)X_N1[k++];
        }
        if (k != this->N) throw std::runtime_error("MS-PRS: modulate consumed " + std::to_string(k) +
                                                   " bits, expected " + std::to_string(this->N));

        for (int t = 0; t < L; t++)
        {
            double acc = h1 * x1[t];
            for (int j = 0; j < L0; j++)
            {
                const int i = t - j;
                acc += h0[j] * ((i >= 0 && i < n0) ? x0[i] : 1.0);
            }
            X_N2[t] = (R)acc;
        }
    }

    void _demodulate(const float* CP, const Q* Y_N1, Q* Y_N2, const size_t /*frame_id*/) override
    {
        std::vector<Q> zero((size_t)this->N, (Q)0);
        bcjr((double)CP[0], Y_N1, zero.data(), Y_N2, false);
    }

    void _tdemodulate(const float* CP, const Q* Y_N1, const Q* Y_N2, Q* Y_N3, const size_t /*frame_id*/) override
    {
        bcjr((double)CP[0], Y_N1, Y_N2, Y_N3, true);
    }

  private:
    /*!
     * \brief BCJR over the MS-PRS trellis.
     *
     * \param sigma      noise standard deviation per real dimension
     * \param Y          received symbols, length L
     * \param La         a-priori LLRs, length N, AFF3CT sign
     * \param Lout       output, length N
     * \param extrinsic  subtract La from the posterior before returning
     *
     * gamma is stored resolved on both bits, (t, s, b0, b1), because b1 does not
     * change the state: the alpha/beta recursions marginalise it away, but the
     * b1 LLR needs it back.
     */
    void bcjr(const double sigma, const Q* Y, const Q* La, Q* Lout, const bool extrinsic)
    {
        switch (algo)
        {
            case Bcjr::map: bcjr_t<bcjr::Map>(sigma, Y, La, Lout, extrinsic); break;
            case Bcjr::log_map: bcjr_t<bcjr::LogMap>(sigma, Y, La, Lout, extrinsic); break;
            case Bcjr::max_log_map: bcjr_t<bcjr::MaxLogMap>(sigma, Y, La, Lout, extrinsic); break;
        }
    }

    // S is the arithmetic (see Bcjr.hpp).
    template<class S>
    void bcjr_t(const double sigma, const Q* Y, const Q* La, Q* Lout, const bool extrinsic)
    {
        const double inv2s2 = 1.0 / (2.0 * sigma * sigma);

        std::fill(gamma.begin(), gamma.end(), S::zero());

        int k = 0;
        for (int t = 0; t < L; t++)
        {
            const double y   = (double)Y[t];
            const double la0 = b0_free(t) ? std::max(-50.0, std::min(50.0, (double)La[k++])) : 0.0;
            const double la1 = b1_free(t) ? std::max(-50.0, std::min(50.0, (double)La[k++])) : 0.0;

            const int b0_hi = b0_free(t) ? 1 : 0; // forced to the boundary bit otherwise
            const int b1_hi = b1_free(t) ? 1 : 0;

            // The a-priori term depends only on (t, bit), not on the state, so
            // it is computed twice per symbol instead of 2*M times. In MAP and
            // log-MAP it costs an exp, which would dominate the gamma loop.
            const double ap0[2] = { b0_free(t) ? S::apriori(la0, 0) : S::one(), b0_free(t) ? S::apriori(la0, 1) : S::one() };
            const double ap1[2] = { b1_free(t) ? S::apriori(la1, 0) : S::one(), b1_free(t) ? S::apriori(la1, 1) : S::one() };

            // Final step of an even-L0 trellis: both bits are known, and the
            // Python reference drops the sample outright (msprs.py:231 sets
            // Gamma0 = 0). That sample still depends on the state, so its
            // metric is NOT a constant and dropping it discards real
            // information about the last x0 bits. Matching the reference is
            // deliberate: it keeps the two implementations bit-comparable,
            // which is worth more right now than the fraction of a dB. Use the
            // real metric instead and even-L0 stops being diffable.
            const bool drop_sample = even_L0 && t == L - 1;

            for (int s = 0; s < M; s++)
                for (int b0 = 0; b0 <= b0_hi; b0++)
                    for (int b1 = 0; b1 <= b1_hi; b1++)
                    {
                        if (drop_sample) { gamma[idx(t, s, b0, b1)] = S::one(); continue; }
                        const double d           = y - symf[((size_t)s * 2 + b0) * 2 + b1];
                        gamma[idx(t, s, b0, b1)] = S::mul(S::mul(S::from_log(-d * d * inv2s2), ap0[b0]), ap1[b1]);
                    }
        }

        // alpha: state 0 is the all-boundary-bit state at t=0.
        std::fill(alpha.begin(), alpha.end(), S::zero());
        alpha[0] = S::one();
        for (int t = 0; t < L; t++)
        {
            double norm = S::zero();
            for (int ns = 0; ns < M; ns++) alpha[(size_t)(t + 1) * M + ns] = S::zero();
            for (int s = 0; s < M; s++)
            {
                const double a = alpha[(size_t)t * M + s];
                if (a == S::zero()) continue;
                for (int b0 = 0; b0 < 2; b0++)
                {
                    const double g = S::add(gamma[idx(t, s, b0, 0)], gamma[idx(t, s, b0, 1)]);
                    if (g == S::zero()) continue;
                    const int ns                    = nxt[(size_t)s * 2 + b0];
                    alpha[(size_t)(t + 1) * M + ns] = S::add(alpha[(size_t)(t + 1) * M + ns], S::mul(a, g));
                }
            }
            for (int ns = 0; ns < M; ns++) norm = S::add(alpha[(size_t)(t + 1) * M + ns], norm);
            if (norm != S::zero())
                for (int ns = 0; ns < M; ns++)
                    if (alpha[(size_t)(t + 1) * M + ns] != S::zero())
                        alpha[(size_t)(t + 1) * M + ns] = S::div(alpha[(size_t)(t + 1) * M + ns], norm);
        }

        // beta: the flushed trellis terminates in state 0.
        std::fill(beta.begin(), beta.end(), S::zero());
        beta[(size_t)L * M] = S::one();
        for (int t = L; t > 0; t--)
        {
            double norm = S::zero();
            for (int s = 0; s < M; s++)
            {
                double acc = S::zero();
                for (int b0 = 0; b0 < 2; b0++)
                {
                    const double g = S::add(gamma[idx(t - 1, s, b0, 0)], gamma[idx(t - 1, s, b0, 1)]);
                    if (g == S::zero()) continue;
                    const int    ns = nxt[(size_t)s * 2 + b0];
                    const double bn = beta[(size_t)t * M + ns];
                    if (bn == S::zero()) continue;
                    acc = S::add(acc, S::mul(bn, g));
                }
                beta[(size_t)(t - 1) * M + s] = acc;
                norm                          = S::add(acc, norm);
            }
            if (norm != S::zero())
                for (int s = 0; s < M; s++)
                    if (beta[(size_t)(t - 1) * M + s] != S::zero())
                        beta[(size_t)(t - 1) * M + s] = S::div(beta[(size_t)(t - 1) * M + s], norm);
        }

        // Soft output. p[b0][b1] accumulates alpha + gamma + beta over the
        // trellis-valid edges only; the two marginals fall out of it.
        k = 0;
        for (int t = 0; t < L; t++)
        {
            double p[2][2] = { { S::zero(), S::zero() }, { S::zero(), S::zero() } };
            for (int s = 0; s < M; s++)
            {
                const double a = alpha[(size_t)t * M + s];
                if (a == S::zero()) continue;
                for (int b0 = 0; b0 < 2; b0++)
                {
                    const int    ns = nxt[(size_t)s * 2 + b0];
                    const double bn = beta[(size_t)(t + 1) * M + ns];
                    if (bn == S::zero()) continue;
                    for (int b1 = 0; b1 < 2; b1++)
                    {
                        const double g = gamma[idx(t, s, b0, b1)];
                        if (g == S::zero()) continue;
                        p[b0][b1] = S::add(p[b0][b1], S::mul(S::mul(a, g), bn));
                    }
                }
            }

            if (b0_free(t))
            {
                const double lp0 = S::add(p[0][0], p[0][1]);
                const double lp1 = S::add(p[1][0], p[1][1]);
                Lout[k]          = clip(S::llr(lp0, lp1) - (extrinsic ? (double)La[k] : 0.0));
                k++;
            }
            if (b1_free(t))
            {
                const double lp0 = S::add(p[0][0], p[1][0]);
                const double lp1 = S::add(p[0][1], p[1][1]);
                Lout[k]          = clip(S::llr(lp0, lp1) - (extrinsic ? (double)La[k] : 0.0));
                k++;
            }
        }
    }

    inline size_t idx(const int t, const int s, const int b0, const int b1) const
    {
        return (((size_t)t * M + s) * 2 + b0) * 2 + b1;
    }

    static Q clip(const double v)
    {
        if (!(v == v)) return (Q)0;                 // NaN guard: -inf minus -inf
        return (Q)std::max(-50.0, std::min(50.0, v));
    }

    const int                 L0;
    const std::vector<double> h0;
    const double              h1;
    const int                 L;       //!< trellis steps == modulated symbols
    const int                 M;       //!< 2^(L0-1) states
    const int                 n0;      //!< steps carrying a free x0 bit
    const bool                even_L0;

    std::vector<double> sym0; //!< h0 contribution,      indexed [s*2 + b0]
    std::vector<double> symf; //!< full constellation,   indexed [(s*2 + b0)*2 + b1]
    std::vector<int>    nxt;  //!< next state,           indexed [s*2 + b0]
    std::vector<double> gamma, alpha, beta;
    Bcjr                algo = Bcjr::map;
};

} // namespace msprs

#endif
