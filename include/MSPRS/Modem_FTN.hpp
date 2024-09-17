// Binary Faster-Than-Nyquist at packing factor tau, as an idealised ISI
// reference only.
//
// NOT A VALID FTN MODEL. It normalises the one-sided pulse autocorrelation to
// unit energy and drives it with white noise, which pins the isolated-error
// squared distance to 4, the ISI-free 2-ASK value, independently of tau. That
// flatters FTN by 2.95 dB at tau = 0.5, where the true MSED is 2.03. A correct
// receiver needs the Ungerboeck observation model with noise covariance
// sigma^2 G.
//
// Energy: taps are normalised to ||h||^2 = 1 and one symbol carries ONE bit,
// so avg_bit_energy = 1.0, not the 0.5 of rate-2 MS-PRS. Getting that wrong
// halves sigma^2 and hands FTN a spurious 3 dB.
#ifndef MSPRS_MODEM_FTN_HPP
#define MSPRS_MODEM_FTN_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <aff3ct.hpp>

#include "MSPRS/Modem_MSPRS.hpp" // detail::log_add, detail::log_p

namespace msprs
{

//! Root-raised-cosine pulse, length 2*span*sps+1, unit energy.
inline std::vector<double> rrc_taps(const double a, const int sps, const int span)
{
    const double        T = 1.0, PI = 3.14159265358979323846;
    const int           n = 2 * span * sps + 1;
    std::vector<double> h(n);
    double              e = 0.0;
    for (int i = 0; i < n; i++)
    {
        const double ti = (double)(i - span * sps) / (double)sps;
        double       v;
        if (std::abs(ti) < 1e-12)
            v = (1.0 + a * (4.0 / PI - 1.0)) / T;
        else if (std::abs(std::abs(ti) - T / (4 * a)) < 1e-12)
            v = (a / (T * std::sqrt(2.0))) *
                ((1 + 2 / PI) * std::sin(PI / (4 * a)) + (1 - 2 / PI) * std::cos(PI / (4 * a)));
        else
        {
            const double num =
              std::sin(PI * ti * (1 - a) / T) + 4 * a * ti / T * std::cos(PI * ti * (1 + a) / T);
            const double den = PI * ti * (1 - std::pow(4 * a * ti / T, 2.0)) / T;
            v                = num / den;
        }
        h[i] = v;
        e += v * v;
    }
    const double s = std::sqrt(e);
    for (auto& v : h) v /= s;
    return h;
}

//! One-sided FTN ISI taps: pulse autocorrelation at lags 0..L_isi-1, unit norm.
inline std::vector<double> ftn_isi(const double tau, const double rolloff, const int L_isi,
                                   const int sps = 32, const int span = 12)
{
    const std::vector<double> p    = rrc_taps(rolloff, sps, span);
    const int                 step = (int)std::lround(sps * tau);
    std::vector<double>       h(L_isi, 0.0);
    for (int k = 0; k < L_isi; k++)
    {
        const int lag       = k * step;
        const int n_overlap = (int)p.size() - std::abs(lag);
        double    acc       = 0.0;
        if (n_overlap > 0)
            for (int i = 0; i < n_overlap; i++) acc += p[i] * p[i + lag];
        h[k] = acc;
    }
    double e = 0.0;
    for (auto v : h) e += v * v;
    const double s = std::sqrt(e);
    for (auto& v : h) v /= s;
    return h;
}

template<typename B = int, typename R = float, typename Q = R>
class Modem_FTN : public aff3ct::module::Modem<B, R, Q>
{
  public:
    //! One symbol per bit, plus the L_isi-1 samples the convolution flushes.
    static int size_mod(const int N, const int L_isi) { return N + L_isi - 1; }

    Modem_FTN(const int N, const double tau = 0.5, const double rolloff = 0.3, const int L_isi = 5)
      : aff3ct::module::Modem<B, R, Q>(N, size_mod(N, L_isi))
      , L_isi(L_isi)
      , h(ftn_isi(tau, rolloff, L_isi))
      , L(size_mod(N, L_isi))
      , memory(L_isi - 1)
      , M(1 << (L_isi - 1))
    {
        if (L_isi < 2) throw std::runtime_error("FTN: L_isi must be >= 2");
        const std::string name = "Modem_FTN";
        this->set_name(name);
        this->set_short_name(name);

        // State holds the previous `memory` bits, MSB oldest. The new bit
        // shifts in on the LSB side and the oldest drops off.
        lab.resize((size_t)M * 2);
        nxt.resize((size_t)M * 2);
        for (int ps = 0; ps < M; ps++)
            for (int b = 0; b < 2; b++)
            {
                double s = h[0] * (1.0 - 2.0 * b);
                for (int k = 1; k < L_isi; k++)
                {
                    // bits_prev[L_isi-1-k] under an MSB-oldest layout is bit
                    // (k-1) of ps counted from the LSB.
                    const int bit = (ps >> (k - 1)) & 1;
                    s += h[k] * (1.0 - 2.0 * bit);
                }
                lab[(size_t)ps * 2 + b] = s;
                nxt[(size_t)ps * 2 + b] = ((ps << 1) | b) & (M - 1);
            }

        gamma.resize((size_t)L * M * 2);
        alpha.resize((size_t)(L + 1) * M);
        beta.resize((size_t)(L + 1) * M);
    }

    virtual ~Modem_FTN() = default;

    Modem_FTN<B, R, Q>* clone() const override { return new Modem_FTN<B, R, Q>(*this); }

    const std::vector<double>& taps() const { return h; }

  protected:
    void _modulate(const B* X_N1, R* X_N2, const size_t /*frame_id*/) override
    {
        // Plain convolution: zero-padded outside the data, matching np.convolve.
        for (int t = 0; t < L; t++)
        {
            double acc = 0.0;
            for (int k = 0; k < L_isi; k++)
            {
                const int i = t - k;
                if (i >= 0 && i < this->N) acc += h[k] * (1.0 - 2.0 * (double)X_N1[i]);
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
    void bcjr(const double sigma, const Q* Y, const Q* La, Q* Lout, const bool extrinsic)
    {
        using detail::log_add;
        using detail::log_p;
        using detail::NEG_INF;

        const double inv2s2 = 1.0 / (2.0 * sigma * sigma);

        for (int t = 0; t < L; t++)
        {
            const double y  = (double)Y[t];
            // Only the first N steps carry a data bit; the flush samples get
            // zero a-priori but stay free in the trellis.
            const double la = (t < this->N) ? std::max(-50.0, std::min(50.0, (double)La[t])) : 0.0;
            const double ap[2] = { log_p(la, 0), log_p(la, 1) };
            for (int ps = 0; ps < M; ps++)
                for (int b = 0; b < 2; b++)
                {
                    const double d                = y - lab[(size_t)ps * 2 + b];
                    gamma[((size_t)t * M + ps) * 2 + b] = -d * d * inv2s2 + ap[b];
                }
        }

        std::fill(alpha.begin(), alpha.end(), NEG_INF);
        alpha[0] = 0.0; // start state 0: the reference assumes zero prior memory
        for (int t = 0; t < L; t++)
        {
            double norm = NEG_INF;
            for (int ns = 0; ns < M; ns++) alpha[(size_t)(t + 1) * M + ns] = NEG_INF;
            for (int ps = 0; ps < M; ps++)
            {
                const double a = alpha[(size_t)t * M + ps];
                if (a == NEG_INF) continue;
                for (int b = 0; b < 2; b++)
                {
                    const int ns                    = nxt[(size_t)ps * 2 + b];
                    alpha[(size_t)(t + 1) * M + ns] = log_add(alpha[(size_t)(t + 1) * M + ns],
                                                              a + gamma[((size_t)t * M + ps) * 2 + b]);
                }
            }
            for (int ns = 0; ns < M; ns++) norm = log_add(alpha[(size_t)(t + 1) * M + ns], norm);
            if (norm != NEG_INF)
                for (int ns = 0; ns < M; ns++)
                    if (alpha[(size_t)(t + 1) * M + ns] != NEG_INF) alpha[(size_t)(t + 1) * M + ns] -= norm;
        }

        // Soft termination: every final state equally likely. The modulator
        // zero-pads rather than driving the trellis to a known state, so there
        // is no state to terminate on.
        for (int ns = 0; ns < M; ns++) beta[(size_t)L * M + ns] = 0.0;
        for (int t = L; t > 0; t--)
        {
            double norm = NEG_INF;
            for (int ps = 0; ps < M; ps++)
            {
                double acc = NEG_INF;
                for (int b = 0; b < 2; b++)
                {
                    const double bn = beta[(size_t)t * M + nxt[(size_t)ps * 2 + b]];
                    if (bn == NEG_INF) continue;
                    acc = log_add(acc, bn + gamma[((size_t)(t - 1) * M + ps) * 2 + b]);
                }
                beta[(size_t)(t - 1) * M + ps] = acc;
                norm                           = log_add(acc, norm);
            }
            if (norm != NEG_INF)
                for (int ps = 0; ps < M; ps++)
                    if (beta[(size_t)(t - 1) * M + ps] != NEG_INF) beta[(size_t)(t - 1) * M + ps] -= norm;
        }

        for (int t = 0; t < this->N; t++)
        {
            double p0 = NEG_INF, p1 = NEG_INF;
            for (int ps = 0; ps < M; ps++)
            {
                const double a = alpha[(size_t)t * M + ps];
                if (a == NEG_INF) continue;
                for (int b = 0; b < 2; b++)
                {
                    const double bn = beta[(size_t)(t + 1) * M + nxt[(size_t)ps * 2 + b]];
                    if (bn == NEG_INF) continue;
                    const double v = a + gamma[((size_t)t * M + ps) * 2 + b] + bn;
                    (b == 0 ? p0 : p1) = log_add(b == 0 ? p0 : p1, v);
                }
            }
            const double e = (p0 - p1) - (extrinsic ? (double)La[t] : 0.0);
            Lout[t]        = (Q)std::max(-50.0, std::min(50.0, e));
        }
    }

    const int                 L_isi;
    const std::vector<double> h;
    const int                 L;      //!< observed samples == N + memory
    const int                 memory; //!< L_isi - 1
    const int                 M;      //!< 2^memory

    std::vector<double> lab, gamma, alpha, beta;
    std::vector<int>    nxt;
};

} // namespace msprs

#endif
