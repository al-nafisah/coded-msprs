// Non-recursive, non-systematic convolutional code as AFF3CT modules.
//
// AFF3CT 4.1.2 ships convolutional support for RSC only, so the SISO decoder
// for the (5,7) K=3 outer code is written here. Switching to RSC would give
// the same codeword set and FER but a different information-to-codeword
// mapping, so a different BER and EXIT curve.
//
// LLR is ln P(b=0)/P(b=1). Termination is an implicit zero tail: memory zero
// bits follow the K information bits, so the trellis starts and ends in state
// 0 and N = (K + memory) * n_out.
#ifndef MSPRS_NSC_HPP
#define MSPRS_NSC_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <aff3ct.hpp>

namespace msprs
{

// Trellis of a rate-1/n non-recursive code. State s holds u[k-1] in its MSB
// down to u[k-memory] in its LSB; generator g[j] reads tap i at delay i, so
// octal 5 = 101b taps delays 0 and 2.
struct NSC_Trellis
{
    int                           K_constraint;
    int                           memory;
    int                           n_out;
    int                           n_states;
    std::vector<std::vector<int>> out_bits; //!< [s * 2 + u][j]
    std::vector<int>              nxt;      //!< [s * 2 + u]

    NSC_Trellis(const int constraint_len, const std::vector<int>& poly_octal)
      : K_constraint(constraint_len)
      , memory(constraint_len - 1)
      , n_out((int)poly_octal.size())
      , n_states(1 << (constraint_len - 1))
    {
        if (constraint_len < 2) throw std::runtime_error("NSC: constraint length must be >= 2");
        if (n_out < 1) throw std::runtime_error("NSC: need at least one generator");

        // Octal literal -> tap vector, tap i at delay i.
        std::vector<std::vector<int>> g(n_out, std::vector<int>(constraint_len, 0));
        for (int j = 0; j < n_out; j++)
        {
            int dec = 0, p = 1;
            for (int v = poly_octal[j]; v > 0; v /= 10) { dec += (v % 10) * p; p *= 8; }
            for (int i = 0; i < constraint_len; i++) { g[j][i] = dec & 1; dec >>= 1; }
        }

        out_bits.assign((size_t)n_states * 2, std::vector<int>(n_out, 0));
        nxt.assign((size_t)n_states * 2, 0);
        for (int s = 0; s < n_states; s++)
            for (int u = 0; u < 2; u++)
            {
                for (int j = 0; j < n_out; j++)
                {
                    int acc = g[j][0] & u;
                    for (int i = 1; i <= memory; i++)
                        acc ^= g[j][i] & ((s >> (memory - i)) & 1);
                    out_bits[(size_t)s * 2 + u][j] = acc;
                }
                nxt[(size_t)s * 2 + u] = (u << (memory - 1)) | (s >> 1);
            }
    }

    int codeword_length(const int K) const { return (K + memory) * n_out; }
};

// ---------------------------------------------------------------------------

template<typename B = int>
class Encoder_NSC : public aff3ct::module::Encoder<B>
{
  public:
    Encoder_NSC(const int K, const NSC_Trellis& tr)
      : aff3ct::module::Encoder<B>(K, tr.codeword_length(K))
      , trellis(tr)
    {
        const std::string name = "Encoder_NSC";
        this->set_name(name);
        this->set_short_name(name);
    }

    virtual ~Encoder_NSC() = default;

    Encoder_NSC<B>* clone() const override { return new Encoder_NSC<B>(*this); }

    bool is_sys() const override { return false; } //!< non-systematic: no bit of U_K appears in X_N

  protected:
    void _encode(const B* U_K, B* X_N, const size_t /*frame_id*/) override
    {
        int s = 0, n = 0;
        for (int t = 0; t < this->K + trellis.memory; t++)
        {
            const int u = (t < this->K) ? (int)U_K[t] : 0; // implicit zero tail
            for (int j = 0; j < trellis.n_out; j++) X_N[n++] = (B)trellis.out_bits[(size_t)s * 2 + u][j];
            s = trellis.nxt[(size_t)s * 2 + u];
        }
    }

  private:
    NSC_Trellis trellis;
};

// ---------------------------------------------------------------------------

/*!
 * \brief BCJR for the NSC code, soft-in soft-out on the CODED bits.
 *
 * `_decode_siso` is N LLRs in, N extrinsic LLRs out, which is the form the
 * turbo-equalisation loop needs: the modem wants extrinsic information about
 * the coded bits, not about the information bits. The sibling
 * `_decode_siso_alt(sys, par, ext)` in the base class is the turbo-code split
 * form and does not apply to a non-systematic code.
 */
template<typename B = int, typename R = float>
class Decoder_NSC_SISO : public aff3ct::module::Decoder_SISO<B, R>
{
  public:
    Decoder_NSC_SISO(const int K, const NSC_Trellis& tr)
      : aff3ct::module::Decoder_SISO<B, R>(K, tr.codeword_length(K))
      , trellis(tr)
      , L(K + tr.memory)
      , M(tr.n_states)
    {
        const std::string name = "Decoder_NSC_SISO";
        this->set_name(name);
        this->set_short_name(name);
        gamma.resize((size_t)L * M * 2);
        alpha.resize((size_t)(L + 1) * M);
        beta.resize((size_t)(L + 1) * M);
    }

    virtual ~Decoder_NSC_SISO() = default;

    Decoder_NSC_SISO<B, R>* clone() const override { return new Decoder_NSC_SISO<B, R>(*this); }

    /*!
     * \brief Extrinsic LLRs and hard decisions from a single BCJR pass.
     *
     * The turbo loop needs both every iteration. Calling decode_siso() then
     * decode_siho() would run the recursions twice for identical input.
     */
    void decode_both(const R* Lin, R* Lext, B* V_K) { bcjr(Lin, Lext, V_K); }

  protected:
    int _decode_siso(const R* Y_N1, R* Y_N2, const size_t /*frame_id*/) override
    {
        bcjr(Y_N1, Y_N2, nullptr);
        return 0;
    }

    int _decode_siho(const R* Y_N, B* V_K, const size_t /*frame_id*/) override
    {
        bcjr(Y_N, nullptr, V_K);
        return 0;
    }

  private:
    static constexpr double NEG_INF = -std::numeric_limits<double>::infinity();

    static double log_add(const double a, const double b)
    {
        if (a == NEG_INF) return b;
        if (b == NEG_INF) return a;
        const double d = std::abs(a - b), m = a > b ? a : b;
        return d > 20.0 ? m : m + std::log1p(std::exp(-d));
    }

    //! log P(bit) from an LLR in AFF3CT sign (positive -> bit 0).
    static double log_p(const double llr, const int bit)
    {
        const double l = (bit == 0) ? llr : -llr;
        if (l > 10.0) return -1e-6;
        if (l < -10.0) return l;
        return l - std::log1p(std::exp(l));
    }

    void bcjr(const R* Lin, R* Lext, B* V_K)
    {
        const int n_out = trellis.n_out;

        // log_p is a log1p+exp pair and depends only on (t, j, bit), never on
        // the state, so it is evaluated 2*n_out times per step rather than
        // 2*M*n_out times. This is the hot loop of the turbo iteration.
        std::vector<double> lp(2 * (size_t)n_out);
        for (int t = 0; t < L; t++)
        {
            const int u_hi = (t < this->K) ? 1 : 0; // tail steps are forced to u = 0
            for (int j = 0; j < n_out; j++)
            {
                const double li      = std::max(-50.0, std::min(50.0, (double)Lin[n_out * t + j]));
                lp[(size_t)j * 2]     = log_p(li, 0);
                lp[(size_t)j * 2 + 1] = log_p(li, 1);
            }
            for (int s = 0; s < M; s++)
            {
                gamma[idx(t, s, 1)] = NEG_INF;
                for (int u = 0; u <= u_hi; u++)
                {
                    const std::vector<int>& ob  = trellis.out_bits[(size_t)s * 2 + u];
                    double                  acc = 0.0;
                    for (int j = 0; j < n_out; j++) acc += lp[(size_t)j * 2 + ob[j]];
                    gamma[idx(t, s, u)] = acc;
                }
            }
        }

        std::fill(alpha.begin(), alpha.end(), NEG_INF);
        alpha[0] = 0.0;
        for (int t = 0; t < L; t++)
        {
            double norm = NEG_INF;
            for (int ns = 0; ns < M; ns++) alpha[(size_t)(t + 1) * M + ns] = NEG_INF;
            for (int s = 0; s < M; s++)
            {
                const double a = alpha[(size_t)t * M + s];
                if (a == NEG_INF) continue;
                for (int u = 0; u < 2; u++)
                {
                    const double g = gamma[idx(t, s, u)];
                    if (g == NEG_INF) continue;
                    const int ns                    = trellis.nxt[(size_t)s * 2 + u];
                    alpha[(size_t)(t + 1) * M + ns] = log_add(alpha[(size_t)(t + 1) * M + ns], a + g);
                }
            }
            for (int ns = 0; ns < M; ns++) norm = log_add(alpha[(size_t)(t + 1) * M + ns], norm);
            if (norm != NEG_INF)
                for (int ns = 0; ns < M; ns++)
                    if (alpha[(size_t)(t + 1) * M + ns] != NEG_INF) alpha[(size_t)(t + 1) * M + ns] -= norm;
        }

        std::fill(beta.begin(), beta.end(), NEG_INF);
        beta[(size_t)L * M] = 0.0; // zero tail terminates the trellis in state 0
        for (int t = L; t > 0; t--)
        {
            double norm = NEG_INF;
            for (int s = 0; s < M; s++)
            {
                double acc = NEG_INF;
                for (int u = 0; u < 2; u++)
                {
                    const double g = gamma[idx(t - 1, s, u)];
                    if (g == NEG_INF) continue;
                    const double bn = beta[(size_t)t * M + trellis.nxt[(size_t)s * 2 + u]];
                    if (bn == NEG_INF) continue;
                    acc = log_add(acc, bn + g);
                }
                beta[(size_t)(t - 1) * M + s] = acc;
                norm                          = log_add(acc, norm);
            }
            if (norm != NEG_INF)
                for (int s = 0; s < M; s++)
                    if (beta[(size_t)(t - 1) * M + s] != NEG_INF) beta[(size_t)(t - 1) * M + s] -= norm;
        }

        std::vector<double> lp0(n_out), lp1(n_out);
        for (int t = 0; t < L; t++)
        {
            std::fill(lp0.begin(), lp0.end(), NEG_INF);
            std::fill(lp1.begin(), lp1.end(), NEG_INF);
            double u0 = NEG_INF, u1 = NEG_INF;

            for (int s = 0; s < M; s++)
            {
                const double a = alpha[(size_t)t * M + s];
                if (a == NEG_INF) continue;
                for (int u = 0; u < 2; u++)
                {
                    const double g = gamma[idx(t, s, u)];
                    if (g == NEG_INF) continue;
                    const double bn = beta[(size_t)(t + 1) * M + trellis.nxt[(size_t)s * 2 + u]];
                    if (bn == NEG_INF) continue;
                    const double v = a + g + bn;
                    (u == 0 ? u0 : u1) = log_add(u == 0 ? u0 : u1, v);
                    for (int j = 0; j < n_out; j++)
                        (trellis.out_bits[(size_t)s * 2 + u][j] == 0 ? lp0[j] : lp1[j]) =
                          log_add(trellis.out_bits[(size_t)s * 2 + u][j] == 0 ? lp0[j] : lp1[j], v);
                }
            }

            if (Lext != nullptr)
                for (int j = 0; j < n_out; j++)
                {
                    const int    n = n_out * t + j;
                    const double e = (lp0[j] - lp1[j]) - (double)Lin[n];
                    Lext[n]        = (R)std::max(-50.0, std::min(50.0, e));
                }

            if (V_K != nullptr && t < this->K) V_K[t] = (B)((u0 > u1) ? 0 : 1);
        }
    }

    size_t idx(const int t, const int s, const int u) const { return ((size_t)t * M + s) * 2 + u; }

    NSC_Trellis         trellis;
    const int           L; //!< trellis steps, K information + memory tail
    const int           M;
    std::vector<double> gamma, alpha, beta;
};

} // namespace msprs

#endif
