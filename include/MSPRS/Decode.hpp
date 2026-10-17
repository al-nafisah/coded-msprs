// The turbo receiver on bursts received elsewhere, and the vectors the HDL is
// tested against.
//
// `decode` reads bursts from stdin and writes the decisions to stdout, so the
// over-the-air front end in ota/ can hand its symbols to the same receiver the
// BER curves come from. Per burst, stdin holds the L received symbols, as int16
// words on the integer receiver's grid or as float32 followed by the float32
// noise sigma; stdout holds the K information-bit decisions, one byte each.
#ifndef MSPRS_DECODE_HPP
#define MSPRS_DECODE_HPP

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <aff3ct.hpp>

#include "MSPRS/Fixed.hpp"
#include "MSPRS/Interleaver.hpp"
#include "MSPRS/Modem_MSPRS.hpp"
#include "MSPRS/NSC.hpp"
#include "MSPRS/Record.hpp"
#include "MSPRS/Taps.hpp"

namespace msprs
{

struct DecodeConfig
{
    int                K       = 4998;
    int                passes  = 8;
    int                threads = 1;
    Bcjr               algo    = Bcjr::map;
    const FixedFormat* fixed   = nullptr;
};

namespace detail
{
//! Bursts [lo, hi) of `y` (L each), decisions into `u` (K each).
inline void decode_range(const DecodeConfig& c, const Taps& taps, const NSC_Trellis& tr,
                         const std::vector<uint32_t>& pi, const std::vector<float>& y,
                         const std::vector<float>& sigma, std::vector<uint8_t>& u, const size_t lo, const size_t hi)
{
    const int N = (int)pi.size(), L = Modem_MSPRS<>::size_mod(N, taps.L0);
    std::vector<int> hard((size_t)c.K);

    if (c.fixed)
    {
        FixedTurbo ft(c.K, tr, taps, *c.fixed, pi);
        std::vector<int> yq((size_t)L);
        for (size_t b = lo; b < hi; b++)
        {
            for (int i = 0; i < L; i++) yq[(size_t)i] = (int)y[b * L + (size_t)i];
            ft.start();
            for (int p = 0; p < c.passes; p++) ft.pass(yq.data(), hard.data());
            for (int i = 0; i < c.K; i++) u[b * c.K + (size_t)i] = (uint8_t)hard[(size_t)i];
        }
        return;
    }

    Modem_MSPRS<>                      modem(N, taps);
    Decoder_NSC_SISO<>                 dec(c.K, tr);
    Interleaver_core_table             core(pi);
    aff3ct::module::Interleaver<float> itl(core);
    modem.set_bcjr(c.algo);
    dec.set_bcjr(c.algo);
    std::vector<float> rx((size_t)L), La((size_t)N), Le((size_t)N), Le_n((size_t)N), De_n((size_t)N);
    for (size_t b = lo; b < hi; b++)
    {
        std::copy(y.begin() + (long)(b * L), y.begin() + (long)((b + 1) * L), rx.begin());
        const std::vector<float> CP = { sigma[b] };
        std::fill(La.begin(), La.end(), 0.f);
        for (int p = 0; p < c.passes; p++)
        {
            modem.tdemodulate(CP, rx, La, Le);
            itl.deinterleave(Le, Le_n);
            dec.decode_both(Le_n.data(), De_n.data(), hard.data());
            itl.interleave(De_n, La);
        }
        for (int i = 0; i < c.K; i++) u[b * c.K + (size_t)i] = (uint8_t)hard[(size_t)i];
    }
}
} // namespace detail

//! stdin to stdout, as described at the top. Returns the number of bursts.
inline size_t decode_stdin(const DecodeConfig& c, const Taps& taps, const NSC_Trellis& tr)
{
    const auto pi = load_interleaver();
    const int  L  = Modem_MSPRS<>::size_mod((int)pi.size(), taps.L0);

    std::vector<float> y, sigma;
    if (c.fixed)
    {
        std::vector<int16_t> w((size_t)L);
        while (std::fread(w.data(), sizeof(int16_t), (size_t)L, stdin) == (size_t)L) y.insert(y.end(), w.begin(), w.end());
    }
    else
    {
        std::vector<float> w((size_t)L + 1);
        while (std::fread(w.data(), sizeof(float), w.size(), stdin) == w.size())
        {
            y.insert(y.end(), w.begin(), w.end() - 1);
            sigma.push_back(w.back());
        }
    }
    const size_t n = y.size() / (size_t)L;

    std::vector<uint8_t>     u(n * (size_t)c.K);
    std::vector<std::thread> pool;
    const size_t             nth = (size_t)std::max(1, c.threads);
    for (size_t t = 0; t < nth; t++)
        pool.emplace_back([&, t] { detail::decode_range(c, taps, tr, pi, y, sigma, u, n * t / nth, n * (t + 1) / nth); });
    for (auto& th : pool) th.join();

    std::fwrite(u.data(), 1, u.size(), stdout);
    std::fflush(stdout);
    return n;
}

namespace detail
{
inline void write_words(const std::string& path, const std::vector<int>& v)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot write '" + path + "'");
    for (int x : v) f << x << "\n";
}
} // namespace detail

/*!
 * \brief One burst through the integer receiver, every block boundary written
 *        out: the constellation table, and per pass the messages in and out of
 *        both SISOs, their state metrics and the decisions. One integer per
 *        line, in the order the receiver produces them.
 */
inline void write_vectors(const std::string& dir, const std::vector<int>& y, const int K, const int passes,
                          const Taps& taps, const NSC_Trellis& tr, const FixedFormat& f)
{
    FixedTurbo ft(K, tr, taps, f, load_interleaver());
    if ((int)y.size() != ft.equaliser().symbols()) throw std::runtime_error("vectors: wrong number of symbols");
    mkdir_p(dir);
    detail::write_words(dir + "/y.txt", y);
    detail::write_words(dir + "/constellation.txt", ft.equaliser().constellation());
    std::ofstream(dir + "/format.json") << f.json().dump(2) << "\n";

    std::vector<int> u((size_t)K);
    ft.start();
    for (int p = 0; p < passes; p++)
    {
        PassTrace t;
        ft.pass(y.data(), u.data(), &t);
        const std::string pre = dir + "/pass" + std::to_string(p) + "_";
        detail::write_words(pre + "eq_in.txt", t.la);
        detail::write_words(pre + "eq_out.txt", t.le);
        detail::write_words(pre + "eq_sm.txt", t.eq_sm);
        detail::write_words(pre + "dec_in.txt", t.le_n);
        detail::write_words(pre + "dec_out.txt", t.ld_n);
        detail::write_words(pre + "dec_sm.txt", t.dec_sm);
        detail::write_words(pre + "bits.txt", u);
    }
}

} // namespace msprs

#endif
