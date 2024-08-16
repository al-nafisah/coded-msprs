// Monte-Carlo BER sweep over an Eb/N0 grid.
//
// Energy convention: Es/N0 = Eb/N0 + 10 log10(m Rc), with m coded bits per
// symbol and code rate Rc. Rate-2 MS-PRS has m = 2; the coded schemes use
// Rc = 1/2 rather than the exact K/N, so every curve shares one abscissa.
#ifndef MSPRS_SWEEP_HPP
#define MSPRS_SWEEP_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <thread>
#include <vector>

#include <aff3ct.hpp>

#include "MSPRS/Modem_MSPRS.hpp"
#include "MSPRS/NSC.hpp"
#include "MSPRS/Taps.hpp"

namespace msprs
{

struct SweepConfig
{
    bool   coded    = true;
    int    K        = 4998;   // information bits per frame
    int    iters    = 7;      // turbo iterations; the loop runs iters+1 passes
    int    be       = 500;    // bit errors before a point is done
    int    min_fra  = 200;
    int    max_fra  = 200000;
    int    chunk    = 25;     // frames per thread between stop-rule checks
    int    threads  = 1;
    int    seed     = 0;
    int    itl_seed = 42;
    double ebn0_min = 0.0, ebn0_max = 7.01, ebn0_step = 0.5;
};

struct Point
{
    double                 eb_no_db = 0.0;
    long long              frames = 0, bits = 0, errors = 0, frame_errors = 0;
    std::vector<long long> per_iter;
    double                 seconds = 0.0;
};

namespace detail
{
// One chain per thread. Every module holds mutable scratch, above all the BCJR
// alpha/beta/gamma buffers, so sharing one across threads would race and
// produce plausible-looking garbage. Frames are independent trials, so the
// result is numerically identical to the serial loop.
struct Chain
{
    spu::module::Source_random<>            source;
    Encoder_NSC<>                           encoder;
    Decoder_NSC_SISO<>                      decoder;
    Modem_MSPRS<>                           modem;
    aff3ct::module::Channel_AWGN_LLR<>      channel;
    aff3ct::module::Interleaver<int>        itl_b;
    aff3ct::module::Interleaver<float>      itl_l;
    std::vector<int>                        ref, dec, cw, cw_i;
    std::vector<uint32_t>                   n_gen;
    std::vector<float>                      sym, rx, Le_i, La_i, Le_n, De_n;

    Chain(const SweepConfig& c, const NSC_Trellis& tr, const Taps& taps, int N_in,
          int N_mod, const aff3ct::tools::Interleaver_core_random<>& core, int stream)
      : source(c.K), encoder(c.K, tr), decoder(c.K, tr), modem(N_in, taps)
      , channel(N_mod), itl_b(core), itl_l(core)
      , ref(c.K), dec(c.K), cw(N_in), cw_i(N_in), n_gen(1)
      , sym(N_mod), rx(N_mod), Le_i(N_in), La_i(N_in), Le_n(N_in), De_n(N_in)
    {
        channel.set_seed(c.seed + 7919 * stream);
        source .set_seed(c.seed + 6271 * stream);
    }
};
} // namespace detail

inline std::vector<Point> run_sweep(const SweepConfig& cfg, const Taps& taps,
                                    const NSC_Trellis& trellis)
{
    const int N_in  = cfg.coded ? trellis.codeword_length(cfg.K) : cfg.K;
    const int N_mod = Modem_MSPRS<>::size_mod(N_in, taps.L0);
    const double m  = 2.0;
    const double Rc = cfg.coded ? 0.5 : 1.0;

    const int n_threads = std::max(1, cfg.threads);
    aff3ct::tools::Interleaver_core_random<> itl_core(N_in, cfg.itl_seed, false);

    std::vector<std::unique_ptr<detail::Chain>> chains;
    for (int t = 0; t < n_threads; t++)
        chains.emplace_back(new detail::Chain(cfg, trellis, taps, N_in, N_mod, itl_core, t + 1));

    auto run_frames = [&](detail::Chain& c, int n, const std::vector<float>& CP, Point& st)
    {
        for (int f = 0; f < n; f++)
        {
            c.source.generate(c.ref, c.n_gen);

            if (!cfg.coded)
            {
                c.modem  .modulate  (c.ref, c.sym);
                c.channel.add_noise (CP, c.sym, c.rx);
                c.modem  .demodulate(CP, c.rx, c.Le_i);
                for (int i = 0; i < cfg.K; i++) c.dec[i] = (c.Le_i[i] < 0.f) ? 1 : 0;
            }
            else
            {
                c.encoder.encode    (c.ref, c.cw);
                c.itl_b  .interleave(c.cw, c.cw_i);
                c.modem  .modulate  (c.cw_i, c.sym);
                c.channel.add_noise (CP, c.sym, c.rx);

                std::fill(c.La_i.begin(), c.La_i.end(), 0.f);
                for (int it = 0; it <= cfg.iters; it++)
                {
                    c.modem  .tdemodulate (CP, c.rx, c.La_i, c.Le_i);
                    c.itl_l  .deinterleave(c.Le_i, c.Le_n);
                    c.decoder.decode_both (c.Le_n.data(), c.De_n.data(), c.dec.data());
                    c.itl_l  .interleave  (c.De_n, c.La_i);

                    long long e = 0;
                    for (int i = 0; i < cfg.K; i++) e += (c.dec[i] != c.ref[i]);
                    st.per_iter[it] += e;
                }
            }

            long long e = 0;
            for (int i = 0; i < cfg.K; i++) e += (c.dec[i] != c.ref[i]);
            st.errors       += e;
            st.frame_errors += (e > 0);
            st.frames       += 1;
            st.bits         += cfg.K;
        }
    };

    std::vector<Point> out;
    for (double ebn0 = cfg.ebn0_min; ebn0 < cfg.ebn0_max; ebn0 += cfg.ebn0_step)
    {
        const double esn0  = ebn0 + 10.0 * std::log10(m * Rc);
        const float  sigma = (float)std::sqrt(1.0 / (2.0 * std::pow(10.0, esn0 / 10.0)));
        const std::vector<float> CP = { sigma };

        Point total;
        total.eb_no_db = ebn0;
        total.per_iter.assign((size_t)cfg.iters + 1, 0);
        const auto t0 = std::chrono::steady_clock::now();

        // Chunked rounds, because the stopping rule depends on the accumulated
        // error count and has to be re-checked between them.
        while (total.frames < cfg.max_fra &&
               (total.frames < cfg.min_fra || total.errors < cfg.be))
        {
            const int chunk = std::max(1, std::min(cfg.chunk,
                (int)((cfg.max_fra - total.frames + n_threads - 1) / n_threads)));

            std::vector<Point> part(n_threads);
            for (auto& s : part) s.per_iter.assign((size_t)cfg.iters + 1, 0);

            std::vector<std::thread> pool;
            for (int t = 0; t < n_threads; t++)
                pool.emplace_back([&, t]() { run_frames(*chains[(size_t)t], chunk, CP, part[(size_t)t]); });
            for (auto& th : pool) th.join();

            for (const auto& s : part)
            {
                total.frames       += s.frames;
                total.bits         += s.bits;
                total.errors       += s.errors;
                total.frame_errors += s.frame_errors;
                for (size_t i = 0; i < total.per_iter.size(); i++) total.per_iter[i] += s.per_iter[i];
            }
        }

        total.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        out.push_back(total);
    }
    return out;
}

} // namespace msprs

#endif
