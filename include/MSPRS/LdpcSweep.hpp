// MS-PRS with an LDPC outer code, turbo equalised.
//
// AFF3CT's BP decoders derive from Decoder_SISO and return the extrinsic on
// the coded bits, which is the interface the turbo loop needs.
#ifndef MSPRS_LDPC_SWEEP_HPP
#define MSPRS_LDPC_SWEEP_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <aff3ct.hpp>

#include "MSPRS/Modem_MSPRS.hpp"
#include "MSPRS/Sweep.hpp"
#include "MSPRS/Taps.hpp"

namespace msprs
{

struct LdpcConfig
{
    std::string h_path;
    int    bp_iters = 20;
    int    iters    = 7;
    int    be       = 500;
    int    min_fra  = 200;
    int    max_fra  = 200000;
    int    chunk    = 25;
    int    threads  = 1;
    int    seed     = 0;
    int    itl_seed = 42;
    double ebn0_min = 0.0, ebn0_max = 7.01, ebn0_step = 0.5;
};

namespace detail
{
struct LdpcChain
{
    spu::module::Source_random<>                          src;
    aff3ct::module::Encoder_LDPC_from_H<>                 enc;
    aff3ct::module::Decoder_LDPC_BP_flooding_SPA<>        dec;
    Modem_MSPRS<>                                         mdm;
    aff3ct::module::Channel_AWGN_LLR<>                    chn;
    aff3ct::module::Interleaver<int>                      ib;
    aff3ct::module::Interleaver<float>                    il;
    std::vector<int>      u, cw, cwi, hard;
    std::vector<uint32_t> ng;
    std::vector<float>    sym, rx, Lei, Lai, Len, Den;

    LdpcChain(int K, int N, int Nm, int bp, const aff3ct::tools::Sparse_matrix& H,
              const std::vector<uint32_t>& ibp, const Taps& t,
              const aff3ct::tools::Interleaver_core_random<>& core, int seed, int stream)
      : src(K), enc(K, N, H, "IDENTITY"), dec(K, N, bp, H, ibp, false)
      , mdm(N, t), chn(Nm), ib(core), il(core)
      , u(K), cw(N), cwi(N), hard(K), ng(1)
      , sym(Nm), rx(Nm), Lei(N), Lai(N), Len(N), Den(N)
    { chn.set_seed(seed + 7919 * stream); src.set_seed(seed + 6271 * stream); }
};
} // namespace detail

inline std::vector<Point> ldpc_sweep(const LdpcConfig& cfg, const Taps& taps)
{
    const auto H = aff3ct::tools::LDPC_matrix_handler::read(cfg.h_path);

    // Dimensions the way AFF3CT's own factory reads them: N is the width of H
    // and K = N - M. Taking them off the Sparse_matrix gets the orientation
    // wrong and yields a negative K.
    int M = 0, N = 0;
    aff3ct::tools::LDPC_matrix_handler::read_matrix_size(cfg.h_path, M, N);
    const int K   = N - M;
    const int Nm  = Modem_MSPRS<>::size_mod(N, taps.L0);
    const int nth = std::max(1, cfg.threads);

    aff3ct::module::Encoder_LDPC_from_H<> probe(K, N, H, "IDENTITY");
    const auto ibp = probe.get_info_bits_pos();

    aff3ct::tools::Interleaver_core_random<> core(N, cfg.itl_seed, false);
    std::vector<std::unique_ptr<detail::LdpcChain>> ch;
    for (int t = 0; t < nth; t++)
        ch.emplace_back(new detail::LdpcChain(K, N, Nm, cfg.bp_iters, H, ibp, taps, core,
                                              cfg.seed, t + 1));

    const double Rl = (double)K / (double)N;

    std::vector<Point> out;
    for (double ebn0 = cfg.ebn0_min; ebn0 < cfg.ebn0_max; ebn0 += cfg.ebn0_step)
    {
        const double esn0 = ebn0 + 10.0 * std::log10(2.0 * Rl);
        const float  sig  = (float)std::sqrt(1.0 / (2.0 * std::pow(10.0, esn0 / 10.0)));
        const std::vector<float> CP = { sig };

        Point pt;
        pt.eb_no_db = ebn0;
        const auto t0 = std::chrono::steady_clock::now();

        while (pt.frames < cfg.max_fra && (pt.frames < cfg.min_fra || pt.errors < cfg.be))
        {
            std::vector<long long> pe((size_t)nth, 0), pf((size_t)nth, 0);
            std::vector<std::thread> pool;
            for (int t = 0; t < nth; t++)
                pool.emplace_back([&, t] {
                    auto& c = *ch[(size_t)t];
                    for (int f = 0; f < cfg.chunk; f++)
                    {
                        c.src.generate(c.u, c.ng);
                        c.enc.encode(c.u, c.cw);
                        c.ib .interleave(c.cw, c.cwi);
                        c.mdm.modulate(c.cwi, c.sym);
                        c.chn.add_noise(CP, c.sym, c.rx);
                        std::fill(c.Lai.begin(), c.Lai.end(), 0.f);
                        for (int it = 0; it <= cfg.iters; it++)
                        {
                            c.mdm.tdemodulate(CP, c.rx, c.Lai, c.Lei);
                            c.il .deinterleave(c.Lei, c.Len);
                            c.dec.decode_siso(c.Len, c.Den);
                            c.il .interleave(c.Den, c.Lai);
                        }
                        c.dec.decode_siho(c.Len, c.hard);
                        long long e = 0;
                        for (int i = 0; i < K; i++) e += (c.hard[(size_t)i] != c.u[(size_t)i]);
                        pe[(size_t)t] += e; pf[(size_t)t] += 1;
                    }
                });
            for (auto& th : pool) th.join();
            for (int t = 0; t < nth; t++) { pt.errors += pe[(size_t)t]; pt.frames += pf[(size_t)t]; }
        }

        pt.bits    = pt.frames * K;
        pt.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        out.push_back(pt);
    }
    return out;
}

} // namespace msprs

#endif
