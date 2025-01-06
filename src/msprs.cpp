// MS-PRS simulation driver.
#include <iomanip>
#include <iostream>
#include <string>

#include "MSPRS/Modem_MSPRS.hpp"
#include "MSPRS/NSC.hpp"
#include "MSPRS/Bounds.hpp"
#include "MSPRS/ExitSweep.hpp"
#include "MSPRS/LdpcSweep.hpp"
#include "MSPRS/Params.hpp"
#include "MSPRS/Record.hpp"
#include "MSPRS/Sweep.hpp"
#include "MSPRS/Taps.hpp"

namespace
{
struct args
{
    std::string mode   = "params";
    int         L0     = 3;
    std::string family = "balanced";
    std::string params = msprs::Params::default_path();
    std::string taps   = msprs::taps_dir();
    double      sigma  = 1.0;
    int         iters  = 7;
    int         be     = 500;
    int         minfra = 200;
    int         maxfra = 200000;
    int         threads = 1;
    int         seed   = 0;
    double      lo = 0.0, hi = 7.01, step = 0.5;
    int         n_ia = 100, n_trials = 50, exit_bits = 4998;
    bool        fresh_rx = false;
    int         bp_ite = 20;
    int         chunk  = 25;
    std::string ldpc_h;
    std::string out;
    std::string stamp;
    std::string scheme;
};

args parse(int argc, char** argv)
{
    args a;
    for (int i = 1; i < argc; i++)
    {
        const std::string s = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if      (s == "--mode"   && i + 1 < argc) a.mode   = next();
        else if (s == "--L0"     && i + 1 < argc) a.L0     = std::stoi(next());
        else if (s == "--family" && i + 1 < argc) a.family = next();
        else if (s == "--params" && i + 1 < argc) a.params = next();
        else if (s == "--taps"   && i + 1 < argc) a.taps   = next();
        else if (s == "--sigma"  && i + 1 < argc) a.sigma  = std::stod(next());
        else if (s == "--iters"     && i + 1 < argc) a.iters   = std::stoi(next());
        else if (s == "--be"        && i + 1 < argc) a.be      = std::stoi(next());
        else if (s == "--min-fra"   && i + 1 < argc) a.minfra  = std::stoi(next());
        else if (s == "--max-fra"   && i + 1 < argc) a.maxfra  = std::stoi(next());
        else if (s == "--threads"   && i + 1 < argc) a.threads = std::stoi(next());
        else if (s == "--seed"      && i + 1 < argc) a.seed    = std::stoi(next());
        else if (s == "--ebn0-min"  && i + 1 < argc) a.lo      = std::stod(next());
        else if (s == "--ebn0-max"  && i + 1 < argc) a.hi      = std::stod(next());
        else if (s == "--ebn0-step" && i + 1 < argc) a.step    = std::stod(next());
        else if (s == "--n-ia"      && i + 1 < argc) a.n_ia     = std::stoi(next());
        else if (s == "--n-trials"  && i + 1 < argc) a.n_trials = std::stoi(next());
        else if (s == "--exit-bits" && i + 1 < argc) a.exit_bits = std::stoi(next());
        else if (s == "--exit-fresh-rx")             a.fresh_rx = true;
        else if (s == "--bp-ite"    && i + 1 < argc) a.bp_ite   = std::stoi(next());
        else if (s == "--chunk"     && i + 1 < argc) a.chunk    = std::stoi(next());
        else if (s == "--ldpc-h"    && i + 1 < argc) a.ldpc_h   = next();
        else if (s == "--out"       && i + 1 < argc) a.out      = next();
        else if (s == "--stamp"     && i + 1 < argc) a.stamp    = next();
        else if (s == "--scheme"    && i + 1 < argc) a.scheme   = next();
        else { std::cerr << "unknown argument: " << s << "\n"; std::exit(2); }
    }
    return a;
}
} // namespace

int main(int argc, char** argv)
{
    const args a = parse(argc, argv);

    if (a.mode == "params")
    {
        const auto p = msprs::load_params(a.params);
        std::cout << "source_bits      " << p.source_bits       << "\n"
                  << "coded_bits       " << p.coded_bits        << "\n"
                  << "turbo_iterations " << p.turbo_iterations  << "\n"
                  << "conv             K=" << p.conv_K << " ("
                  << p.conv_octal[0] << "," << p.conv_octal[1] << ")\n"
                  << "ldpc             " << p.ldpc.k << "/" << p.ldpc.n << "\n";
        return 0;
    }

    if (a.mode == "taps")
    {
        const auto t = msprs::load_taps(a.taps, a.L0, a.family);
        std::cout << std::setprecision(12);
        for (double v : t.h0) std::cout << v << "\n";
        std::cout << t.h1 << "\n";
        return 0;
    }

    // Parity harness: identical input in, numbers out, so the modem can be
    // diffed against a reference implementation.
    if (a.mode == "modtest" || a.mode == "demodtest" || a.mode == "tdemodtest")
    {
        const auto t = msprs::load_taps(a.taps, a.L0, a.family);
        const std::vector<float> CP = { (float)a.sigma };
        std::cout << std::setprecision(12);

        if (a.mode == "modtest")
        {
            std::vector<int> bits;
            int b;
            while (std::cin >> b) bits.push_back(b);
            msprs::Modem_MSPRS<> m((int)bits.size(), t);
            std::vector<float> out(msprs::Modem_MSPRS<>::size_mod((int)bits.size(), a.L0));
            m.modulate(bits, out);
            for (auto v : out) std::cout << v << "\n";
            return 0;
        }

        if (a.mode == "demodtest")
        {
            std::vector<float> y;
            double v;
            while (std::cin >> v) y.push_back((float)v);
            const int Nb = 2 * (int)y.size() - (a.L0 - 1) - (a.L0 % 2 == 0 ? 1 : 0);
            msprs::Modem_MSPRS<> m(Nb, t);
            std::vector<float> llr(Nb);
            m.demodulate(CP, y, llr);
            for (auto z : llr) std::cout << z << "\n";
            return 0;
        }

        int n_sym = 0;
        std::cin >> n_sym;
        std::vector<float> y(n_sym);
        for (int i = 0; i < n_sym; i++) std::cin >> y[i];
        std::vector<float> la;
        double v;
        while (std::cin >> v) la.push_back((float)v);
        msprs::Modem_MSPRS<> m((int)la.size(), t);
        std::vector<float> ext(la.size());
        m.tdemodulate(CP, y, la, ext);
        for (auto z : ext) std::cout << z << "\n";
        return 0;
    }

    if (a.mode == "enctest" || a.mode == "dectest")
    {
        const auto p = msprs::load_params(a.params);
        const msprs::NSC_Trellis tr(p.conv_K, { p.conv_octal[0], p.conv_octal[1] });
        std::cout << std::setprecision(12);

        if (a.mode == "enctest")
        {
            std::vector<int> u;
            int b;
            while (std::cin >> b) u.push_back(b);
            msprs::Encoder_NSC<> enc((int)u.size(), tr);
            std::vector<int> x(tr.codeword_length((int)u.size()));
            enc.encode(u, x);
            for (auto v : x) std::cout << v << "\n";
            return 0;
        }

        std::vector<float> lin;
        double v;
        while (std::cin >> v) lin.push_back((float)v);
        const int K = (int)lin.size() / tr.n_out - tr.memory;
        msprs::Decoder_NSC_SISO<> d(K, tr);
        std::vector<float> ext(lin.size());
        std::vector<int> hard(K);
        d.decode_both(lin.data(), ext.data(), hard.data());
        for (auto z : ext) std::cout << z << "\n";
        return 0;
    }

    if (a.mode == "uncoded-msprs" || a.mode == "coded-msprs")
    {
        const auto pr = msprs::load_params(a.params);
        const auto taps = msprs::load_taps(a.taps, a.L0, a.family);
        const msprs::NSC_Trellis tr(pr.conv_K, { pr.conv_octal[0], pr.conv_octal[1] });

        msprs::SweepConfig cfg;
        cfg.coded   = (a.mode == "coded-msprs");
        cfg.K       = cfg.coded ? pr.source_bits : 2 * pr.source_bits + 2;
        cfg.iters   = a.iters;
        cfg.be      = a.be;
        cfg.min_fra = a.minfra;
        cfg.max_fra = a.maxfra;
        cfg.threads = a.threads;
        cfg.chunk   = a.chunk;
        cfg.seed    = a.seed;
        cfg.itl_seed  = pr.interleaver_seed;
        cfg.ebn0_min  = a.lo;
        cfg.ebn0_max  = a.hi;
        cfg.ebn0_step = a.step;

        std::cout << "#     Eb/N0 |        FRA |         BE |       BER |    s\n";
        for (const auto& pt : msprs::run_sweep(cfg, taps, tr))
        {
            std::cout << std::fixed << std::setprecision(2) << std::setw(11) << pt.eb_no_db
                      << " |" << std::setw(11) << pt.frames << " |" << std::setw(11) << pt.errors
                      << " |" << std::scientific << std::setprecision(3) << std::setw(11)
                      << (double)pt.errors / (double)pt.bits
                      << " |" << std::fixed << std::setprecision(1) << std::setw(5) << pt.seconds << "\n";
            if (cfg.coded)
            {
                std::cout << "# per-iter " << std::fixed << std::setprecision(1) << pt.eb_no_db;
                for (auto e : pt.per_iter) std::cout << " " << e;
                std::cout << "\n";
            }
            if (!a.out.empty())
            {
                nlohmann::json meta;
                meta["L0"] = a.L0;
                meta["filter"] = a.family;
                meta["source_bits"] = cfg.K;
                meta["metric_convention"] = "llr-2sigma2";
                if (cfg.coded) { meta["iters"] = cfg.iters; meta["code"] = "conv_K3_57"; }
                const std::string scheme = a.scheme.empty()
                    ? ("nsm_L" + std::to_string(a.L0) + "_" + a.family +
                       (cfg.coded ? "_conv_K3_" + std::to_string(cfg.iters) + "iters" : "_uncoded"))
                    : a.scheme;
                msprs::write_point(a.out, scheme, pt, meta, "aff3ct-4.1.2", a.stamp);
            }
        }
        std::cout << "# done\n";
        return 0;
    }

    if (a.mode == "exit")
    {
        const auto taps = msprs::load_taps(a.taps, a.L0, a.family);
        msprs::ExitConfig cfg;
        cfg.bits = a.exit_bits; cfg.n_ia = a.n_ia; cfg.n_trials = a.n_trials;
        cfg.threads = a.threads; cfg.fresh_rx = a.fresh_rx;
        cfg.ebn0_min = a.lo; cfg.ebn0_max = a.hi; cfg.ebn0_step = a.step;

        std::cout << "# exit L0=" << a.L0 << " " << a.family << " trials=" << cfg.n_trials
                  << " n_ia=" << cfg.n_ia << " bits=" << cfg.bits << "\n"
                  << std::setprecision(10);
        const auto pts = msprs::exit_sweep(cfg, taps);
        for (const auto& e : pts)
            std::cout << "E " << e.eb_no_db << " " << e.ia << " " << e.ie_avg << " "
                      << e.ie_hist << " " << e.ie_mag << " " << e.ia_measured << "\n";

        if (!a.out.empty())
        {
            nlohmann::json j;
            std::vector<double> ia;
            for (int k = 0; k < cfg.n_ia; k++) ia.push_back(pts[(size_t)k].ia);
            j["IA"] = ia;
            std::vector<double> snrs;
            for (const auto& e : pts)
            {
                std::ostringstream key; key << e.eb_no_db;
                if (!j["results"].contains(key.str()))
                {
                    snrs.push_back(e.eb_no_db);
                    j["results"][key.str()] = { {"IE_avg", nlohmann::json::array()},
                                                {"IE_hist", nlohmann::json::array()},
                                                {"IE_mag", nlohmann::json::array()},
                                                {"IA_measured", nlohmann::json::array()} };
                }
                auto& r = j["results"][key.str()];
                r["IE_avg"].push_back(e.ie_avg);
                r["IE_hist"].push_back(e.ie_hist);
                r["IE_mag"].push_back(e.ie_mag);
                r["IA_measured"].push_back(e.ia_measured);
            }
            j["eb_no_db"] = snrs;
            j["meta"] = { {"L0", a.L0}, {"filter", a.family}, {"N_TRIALS", cfg.n_trials},
                          {"source_bits", cfg.bits}, {"implementation", "aff3ct-4.1.2"},
                          {"metric_convention", "llr-2sigma2"} };
            if (!a.stamp.empty()) j["meta"]["timestamp"] = a.stamp;
            const std::string scheme = a.scheme.empty()
                ? ("nsm_L" + std::to_string(a.L0) + "_" + a.family) : a.scheme;
            msprs::mkdir_p(a.out);
            std::ofstream of(a.out + "/" + scheme + ".json");
            of << j.dump(2) << "\n";
        }
        std::cout << "# done\n";
        return 0;
    }

    if (a.mode == "ldpc-msprs")
    {
        const auto pr = msprs::load_params(a.params);
        const auto taps = msprs::load_taps(a.taps, a.L0, a.family);
        msprs::LdpcConfig cfg;
        cfg.h_path = a.ldpc_h;
        cfg.bp_iters = a.bp_ite ? a.bp_ite : pr.ldpc.bp_iterations;
        cfg.iters = a.iters; cfg.be = a.be; cfg.min_fra = a.minfra; cfg.max_fra = a.maxfra;
        cfg.threads = a.threads; cfg.seed = a.seed; cfg.itl_seed = pr.interleaver_seed;
        cfg.chunk = a.chunk;
        cfg.ebn0_min = a.lo; cfg.ebn0_max = a.hi; cfg.ebn0_step = a.step;
        if (cfg.h_path.empty()) { std::cerr << "ldpc-msprs needs --ldpc-h\n"; return 2; }

        std::cout << "#     Eb/N0 |        FRA |         BE |       BER |    s\n";
        for (const auto& pt : msprs::ldpc_sweep(cfg, taps))
            std::cout << std::fixed << std::setprecision(2) << std::setw(11) << pt.eb_no_db
                      << " |" << std::setw(11) << pt.frames << " |" << std::setw(11) << pt.errors
                      << " |" << std::scientific << std::setprecision(3) << std::setw(11)
                      << (double)pt.errors / (double)pt.bits
                      << " |" << std::fixed << std::setprecision(1) << std::setw(5) << pt.seconds << "\n";
        std::cout << "# done\n";
        return 0;
    }

    if (a.mode == "bounds")
    {
        const auto pr = msprs::load_params(a.params);
        std::cout << "# family L0 d2_es1 d2_es5 gain_db\n" << std::fixed << std::setprecision(6);
        for (const auto& fam : pr.families)
            for (int L0 : pr.l0_values)
            {
                const auto t = msprs::load_taps(a.taps, L0, fam);
                const double d2 = msprs::min_squared_distance(t);
                std::cout << fam << " " << L0 << " " << d2 << " " << pr.es * d2 << " "
                          << 10.0 * std::log10(pr.es * d2 / 4.0) << "\n";
            }
        return 0;
    }

    std::cerr << "unknown mode: " << a.mode << "\n";
    return 2;
}
