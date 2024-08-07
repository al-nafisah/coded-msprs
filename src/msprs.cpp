// MS-PRS simulation driver.
#include <iomanip>
#include <iostream>
#include <string>

#include "MSPRS/Modem_MSPRS.hpp"
#include "MSPRS/NSC.hpp"
#include "MSPRS/Params.hpp"
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

    std::cerr << "unknown mode: " << a.mode << "\n";
    return 2;
}
