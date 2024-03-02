// MS-PRS simulation driver.
#include <iomanip>
#include <iostream>
#include <string>

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

    std::cerr << "unknown mode: " << a.mode << "\n";
    return 2;
}
