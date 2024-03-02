// Simulation parameters, read from config.json. One source for the C++ and
// the notebooks; there is no second copy.
#ifndef MSPRS_PARAMS_HPP
#define MSPRS_PARAMS_HPP

#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef MSPRS_REPO_ROOT
#define MSPRS_REPO_ROOT "."
#endif

namespace msprs
{

struct Params
{
    int              conv_K;
    std::vector<int> conv_octal;
    double           conv_rate;

    int    source_bits, coded_bits, interleaver_seed, turbo_iterations;
    double es;

    std::map<std::string, int>    bits_per_symbol;
    std::map<std::string, double> avg_bit_energy;

    std::vector<int>         l0_values;
    std::vector<std::string> families;

    struct { int k, n, bp_iterations; std::string matrix; } ldpc{};
    struct { int min_bit_errors, min_packets, reliability_floor; } monte_carlo{};

    std::vector<double> ber_grid, exit_grid, exit_standout, convergence;

    double exit_ia_scale, exit_ia_lo, exit_ia_hi;
    int    exit_ia_n, exit_n_trials;

    struct { double tau, rolloff; } ftn{};

    std::vector<double> ia_grid() const
    {
        std::vector<double> ia((size_t)exit_ia_n);
        for (int i = 0; i < exit_ia_n; i++)
            ia[(size_t)i] = exit_ia_scale *
                (exit_ia_lo + (exit_ia_hi - exit_ia_lo) * (double)i / (double)(exit_ia_n - 1));
        return ia;
    }

    static std::string default_path() { return std::string(MSPRS_REPO_ROOT) + "/config.json"; }
};

// Every field is required. A silently defaulted packet length or iteration
// count would produce a cache that disagrees with every other one.
inline Params load_params(const std::string& path = Params::default_path())
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open '" + path + "'");
    nlohmann::json j;
    in >> j;

    Params p;
    p.conv_K     = j.at("conv_coder").at("K");
    p.conv_octal = j.at("conv_coder").at("octal_code").get<std::vector<int>>();
    p.conv_rate  = j.at("conv_rate");

    p.source_bits      = j.at("source_bits");
    p.coded_bits       = j.at("coded_bits");
    p.interleaver_seed = j.at("interleaver_seed");
    p.turbo_iterations = j.at("turbo_iterations");
    p.es               = j.at("es");

    p.bits_per_symbol = j.at("bits_per_symbol").get<std::map<std::string, int>>();
    p.avg_bit_energy  = j.at("avg_bit_energy").get<std::map<std::string, double>>();
    p.l0_values       = j.at("l0_values").get<std::vector<int>>();
    p.families        = j.at("families").get<std::vector<std::string>>();

    p.ldpc.k             = j.at("ldpc").at("k");
    p.ldpc.n             = j.at("ldpc").at("n");
    p.ldpc.matrix        = j.at("ldpc").at("matrix");
    p.ldpc.bp_iterations = j.at("ldpc").at("bp_iterations");

    p.monte_carlo.min_bit_errors    = j.at("monte_carlo").at("min_bit_errors");
    p.monte_carlo.min_packets       = j.at("monte_carlo").at("min_packets");
    p.monte_carlo.reliability_floor = j.at("monte_carlo").at("reliability_floor");

    p.ber_grid      = j.at("ebno_grid_db").at("ber").get<std::vector<double>>();
    p.exit_grid     = j.at("ebno_grid_db").at("exit").get<std::vector<double>>();
    p.exit_standout = j.at("ebno_grid_db").at("exit_standout").get<std::vector<double>>();
    p.convergence   = j.at("ebno_grid_db").at("convergence").get<std::vector<double>>();

    p.exit_ia_scale = j.at("exit_ia_grid").at("scale");
    p.exit_ia_lo    = j.at("exit_ia_grid").at("lo");
    p.exit_ia_hi    = j.at("exit_ia_grid").at("hi");
    p.exit_ia_n     = j.at("exit_ia_grid").at("n");
    p.exit_n_trials = j.at("exit_n_trials");

    p.ftn.tau     = j.at("ftn").at("tau");
    p.ftn.rolloff = j.at("ftn").at("rolloff");

    const int expect = 2 * (p.source_bits + p.conv_K - 1);
    if (p.coded_bits != expect)
        throw std::runtime_error("coded_bits disagrees with 2*(source_bits + K - 1)");
    if (p.conv_octal.size() != 2) throw std::runtime_error("need exactly 2 generators");

    return p;
}

} // namespace msprs

#endif
