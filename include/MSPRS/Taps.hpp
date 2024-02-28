// MS-PRS filter taps, read from JSON.
#ifndef MSPRS_TAPS_HPP
#define MSPRS_TAPS_HPP

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef MSPRS_REPO_ROOT
#define MSPRS_REPO_ROOT "."
#endif

namespace msprs
{

struct Taps
{
    int                 L0;
    std::vector<double> h0;  // FIR taps of the memory stream
    double              h1;  // scalar tap of the memoryless stream
};

inline std::string taps_dir() { return std::string(MSPRS_REPO_ROOT) + "/filters"; }

// The JSON stores unit-norm h0, h1 and eta, the energy on the FIR stream at
// Es = 5. Scaling by sqrt(eta/5) and sqrt((5-eta)/5) normalises the pair to
// ||h0||^2 + h1^2 = 1. Swapping the two was a real bug once; the check catches it.
inline Taps load_taps(const std::string& dir, int L0, const std::string& family)
{
    const std::string path = dir + "/" + family + ".json";
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open tap file '" + path + "'");

    nlohmann::json j;
    in >> j;
    const auto key = std::to_string(L0);
    if (!j.contains(key)) throw std::runtime_error("no L0=" + key + " in " + path);

    Taps t;
    t.L0 = L0;
    t.h0 = j[key].at("h0").get<std::vector<double>>();
    const auto h1v = j[key].at("h1").get<std::vector<double>>();
    const double eta = j[key].at("eta").get<double>();

    if ((int)t.h0.size() != L0) throw std::runtime_error("h0 length != L0");
    if (h1v.size() != 1) throw std::runtime_error("h1 must be scalar");

    for (auto& v : t.h0) v *= std::sqrt(eta / 5.0);
    t.h1 = h1v[0] * std::sqrt((5.0 - eta) / 5.0);

    double e = t.h1 * t.h1;
    for (double v : t.h0) e += v * v;
    if (std::abs(e - 1.0) > 1e-9) throw std::runtime_error("tap energy != 1");

    return t;
}

inline Taps load_taps(int L0, const std::string& family) { return load_taps(taps_dir(), L0, family); }

} // namespace msprs

#endif
