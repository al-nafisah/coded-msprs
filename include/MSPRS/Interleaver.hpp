// The interleaver of the coded block, read from rom/interleaver.hex.
//
// One table serves the simulator, ota/ and the HDL, so a permutation drawn by
// a library's shuffle cannot drift between them. Entry i is the position in
// the coded block of the i-th bit sent, out[i] = in[pi[i]], the convention of
// AFF3CT's interleavers.
#ifndef MSPRS_INTERLEAVER_HPP
#define MSPRS_INTERLEAVER_HPP

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <aff3ct.hpp>

#ifndef MSPRS_REPO_ROOT
#define MSPRS_REPO_ROOT "."
#endif

namespace msprs
{

inline std::string interleaver_path() { return std::string(MSPRS_REPO_ROOT) + "/rom/interleaver.hex"; }

//! The permutation, checked to hold every position once. Lines starting with
//! "//" are comments, as in Verilog's $readmemh.
inline std::vector<uint32_t> load_interleaver(const std::string& path = interleaver_path())
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open '" + path + "'");
    std::vector<uint32_t> pi;
    for (std::string line; std::getline(in, line);)
        if (!line.empty() && line.compare(0, 2, "//") != 0) pi.push_back((uint32_t)std::stoul(line, nullptr, 16));
    std::vector<bool> seen(pi.size(), false);
    for (const auto v : pi)
    {
        if (v >= pi.size() || seen[v]) throw std::runtime_error("'" + path + "' is not a permutation");
        seen[v] = true;
    }
    return pi;
}

class Interleaver_core_table : public aff3ct::tools::Interleaver_core<uint32_t>
{
  public:
    explicit Interleaver_core_table(const std::vector<uint32_t>& table)
      : aff3ct::tools::Interleaver_core<uint32_t>((int)table.size(), "table", false)
      , table(table)
    {
        this->init();
    }

    Interleaver_core_table* clone() const override { return new Interleaver_core_table(*this); }

  protected:
    void gen_lut(uint32_t* lut, const size_t /*frame_id*/) override { std::copy(table.begin(), table.end(), lut); }

  private:
    std::vector<uint32_t> table;
};

} // namespace msprs

#endif
