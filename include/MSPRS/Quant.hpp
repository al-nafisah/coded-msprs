// Fixed-point quantiser for the LLRs the turbo loop passes between the
// equaliser and the outer decoder: the words a hardware receiver keeps in its
// interleaver memory.
//
// b bits, two's complement restricted to the symmetric codes
// -(2^(b-1)-1) .. 2^(b-1)-1. Zero, the "no information" message of the first
// pass, is a level, and both signs saturate at the same magnitude. Code c
// stands for the LLR c * step, so a power-of-two step is a fixed-point format
// with -log2(step) fractional bits. Rounding is to the nearest level.
#ifndef MSPRS_QUANT_HPP
#define MSPRS_QUANT_HPP

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace msprs
{

struct LlrQuant
{
    int    bits = 0;   // 0 leaves the messages in floating point
    double step = 1.0;

    LlrQuant() = default;
    LlrQuant(const int b, const double s) : bits(b), step(s)
    {
        if (b != 0 && (b < 2 || b > 16)) throw std::invalid_argument("--llr-bits must be 0 or 2..16");
        if (!(s > 0.0)) throw std::invalid_argument("--llr-step must be positive");
    }

    bool on() const { return bits > 0; }
    int  top() const { return (1 << (bits - 1)) - 1; }   // largest code
    int  levels() const { return 2 * top() + 1; }

    int code(const float v) const
    {
        const double c = std::round((double)v / step);
        return (int)std::max((double)-top(), std::min((double)top(), c));
    }

    void apply(std::vector<float>& v) const
    {
        if (!on()) return;
        for (auto& x : v) x = (float)(code(x) * step);
    }

    //! Record suffix, e.g. "_q4_step0.5"; empty in floating point.
    std::string suffix() const
    {
        if (!on()) return "";
        std::ostringstream s;
        s << "_q" << bits << "_step" << step;
        return s.str();
    }
};

} // namespace msprs

#endif
