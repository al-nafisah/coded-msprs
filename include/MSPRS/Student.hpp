// The trained equaliser, running inside the C++ receiver.
//
// Loads the weights the training script exported and evaluates the same
// windowed forward pass, so the learned and exact equalisers are measured by
// one BER sweep. No libtorch: the network is a few dense layers.
#ifndef MSPRS_STUDENT_HPP
#define MSPRS_STUDENT_HPP

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "MSPRS/Dataset.hpp"

namespace msprs
{

struct Dense
{
    int                in, out;
    std::vector<float> w, b;   // w is out x in, row-major
};

class Student
{
  public:
    Student(const std::string& weights, const std::string& norm)
    {
        std::ifstream f(weights);
        if (!f) throw std::runtime_error("cannot open '" + weights + "'");
        int n_hidden = 0, hidden = 0;
        f >> features_ >> n_hidden >> hidden;

        while (true)
        {
            Dense d;
            if (!(f >> d.in >> d.out)) break;
            d.w.resize((size_t)d.in * d.out);
            d.b.resize((size_t)d.out);
            for (auto& v : d.w) f >> v;
            for (auto& v : d.b) f >> v;
            layers_.push_back(std::move(d));
        }
        if (layers_.empty()) throw std::runtime_error("no layers in '" + weights + "'");

        std::ifstream n(norm);
        if (!n) throw std::runtime_error("cannot open '" + norm + "'");
        mu_.resize((size_t)features_);
        sd_.resize((size_t)features_);
        for (auto& v : mu_) n >> v;
        for (auto& v : sd_) n >> v;
    }

    int features() const { return features_; }

    float forward(const float* x) const
    {
        std::vector<float> a((size_t)features_);
        for (int i = 0; i < features_; i++) a[(size_t)i] = (x[i] - mu_[(size_t)i]) / sd_[(size_t)i];

        std::vector<float> out;
        for (size_t l = 0; l < layers_.size(); l++)
        {
            const auto& d = layers_[l];
            out.assign((size_t)d.out, 0.0f);
            for (int o = 0; o < d.out; o++)
            {
                float acc = d.b[(size_t)o];
                const float* row = &d.w[(size_t)o * d.in];
                for (int i = 0; i < d.in; i++) acc += row[i] * a[(size_t)i];
                // ReLU on every layer but the last
                out[(size_t)o] = (l + 1 < layers_.size()) ? std::max(0.0f, acc) : acc;
            }
            a.swap(out);
        }
        return a[0];
    }

    // Same role as Modem_MSPRS::tdemodulate: a priori in, extrinsic out.
    void tdemodulate(const std::vector<float>& rx, const std::vector<float>& la,
                     std::vector<float>& ext, const DatasetConfig& cfg, float sigma) const
    {
        const int N  = (int)la.size();
        const int Ns = (int)rx.size();
        std::vector<float> feat((size_t)features_);

        for (int i = 0; i < N; i++)
        {
            const int c = i / 2;
            int k = 0;
            for (int s = -cfg.half_sym; s <= cfg.half_sym; s++)
            {
                const int idx = c + s;
                feat[(size_t)k++] = (idx >= 0 && idx < Ns) ? rx[(size_t)idx] : 0.0f;
            }
            for (int b = -cfg.half_bit; b <= cfg.half_bit; b++)
            {
                const int idx = i + b;
                feat[(size_t)k++] = (b == 0 || idx < 0 || idx >= N) ? 0.0f : la[(size_t)idx];
            }
            feat[(size_t)k++] = sigma;
            feat[(size_t)k++] = (i % 2 == 0) ? 1.0f : -1.0f;
            ext[(size_t)i] = forward(feat.data());
        }
    }

  private:
    int                features_ = 0;
    std::vector<Dense> layers_;
    std::vector<float> mu_, sd_;
};

} // namespace msprs

#endif
