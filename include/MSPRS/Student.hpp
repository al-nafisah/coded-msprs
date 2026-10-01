// The trained equaliser, inside the C++ receiver.
//
// Loads the weights the training script exported and evaluates the network on
// the same Window the dataset was built from. It plays the BCJR's role, a
// priori in and extrinsic out, so the BER and EXIT sweeps can swap one for the
// other. No libtorch: the network is a few dense layers.
#ifndef MSPRS_STUDENT_HPP
#define MSPRS_STUDENT_HPP

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "MSPRS/Dataset.hpp"

namespace msprs
{

struct Dense
{
    int                in, out;
    std::vector<float> wt, b;   // wt is in x out, so the inner loop runs over outputs and vectorises
};

class Student
{
  public:
    //! `weights` as train.py writes it; the input normalisation is read from
    //! the same path with the suffix .norm.
    explicit Student(const std::string& weights)
    {
        std::ifstream f(weights);
        if (!f) throw std::runtime_error("cannot open '" + weights + "'");
        int n_hidden = 0, hidden = 0;
        f >> features_ >> n_hidden >> hidden;

        while (true)
        {
            Dense d;
            if (!(f >> d.in >> d.out)) break;
            std::vector<float> w((size_t)d.in * d.out);   // out x in, as torch stores it
            for (auto& v : w) f >> v;
            d.b.resize((size_t)d.out);
            for (auto& v : d.b) f >> v;
            d.wt.resize(w.size());
            for (int o = 0; o < d.out; o++)
                for (int i = 0; i < d.in; i++) d.wt[(size_t)i * d.out + o] = w[(size_t)o * d.in + i];
            width_ = std::max(width_, d.out);
            layers_.push_back(std::move(d));
        }
        if (layers_.empty()) throw std::runtime_error("no layers in '" + weights + "'");
        if (layers_.front().in != features_ || layers_.back().out != 1)
            throw std::runtime_error("layer sizes in '" + weights + "' do not chain");

        const std::string norm = weights.substr(0, weights.rfind('.')) + ".norm";
        std::ifstream n(norm);
        if (!n) throw std::runtime_error("cannot open '" + norm + "'");
        mu_.resize((size_t)features_);
        sd_.resize((size_t)features_);
        for (auto& v : mu_) n >> v;
        for (auto& v : sd_) n >> v;
        if (!n) throw std::runtime_error("'" + norm + "' is short");
    }

    int features() const { return features_; }

    //! Same role as Modem_MSPRS::tdemodulate: a priori in, extrinsic out.
    void tdemodulate(const Window& w, const float sigma, const std::vector<float>& rx,
                     const std::vector<float>& la, std::vector<float>& ext) const
    {
        if (w.dim() != features_)
            throw std::runtime_error("network takes " + std::to_string(features_) +
                                     " features, window gives " + std::to_string(w.dim()));
        std::vector<float> x((size_t)features_), a((size_t)width_), z((size_t)width_);
        for (size_t i = 0; i < w.sites.size(); i++)
        {
            w.fill((int)i, rx, la, sigma, x.data());
            for (int k = 0; k < features_; k++) x[(size_t)k] = (x[(size_t)k] - mu_[(size_t)k]) / sd_[(size_t)k];
            ext[i] = forward(x.data(), a.data(), z.data());
        }
    }

  private:
    // a and z are scratch, at least as wide as the widest layer.
    float forward(const float* x, float* a, float* z) const
    {
        const float* in = x;
        for (size_t l = 0; l < layers_.size(); l++)
        {
            const Dense& d = layers_[l];
            std::copy(d.b.begin(), d.b.end(), z);
            for (int i = 0; i < d.in; i++)
            {
                const float  v   = in[i];
                const float* row = &d.wt[(size_t)i * d.out];
                for (int o = 0; o < d.out; o++) z[o] += v * row[o];
            }
            if (l + 1 < layers_.size())   // ReLU on every layer but the last
                for (int o = 0; o < d.out; o++) a[o] = std::max(0.0f, z[o]);
            in = a;
        }
        return z[0];
    }

    int                features_ = 0;
    int                width_    = 0;
    std::vector<Dense> layers_;
    std::vector<float> mu_, sd_;
};

} // namespace msprs

#endif
