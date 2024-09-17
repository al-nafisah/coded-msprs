// Minimum squared Euclidean distance of the rate-2 trellis, and the union bound.
//
// Distances are returned on the internal Es = 1 normalisation. Multiply by 5
// to compare against the Es = 5 scale, where 4-ASK has d^2 = 4.
//
// MSED is a shortest path on the error-state-pair graph. A node is an ordered
// pair of trellis states; an edge weight is the squared difference of the two
// branch outputs. Every diagonal node feeds a virtual sink at zero cost, so
// Dijkstra from the sink over the transposed graph gives, for every pair, the
// cheapest way back to re-merging.
#ifndef MSPRS_BOUNDS_HPP
#define MSPRS_BOUNDS_HPP

#include <cmath>
#include <limits>
#include <map>
#include <queue>
#include <utility>
#include <vector>

#include "MSPRS/Taps.hpp"

namespace msprs
{

// Trellis outputs and transitions, in the same convention Modem_MSPRS uses:
// state bits are newest-first, so bit (L0-2) is x0[k-1].
struct PairTrellis
{
    int                 M;
    std::vector<double> out;   // [(s*2 + j)*2 + b] -> symbol
    std::vector<int>    next;  // [s*2 + j] -> next state

    explicit PairTrellis(const Taps& t)
      : M(1 << (t.L0 - 1)), out((size_t)(1 << (t.L0 - 1)) * 4), next((size_t)(1 << (t.L0 - 1)) * 2)
    {
        for (int s = 0; s < M; s++)
            for (int j = 0; j < 2; j++)
            {
                double acc = t.h0[0] * (1.0 - 2.0 * j);
                for (int m = 0; m < t.L0 - 1; m++)
                    acc += t.h0[(size_t)m + 1] * (1.0 - 2.0 * ((s >> (t.L0 - 2 - m)) & 1));
                next[(size_t)s * 2 + j] = (s >> 1) | (j << (t.L0 - 2));
                for (int b = 0; b < 2; b++)
                    out[((size_t)s * 2 + j) * 2 + b] = acc + (b == 0 ? t.h1 : -t.h1);
            }
    }
};

// Shortest distance from every pair node back to the re-merged set.
inline std::vector<double> dist_to_merge(const PairTrellis& tr)
{
    const int M = tr.M, N = M * M, sink = N;
    const double INF = std::numeric_limits<double>::infinity();

    // Transposed graph: an edge src->dst of the pair graph is stored dst->src,
    // with parallel edges collapsed to their minimum weight.
    std::map<std::pair<int, int>, double> best;
    for (int s1 = 0; s1 < M; s1++)
        for (int s2 = 0; s2 < M; s2++)
        {
            const int src = s1 * M + s2;
            for (int j1 = 0; j1 < 2; j1++)
                for (int b1 = 0; b1 < 2; b1++)
                    for (int j2 = 0; j2 < 2; j2++)
                        for (int b2 = 0; b2 < 2; b2++)
                        {
                            const double d = tr.out[((size_t)s1 * 2 + j1) * 2 + b1]
                                           - tr.out[((size_t)s2 * 2 + j2) * 2 + b2];
                            const int dst = tr.next[(size_t)s1 * 2 + j1] * M
                                          + tr.next[(size_t)s2 * 2 + j2];
                            const auto key = std::make_pair(dst, src);
                            const double w = d * d;
                            auto it = best.find(key);
                            if (it == best.end() || it->second > w) best[key] = w;
                        }
        }

    std::vector<std::vector<std::pair<int, double>>> adj((size_t)N + 1);
    for (const auto& kv : best) adj[(size_t)kv.first.first].push_back({ kv.first.second, kv.second });
    for (int s = 0; s < M; s++) adj[(size_t)sink].push_back({ s * M + s, 0.0 });

    std::vector<double> dist((size_t)N + 1, INF);
    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>,
                        std::greater<std::pair<double, int>>> pq;
    dist[(size_t)sink] = 0.0;
    pq.push({ 0.0, sink });
    while (!pq.empty())
    {
        const auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[(size_t)u]) continue;
        for (const auto& [v, w] : adj[(size_t)u])
            if (d + w < dist[(size_t)v]) { dist[(size_t)v] = d + w; pq.push({ d + w, v }); }
    }
    return dist;
}

// Full MSED over all error events, including parallel transitions that flip
// only the memoryless stream-1 bit.
inline double min_squared_distance(const Taps& t)
{
    const PairTrellis tr(t);
    const auto dist = dist_to_merge(tr);
    const int  M = tr.M;

    double best = std::numeric_limits<double>::infinity();
    for (int s = 0; s < M; s++)
        for (int j1 = 0; j1 < 2; j1++)
            for (int b1 = 0; b1 < 2; b1++)
                for (int j2 = 0; j2 < 2; j2++)
                    for (int b2 = 0; b2 < 2; b2++)
                    {
                        if (j1 == j2 && b1 == b2) continue;
                        const double d = tr.out[((size_t)s * 2 + j1) * 2 + b1]
                                       - tr.out[((size_t)s * 2 + j2) * 2 + b2];
                        const int dst = tr.next[(size_t)s * 2 + j1] * M
                                      + tr.next[(size_t)s * 2 + j2];
                        const double total = d * d + dist[(size_t)dst];
                        if (total > 0.0 && total < best) best = total;
                    }
    return best;
}

// Squared distance of the smallest parallel-transition event: one that flips
// only the stream-1 bit and leaves the state trajectory alone.
inline double parallel_min_squared_distance(const Taps& t) { return 4.0 * t.h1 * t.h1; }

// Single-term union bound, BER ~ (k_e/2) erfc(sqrt(d2 Rc m Eb/N0 / 4)).
inline double union_bound_ber(double d2, double eb_no_db, double k_e = 1.0,
                              double rate = 1.0, double bits_per_symbol = 2.0)
{
    const double snr = std::pow(10.0, eb_no_db / 10.0);
    return 0.5 * k_e * std::erfc(std::sqrt(d2 * rate * bits_per_symbol * snr / 4.0));
}

} // namespace msprs

#endif
