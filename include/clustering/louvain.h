#ifndef LOUVAIN_H
#define LOUVAIN_H

#include <unordered_map>
#include "clustering/graph.h"

class Louvain
{
public:
    Louvain();

    Louvain(std::vector<size_t> const & nodes,
            std::unordered_map<size_t, std::unordered_map<size_t, double> > const & edges);

    Louvain(std::vector<size_t> const & nodes,
            std::vector<std::pair<size_t, size_t> > const & edges);

    ~Louvain();

    void Cluster();
    void Cluster(std::vector<std::pair<size_t, size_t> > const & initial_pairs);
    void StochasticCluster();
    void StochasticCluster(std::vector<std::pair<size_t, size_t> > const & initial_pairs);

    void MCMCSampling();

    void Initialize(std::vector<size_t> const & nodes,
                    std::vector<std::pair<size_t, size_t> > const & edges);
    void Initialize(std::vector<size_t> const & nodes,
                    std::unordered_map<size_t, std::unordered_map<size_t, double> > const & edges);
    void Reinitialize();

    void Print();

    double Modularity(size_t const community) const;
    double Modularity() const;

    size_t NodeNum() const { return node_graph_.NodeNum(); }
    size_t EdgeNum() const { return node_graph_.EdgeNum(); }

    inline double SumEdgeWeight() const { return sum_edge_weight_; }
    double EdgeWeight(size_t const, size_t const) const;

    void SetMaxCommunity(size_t s) { max_community_ = s; }
    size_t GetMaxCommunity() const { return max_community_; }
    void SetTemperature(double t) { temperature_ = t; }
    double GetTemperature() const { return temperature_; }

    void GetClusters(std::vector<std::vector<size_t> > &) const;
    void GetEdgesAcrossClusters(std::vector<std::pair<size_t, size_t> > & pairs) const;

private:
    void RearrangeCommunities();
    bool Merge();
    bool Merge(std::vector<std::pair<size_t, size_t> > const & initial_pairs);
    bool StochasticMerge();
    void Rebuild();

private:
    Graph graph_;
    Graph node_graph_;
    std::vector<int> node_map_;                                 // node index to community index
    std::vector<int> community_map_;                        // prev community index to community index
    std::vector<size_t> community_size_;                    // size of community
    std::vector<double> community_in_weight_;          // sum of edge weights inside a community
    std::vector<double> community_total_weight_;      // sum of edge weights incident to nodes of a community, including those inside a community
    double sum_edge_weight_;    // 2m
    size_t max_community_;
    double temperature_;

};

void TestMCMCSampling();
void Test();

#endif // LOUVAIN_H
