#ifndef _NETWORK_HEADER
#define _NETWORK_HEADER

#include "utility.h"
#include <vector>

struct EdgeData
{
    EdgeData() : node(-1), weight(0.0) {}
    EdgeData(size_t n, double w) : node(n), weight(w) {}

    size_t node;
    double weight;
};

struct Graph
{
    Graph();

    void AddSelfEdge(size_t i, double weight = 0.0);
    void AddUndirectedEdge(size_t i, size_t j, double weight = 1.0);
    void AddDirectedEdge(size_t i, size_t j, double weight = 1.0);

    size_t EdgeNum() const;
    double EdgeWeight(size_t, size_t) const;

    inline size_t NodeNum() const { return incidences_.size(); }
    inline const std::vector<EdgeData>& GetIncidentEdges(int i) const { return incidences_[i]; }
    inline double GetSelfWeight(int i) const { return selfs_[i]; }
    inline void Swap(Graph& other)
    {
        other.selfs_.swap(selfs_);
        other.incidences_.swap(incidences_);
    }
    void Resize(size_t num)
    {
        selfs_.resize(num);
        incidences_.resize(num);
    }

    std::vector<double> selfs_;
    std::vector<std::vector<EdgeData> > incidences_;
};

#endif
