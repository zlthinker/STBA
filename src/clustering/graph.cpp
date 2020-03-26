#include <cassert>
#include <iostream>

#include "STBA/clustering/graph.h"

Graph::Graph()
{
}

void Graph::AddSelfEdge(size_t i, double weight)
{
    assert(i < selfs_.size());
    selfs_[i] = weight;
}

void Graph::AddUndirectedEdge(size_t i, size_t j, double weight)
{
	assert(i != j);
    assert(i < incidences_.size() && j < incidences_.size());
	// Do not check if an edge is added more than once
    incidences_[i].push_back({j, weight});
    incidences_[j].push_back({i, weight});
}

void Graph::AddDirectedEdge(size_t i, size_t j, double weight)
{
	assert(i != j);
    assert(i < incidences_.size() && j < incidences_.size());
	// Do not check if an edge is added more than once
    incidences_[i].push_back({j, weight});
}

size_t Graph::EdgeNum() const
{
    size_t node_num = NodeNum();
    size_t edge_num = 0;
    for (size_t i = 0; i < node_num; i++)
    {
        edge_num += incidences_[i].size();
    }
    return edge_num / 2;
}

double Graph::EdgeWeight(size_t i, size_t j) const
{
    assert(i != j);
    assert(i < incidences_.size() && j < incidences_.size());
    double weight = 0.0;
    std::vector<EdgeData> const & edges = incidences_[i];
    bool found = false;
    for (size_t k = 0; k < edges.size(); k++)
    {
        EdgeData const & edge = edges[k];
        if (edge.node == j)
        {
            weight = edge.weight;
            found = true;
            break;
        }
    }
    assert(found && "Edge not exists");
    return weight;
}
