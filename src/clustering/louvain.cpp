#include "STBA/clustering/louvain.h"
#include "STBA/utility.h"

#include <queue>
#include <chrono>
#include <unordered_map>
#include <iostream>

size_t RandomPick(std::unordered_map<size_t, double> const & prob_map)
{
    assert(!prob_map.empty());
    double sum = 0.0;
    std::unordered_map<size_t, double>::const_iterator it = prob_map.begin();
    for (; it != prob_map.end(); it++)
    {
        sum += it->second;
    }

    double r = ((double) rand() / (RAND_MAX)) * sum;
    double accu_sum = 0.0;
    it = prob_map.begin();
    size_t index = it->first;
    for (; it != prob_map.end(); it++)
    {
        accu_sum += it->second;
        if (accu_sum >= r)
        {
            index = it->first;
            break;
        }
    }
    return index;
}

Louvain::Louvain() : max_community_(std::numeric_limits<size_t>::max()), temperature_(10.0)
{

}

Louvain::Louvain(std::vector<size_t> const & nodes,
                 std::unordered_map<size_t, std::unordered_map<size_t, double> > const & edges)
    : max_community_(std::numeric_limits<size_t>::max()), temperature_(10.0)
{
    Initialize(nodes, edges);
}

Louvain::Louvain(std::vector<size_t> const & nodes,
                 std::vector<std::pair<size_t, size_t> > const & edges)
    : max_community_(std::numeric_limits<size_t>::max()), temperature_(10.0)
{
    Initialize(nodes, edges);
}

Louvain::~Louvain()
{

}

void Louvain::Initialize(std::vector<size_t> const & nodes,
                         std::vector<std::pair<size_t, size_t> > const & edges)
{
    size_t node_num = nodes.size();
    graph_.Resize(node_num);
    node_graph_.Resize(node_num);
    node_map_.resize(node_num);
    community_map_.resize(node_num);
    community_size_.resize(node_num);
    community_in_weight_.resize(node_num);
    community_total_weight_.resize(node_num);

    std::unordered_map<size_t, size_t> node_index_map;
    for (size_t i = 0; i < node_num; i++)
    {
        node_index_map[nodes[i]] = i;
        double node_weight = 0.0;
        graph_.AddSelfEdge(node_weight);
        node_graph_.AddSelfEdge(node_weight);
        node_map_[i] = i;
        community_map_[i] = i;
        community_size_[i] = 1;
        community_in_weight_[i] = node_weight * 2;
        community_total_weight_[i] = node_weight * 2;
    }

    sum_edge_weight_ = 0.0;
    for (size_t i = 0; i < edges.size(); i++)
    {
        std::pair<size_t, size_t> const & edge = edges[i];
        size_t index1 = edge.first;
        size_t index2 = edge.second;
        assert(node_index_map.find(index1) != node_index_map.end());
        assert(node_index_map.find(index2) != node_index_map.end());
        size_t node_index1 = node_index_map[index1];
        size_t node_index2 = node_index_map[index2];
        if (node_index1 > node_index2)  continue;
        double weight = 1.0;
        graph_.AddUndirectedEdge(node_index1, node_index2, weight);
        node_graph_.AddUndirectedEdge(node_index1, node_index2, weight);

        sum_edge_weight_ += 2 * weight;
        community_total_weight_[node_index1] += weight;
        community_total_weight_[node_index2] += weight;
    }

    double avg_weight = sum_edge_weight_ / edges.size();
    for (size_t i = 0; i < node_num; i++)
    {
        node_graph_.AddSelfEdge(i, avg_weight);
        graph_.AddSelfEdge(i, avg_weight);
    }

    std::cout << "------------------ Build graph --------------------\n"
              << "# nodes: " << graph_.NodeNum() << ", # edges: " << edges.size() << "\n"
              << "---------------------------------------------------\n";
}

void Louvain::Initialize(std::vector<size_t> const & nodes,
                         std::unordered_map<size_t, std::unordered_map<size_t, double> > const & edges)
{
    size_t node_num = nodes.size();
    graph_.Resize(node_num);
    node_graph_.Resize(node_num);
    node_map_.resize(node_num);
    community_map_.resize(node_num);
    community_size_.resize(node_num);
    community_in_weight_.resize(node_num);
    community_total_weight_.resize(node_num);

    std::unordered_map<size_t, size_t> node_index_map;
    for (size_t i = 0; i < node_num; i++)
    {
        node_index_map[nodes[i]] = i;
        double node_weight = 0.0;
        graph_.AddSelfEdge(node_weight);
        node_graph_.AddSelfEdge(node_weight);
        node_map_[i] = i;
        community_map_[i] = i;
        community_size_[i] = 1;
        community_in_weight_[i] = node_weight * 2;
        community_total_weight_[i] = node_weight * 2;
    }

    sum_edge_weight_ = 0.0;
    size_t edge_num = 0;
    std::unordered_map<size_t, std::unordered_map<size_t, double> >::const_iterator it1 = edges.begin();
    for (; it1 != edges.end(); it1++)
    {
        size_t index1 = it1->first;
        assert(node_index_map.find(index1) != node_index_map.end());
        size_t node_index1 = node_index_map[index1];
        std::unordered_map<size_t, double> const & submap = it1->second;
        std::unordered_map<size_t, double>::const_iterator it2 = submap.begin();
        for (; it2 != submap.end(); it2++)
        {
            size_t index2 = it2->first;
            if (index1 == index2)   continue;
            edge_num++;
           assert(node_index_map.find(index2) != node_index_map.end());
           size_t node_index2 = node_index_map[index2];
           if (node_index1 > node_index2)   continue;
            double weight = it2->second;

            graph_.AddUndirectedEdge(node_index1, node_index2, weight);
            node_graph_.AddUndirectedEdge(node_index1, node_index2, weight);

            sum_edge_weight_ += 2 * weight;
            community_total_weight_[node_index1] += weight;
            community_total_weight_[node_index2] += weight;
        }
    }

    double avg_weight = sum_edge_weight_ / edge_num;
    for (size_t i = 0; i < node_num; i++)
    {
        node_graph_.AddSelfEdge(i, avg_weight);
        graph_.AddSelfEdge(i, avg_weight);
    }

    std::cout << "------------------ Build graph --------------------\n"
              << "# nodes: " << nodes.size() << ", # edges: " << edge_num << "\n"
              << "---------------------------------------------------\n";
}

void Louvain::Reinitialize()
{
    graph_ = node_graph_;

    size_t node_num = node_graph_.NodeNum();
    node_map_.resize(node_num);
    community_map_.resize(node_num);
    community_size_.resize(node_num);
    community_in_weight_.resize(node_num);
    community_total_weight_.resize(node_num);
    for (size_t i = 0; i < node_num; i++)
    {
        double node_weight = 0.0;
        node_map_[i] = i;
        community_map_[i] = i;
        community_size_[i] = 1;
        community_in_weight_[i] = node_weight * 2;
        community_total_weight_[i] = node_weight * 2;
    }
}

double Louvain::Modularity(size_t const community) const
{
    assert(community < community_in_weight_.size());
    double in_weight = community_in_weight_[community];
    double total_weight = community_total_weight_[community];
    double modularity = in_weight / sum_edge_weight_ - std::pow(total_weight / sum_edge_weight_, 2);
    return modularity;
}

double Louvain::Modularity() const
{
    double modularity = 0.0;
    size_t num_communities = community_in_weight_.size();
    for (size_t i = 0; i < num_communities; i++)
    {
        double in_weight = community_in_weight_[i];
        double total_weight = community_total_weight_[i];
        modularity += in_weight / sum_edge_weight_ - std::pow(total_weight / sum_edge_weight_, 2);
    }
    return modularity;
}

double Louvain::EdgeWeight(size_t const index1, size_t const index2) const
{
    return node_graph_.EdgeWeight(index1, index2);
}

void Louvain::GetClusters(std::vector<std::vector<size_t> > & clusters) const
{
    clusters.clear();
    size_t num_communities = community_in_weight_.size();
    size_t num_nodes = node_map_.size();
    clusters.resize(num_communities, std::vector<size_t>());
    for (size_t i = 0; i < num_nodes; i++)
    {
        size_t community_index = node_map_[i];
        assert(community_index < num_communities && "[GetClusters] Community index out of range");
        clusters[community_index].push_back(i);
    }
}

void Louvain::GetEdgesAcrossClusters(std::vector<std::pair<size_t, size_t> > & pairs) const
{
    pairs.clear();

    size_t node_num = node_graph_.NodeNum();
    for (size_t i = 0; i < node_num; i++)
    {
        size_t node_index1 = i;
        size_t community_index1 = node_map_[node_index1];
        std::vector<EdgeData> const & edges = node_graph_.GetIncidentEdges(i);
        for (size_t j = 0; j < edges.size(); j++)
        {
            EdgeData const & edge = edges[j];
            size_t node_index2 = edge.node;
            size_t community_index2 = node_map_[node_index2];
            if (node_index1 > node_index2)
                continue;
            if (community_index1 != community_index2)
            {
                pairs.push_back(std::make_pair(node_index1, node_index2));
            }
        }
    }
}

void Louvain::Print()
{
    std::cout << "------------------------ Print ------------------------\n";
    std::cout << "# communities = " << community_in_weight_.size() << ", modularity = " << Modularity() << "\n";

    std::vector<std::vector<size_t> > clusters;
    GetClusters(clusters);
    for (size_t i = 0; i < clusters.size(); i++)
    {
        std::vector<size_t> const & nodes = clusters[i];
        std::cout << "Community " << i << " of size " << nodes.size() << ": ";
        for (size_t j = 0; j < nodes.size(); j++)
        {
            size_t node_index = nodes[j];
             std::cout << node_index << " ";
        }
        std::cout << "\n";
    }
    std::cout << "-------------------------------------------------------\n";
}

void Louvain::Cluster()
{
    size_t pass = 0;

    while(Merge())
    {
        Rebuild();
        pass++;
    }
}

/*!
 * @param initial_pairs - The end points of initial pairs must be in the same cluster.
 */
void Louvain::Cluster(std::vector<std::pair<size_t, size_t> > const & initial_pairs)
{
    size_t pass = 0;

    Merge(initial_pairs);
    Rebuild();

    while(Merge())
    {
        Rebuild();
        pass++;
    }
}

/*!
 * @brief StochasticCluster introduces stochasticity to clustering by merging clusters with some probability
 * rather than greedily like louvain's algorithm.
 */
void Louvain::StochasticCluster()
{
    size_t pass = 0;

    std::srand(unsigned(std::time(0)));
    while (StochasticMerge())
    {
        Rebuild();
        pass++;
    }
}

void Louvain::StochasticCluster(std::vector<std::pair<size_t, size_t> > const & initial_pairs)
{
    size_t pass = 0;

    Merge(initial_pairs);
    Rebuild();

    std::srand(unsigned(std::time(0)));
    while (StochasticMerge())
    {
        Rebuild();
        pass++;
    }
}

bool Louvain::Merge()
{
    bool improved = false;
    std::queue<size_t> queue;

    size_t node_num = graph_.NodeNum();
    std::unordered_map<size_t, bool> visited;
    for (size_t i = 0; i < node_num; i++)
    {
        size_t community_index = i;
        if (community_size_[community_index] < max_community_)
        {
            queue.push(community_index);
            visited[community_index] = false;
        }
        else
        {
            visited[community_index] = true;
        }
    }

     std::vector<size_t> prev_community_size = community_size_;

    size_t loop_count = 0;
    while(!queue.empty())
    {
        size_t node_index = queue.front();
        queue.pop();
        visited[node_index] = true;

        double self_weight = graph_.GetSelfWeight(node_index);
        double total_weight = self_weight;

        std::vector<EdgeData> const & edges = graph_.GetIncidentEdges(node_index);
        std::unordered_map<size_t, double> neighb_weights;
        for (size_t i = 0; i < edges.size(); i++)
        {
            EdgeData const & edge = edges[i];
            size_t neighb_index = edge.node;
            size_t neighb_community_index = community_map_[neighb_index];
            neighb_weights[neighb_community_index] += edge.weight;
            total_weight += edge.weight;
        }

        size_t prev_community = community_map_[node_index];
        double prev_neighb_weight = neighb_weights[prev_community];
        community_map_[node_index] = -1;
        community_size_[prev_community] -= prev_community_size[node_index];
        community_in_weight_[prev_community] -= 2 * prev_neighb_weight + self_weight;
        community_total_weight_[prev_community] -= total_weight;

        double max_inc = 0.0;
        size_t best_community = prev_community;
        double best_neighb_weight = prev_neighb_weight;
        std::unordered_map<size_t, double>::const_iterator it = neighb_weights.begin();
        for (; it != neighb_weights.end(); it++)
        {
            size_t neighb_community_index = it->first;
            if (community_size_[neighb_community_index] >= max_community_)
                continue;
            double neighb_weight = it->second;
            double neighb_community_total_weight = community_total_weight_[neighb_community_index];
            double inc = (neighb_weight - (neighb_community_total_weight * total_weight) / sum_edge_weight_) / sum_edge_weight_ * 2;

            if (inc > max_inc)
            {
                max_inc = inc;
                best_community = neighb_community_index;
                best_neighb_weight = neighb_weight;
            }
        }

        community_map_[node_index] = best_community;
        community_size_[best_community] += prev_community_size[node_index];
        community_in_weight_[best_community] += 2 * best_neighb_weight  + self_weight;
        community_total_weight_[best_community] += total_weight;

        if (best_community != prev_community)
        {
            for (size_t i = 0; i < edges.size(); i++)
            {
                EdgeData const & edge = edges[i];
                size_t neighb_index = edge.node;
                size_t neighb_community_index = community_map_[neighb_index];
                if (visited[neighb_index] && community_size_[neighb_community_index] < max_community_)
                {
                    queue.push(neighb_index);
                    visited[neighb_index] = false;
                }
            }
            improved = true;
        }

        if (++loop_count > 3 * node_num)
            break;
    }
    return improved;
}

bool Louvain::Merge(std::vector<std::pair<size_t, size_t> > const & initial_pairs)
{
    size_t node_num = graph_.NodeNum();

    std::vector<double> weights;
    weights.reserve(initial_pairs.size());
    for (size_t i = 0; i < initial_pairs.size(); i++)
    {
        size_t node_index1 = initial_pairs[i].first;
        size_t node_index2 = initial_pairs[i].second;
        double weight = graph_.EdgeWeight(node_index1, node_index2);
        weights.push_back(weight);
    }
    std::vector<size_t> pair_indexes = SortIndexes(weights, false);

    UnionFind union_find;
    union_find.InitSets(node_num);

    std::vector<size_t> node_visit_times;
    node_visit_times.resize(node_num, 0);
    for (size_t i = 0; i < pair_indexes.size(); i++)
    {
        size_t pair_index = pair_indexes[i];
        size_t node_index1 = initial_pairs[pair_index].first;
        size_t node_index2 = initial_pairs[pair_index].second;
        if (node_visit_times[node_index1] < 1 && node_visit_times[node_index2] < 1)
        {
            union_find.Union(node_index1, node_index2);
            node_visit_times[node_index1]++;
            node_visit_times[node_index2]++;
        }
    }

    for (size_t i = 0; i < node_num; i++)
    {
        size_t node_index = i;
        community_in_weight_[node_index] = 0.0;
        community_total_weight_[node_index] = 0.0;
        community_size_[node_index] = 0;
    }
    for (size_t i = 0; i < node_num; i++)
    {
        size_t node_index = i;
        size_t community_index = union_find.Find(node_index);
        community_map_[node_index] = community_index;
        community_in_weight_[community_index] += graph_.GetSelfWeight(node_index);
        community_total_weight_[community_index] += graph_.GetSelfWeight(node_index);
        community_size_[community_index] += 1;
    }

    for (size_t i = 0; i < node_num; i++)
    {
        size_t node_index1 = i;
        size_t community_index1 = community_map_[node_index1];
        std::vector<EdgeData> const & edges = graph_.GetIncidentEdges(node_index1);
        for (size_t j = 0; j < edges.size(); j++)
        {
            EdgeData const & edge = edges[j];
            size_t node_index2 = edge.node;
            if (node_index1 > node_index2) continue;
            size_t community_index2 = community_map_[node_index2];
            double weight = edge.weight;
            if (community_index1 == community_index2)
            {
                community_in_weight_[community_index1] += 2 * weight;
                community_total_weight_[community_index1] += 2 * weight;
            }
            else
            {
                community_total_weight_[community_index1] += weight;
                community_total_weight_[community_index2] += weight;
            }
        }
    }

    return true;
}

bool Louvain::StochasticMerge()
{
    bool improve = false;
    std::queue<size_t> queue;

    size_t community_num = graph_.NodeNum();
    for (size_t i = 0; i < community_num; i++)
    {
        size_t community_index = i;
        if (community_size_[community_index] < max_community_)
        {
            queue.push(community_index);
        }
    }

    size_t node_num = node_graph_.NodeNum();
    size_t node_edge_num = node_graph_.EdgeNum();
    size_t community_edge_num = graph_.EdgeNum();
    double factor = (community_num + community_edge_num) / double(node_num + node_edge_num);

    std::vector<size_t> prev_community_size = community_size_;

    while(!queue.empty())
    {
        size_t node_index = queue.front();
        queue.pop();

        double self_weight = graph_.GetSelfWeight(node_index);
        double total_weight = self_weight;

        std::unordered_map<size_t, double> neighb_weights;  // the weight of edge to neighboring community
        std::vector<EdgeData> const & edges = graph_.GetIncidentEdges(node_index);
        for (size_t i = 0; i < edges.size(); i++)
        {
            EdgeData const & edge = edges[i];
            size_t neighb_index = edge.node;
            size_t neighb_community_index = community_map_[neighb_index];
            neighb_weights[neighb_community_index] += edge.weight;
            total_weight += edge.weight;
        }

        size_t prev_community = community_map_[node_index];
        double prev_neighb_weight = neighb_weights[prev_community];
        community_map_[node_index] = -1;
        community_size_[prev_community] -= prev_community_size[node_index];
        community_in_weight_[prev_community] -= 2 * prev_neighb_weight + self_weight;
        community_total_weight_[prev_community] -= total_weight;

        std::unordered_map<size_t, double> prob_map;
        std::unordered_map<size_t, double>::const_iterator it = neighb_weights.begin();
        for (; it != neighb_weights.end(); it++)
        {
            size_t neighb_community_index = it->first;
            size_t neighb_community_size = community_size_[neighb_community_index];
            if (prev_community_size[node_index] + neighb_community_size >= max_community_)
                continue;
            double neighb_weight = it->second;
            double neighb_community_total_weight = community_total_weight_[neighb_community_index];
            double inc = factor * (neighb_weight - (neighb_community_total_weight * total_weight) / sum_edge_weight_);

            double prob = std::exp(temperature_ * inc);
            prob_map[neighb_community_index] = prob;
        }
        assert(!prob_map.empty());

        size_t best_community = RandomPick(prob_map);
        double best_neighb_weight = neighb_weights[best_community];
        if (best_community != prev_community)
            improve = true;

        community_map_[node_index] = best_community;
        community_size_[best_community] += prev_community_size[node_index];
        community_in_weight_[best_community] += 2 * best_neighb_weight  + self_weight;
        community_total_weight_[best_community] += total_weight;
    }
    return improve;
}

void Louvain::RearrangeCommunities()
{
    std::unordered_map<size_t, size_t> renumbers;   // map from old cluster index to organized cluster index

    size_t num = 0;
    for (size_t i = 0; i < community_map_.size(); i++)
    {
        std::unordered_map<size_t, size_t>::const_iterator it = renumbers.find(community_map_[i]);
        if (it == renumbers.end())
        {
            renumbers[community_map_[i]] = num;
            community_map_[i] = num;
            num++;
        }
        else
        {
            community_map_[i] = it->second;
        }
    }

    for (size_t i = 0; i < node_map_.size(); i++)
    {
        node_map_[i] = community_map_[node_map_[i]];
    }

    std::vector<size_t> community_size_new(num);
    std::vector<double> community_in_weight_new(num);
    std::vector<double> community_total_weight_new(num);
    for (size_t i = 0; i < community_in_weight_.size(); i++)
    {
         std::unordered_map<size_t, size_t>::const_iterator it = renumbers.find(i);
         if (it != renumbers.end())
         {
             size_t new_community_index = it->second;
             community_size_new[new_community_index] = community_size_[i];
             community_in_weight_new[new_community_index] = community_in_weight_[i];
             community_total_weight_new[new_community_index] = community_total_weight_[i];
         }
    }
    community_size_new.swap(community_size_);
    community_in_weight_new.swap(community_in_weight_);
    community_total_weight_new.swap(community_total_weight_);
}

void Louvain::Rebuild()
{
    RearrangeCommunities();

    size_t num_communities = community_in_weight_.size();
    std::vector<std::vector<size_t>> community_nodes(num_communities);
    for (size_t i = 0; i < graph_.NodeNum(); i++)
    {
        community_nodes[community_map_[i]].push_back(i);
    }

    Graph graph_new;
    graph_new.Resize(num_communities);
    for (size_t i = 0; i < num_communities; i++)
    {
        std::vector<size_t> const & nodes = community_nodes[i];
        double self_weight = 0.0;
        std::unordered_map<size_t, double> edges_new;
        for (size_t j = 0; j < nodes.size(); j++)
        {
            size_t node_index = nodes[j];
            self_weight += graph_.GetSelfWeight(node_index);
            std::vector<EdgeData> const & edges = graph_.GetIncidentEdges(node_index);
            for (size_t k = 0; k < edges.size(); k++)
            {
                EdgeData const & edge = edges[k];
                edges_new[community_map_[edge.node]] += edge.weight;
            }
        }
        self_weight += edges_new[i];
        graph_new.AddSelfEdge(i, self_weight);

         std::unordered_map<size_t, double>::const_iterator it = edges_new.begin();
         for (; it != edges_new.end(); it++)
         {
             size_t neighb_community_index = it->first;
             double weight = it->second;
             if (i != neighb_community_index)
             {;
                 graph_new.AddDirectedEdge(i, neighb_community_index, weight);
             }
         }
    }

    graph_.Swap(graph_new);

    community_map_.resize(num_communities);
    for (size_t i = 0; i < num_communities; i++)
    {
        community_map_[i] = i;
    }
}
