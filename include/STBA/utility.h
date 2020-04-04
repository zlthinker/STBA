#ifndef UTILITY_H
#define UTILITY_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unordered_map>
#include <numeric>

typedef double DT;
#define EPSILON double(1e-12)
#define MIN_DEPTH double(1e-12)
#define MAX_REPROJ_ERROR double(1e6)

using namespace Eigen;

typedef Eigen::Matrix<DT, 2, 1, Eigen::ColMajor> Vec2;
typedef Eigen::Matrix<DT, 3, 1, Eigen::ColMajor> Vec3;
typedef Eigen::Matrix<DT, 4, 1, Eigen::ColMajor> Vec4;
typedef Eigen::Matrix<DT, 6, 1, Eigen::ColMajor> Vec6;
typedef Eigen::Matrix<DT, Eigen::Dynamic, 1> VecX;
typedef Eigen::Matrix<DT, 2, 3, Eigen::RowMajor> Mat23;
typedef Eigen::Matrix<DT, 2, 6, Eigen::RowMajor> Mat26;
typedef Eigen::Matrix<DT, 2, 2, Eigen::RowMajor> Mat2;
typedef Eigen::Matrix<DT, 3, 3, Eigen::RowMajor> Mat3;
typedef Eigen::Matrix<DT, 6, 6, Eigen::RowMajor> Mat6;
typedef Eigen::Matrix<DT, 6, 3, Eigen::RowMajor> Mat63;
typedef Eigen::Matrix<DT, 9, 3, Eigen::RowMajor> Mat93;
typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatX;
typedef Eigen::SparseMatrix<DT> SMat;
typedef AngleAxis<DT> AxisAngle;

DT Determinant(Mat3 const & M);
Mat3 InverseMat(Mat3 const & mat);
Mat3 AngleAxis2Matrix(Vec3 const & angle_axis);
Vec3 Quaternion2AngleAxis(Vec4 const & quaternion);
Vec4 AngleAxis2Quaternion(Vec3 const & angle_axis);

Vec3 RotatePoint(Vec3 const & angle_axis, Vec3 const & point);

Mat26 Projection2IntrinsicJacobian(double const focal,
                                   Vec3 const & radial_distortion,
                                   Vec3 const & angle_axis,
                                   Vec3 const & translation,
                                   Vec3 const & global_point);

Mat23 Projection2RotationJacobian(double const focal,
                                  Vec3 const & radial_distortion,
                                  Vec3 const & angle_axis,
                                  Vec3 const & translation,
                                  Vec3 const & global_point);

Mat23 Projection2TranslationJacobian(double const focal,
                                     Vec3 const & radial_distortion,
                                     Vec3 const & angle_axis,
                                     Vec3 const & translation,
                                     Vec3 const & global_point);

Mat23 Projection2GlobalPointJacobian(double const focal,
                                     Vec3 const & radial_distortion,
                                     Vec3 const & angle_axis,
                                     Vec3 const & translation,
                                     Vec3 const & global_point);

void ProjectAndGradient(Vec3 const & rotation, Vec3 const & translation, Vec3 const & point,
                        double const focal, double const u, double const v,
                        Vec3 const & radial_distortion, Vec2 const & projection,
                        Mat23 & rotation_jacobian,
                        Mat23 & translation_jacobian,
                        Mat23 & point_jacobian,
                        Mat26 & intrinsic_jacobian);

bool Project(double const focal, double const u, double const v,
             Vec3 const & angle_axis, Vec3 const & translation, Vec3 const & global_point,
             Vec3 const & radial_distortion,
             Vec2 & reprojection);

double RandomNoise(double min, double max);

template <class T>
T const & RandomNoise(T & data, double min, double max)
{
    for (int i = 0; i < data.size(); i++)
    {
        *(data.data() + i) += DT(RandomNoise(min, max));
    }
    return data;
}


double GaussianNoise(double mean, double sigma);

template <class T>
bool IsNumericalValid(T const & data)
{
    for (int i = 0; i < data.size(); i++)
    {
        DT val = *(data.data() + i);
        if (std::isnan(val) || std::isinf(val))
            return false;
    }
    return true;
}

struct UnionFind
{
    // Represent the DS/UF forest thanks to two array:
    // A parent 'pointer tree' where each node holds a reference to its parent node
    std::vector<unsigned int> m_cc_parent;
    // A rank array used for union by rank
    std::vector<unsigned int> m_cc_rank;
    // A 'size array' to know the size of each connected component
    std::vector<unsigned int> m_cc_size;

    // Init the UF structure with num_cc nodes
    void InitSets(const unsigned int num_cc)
    {
        // all set size are 1 (independent nodes)
        m_cc_size.resize(num_cc, 1);
        // Parents id have their own CC id {0,n}
        m_cc_parent.resize(num_cc);
        for(size_t i = 0; i < num_cc; i++)
        {
            m_cc_parent[i] = i;
        }

        // Rank array (0)
        m_cc_rank.resize(num_cc, 0);
    }

    // Return the number of nodes that have been initialized in the UF tree
    unsigned int GetNumNodes() const
    {
        return m_cc_size.size();
    }

    // Return the representative set id of I nth component
    unsigned int Find(unsigned int i)
    {
        // Recursively set all branch as children of root (Path compression)
        if (m_cc_parent[i] != i)
            m_cc_parent[i] = Find(m_cc_parent[i]);
        return m_cc_parent[i];
    }

    // Replace sets containing I and J with their union
    void Union(unsigned int i, unsigned int j)
    {
        i = Find(i);
        j = Find(j);
        if(i==j)
        {
            // Already in the same set. Nothing to do
            return;
        }

        // x and y are not already in same set. Merge them.
        // Perform an union by rank:
        //  - always attach the smaller tree to the root of the larger tree
        if(m_cc_rank[i] < m_cc_rank[j])
        {
            m_cc_parent[i] = j;
            m_cc_size[j] += m_cc_size[i];
        }
        else if(m_cc_rank[i] > m_cc_rank[j])
        {
            m_cc_parent[j] = i;
            m_cc_size[i] += m_cc_size[j];
        }
        else {
            m_cc_parent[j] = i;
            m_cc_size[i] += m_cc_size[j];
            m_cc_rank[i]++;
        }
    }
};

template <typename T>
std::vector<size_t> SortIndexes(const std::vector<T> &v, bool increase = true)
{
    // initialize original index locations
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    if (increase)
        std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    else
        std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

    return idx;
}

bool ReadCameraGroup(std::string const & camera_group_file,
                     std::unordered_map<size_t, size_t> & camera_group_map);

bool ReadLinesFromFile(std::string const & file_path, std::vector<std::string> & lines);

std::string JoinPath(std::string const & folder, std::string const & file);

#endif // UTILITY_H
