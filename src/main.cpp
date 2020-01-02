#include "clustering/louvain.h"
#include "stochasticbaproblem.h"
#include "lmbaproblem.h"
#include "dlbaproblem.h"


int main(int argc, char **argv)
{
    std::string cameras_path = argv[1];
    std::string images_path = argv[2];
    std::string points_path = argv[3];
    std::cout << cameras_path << "\n"
              << images_path << "\n"
              << points_path << "\n";

    BundleBlock bundle_block;
    bundle_block.LoadColmapTxt(cameras_path, images_path, points_path);

    std::unordered_map<size_t, size_t> camera_group_map;
    std::vector<size_t> camera_indexes = bundle_block.CameraIndexes();
    for (size_t i = 0; i < camera_indexes.size(); i++)
    {
        camera_group_map[camera_indexes[i]] = 0;
    }

    std::cout << "Before LMBAProblem.\n";
    LMBAProblem problem;
    problem.SetIntrinsicFixed(true);
    std::cout << "Before initialize.\n";
    problem.Initialize(bundle_block, camera_group_map);
    std::cout << "Before solve.\n";
    problem.Solve();
}
