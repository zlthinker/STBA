#include "clustering/louvain.h"
#include "stochasticbaproblem.h"
#include "lmbaproblem.h"
#include "dlbaproblem.h"


int main(int argc, char **argv)
{
    std::string cameras_path = argv[1];
    std::string images_path = argv[2];
    std::string points_path = argv[3];
    std::string output_folder = argv[4];
    std::cout << cameras_path << "\n"
              << images_path << "\n"
              << points_path << "\n"
              << output_folder << "\n";

    BundleBlock bundle_block;
    bundle_block.LoadColmapTxt(cameras_path, images_path, points_path);

    std::cout << "Before LMBAProblem.\n";
    LMBAProblem problem;
    problem.SetIntrinsicFixed(true);
    problem.SetMaxIteration(10);
    std::cout << "Before initialize.\n";
    problem.Initialize(bundle_block);
    std::cout << "Before solve.\n";
    problem.Solve();

    problem.Update(bundle_block);
    bundle_block.SaveColmapTxt(JoinPath(output_folder, "cameras.txt"), JoinPath(output_folder, "images.txt"), JoinPath(output_folder, "points.txt"));

    return 0;
}
