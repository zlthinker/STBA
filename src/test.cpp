#include "STBA/test.h"
#include "STBA/lmbaproblem.h"
#include "STBA/stochasticbaproblem.h"


void SynthesizeBundleBlock(BundleBlock & bundle_block)
{
    double focal = 1000;
    double u = 2000;
    double v = 1500;
    Vec3 distortion = {0.0, 0.0, 0.0};
    Vec3 aa = {0.0001, 0.0001, 0.0001};
    size_t group_id = 0;
    Vec6 intrinsic;
    intrinsic << focal, u, v, distortion[0], distortion[1], distortion[2];
    BundleBlock::DGroup group(group_id, intrinsic);
    bundle_block.InsertGroup(group);
    const size_t camera_num = 6;
    const size_t track_num = 20;

    for (size_t i = 0; i < camera_num; i++)
    {
        Vec3 trans = {i * 1.0, 0.0, 0.0};
        BundleBlock::DCamera camera(i, group_id, aa, trans);
        bundle_block.InsertCamera(camera);
    }

    size_t proj_id = 0;
    for (size_t i = 0; i < track_num; i++)
    {
        Vec3 point = {i * 0.2, 0.0, 10.0};
        BundleBlock::DTrack track(i, point);
        bundle_block.InsertTrack(track);

        for (size_t j = 0; j < camera_num; j++)
        {
            BundleBlock::DCamera const & camera = bundle_block.GetCamera(j);
            Vec2 proj;
            Project(focal, u, v, camera.axis_angle, camera.translation, point, distortion, proj);
            proj[0] += 5.0;
            proj[1] += 5.0;
            BundleBlock::DProjection projection(proj_id++, j, i, proj);
            bundle_block.InsertProjection(projection);
        }
    }
    for (size_t i = 0; i < camera_num; i++)
    {
        BundleBlock::DCamera & camera = bundle_block.GetCamera(i);
        for (size_t j = 0; j < camera_num; j++)
        {
            if (i != j)
            {
                camera.linked_cameras.insert(j);
            }
        }
    }
}

void TestLM()
{
    BundleBlock bundle_block;
    SynthesizeBundleBlock(bundle_block);

    LMBAProblem problem(1, 1e4, static_cast<LossType>(0));
    problem.SetIntrinsicFixed(true);
    problem.Initialize(bundle_block);
    problem.Solve();
    problem.Update(bundle_block);

    std::vector<size_t> group_indexes = bundle_block.GroupIndexes();
    std::vector<size_t> camera_indexes = bundle_block.CameraIndexes();
    std::vector<size_t> track_indexes = bundle_block.TrackIndexes();

    for (size_t i = 0; i < group_indexes.size(); i++)
    {
        bundle_block.GetGroup(group_indexes[i]).Print();
    }
    for (size_t i = 0; i < camera_indexes.size(); i++)
    {
        bundle_block.GetCamera(camera_indexes[i]).Print();
    }
    for (size_t i = 0; i < track_indexes.size(); i++)
    {
        bundle_block.GetTrack(track_indexes[i]).Print();
    }
}

void TestSTBA()
{
    BundleBlock bundle_block;
    SynthesizeBundleBlock(bundle_block);

    size_t max_cluster = 100;
    StochasticBAProblem problem(1, 1e4, static_cast<LossType>(0), max_cluster, 4);
    problem.Initialize(bundle_block);
    problem.Solve();
    problem.Update(bundle_block);

    std::vector<size_t> group_indexes = bundle_block.GroupIndexes();
    std::vector<size_t> camera_indexes = bundle_block.CameraIndexes();
    std::vector<size_t> track_indexes = bundle_block.TrackIndexes();

    for (size_t i = 0; i < group_indexes.size(); i++)
    {
        bundle_block.GetGroup(group_indexes[i]).Print();
    }
    for (size_t i = 0; i < camera_indexes.size(); i++)
    {
        bundle_block.GetCamera(camera_indexes[i]).Print();
    }
    for (size_t i = 0; i < track_indexes.size(); i++)
    {
        bundle_block.GetTrack(track_indexes[i]).Print();
    }
}
