#include "datablock.h"

#include <fstream>

std::vector<size_t> BundleBlock::GroupIndexes() const
{
    std::vector<size_t> indexes;
    indexes.reserve(groups_.size());
    std::unordered_map<size_t, DGroup>::const_iterator it = groups_.begin();
    for (; it != groups_.end(); it++)
    {
        indexes.push_back(it->first);
    }
    return indexes;
}

std::vector<size_t> BundleBlock::CameraIndexes() const
{
    std::vector<size_t> indexes;
    indexes.reserve(cameras_.size());
    std::unordered_map<size_t, DCamera>::const_iterator it = cameras_.begin();
    for (; it != cameras_.end(); it++)
    {
        indexes.push_back(it->first);
    }
    return indexes;
}

std::vector<size_t> BundleBlock::TrackIndexes() const
{
    std::vector<size_t> indexes;
    indexes.reserve(tracks_.size());
    std::unordered_map<size_t, DTrack>::const_iterator it = tracks_.begin();
    for (; it != tracks_.end(); it++)
    {
        indexes.push_back(it->first);
    }
    return indexes;
}

std::vector<size_t> BundleBlock::ProjectionIndexes() const
{
    std::vector<size_t> indexes;
    indexes.reserve(projections_.size());
    std::unordered_map<size_t, DProjection>::const_iterator it = projections_.begin();
    for (; it != projections_.end(); it++)
    {
        indexes.push_back(it->first);
    }
    return indexes;
}

BundleBlock::DGroup const & BundleBlock::GetGroup(size_t id) const
{
    std::unordered_map<size_t, DGroup>::const_iterator it = groups_.find(id);
    assert(it != groups_.end() && "[BundleBlock::GetGroup] Group id not found");
    return it->second;
}

BundleBlock::DGroup & BundleBlock::GetGroup(size_t id)
{
    std::unordered_map<size_t, DGroup>::iterator it = groups_.find(id);
    assert(it != groups_.end() && "[BundleBlock::GetGroup] Group id not found");
    return it->second;
}

BundleBlock::DCamera const & BundleBlock::GetCamera(size_t id) const
{
    std::unordered_map<size_t, DCamera>::const_iterator it = cameras_.find(id);
    assert(it != cameras_.end() && "[BundleBlock::GetCamera] Camera id not found");
    return it->second;
}

BundleBlock::DCamera & BundleBlock::GetCamera(size_t id)
{
    std::unordered_map<size_t, DCamera>::iterator it = cameras_.find(id);
    assert(it != cameras_.end() && "[BundleBlock::GetCamera] Camera id not found");
    return it->second;
}

BundleBlock::DTrack const & BundleBlock::GetTrack(size_t id) const
{
    std::unordered_map<size_t, DTrack>::const_iterator it = tracks_.find(id);
    assert(it != tracks_.end() && "[BundleBlock::GetTrack] Track id not found");
    return it->second;
}

BundleBlock::DTrack & BundleBlock::GetTrack(size_t id)
{
    std::unordered_map<size_t, DTrack>::iterator it = tracks_.find(id);
    assert(it != tracks_.end() && "[BundleBlock::GetTrack] Track id not found");
    return it->second;
}

BundleBlock::DProjection const & BundleBlock::GetProjection(size_t id) const
{
    std::unordered_map<size_t, DProjection>::const_iterator it = projections_.find(id);
    assert(it != projections_.end() && "[BundleBlock::GetProjection] Projection id not found");
    return it->second;
}

BundleBlock::DProjection & BundleBlock::GetProjection(size_t id)
{
    std::unordered_map<size_t, DProjection>::iterator it = projections_.find(id);
    assert(it != projections_.end() && "[BundleBlock::GetProjection] Projection id not found");
    return it->second;
}

void BundleBlock::GetCommonPoints(std::unordered_map<size_t, std::unordered_map<size_t, std::vector<size_t> > > & common_point_map) const
{
    std::vector<size_t> camera_indexes = CameraIndexes();
    for (size_t i = 0; i < camera_indexes.size(); i++)
    {
        size_t camera_index = camera_indexes[i];
        DCamera const & camera = GetCamera(camera_index);

        std::unordered_map<size_t, std::vector<size_t> > local_map;
        local_map[camera_index] = std::vector<size_t>();
        std::unordered_set<size_t> const & link_camera_indexes = camera.linked_cameras;
        std::unordered_set<size_t>::const_iterator it = link_camera_indexes.begin();
        for (; it != link_camera_indexes.end(); it++)
        {
            size_t link_camera_index = *it;
            local_map[link_camera_index] = std::vector<size_t>();
        }
        common_point_map[camera_index] = local_map;
    }

    std::vector<size_t> track_indexes = TrackIndexes();
    for (size_t i = 0; i < track_indexes.size(); i++)
    {
        size_t track_index = track_indexes[i];
        DTrack const & track = GetTrack(track_index);

        std::unordered_set<size_t> const & proj_indexes = track.linked_projections;
        std::vector<size_t> local_camera_indexes;
        std::unordered_set<size_t>::const_iterator it = proj_indexes.begin();
        for (; it != proj_indexes.end(); it++)
        {
            size_t proj_index = *it;
            DProjection const & projection = GetProjection(proj_index);
            local_camera_indexes.push_back(projection.camera_id);
        }
        for (size_t j = 0; j < local_camera_indexes.size(); j++)
        {
            size_t camera_index1 = local_camera_indexes[j];
            std::unordered_map<size_t, std::unordered_map<size_t, std::vector<size_t> > >::iterator it1;
            it1 = common_point_map.find(camera_index1);
            assert(it1 != common_point_map.end() && "[BundleBlock::GetCommonPoints] Camera index1 not found");
            std::unordered_map<size_t, std::vector<size_t> > & local_map = it1->second;
            for (size_t k = 0; k < local_camera_indexes.size(); k++)
            {
                size_t camera_index2 = local_camera_indexes[k];
                std::unordered_map<size_t, std::vector<size_t> >::iterator it2 = local_map.find(camera_index2);
                assert(it2 != local_map.end() && "[BundleBlock::GetCommonPoints] Camera index2 not found");
                std::vector<size_t> & common_points = it2->second;
                common_points.push_back(track_index);
            }
        }
    }
}

bool BundleBlock::LoadColmapTxt(std::string const & cameras_path, std::string const & images_path, std::string const & points_path)
{
    // Read camera intrinsics
    std::cout << "[BundleBlock::LoadColmapTxt] Load intrinsics: " << cameras_path << "\n";
    {
        std::ifstream cameras_file_stream(cameras_path, std::ios::in);
        if (!cameras_file_stream)
            return false;

        while (!cameras_file_stream.eof() && !cameras_file_stream.bad())
        {
            std::string line;
            std::getline(cameras_file_stream, line);
            if (line.size() == 0 || line[0] == '#')
            {
                continue;
            }

            std::istringstream line_stream(line);
            size_t cam_idx;
            std::string model_name;
            int width, height;
            double fx, fy, u, v;
            double k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0, k4 = 0, sx1 = 0, sy1 = 0;

            line_stream >> cam_idx >> model_name >> width >> height;
            if (model_name != "PINHOLE"
                    && model_name != "THIN_PRISM_FISHEYE"
                    && model_name != "SIMPLE_RADIAL"
                    && model_name != "OPENCV")
            {
                std::cout << "Cannot deal with camera models other than PINHOLE and THIN_PRISM_FISHEYE."
                          << std::endl;
                return false;
            }
            if (model_name == "PINHOLE")
            {
                line_stream >> fx >> fy >> u >> v;
            }
            else if (model_name == "THIN_PRISM_FISHEYE")
            {
                line_stream >> fx >> fy >> u >> v;
                line_stream >> k1 >> k2 >> p1 >> p2 >> k3 >> k4 >> sx1 >> sy1;
            }
            else if (model_name == "SIMPLE_RADIAL")
            {
                line_stream >> fx >> u >> v >> k1;
                fy = fx;
            }
            else if (model_name == "OPENCV")
            {
                line_stream >>  fx >> fy >> u >> v >> k1 >> k2 >> p1 >> p2;
            }

            Vec6 intrinsic;
            intrinsic << fx, u, v, k1, k2, k3;
            groups_[cam_idx] = DGroup(cam_idx, intrinsic, width, height);
        }
        cameras_file_stream.close();
    }

    // Read camera extrinsics
    std::cout << "[BundleBlock::LoadColmapTxt] Load extrinsics: " << images_path << "\n";
    {
        std::ifstream images_file_stream(images_path, std::ios::in);
        if (!images_file_stream)
        {
            return false;
        }

        size_t projection_id = 0;
        while (!images_file_stream.eof() && !images_file_stream.bad())
        {
            std::string line;
            std::getline(images_file_stream, line);
            if (line.size() == 0 || line[0] == '#')
            {
                continue;
            }

            // Read image info line.
            size_t image_idx, camera_idx;
            double qw, qx, qy, qz, tx, ty, tz;
            std::string image_path;

            std::istringstream image_stream(line);
            image_stream >> image_idx >> qw >> qx >> qy >> qz
                    >> tx >> ty >> tz >> camera_idx >> image_path;

            Vec3 angle_axis = Quaternion2AngleAxis(Vec4(qw, qx, qy, qz));

            DCamera camera(image_idx, camera_idx, angle_axis, Vec3(tx, ty, tz), image_path);

            // read projections
            std::unordered_set<size_t> track_ids;
            std::getline(images_file_stream, line);
            std::istringstream observations_stream(line);
            while (!observations_stream.eof() && !observations_stream.bad())
            {
                double px, py;
                int track_idx;
                observations_stream >> px >> py >> track_idx;

                if (track_idx == -1 || track_ids.find(track_idx) != track_ids.end()) continue;

                // init track
                if (tracks_.find(track_idx) == tracks_.end())
                {
                    tracks_[track_idx] = DTrack(track_idx);
                }

                // add projection
                DProjection projection(projection_id, image_idx, track_idx, Vec2(px, py));
                projections_[projection_id] = projection;
                camera.linked_projections.insert(projection_id);
                DTrack & track = tracks_[track_idx];
                track.linked_projections.insert(projection_id);
                projection_id++;
                cameras_[image_idx] = camera;
            }
        }
    }

    // Read points
    std::cout << "[BundleBlock::LoadColmapTxt] Load points: " << points_path << "\n";
    {
        std::ifstream point_file_stream(points_path, std::ios::in);
        if (!point_file_stream)
        {
            return false;
        }
        while (!point_file_stream.eof() && !point_file_stream.bad())
        {
            std::string line;
            std::getline(point_file_stream, line);
            if (line.size() == 0 || line[0] == '#')
            {
                continue;
            }

            // Read image info line.
            size_t track_idx;
            double px, py, pz, cx, cy, cz;

            // set intrisic from intrisic_xms
            std::istringstream track_stream(line);
            track_stream >> track_idx >> px >> py >> pz >> cx >> cy >> cz;
            if (tracks_.find(track_idx) == tracks_.end())
            {
                std::cout << "cannot find the track " << track_idx << std::endl;
            }
            DTrack & track = tracks_[track_idx];
            track.position = Vec3(px, py, pz);
            track.color = Vec3(cx, cy, cz);
        }
    }

    // Build linked cameras
    {
        std::vector<size_t> track_indexes = TrackIndexes();
        for (size_t i = 0; i < track_indexes.size(); i++)
        {
            size_t track_index = track_indexes[i];
            DTrack const & track = GetTrack(track_index);
            std::unordered_set<size_t> const & linked_projections = track.linked_projections;
            std::unordered_set<size_t>::const_iterator it = linked_projections.begin();
            std::unordered_set<size_t> linked_cameras;
            for (; it != linked_projections.end(); it++)
            {
                size_t proj_index = *it;
                DProjection const & projection = GetProjection(proj_index);
                linked_cameras.insert(projection.camera_id);
            }
            std::unordered_set<size_t>::const_iterator it1 = linked_cameras.begin();
            for (; it1 != linked_cameras.end(); it1++)
            {
                size_t camera_index1 = *it1;
                DCamera & camera1 = GetCamera(camera_index1);
                std::unordered_set<size_t>::const_iterator it2 = linked_cameras.begin();
                for (; it2 != linked_cameras.end(); it2++)
                {
                    size_t camera_index2 = *it2;
                    if (camera_index1 != camera_index2)
                        camera1.linked_cameras.insert(camera_index2);
                }
            }
        }
    }

    Print();

    return true;
}

void BundleBlock::SaveColmapTxt(std::string const & cameras_path, std::string const & images_path, std::string const & points_path) const
{
    // save camera intrinsics
    {
        std::ofstream stream(cameras_path);
        std::unordered_map<size_t, DGroup>::const_iterator it = groups_.begin();
        for (; it != groups_.end(); it++)
        {
            DGroup const & group = it->second;
            /* some mis-alignment here.
             * Our intrinsic is parameterized by <focal u v k1 k2 k3>, while the OPENCV model is <fx fy u v k1 k2 p1 p2>
             */
            stream << group.id << " OPENCV " << group.width << " " << group.height << " " << group.intrinsic[0] << " " << group.intrinsic[0] << " "
                   << group.intrinsic[1] << " " << group.intrinsic[2] << " " << group.intrinsic[3] << " " << group.intrinsic[4] << " 0 0\n";
        }
        stream.close();
    }

    // save camera extrinsics
    {
        std::ofstream stream(images_path);
        std::unordered_map<size_t, DCamera>::const_iterator it = cameras_.begin();
        for (; it != cameras_.end(); it++)
        {
            DCamera const & camera = it->second;
            Vec4 quaternion = AngleAxis2Quaternion(camera.axis_angle);
            stream << camera.id << " ";
            for (size_t i = 0; i < 4; i++)  stream << quaternion[i] << " ";
            for (size_t i = 0; i < 3; i++) stream << camera.translation[i] << " ";
            stream << camera.group_id << " " << camera.image_path << "\n";
            std::unordered_set<size_t> const & linked_projections = camera.linked_projections;
            std::unordered_set<size_t>::const_iterator it = linked_projections.begin();
            for (; it != linked_projections.end(); it++)
            {
                size_t proj_index = *it;
                DProjection const & projection = GetProjection(proj_index);
                stream << projection.projection[0] << " " << projection.projection[1] << " " << projection.track_id << " ";
            }
            stream << "\n";
        }
        stream.close();
    }

    // save points
    {
        std::ofstream stream(points_path);
        std::unordered_map<size_t, DTrack>::const_iterator it = tracks_.begin();
        for (; it != tracks_.end(); it++)
        {
            DTrack const & track = it->second;
            /* some mis-alignment here.
             * The error and the track info are not saved here, since they are a bit redundant.
             */
            stream << track.id << " ";
            for (size_t i = 0; i < 3; i++)  stream << track.position[i] << " ";
            for (size_t i = 0; i < 3; i++)  stream << track.color[i] << " ";
            stream << "\n";
        }
        stream.close();
    }
}

void BundleBlock::Print() const
{
    std::cout << "[BundleBlock::Print] # groups = " << groups_.size() << "\n"
              << "# cameras = " << cameras_.size() << "\n"
              << "# tracks = " << tracks_.size() << "\n"
              << "# projections = " << projections_.size() << "\n";
}

void BundleBlock::AddGaussianNoiseToTrack(DT mean, DT sigma)
{
    std::cout << "[BundleBlock::AddGaussianNoiseToTrack] mean = " << mean << ", sigma = " << sigma << "\n";
    std::unordered_map<size_t, DTrack>::iterator it = tracks_.begin();
    for (; it != tracks_.end(); it++)
    {
        DTrack & track = it->second;
        for (size_t i = 0; i < 3; i++)
            track.position[i] += GaussianNoise(mean, sigma);
    }
}

void BundleBlock::AddGaussianNoiseToCameraTranslation(DT mean, DT sigma)
{
    std::cout << "[BundleBlock::AddGaussianNoiseToCameraTranslation] mean = " << mean << ", sigma = " << sigma << "\n";
    std::unordered_map<size_t, DCamera>::iterator it = cameras_.begin();
    for (; it != cameras_.end(); it++)
    {
        DCamera & camera = it->second;
        for (size_t i = 0; i < 3; i++)
            camera.translation[i] += GaussianNoise(mean, sigma);
    }
}
