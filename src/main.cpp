#include "STBA/clustering/louvain.h"
#include "STBA/stochasticbaproblem.h"
#include "STBA/lmbaproblem.h"
#include "STBA/dlbaproblem.h"
#include "STBA/test.h"

void PrintHelp()
{
    std::cout << "<exe> <cameras.txt> <images.txt> <points.txt> <output_folder>\n"
              << "--iteration <val> : Set maximum iteration, default val = 100 \n"
              << "--cluster <val> : Set maximum cluster size, default val = 100 \n"
              << "--inner_step <val> : Set number of correction steps, default val = 4 \n"
              << "--thread_num <val> : Set thread number, default val = 1 \n"
              << "--radius <val> : Set intial radius of trust region, default val = 10000 \n"
              << "--loss <val> : Set loss type (0 - NULL, 1 - Huber, 2 - Cauchy), default val = 2 \n"
              << "--noise <val> : Set sigma of Gaussian noise, default val = 0.0 \n"
              << "--lm : Use Levenberg Marquardt \n"
              << "--dl : Use DogLeg \n";
}

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        PrintHelp();
        return -1;
    }
    std::string cameras_path = argv[1];
    std::string images_path = argv[2];
    std::string points_path = argv[3];
    std::string output_folder = argv[4];
    size_t iteration = 100;
    size_t cluster = 100;
    size_t inner_step = 4;
    size_t thread_num = 1;
    double radius = 10000;
    LossType loss_type = CauchyLossType;
    double noise = 0.0;
    bool lm = false;
    bool dl = false;

    size_t i = 5;
    while (i < argc)
    {
        std::string option = argv[i];
        if (option == "--iteration")
        {
            i++;
            if (i >= argc)
            {
                PrintHelp();
                return -1;
            }
            iteration = static_cast<size_t>(std::stoi(argv[i]));
        }
        else if (option == "--cluster")
        {
            i++;
            if (i >= argc)
            {
                PrintHelp();
                return -1;
            }
            cluster = static_cast<size_t>(std::stoi(argv[i]));
        }
        else if (option == "--inner_step")
        {
            i++;
            if (i >= argc)
            {
                PrintHelp();
                return -1;
            }
            inner_step = static_cast<size_t>(std::stoi(argv[i]));
        }
        else if (option == "--thread_num")
        {
            i++;
            if (i >= argc)
            {
                PrintHelp();
                return -1;
            }
            thread_num = static_cast<size_t>(std::stoi(argv[i]));
        }
        else if (option == "--radius")
        {
            i++;
            if (i >= argc)
            {
                PrintHelp();
                return -1;
            }
            radius = static_cast<double>(std::stod(argv[i]));
        }
        else if (option == "--loss")
        {
            i++;
            if (i >= argc)
            {
                PrintHelp();
                return -1;
            }
            if (loss_type != 0 && loss_type != 1 && loss_type != 2)
            {
                std::cout << "Invalid loss type: " << loss_type << "\n";
                PrintHelp();
                return -1;
            }
            loss_type = static_cast<LossType>(std::stoi(argv[i]));
        }
        else if (option == "--noise")
        {
            i++;
            if (i >= argc)
            {
                PrintHelp();
                return -1;
            }
            noise = std::abs(static_cast<double>(std::stod(argv[i])));
        }
        else if (option == "--lm")
        {
            lm = true;
        }
        else if (option == "--dl")
        {
            dl = true;
        }
        else if (option == "--help" || option == "--h")
        {
            PrintHelp();
            return 0;
        }
        else
        {
            std::cout << "Invalid option: " << option << "\n";
            PrintHelp();
            return -1;
        }
        i++;
    }


    std::cout << cameras_path << "\n"
              << images_path << "\n"
              << points_path << "\n"
              << output_folder << "\n"
              << "iteration = " << iteration << "\n"
              << "cluster = " << cluster << "\n"
              << "inner_step = " << inner_step << "\n"
              << "thread_num = " << thread_num << "\n"
              << "radius = " << radius << "\n"
              << "loss_type = " << loss_type << "\n"
              << "noise = " << noise << "\n"
              << "lm = " << lm << "\n"
              << "dl = " << dl << "\n";

    BundleBlock bundle_block;
    bundle_block.LoadColmapTxt(cameras_path, images_path, points_path);
    if (noise > 0)
    {
        bundle_block.AddGaussianNoiseToTrack(0, noise);
        bundle_block.AddGaussianNoiseToCameraTranslation(0, noise);
    }

    BAProblem * problem;
    if (lm)
    {
        problem = new LMBAProblem(iteration, radius, loss_type);
    }
    else if (dl)
    {
        problem = new DLBAProblem(iteration, radius, loss_type);
    }
    else
    {
        problem = new StochasticBAProblem(iteration, radius, loss_type, cluster, inner_step);
    }
    problem->SetThreadNum(thread_num);

    if (!problem->Initialize(bundle_block))
    {
        std::cout << "Fail to initialize bundle problem.\n";
        delete problem;
        return -1;
    }
    problem->Solve();
    problem->Update(bundle_block);
    problem->SaveReport(JoinPath(output_folder, "report.txt"));
    delete problem;

    bundle_block.SaveColmapTxt(JoinPath(output_folder, "cameras.txt"), JoinPath(output_folder, "images.txt"), JoinPath(output_folder, "points.txt"));

    return 0;
}
