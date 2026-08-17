#include <igl/readOFF.h>
#include <igl/marching_cubes.h>
#include <igl/writeOFF.h>
#include <vector>
#include <ctime>

using namespace std;

Eigen::MatrixXd P;
Eigen::MatrixXd N;
Eigen::MatrixXd constrained_points;
Eigen::VectorXd constrained_values;
double wendlandRadius = 1.1;
unsigned int resolution = 4;
unsigned int sampling_resolution = 25;

Eigen::MatrixXd sampling_pts;
Eigen::VectorXd sampling_values;
Eigen::MatrixXd grid_mins;
Eigen::MatrixXd grid_maxs;
Eigen::VectorXd bb_min;
Eigen::VectorXd bb_max;
Eigen::VectorXd grid_size;

vector< vector<size_t> > grid_hist;
vector<size_t> grid_range;

void init_uniform_grid()
{
    grid_hist.clear();
    grid_range.clear();
    grid_mins.resize(resolution*resolution*resolution, 3);
    grid_maxs.resize(resolution*resolution*resolution, 3);
    grid_size = (bb_max - bb_min) / resolution;
    // cout << grid_size << endl;
    for (int z = 0; z < resolution; ++z)
    {
        for (int y = 0; y < resolution; ++y)
        {
            for (int x = 0; x < resolution; ++x)
            {
                int idx = x + resolution * y + resolution * resolution * z;
                vector <size_t> t_vec;
                t_vec.clear();
                grid_hist.push_back(t_vec);
                grid_mins(idx, 0) = bb_min(0) + x * grid_size(0);
                grid_mins(idx, 1) = bb_min(1) + y * grid_size(1);
                grid_mins(idx, 2) = bb_min(2) + z * grid_size(2);
                grid_maxs(idx, 0) = bb_min(0) + (x + 1) * grid_size(0);
                grid_maxs(idx, 1) = bb_min(1) + (y + 1) * grid_size(1);
                grid_maxs(idx, 2) = bb_min(2) + (z + 1) * grid_size(2);
            }
        }
    }
}

inline double wendland_fun(double r)
{
    return r > wendlandRadius ? 0 : pow(1 - r / wendlandRadius, 4) * (4 * r / wendlandRadius + 1);
}

void insertSpatial(Eigen::MatrixXd points, int id)
{
    Eigen::VectorXd pt = points.row(id);
    int idx = (pt(0) - bb_min(0)) / grid_size(0);
    int idy = (pt(1) - bb_min(1)) / grid_size(1);
    int idz = (pt(2) - bb_min(2)) / grid_size(2);
    grid_hist[idx+idy*resolution+idz*resolution*resolution].push_back(id);
}

inline double pt_box_dist(Eigen::VectorXd pt, Eigen::VectorXd grid_min, Eigen::VectorXd grid_max)
{
    // return 0;
    double dist = 0.0;
    for (size_t i = 0; i < grid_min.size(); ++i)
    {
        dist += pow(max(max(pt(i)-grid_max(i), 0.0), grid_min(i)-pt(i)), 2);
    }
    return sqrt(dist);
}

inline double pt_pt_dist(Eigen::VectorXd pt1, Eigen::VectorXd pt2)
{
    return (pt1-pt2).norm();
}

bool closest_pt(Eigen::VectorXd pt, size_t supposed_closest_index, double eps)
{
    for (size_t grid_id = 0; grid_id < resolution*resolution*resolution; ++grid_id)
    {
        if (pt_box_dist(pt, grid_mins.row(grid_id), grid_maxs.row(grid_id)) <= eps)
        {
            for (size_t i = grid_range[grid_id]; i < grid_range[grid_id+1]; ++i)
            {
                if (i != supposed_closest_index && pt_pt_dist(pt, P.row(i)) <= eps)
                {
                    return false;
                }
            }
        }
    }
    return true;
}

bool nearest_neighbors(Eigen::VectorXd pt, Eigen::Matrix<double, 4, 4> &A, Eigen::Vector4d &b)
{
    size_t count = 0;
    for (size_t i = 0; i < resolution*resolution*resolution; ++i)
    {
        if (pt_box_dist(pt, grid_mins.row(i), grid_maxs.row(i)) <= wendlandRadius)
        {
            for (size_t j = grid_range[i]; j < grid_range[i+1]; ++j)
            {
                double dist = pt_pt_dist(pt, constrained_points.row(j));
                if (dist <= wendlandRadius)
                {
                    ++count;
                    double ww = wendland_fun(dist);
                    double cv = constrained_values[j];
                    double cx = constrained_points(j, 0);
                    double cy = constrained_points(j, 1);
                    double cz = constrained_points(j, 2);
                    A(0, 0) += ww;
                    A(0, 1) += ww * cx;
                    A(0, 2) += ww * cy;
                    A(0, 3) += ww * cz;
                    A(1, 0) += ww * cx;
                    A(1, 1) += ww * cx * cx;
                    A(1, 2) += ww * cy * cx;
                    A(1, 3) += ww * cz * cx;
                    A(2, 0) += ww * cy;
                    A(2, 1) += ww * cx * cy;
                    A(2, 2) += ww * cy * cy;
                    A(2, 3) += ww * cz * cy;
                    A(3, 0) += ww * cz;
                    A(3, 1) += ww * cx * cz;
                    A(3, 2) += ww * cy * cz;
                    A(3, 3) += ww * cz * cz;
                    b(0) += ww * cv;
                    b(1) += ww * cv * cx;
                    b(2) += ww * cv * cy;
                    b(3) += ww * cv * cz;
                }
            }
        }
    }
    return count >= 8;
}

void evaluate_with_mls()
{
    Eigen::Matrix<double, 4, 4> A = Eigen::Matrix<double, 4, 4>::Zero();
    Eigen::Vector4d b = Eigen::Vector4d::Zero();
    for (int i = 0; i < sampling_resolution*sampling_resolution*sampling_resolution; ++i)
    {
        A = Eigen::Matrix<double, 4, 4>::Zero();
        b = Eigen::Vector4d::Zero(); 
        if (nearest_neighbors(sampling_pts.row(i), A, b)) 
        {
            Eigen::Vector4d x = A.colPivHouseholderQr().solve(b);
            sampling_values(i) = x(0) + sampling_pts(i, 0) * x(1) 
                                 + sampling_pts(i, 1) * x(2) + sampling_pts(i, 2) * x(3); 
        } else 
        {
            sampling_values(i) = 100;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage ex2_bin mesh.off [mesh_out.off]\n" << endl;
        exit(0);
    }
    clock_t start, end;
    // Read points and normals
    Eigen::MatrixXi facets;
    igl::readOFF(argv[1], P, facets, N);
    bb_min = P.colwise().minCoeff().eval();
    bb_max = P.colwise().maxCoeff().eval();
    Eigen::VectorXd diag = bb_max-bb_min;
    bb_min -= diag * 0.02;
    bb_max += diag * 0.02;
    init_uniform_grid();
    for (size_t i = 0; i < P.rows(); ++i)
    {
        insertSpatial(P, i);
    }

    // counting sort
    Eigen::MatrixXd sorted_P(P.rows(), P.cols());
    Eigen::MatrixXd sorted_N(N.rows(), N.cols());
    size_t idx = 0;
    for (size_t i = 0; i < resolution*resolution*resolution; ++i)
    {
        grid_range.push_back(idx);
        for (size_t j = 0; j < grid_hist[i].size(); ++j)
        {
            sorted_P.row(idx) = P.row(grid_hist[i][j]);
            sorted_N.row(idx) = N.row(grid_hist[i][j]);
            ++idx;
        }
    }
    grid_range.push_back(idx);

    P = sorted_P;
    N = sorted_N;
    double eps = diag.norm() * 0.01;
    // cout << "init eps: " << eps << endl;
    for (size_t i = 0; i < P.rows(); ++i)
    {
        Eigen::VectorXd p_plus = P.row(i)+N.row(i)*eps;
        while (!closest_pt(p_plus, i, eps))
        {
            eps /= 2;
        }
        Eigen::VectorXd p_minus = P.row(i)-N.row(i)*eps;
        while (!closest_pt(p_minus, i, eps))
        {
            eps /= 2;
        }
    }
    // cout << "eps: " << eps << endl;

    constrained_points.resize(3*P.rows(), 3);
    constrained_values.resize(3*P.rows());
    for (size_t i = 0; i < P.rows(); ++i)
    {
        constrained_points.row(i) = P.row(i);
        constrained_values(i) = 0;
        constrained_points.row(i+P.rows()) = P.row(i)+eps*N.row(i);
        constrained_values(i+P.rows()) = eps;
        constrained_points.row(i+P.rows()*2) = P.row(i)-eps*N.row(i);
        constrained_values(i+P.rows()*2) = -eps;
    }
    bb_min = constrained_points.colwise().minCoeff().eval();
    bb_max = constrained_points.colwise().maxCoeff().eval();
    diag = bb_max-bb_min;
    bb_min -= diag * 0.02;
    bb_max += diag * 0.02;
    init_uniform_grid();
    for (size_t i = 0; i < constrained_points.rows(); ++i)
    {
        insertSpatial(constrained_points, i);
    }
    // counting sort
    Eigen::MatrixXd sorted_CP(constrained_points.rows(), constrained_points.cols());
    Eigen::VectorXd sorted_CV(constrained_values.size());
    idx = 0;
    for (size_t i = 0; i < resolution*resolution*resolution; ++i)
    {
        grid_range.push_back(idx);
        for (size_t j = 0; j < grid_hist[i].size(); ++j)
        {
            sorted_CP.row(idx) = constrained_points.row(grid_hist[i][j]);
            sorted_CV(idx) = constrained_values(grid_hist[i][j]);
            ++idx;
        }
    }
    grid_range.push_back(idx);

    constrained_points = sorted_CP;
    constrained_values = sorted_CV;
    // for (size_t z = 0; z < resolution; ++z)
    // {
    //     for (size_t y = 0; y < resolution; ++y)
    //     {
    //         for (size_t x = 0; x < resolution; ++x)
    //         {
    //             size_t index = x + y * resolution + z * resolution * resolution;
    //             for (size_t j = grid_range[index]; j < grid_range[index+1]; ++j)
    //             {
    //                 if (constrained_points(j, 0) < grid_mins(index, 0) || 
    //                     constrained_points(j, 0) > grid_maxs(index, 0) ||
    //                     constrained_points(j, 1) < grid_mins(index, 1) || 
    //                     constrained_points(j, 1) > grid_maxs(index, 1) ||
    //                     constrained_points(j, 2) < grid_mins(index, 2) || 
    //                     constrained_points(j, 2) > grid_maxs(index, 2))
    //                     {
    //                         cout << "uniform grid error" << endl;
    //                     }
    //             }
    //         }
    //     }
    // }

    Eigen::VectorXd sampling_dist = (bb_max - bb_min) / (sampling_resolution - 1);
    sampling_pts.resize(sampling_resolution*sampling_resolution*sampling_resolution, 3);
    sampling_values.resize(sampling_resolution*sampling_resolution*sampling_resolution);
    // create sampling grid
    for (size_t z = 0; z < sampling_resolution; ++z)
    {
        for (size_t y = 0; y < sampling_resolution; ++y)
        {
            for (size_t x = 0; x < sampling_resolution; ++x)
            {
                size_t index = x + y * sampling_resolution + z * sampling_resolution * sampling_resolution;
                
                Eigen::VectorXd sampling_pt(3);
                sampling_pt(0) = bb_min(0)+x*sampling_dist(0);
                sampling_pt(1) = bb_min(1)+y*sampling_dist(1);
                sampling_pt(2) = bb_min(2)+z*sampling_dist(2);
                sampling_pts.row(index) = sampling_pt;
            }
        }
    }
    // evaluate implicit surfaces
    // start = clock();
    evaluate_with_mls();
    // end = clock();
    // printf("Elapsed: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    if (argc == 3)
    {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        igl::marching_cubes(sampling_values, sampling_pts, sampling_resolution, sampling_resolution, sampling_resolution, 0.0, V, F);
        igl::writeOFF(argv[2], V, F);
    }



}