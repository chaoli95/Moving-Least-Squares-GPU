#include <cstdlib>
#include <cstdio>
#include <ctime> 
#include <string>
#include <vector>
#include <assert.h>
#include <cmath>
#include <cusolverSp.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>
// #include <igl/marching_cubes.h>
// #include <igl/writeOFF.h>

using namespace std;

#define index(i, j, N)  ((i)*(N)) + (j)
#define check_uniform_grid

// resolution for uniform grid
#define X_resolution 8
#define Y_resolution 8
#define Z_resolution 8
// Constraint: bin_size <= 2048
// Also, bin_size should be a power of two
#define bin_size (X_resolution * Y_resolution * Z_resolution)

#define BLOCK_SIZE 128
// sampling points along x, y, z axis
#define NX 200
#define NY 200
#define NZ 200
#define sampling_size (NX*NY*NZ)
// wendland radius
#define H 1.1

#define imin( x, y ) ((x)<(y))? (x) : (y) 

__global__ void fill_box_mins_maxs(double* omins, double* omaxs, double3 bb_min, double3 grid_size);
__global__ void counting_sort_histogram(double* data, int* bin, int* offset, int* ids, double3 aabb_min, double3 box_size, int data_size);
__global__ void counting_sort_prefix_sum(int* g_odata, int* g_idata, int n);
__global__ void counting_sort_copy_3d(double* odata, double* idata, int* bin_range, int* offsets, int* ids, int data_size);
__global__ void counting_sort_copy(double* odata, double* idata, int* bin_range, int* offsets, int* ids, int data_size);
__global__ void find_eps(double* result, double* P, double* N, int number_of_vertices,
    int* box_range, double* aabb_mins, double* aabb_maxs, double initial_eps);

__global__ void generate_constraint_points(double* odata, double* idata, double* normals, double eps, int N);
__global__ void generate_constraint_values(double* odata, double eps, int N);
void find_bounding_box(double3& min, double3& max, double* points, int N);

template <typename Scalar>
bool read_off(const std::string off_file, vector<Scalar>& verticies, vector<Scalar>& normals)
{
    std::ifstream input(off_file);
    if(!input) { return false; }
    std::string line;
    if(!getline(off_file, line) || line != "NOFF") 
    {
        return false;
    }
    int nv, nf, ne;
    off_file >> nv, nf, ne;
    Scalar tmp;
    for(int v=0; v<nv; ++v)
    {
        for (int i=0; i<3; ++i) 
        { 
            off_file >> tmp;
            verticies.push_back(tmp); 
        }
        for (int i=0; i<3; ++i) 
        { 
            off_file >> tmp;
            normals.push_back(tmp); 
        }
    }
    // we don't need faces
    return true;
}

// bool readPointCloud(double** vertices, double** normals, int* num_v, const char* filename)
// {
//     FILE* off_file = fopen(filename, "r");
//     const std::string NOFF("NOFF");
//     char header[100];
//     if (fscanf(off_file, "%s\n", header) != 1
//         || !(string(header).compare(0, NOFF.length(), NOFF) == 0))
//     {
//         printf("Error: readOFF() first line should be NOFF, not %s...", header);
//         fclose(off_file);
//         return false;
//     }
//     int number_of_faces;
//     int number_of_edges;
//     fscanf(off_file, "%d %d %d", num_v, &number_of_faces, &number_of_edges);
//     // times 3 because there's x, y, and z
//     // cudaMallocManaged(&P, sizeof(double) * number_of_vertices * 3);
//     // cudaMallocManaged(&N, sizeof(double) * number_of_vertices * 3);
//     *vertices = (double*)malloc(sizeof(double) * (*num_v) * 3);
//     *normals = (double*)malloc(sizeof(double) * (*num_v) * 3);

//     for (int i = 0; i < *num_v; ++i)
//     {
//         for (int j = 0; j < 3; ++j)
//         {
//             fscanf(off_file, "%lf ", (*vertices) + i * 3 + j);
//             // P[i*3+j] *= 100;
//         }
//         for (int j = 0; j < 3; ++j)
//         {
//             // TODO: normals are not guaranteed normalized
//             fscanf(off_file, "%lf ", (*normals) + i * 3 + j);
//         }
//     }
//     fclose(off_file);
//     return true;
// }

void build_uniform_grid(double* sorted_P, int* grid_range, double* grid_mins, double* grid_maxs,
    int* ids, int* offsets, double3* bb_min, double3* bb_max, double* diag_length, double* P,
    int num_of_vertices)
{
    cudaError_t error;
    int* grid_hist;
    cudaMalloc((void**)&grid_hist, sizeof(int) * bin_size);
    cudaMemset(grid_hist, 0, sizeof(int) * bin_size);
    double3 Pmin, Pmax;
    find_bounding_box(Pmin, Pmax, P, num_of_vertices);
    double diag_x = Pmax.x - Pmin.x;
    double diag_y = Pmax.y - Pmin.y;
    double diag_z = Pmax.z - Pmin.z;
    *diag_length = sqrt(diag_x*diag_x + diag_y*diag_y + diag_z*diag_z);
    *bb_min = make_double3(Pmin.x-0.02*(*diag_length), 
                          Pmin.y-0.02*(*diag_length), 
                          Pmin.z-0.02*(*diag_length));
    *bb_max = make_double3(Pmax.x+0.02*(*diag_length), 
                          Pmax.y+0.02*(*diag_length), 
                          Pmax.z+0.02*(*diag_length));
    double3 grid_size = make_double3(((*bb_max).x-(*bb_min).x)/X_resolution,
                                    ((*bb_max).y-(*bb_min).y)/Y_resolution,
                                    ((*bb_max).z-(*bb_min).z)/Z_resolution);
    int dimBlock = 64;
    int dimGrid = ceil((double)num_of_vertices/4/dimBlock);
    fill_box_mins_maxs<<<ceil((double)bin_size/dimBlock), dimBlock>>>(grid_mins, grid_maxs, *bb_min, grid_size);
    counting_sort_histogram<<<dimGrid, dimBlock>>>(P, grid_hist, offsets, ids, *bb_min, grid_size, num_of_vertices);
    error = cudaGetLastError(); 
    if(error != cudaSuccess)
    { 
        printf("counting sort histogram error: %s\n", cudaGetErrorString(error));
    }

    // // check hist bin
    // int *hist;
    // hist = (int *)malloc(sizeof(int)*bin_size);
    // cudaMemcpy(hist, grid_hist, sizeof(int)*bin_size, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < bin_size; ++i)
    // {
    //     printf("%d ", hist[i]);
    // }
    // printf("\n");

    // // check ids and offsets
    // int *hid, *hoffset;
    // hid = (int *)malloc(sizeof(int)*num_of_vertices);
    // hoffset = (int *)malloc(sizeof(int)*num_of_vertices);
    // cudaMemcpy(hid, ids, sizeof(int)*num_of_vertices, cudaMemcpyDeviceToHost);
    // cudaMemcpy(hoffset, offsets, sizeof(int)*num_of_vertices, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < num_of_vertices; ++i)
    // {
    //     printf("id %d offset %d\n", hid[i], hoffset[i]);
    // }

    cudaDeviceSynchronize();
    counting_sort_prefix_sum<<<1, bin_size/2, (bin_size+1)*sizeof(int)>>>(grid_range, grid_hist, bin_size);
    error = cudaGetLastError(); 
    if(error != cudaSuccess)
    { 
        printf("counting sort prefix sum error: %s\n", cudaGetErrorString(error));
    }

    // int *range;
    // range = (int *)malloc(sizeof(int)*(bin_size+1));
    // cudaMemcpy(range, grid_range, sizeof(int)*(1+bin_size), cudaMemcpyDeviceToHost);
    // for (int i = 0; i <= bin_size; ++i)
    // {
    //     printf("%d ", range[i]);
    // }
    // printf("\n");

    cudaDeviceSynchronize();
    counting_sort_copy_3d<<<dimGrid, dimBlock>>>(sorted_P, P, grid_range, offsets, ids, num_of_vertices);
    error = cudaGetLastError(); 
    if(error != cudaSuccess)
    { 
        printf("counting sort copy error: %s\n", cudaGetErrorString(error));
    }
    cudaFree(grid_hist);
    cudaDeviceSynchronize();
}

__global__ void fill_box_mins_maxs(double *omins, double *omaxs, double3 bb_min, double3 grid_size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < bin_size)
    {
        int x = id / (Y_resolution * Z_resolution);
        int y = (id - x * Y_resolution * Z_resolution) / Z_resolution;
        int z = id - x * Y_resolution * Z_resolution - y * Z_resolution;
        // printf("tid: %d, (%d, %d, %d)\n", id, x,y,z);
        omins[index(id, 0, 3)] = bb_min.x + grid_size.x * x;
        omins[index(id, 1, 3)] = bb_min.y + grid_size.y * y;
        omins[index(id, 2, 3)] = bb_min.z + grid_size.z * z;
        omaxs[index(id, 0, 3)] = bb_min.x + grid_size.x * (x + 1);
        omaxs[index(id, 1, 3)] = bb_min.y + grid_size.y * (y + 1);
        omaxs[index(id, 2, 3)] = bb_min.z + grid_size.z * (z + 1);
    }
}

void set_up_constraints(double **constraint_points, double **constraint_values, int **grid_range_o, double **grid_mins_o, double **grid_maxs_o, 
                        double3 *bb_min, double3 *bb_max, double *vertices, double *normals, int num_of_vertices)
{
    cudaStream_t s1;
    cudaStreamCreate(&s1);
    int num_of_constraints = 3 * num_of_vertices;

    // int *vertices_dev, *normals_dev;
    double *v_dev1, *v_dev2;
    double *n_dev1, *n_dev2;
    double *grid_mins, *grid_maxs;
    cudaMalloc((void **)&grid_mins, sizeof(double)*bin_size*3);
    cudaMalloc((void **)&grid_maxs, sizeof(double)*bin_size*3);
    cudaMalloc((void **)&v_dev1, sizeof(double)*num_of_vertices*9);
    cudaMalloc((void **)&v_dev2, sizeof(double)*num_of_vertices*9);
    cudaMalloc((void **)&n_dev1, sizeof(double)*num_of_vertices*3);
    cudaMalloc((void **)&n_dev2, sizeof(double)*num_of_vertices*3);
    cudaMemcpyAsync(v_dev1, vertices, sizeof(double)*num_of_vertices*3, cudaMemcpyHostToDevice, s1);
    cudaMemcpyAsync(n_dev1, normals, sizeof(double)*num_of_vertices*3, cudaMemcpyHostToDevice, s1);
    int *grid_range;
    int *ids, *offsets;
    // double3 bb_min, bb_max;
    cudaMalloc((void **)&grid_range, sizeof(int)*(bin_size+1));
    cudaMalloc((void **)&ids, sizeof(int)*num_of_vertices*3);
    cudaMalloc((void **)&offsets, sizeof(int)*num_of_vertices*3);
    double diag_length;
    cudaDeviceSynchronize();

    // build grid for original data
    build_uniform_grid(v_dev2, grid_range, grid_mins, grid_maxs, ids, offsets, bb_min, bb_max, &diag_length, v_dev1, num_of_vertices);
    int dimBlock = 64;
    int dimGrid = ceil((double)num_of_vertices/4/dimBlock);
    counting_sort_copy_3d<<<dimGrid, dimBlock>>>(n_dev2, n_dev1, grid_range, offsets, ids, num_of_vertices);
    cudaDeviceSynchronize();


    // find eps with grid
    double *eps, *eps_dev;
    eps = (double *)malloc(sizeof(double)*num_of_vertices);
    cudaMalloc((void **)&eps_dev, sizeof(double)*num_of_vertices);

    // printf("diag length: %lf\n", diag_length);
    double init_eps = 0.01 * diag_length;
    find_eps<<<ceil((double)num_of_vertices/256), 256>>>(eps_dev, v_dev2, n_dev2, num_of_vertices, 
                                                grid_range, grid_mins, grid_maxs, init_eps);
    cudaMemcpy(eps, eps_dev, sizeof(double)*num_of_vertices, cudaMemcpyDeviceToHost);
    double min_eps = init_eps;
    for (int i = 0; i < num_of_vertices; ++i)
    {
        min_eps = min(min_eps, eps[i]);
    }
    // printf("min eps: %lf\n", min_eps);

    // generate constraints points
    generate_constraint_points<<<ceil((double)num_of_vertices/256/2), 256>>>(v_dev1, v_dev2, n_dev2, min_eps, num_of_vertices);
    double *CV_1, *CV_2;
    cudaMalloc((void **)&CV_1, sizeof(double)*num_of_constraints);
    cudaMalloc((void **)&CV_2, sizeof(double)*num_of_constraints);
    cudaMemset(CV_1, 0, sizeof(double)*num_of_vertices);
    generate_constraint_values<<<ceil((double)num_of_vertices/256/2), 256>>>(CV_1, min_eps, num_of_vertices);
    cudaDeviceSynchronize();

    // build uniform grid for constraints points
    build_uniform_grid(v_dev2, grid_range, grid_mins, grid_maxs, ids, offsets, bb_min, bb_max, &diag_length, v_dev1, num_of_constraints);
    counting_sort_copy<<<dimGrid, dimBlock>>>(CV_2, CV_1, grid_range, offsets, ids, num_of_constraints);
    cudaDeviceSynchronize();

    // clean up
    cudaStreamDestroy(s1);
    cudaFree(v_dev1);
    cudaFree(n_dev1);
    cudaFree(n_dev2);
    cudaFree(ids);
    cudaFree(offsets);
    cudaFree(eps_dev);
    cudaFree(CV_1);
    free(eps);

    // assign to output
    *constraint_points = v_dev2;
    *constraint_values = CV_2;
    *grid_range_o = grid_range;
    *grid_mins_o = grid_mins;
    *grid_maxs_o = grid_maxs;


// #ifdef check_uniform_grid
//     int *grid_range_host;
//     grid_range_host = (int *)malloc(sizeof(int)*(1+bin_size));
//     double *sorted_P;
//     sorted_P = (double *)malloc(sizeof(double)*num_of_constraints*3);
//     cudaMemcpy(grid_range_host, grid_range, sizeof(int)*(bin_size+1), cudaMemcpyDeviceToHost);
//     cudaMemcpy(sorted_P, v_dev2, sizeof(double)*num_of_constraints*3, cudaMemcpyDeviceToHost);
//     double3 grid_size = make_double3(((bb_max).x-(bb_min).x)/X_resolution,
//                                    ((bb_max).y-(bb_min).y)/Y_resolution,
//                                    ((bb_max).z-(bb_min).z)/Z_resolution);
//     double *CV_host;
//     CV_host = (double *)malloc(sizeof(double)*num_of_constraints);
//     cudaMemcpy(CV_host, CV_2, sizeof(double)*num_of_constraints, cudaMemcpyDeviceToHost);
//     // printf("grid size: %lf %lf %lf\n", grid_size.x, grid_size.y, grid_size.z);
//     for (int i = 0; i < bin_size; ++i)
//     {
//         // printf("bin id %d, [%d, %d]\n", i, grid_range_host[i], grid_range_host[i+1]);
//         for (int j = grid_range_host[i]; j < grid_range_host[i+1]; ++j)
//         {
//             double x = sorted_P[index(j, 0, 3)];
//             double y = sorted_P[index(j, 1, 3)];
//             double z = sorted_P[index(j, 2, 3)];
//             printf("(%lf %lf %lf) %lf\n", x,y,z,CV_host[j]);
//             int idx = (x - bb_min.x) / grid_size.x;
//             int idy = (y - bb_min.y) / grid_size.y;
//             int idz = (z - bb_min.z) / grid_size.z;
//             // printf("%d ", idx * Y_resolution * Z_resolution + idy * Z_resolution + idz);
//             if (idx * Y_resolution * Z_resolution + idy * Z_resolution + idz != i)
//             {
//                 printf("box check fail\n");
//             }
//         }
//         // printf("\n");
//     }
//     double *host_mins, *host_maxs;
//     host_mins = (double *)malloc(sizeof(double)*bin_size*3);
//     host_maxs = (double *)malloc(sizeof(double)*bin_size*3);
//     cudaMemcpy(host_mins, grid_mins, sizeof(double)*bin_size*3, cudaMemcpyDeviceToHost);
//     cudaMemcpy(host_maxs, grid_maxs, sizeof(double)*bin_size*3, cudaMemcpyDeviceToHost);
//     for (int i = 0; i < bin_size; ++i)
//     {
//         printf("bin (%lf %lf %lf) (%lf %lf %lf)\n", host_mins[i*3], host_mins[i*3+1], host_mins[i*3+2],
//                                               host_maxs[i*3], host_maxs[i*3+1], host_maxs[i*3+2]);
//     }
// #endif
}


__global__ void counting_sort_histogram(double *data, int *bin, int *offset, int *box_id, double3 aabb_min, double3 box_size, int data_size)
{
    __shared__ int private_bin[bin_size];
    // max thread granularity: 4
    int local_ids[4];
    int local_offset[4];
    
    // initialize private_bin
    for (int i = threadIdx.x; i < bin_size; i += blockDim.x)
    {
        private_bin[i] = 0;
    }
    __syncthreads();

    // load data and update private bin
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/
    #pragma unroll 
    for (int i = 0; i < 4; ++i)
    {   
        if (id < data_size)
        {
            double3 point = make_double3(data[index(id, 0, 3)],
                                        data[index(id, 1, 3)],
                                        data[index(id, 2, 3)]);
            // TODO: multiple is faster than divide
            int3 ids = make_int3(static_cast<int>((point.x - aabb_min.x)/box_size.x),
                                static_cast<int>((point.y - aabb_min.y)/box_size.y),
                                static_cast<int>((point.z - aabb_min.z)/box_size.z));
            local_ids[i] = ids.x * Y_resolution * Z_resolution + ids.y * Z_resolution + ids.z;
            local_offset[i] = atomicAdd(&(private_bin[local_ids[i]]), 1);
            box_id[id] = local_ids[i];
            id += stride;
        }
    }
    __syncthreads();

    // update global bin with private bin
    for (int i = threadIdx.x; i < bin_size; i += blockDim.x)
    {
        if (private_bin[i])
        {
            private_bin[i] = atomicAdd(bin+i, private_bin[i]);
        }
    }
    __syncthreads();

    // update offset for copying later
    id = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll 
    for (int i = 0; i < 4; ++i)
    {   
        if (id < data_size)
        {
            offset[id] = local_offset[i] + private_bin[local_ids[i]];
            id += stride;
        }
    } 
}


// GPU Gems 3
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
#define NUM_BANKS 16 
#define LOG_NUM_BANKS 4 
#define CONFLICT_FREE_OFFSET(n) \
((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 

__global__ void counting_sort_prefix_sum(int *g_odata, int *g_idata, int n)
{
    extern __shared__ int temp[];
    int thid = threadIdx.x;
    int offset = 1;
    int ai = thid; 
    int bi = thid + (n/2); 
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    temp[ai + bankOffsetA] = g_idata[ai];
    temp[bi + bankOffsetB] = g_idata[bi];

    // temp[2*thid] = g_idata[2*thid];
    // temp[2*thid+1] = g_idata[2*thid+1]; 
    for (int d = n >> 1; d > 0; d >>= 1)
    {
        __syncthreads();    
        if (thid < d)    
        {
            int ai = offset*(2*thid+1)-1; 
            int bi = offset*(2*thid+2)-1; 
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            // int ai = offset*(2*thid+1)-1;    
            // int bi = offset*(2*thid+2)-1;  
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    if (thid == 0) 
    { 
        g_odata[n] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
        temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
        // temp[n - 1] = 0; 
    }
    for (int d = 1; d < n; d *= 2)
    {
        offset >>= 1;      
        __syncthreads();      
        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1; 
            int bi = offset*(2*thid+2)-1; 
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            // int ai = offset*(2*thid+1)-1;     
            // int bi = offset*(2*thid+2)-1; 
            double t = temp[ai]; 
            temp[ai] = temp[bi]; 
            temp[bi] += t;
        }
    }
    __syncthreads(); 
    g_odata[ai] = temp[ai + bankOffsetA]; 
    g_odata[bi] = temp[bi + bankOffsetB]; 

    // g_odata[2*thid] = temp[2*thid];  
    // g_odata[2*thid+1] = temp[2*thid+1]; 
}

__global__ void counting_sort_copy_3d(double *odata, double *idata, int *bin_range, int *offsets, int *ids, int data_size)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    while (id < data_size)
    {
        int new_id = bin_range[ids[id]] + offsets[id];
        odata[index(new_id, 0, 3)] = idata[index(id, 0, 3)];
        odata[index(new_id, 1, 3)] = idata[index(id, 1, 3)];
        odata[index(new_id, 2, 3)] = idata[index(id, 2, 3)];
        id += stride;
    }
}

__global__ void counting_sort_copy(double *odata, double *idata, int *bin_range, int *offsets, int *ids, int data_size)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    while (id < data_size)
    {
        int new_id = bin_range[ids[id]] + offsets[id];
        odata[new_id] = idata[id];
        id += stride;
    }
}


__global__ void find_max_and_min(double *points, double *maxs_and_mins, int n)
{
    // tid * 6
    extern __shared__ double local_data[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;
    int grid_size = blockDim.x * 2 * gridDim.x;
    local_data[tid * 6 + 0] = -INFINITY;
    local_data[tid * 6 + 1] = -INFINITY;
    local_data[tid * 6 + 2] = -INFINITY;
    local_data[tid * 6 + 3] = INFINITY;
    local_data[tid * 6 + 4] = INFINITY;
    local_data[tid * 6 + 5] = INFINITY;
    double x0, y0, z0, x1, y1, z1;
    while (i < n)
    {
        x0 = points[index(i, 0, 3)]; y0 = points[index(i, 1, 3)]; z0 = points[index(i, 2, 3)];  
        if (i + blockDim.x < n)
        {
            x1 = points[index(i+blockDim.x, 0, 3)]; 
            y1 = points[index(i+blockDim.x, 1, 3)]; 
            z1 = points[index(i+blockDim.x, 2, 3)]; 
        } else
        {
            x1 = x0; y1 = y0; z1 = z0;
        }
        local_data[tid * 6 + 0] = max(max(local_data[tid * 6 + 0], x0), x1);
        local_data[tid * 6 + 1] = max(max(local_data[tid * 6 + 1], y0), y1);
        local_data[tid * 6 + 2] = max(max(local_data[tid * 6 + 2], z0), z1);
        local_data[tid * 6 + 3] = min(min(local_data[tid * 6 + 3], x0), x1);
        local_data[tid * 6 + 4] = min(min(local_data[tid * 6 + 4], y0), y1);
        local_data[tid * 6 + 5] = min(min(local_data[tid * 6 + 5], z0), z1);   
        i += grid_size; 
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            local_data[tid * 6 + 0] = max(local_data[tid * 6 + 0], local_data[(tid+stride)*6+0]);
            local_data[tid * 6 + 1] = max(local_data[tid * 6 + 1], local_data[(tid+stride)*6+1]);
            local_data[tid * 6 + 2] = max(local_data[tid * 6 + 2], local_data[(tid+stride)*6+2]);
            local_data[tid * 6 + 3] = min(local_data[tid * 6 + 3], local_data[(tid+stride)*6+3]);
            local_data[tid * 6 + 4] = min(local_data[tid * 6 + 4], local_data[(tid+stride)*6+4]);
            local_data[tid * 6 + 5] = min(local_data[tid * 6 + 5], local_data[(tid+stride)*6+5]);     
        }
    }
    if (tid == 0)
    {
        maxs_and_mins[index(blockIdx.x, 0, 6)] = local_data[0];
        maxs_and_mins[index(blockIdx.x, 1, 6)] = local_data[1];
        maxs_and_mins[index(blockIdx.x, 2, 6)] = local_data[2];
        maxs_and_mins[index(blockIdx.x, 3, 6)] = local_data[3];
        maxs_and_mins[index(blockIdx.x, 4, 6)] = local_data[4];
        maxs_and_mins[index(blockIdx.x, 5, 6)] = local_data[5];
    }
}

void find_bounding_box(double3 &box_min, double3 &box_max, double *dev_points, int N)
{
    int grid_size = ceil((double)N/BLOCK_SIZE/2);
    double *result, *dev_result;
    result = (double *)malloc(sizeof(double)*grid_size*6);
    cudaMalloc((void **)&dev_result, sizeof(double)*grid_size*6);

    find_max_and_min<<<grid_size, BLOCK_SIZE, sizeof(double)*BLOCK_SIZE*6>>>(dev_points, dev_result, N);
    cudaMemcpy(result, dev_result, sizeof(double)*grid_size*6, cudaMemcpyDeviceToHost);
    double xmax = result[0];
    double ymax = result[1];
    double zmax = result[2];
    double xmin = result[3];
    double ymin = result[4];
    double zmin = result[5];
    for (int i = 1; i < grid_size; ++i)
    {
        xmax = max(xmax, result[index(i, 0, 6)]);
        ymax = max(ymax, result[index(i, 1, 6)]);
        zmax = max(zmax, result[index(i, 2, 6)]);
        xmin = min(xmin, result[index(i, 3, 6)]);
        ymin = min(ymin, result[index(i, 4, 6)]);
        zmin = min(zmin, result[index(i, 5, 6)]);
    }
    free(result);
    cudaFree(dev_result);
    box_max = make_double3(xmax, ymax, zmax);
    box_min = make_double3(xmin, ymin, zmin);
}

__global__ void generate_constraint_points(double *odata, double *idata, double *normals, double eps, int N)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    double d, n;
    while (id < N * 3)
    {
        d = idata[id];
        n = normals[id];
        odata[id] = d;
        odata[id + N * 3] = d + eps * n;
        odata[id + N * 6] = d - eps * n;
        id += stride;
    }
}

__global__ void generate_constraint_values(double *odata, double eps, int N)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (id < N)
    {
        odata[id+N] = eps;
        odata[id+2*N] = -eps;
        id += stride;
    }
}

inline __device__ double point_box_dist(double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, double3 p)
{
    double dx = max(max(xmin - p.x, 0.0), p.x - xmax);
    double dy = max(max(ymin - p.y, 0.0), p.y - ymax);
    double dz = max(max(zmin - p.z, 0.0), p.z - zmax);
    return sqrt(dx*dx + dy*dy + dz*dz);
}

inline __device__ double point_point_dist(double x, double y, double z, double3 p)
{
    double dx = x-p.x;
    double dy = y-p.y;
    double dz = z-p.z;
    return sqrt(dx*dx + dy*dy + dz*dz); 
}

inline __device__ double point_point_dist(double3 p1, double3 p2)
{
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    double dz = p1.z - p2.z;
    return sqrt(dx*dx + dy*dy + dz*dz);
}

// result: #vertices * 2 because we only validate the added constraint points
// C: constraints points. #vx3 points
__global__ void find_eps(double *result, double *P, double *N, int number_of_vertices, 
                         int *box_range, double *aabb_mins, double *aabb_maxs, double initial_eps)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < number_of_vertices)
    {
        double eps = initial_eps;
        double3 pi = make_double3(P[index(id, 0, 3)], P[index(id, 1, 3)], P[index(id, 2, 3)]);
        double3 ni = make_double3(N[index(id, 0, 3)], N[index(id, 1, 3)], N[index(id, 2, 3)]); 
        double3 pi_new = make_double3(pi.x + eps * ni.x, 
                                     pi.y + eps * ni.y,
                                     pi.z + eps * ni.z);
        for (int i = 0; i < bin_size; ++i)
        {
            if (point_box_dist(aabb_mins[index(i, 0, 3)], 
                                     aabb_mins[index(i, 1, 3)],
                                     aabb_mins[index(i, 2, 3)],
                                     aabb_maxs[index(i, 0, 3)],
                                     aabb_maxs[index(i, 1, 3)],
                                     aabb_maxs[index(i, 2, 3)],
                                     pi_new) <= eps)
            {
                for (int j = box_range[i]; j < box_range[i+1]; ++j)
                {
                    if (j != id && point_point_dist(P[index(j, 0, 3)], P[index(j, 1, 3)], P[index(j, 2, 3)], pi_new) <= eps)
                    {
                        eps /= 2;
                        pi_new = make_double3(pi.x + eps * ni.x, 
                                     pi.y + eps * ni.y,
                                     pi.z + eps * ni.z);
                        j -= 1;
                    }
                }
            }
        }
        pi_new = make_double3(pi.x - eps * ni.x, 
                                     pi.y - eps * ni.y,
                                     pi.z - eps * ni.z);
        for (int i = 0; i < bin_size; ++i)
        {
            if (point_box_dist(aabb_mins[index(i, 0, 3)], 
                                     aabb_mins[index(i, 1, 3)],
                                     aabb_mins[index(i, 2, 3)],
                                     aabb_maxs[index(i, 0, 3)],
                                     aabb_maxs[index(i, 1, 3)],
                                     aabb_maxs[index(i, 2, 3)],
                                     pi_new) <= eps)
            {
                for (int j = box_range[i]; j < box_range[i+1]; ++j)
                {
                    if (j != id && point_point_dist(P[index(j, 0, 3)], P[index(j, 1, 3)], P[index(j, 2, 3)], pi_new) <= eps)
                    {
                        eps /= 2;
                        pi_new = make_double3(pi.x - eps * ni.x, 
                                     pi.y - eps * ni.y,
                                     pi.z - eps * ni.z);
                        j -= 1;
                    }
                }
            }
        }
        result[id] = eps;
    }
}

__global__ void create_sampling_points(double *odata, double3 box_min, double3 box_size)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int z = id / (NX * NY);
    unsigned int y = (id - z * NX * NY) / NX;
    unsigned int x = id - z * NX * NY - y * NX;
    // printf("tid: %u, (%u,%u,%u)\n", id, x, y, z);
    if (x < NX && y < NY && z < NZ)
    {
        odata[index(id, 0, 3)] = box_min.x + x * box_size.x;
        odata[index(id, 1, 3)] = box_min.y + y * box_size.y;
        odata[index(id, 2, 3)] = box_min.z + z * box_size.z;
    }
}

inline __device__ double wendland(double r)
{
    // assuming r <= h
    return pow(1-r/H, 4) * (4*r/H+1);
}

__global__ void set_up_linear_systems(double *As, double *Bs, double *sampling_points, double *constraint_points,
                                      double *constraint_values, int *box_range, double *box_mins, double *box_maxs)
{
    unsigned int pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid < NX*NY*NZ)
    {
        double3 pi = make_double3(sampling_points[index(pid, 0, 3)],
                                sampling_points[index(pid, 1, 3)],
                                sampling_points[index(pid, 2, 3)]); 
        double cx, cy, cz, cv;
        double r, ww;
        int count = 0;
        double localA[16];
        double localB[4];
        #pragma unroll 
        for (int i = 0; i < 16; ++i)
        {
            localA[i] = 0;
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            localB[i] = 0;
        }
        for (int box_id = 0; box_id < bin_size; ++box_id)
        {
            if (point_box_dist(box_mins[index(box_id, 0, 3)], 
                            box_mins[index(box_id, 1, 3)],
                            box_mins[index(box_id, 2, 3)],
                            box_maxs[index(box_id, 0, 3)],
                            box_maxs[index(box_id, 1, 3)],
                            box_maxs[index(box_id, 2, 3)],
                            pi) <= H)
            {
                for (unsigned int i = box_range[box_id]; i < box_range[box_id+1]; ++i)
                {
                    cx = constraint_points[index(i, 0, 3)];
                    cy = constraint_points[index(i, 1, 3)];
                    cz = constraint_points[index(i, 2, 3)];
                    r = point_point_dist(cx, cy, cz, pi);
                    if (r <= H)
                    {
                        ++ count;
                        ww = wendland(r);
                        cv = constraint_values[i];
                        // TODO: no need to atomic add
                        localA[0] += ww;
                        localA[1] += ww*cx;
                        localA[2] += ww*cy;
                        localA[3] += ww*cz;
                        localA[4] += ww*cx;
                        localA[5] += ww*cx*cx;
                        localA[6] += ww*cy*cx;
                        localA[7] += ww*cz*cx;
                        localA[8] += ww*cy;
                        localA[9] += ww*cx*cy;
                        localA[10] += ww*cy*cy;
                        localA[11] += ww*cz*cy;
                        localA[12] += ww*cz;
                        localA[13] += ww*cx*cz;
                        localA[14] += ww*cy*cz;
                        localA[15] += ww*cz*cz;
                        localB[0] += ww*cv;
                        localB[1] += ww*cv*cx;
                        localB[2] += ww*cv*cy;
                        localB[3] += ww*cv*cz;

                    }
                }
            }
        }
        // neighbors_count[pid] = count;
        if (count >= 8)
        {
            #pragma unroll
            for (int i = 0; i < 16; ++i)
            {
                As[index(pid, i, 16)] = localA[i];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                Bs[index(pid, i, 4)] = localB[i];
            }
        } else {
            As[index(pid, 0, 16)] = 1;
            As[index(pid, 5, 16)] = 1;
            As[index(pid, 10, 16)] = 1;
            As[index(pid, 15, 16)] = 1;
            Bs[index(pid, 0, 4)] = 100;
        }
        
    }
}

__global__ void evaluate_implicit_function(double *odata, double *sampling_points, double *xs)
{
    unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < NX*NY*NZ)
    {
        odata[id] = xs[index(id, 0, 4)] + 
                    xs[index(id, 1, 4)] * sampling_points[index(id, 0, 3)] + 
                    xs[index(id, 2, 4)] * sampling_points[index(id, 1, 3)] + 
                    xs[index(id, 3, 4)] * sampling_points[index(id, 2, 3)];
    }
}

void evaluate_with_mls(double *X, double *FX, double *CP, double *CV, int *grid_range, double *grid_mins, double *grid_maxs, double3 bb_min, double3 bb_max)
{

    // https://docs.nvidia.com/cuda/cusolver/index.html#csrqrbatch-example1
    cusolverSpHandle_t cusolverH = NULL;
    csrqrInfo_t info = NULL;
    cusparseMatDescr_t descrA = NULL;

    cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;

    int *d_csrRowPtrA = NULL;
    int *d_csrColIndA = NULL;
    double *d_csrValA = NULL;
    double *d_b = NULL; // batchSize * m
    double *d_x = NULL; // batchSize * m

    size_t size_qr = 0;
    size_t size_internal = 0;
    void *buffer_qr = NULL; // working space for numerical factorization

    const int m = 4;
    const int nnzA = 16;
    const int csrRowPtrA[m+1]  = { 1, 5, 9, 13, 17};
    const int csrColIndA[nnzA] = { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
    const int batchSize = sampling_size;
    
    // create cusolver handle, qr info and matrix descriptor
    cusolver_status = cusolverSpCreate(&cusolverH);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cusparse_status = cusparseCreateMatDescr(&descrA); 
    assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE); // base-1

    cusolver_status = cusolverSpCreateCsrqrInfo(&info);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    

    double *X_device;
    cudaMalloc((void **)&X_device, sizeof(double)*sampling_size*3);
    double3 grid_size = make_double3((bb_max.x-bb_min.x)/(NX-1), 
                                  (bb_max.y-bb_min.y)/(NY-1),
                                  (bb_max.z-bb_min.z)/(NZ-1));
    create_sampling_points<<<ceil((double)sampling_size/256), 256>>>(X_device, bb_min, grid_size);

    cudaStat1 = cudaMalloc ((void**)&d_csrValA   , sizeof(double) * nnzA * batchSize);
    cudaStat2 = cudaMalloc ((void**)&d_csrColIndA, sizeof(int) * nnzA);
    cudaStat3 = cudaMalloc ((void**)&d_csrRowPtrA, sizeof(int) * (m+1));
    cudaStat4 = cudaMalloc ((void**)&d_b         , sizeof(double) * m * batchSize);
    cudaStat5 = cudaMalloc ((void**)&d_x         , sizeof(double) * m * batchSize);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);
    assert(cudaStat5 == cudaSuccess);

    cudaStat2 = cudaMemcpy(d_csrColIndA, csrColIndA, sizeof(int) * nnzA, cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(d_csrRowPtrA, csrRowPtrA, sizeof(int) * (m+1), cudaMemcpyHostToDevice);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);

    
    cudaDeviceSynchronize();
    // polynomial degree is 1
    // double *As, *xs, *Bs;
    cudaMemset(d_b, 0, sizeof(double)*m*batchSize);
    cudaMemset(d_csrValA, 0, sizeof(double)* nnzA * batchSize);
    set_up_linear_systems<<<ceil((double)sampling_size/256), 256>>>(d_csrValA, d_b, X_device, CP, CV, grid_range,
                                    grid_mins, grid_maxs);

    // step 4: symbolic analysis
    cusolver_status = cusolverSpXcsrqrAnalysisBatched(
        cusolverH, m, m, nnzA,
        descrA, d_csrRowPtrA, d_csrColIndA,
        info);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    
    // step 5: find proper batch size
    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaStat1 = cudaMemGetInfo( &free_mem, &total_mem );
    assert( cudaSuccess == cudaStat1 );

    int batchSizeMax = 2;
    while(batchSizeMax/2 < batchSize){
        // printf("batchSizeMax = %d\n", batchSizeMax);
        cusolver_status = cusolverSpDcsrqrBufferInfoBatched(
            cusolverH, m, m, nnzA,
            // d_csrValA is don't care 
            descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
            batchSizeMax, // WARNING: use batchSizeMax
            info,
            &size_internal,
            &size_qr);
        assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

        if ( (size_internal + size_qr) > free_mem ){ 
            // current batchSizeMax exceeds hardware limit, so cut it by half. 
            batchSizeMax /= 2; break; 
        } 
        batchSizeMax *= 2; // double batchSizMax and try it again. 
    }
    // correct batchSizeMax such that it is not greater than batchSize. 
    batchSizeMax = imin(batchSizeMax, batchSize);
    // printf("batchSizeMax = %d\n", batchSizeMax);

    // step 6: prepare working space
    cusolver_status = cusolverSpDcsrqrBufferInfoBatched(
         cusolverH, m, m, nnzA,
         // d_csrValA is don't care 
         descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
         batchSizeMax, // WARNING: use batchSizeMax
         info,
         &size_internal,
         &size_qr);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    printf("numerical factorization needs internal data %lld bytes\n", (long long)size_internal);      
    printf("numerical factorization needs working space %lld bytes\n", (long long)size_qr);      

    cudaStat1 = cudaMalloc((void**)&buffer_qr, size_qr);
    assert(cudaStat1 == cudaSuccess);

    // step 7: solve linear systems
    cudaDeviceSynchronize();
    for(int idx = 0 ; idx < batchSize; idx += batchSizeMax){
        // current batchSize 'cur_batchSize' is the batchSize used in numerical factorization
        const int cur_batchSize = imin(batchSizeMax, batchSize - idx);
        // printf("current batchSize = %d\n", cur_batchSize);
        // solve part of Aj*xj = bj 
        cusolver_status = cusolverSpDcsrqrsvBatched(
            cusolverH, m, m, nnzA,
            descrA, d_csrValA+idx*nnzA, d_csrRowPtrA, d_csrColIndA,
            d_b+idx*m, d_x+idx*m,
            cur_batchSize, // WARNING: use current batchSize
            info,
            buffer_qr);
        assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
        // // copy part of xj back to host
        // cudaStat1 = cudaMemcpy(xBatch + idx*m, d_x, 
        //     sizeof(double) * m * cur_batchSize, cudaMemcpyDeviceToHost);
        // assert(cudaStat1 == cudaSuccess);
    }
    
    // cusolver_status = cusolverSpDcsrqrsvBatched(
    //     cusolverH, m, m, nnzA,
    //     descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
    //     d_b, d_x,
    //     batchSize,
    //     info,
    //     buffer_qr);
    // printf("%s\n", _cudaGetErrorEnum(cusolver_status));
    // assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    
    cudaFree(d_csrValA);
    cudaFree(d_csrRowPtrA);
    cudaFree(d_csrColIndA);
    cudaFree(d_b);
    cudaFree(buffer_qr);

    double *Fx_device;
    cudaMalloc((void **)&Fx_device, sizeof(double)*sampling_size);

    evaluate_implicit_function<<<ceil((double)batchSize/128), 128>>>(Fx_device, X_device, d_x);
    cudaMemcpy(X, X_device, sizeof(double)*batchSize*3, cudaMemcpyDeviceToHost);
    cudaMemcpy(FX, Fx_device, sizeof(double)*batchSize, cudaMemcpyDeviceToHost);

    cudaFree(X_device);
    cudaFree(Fx_device);
    cusolverSpDestroy(cusolverH);

}

// __host__ void export_result(double *X, double *FX, const char *res_name)
// {
//     Eigen::MatrixXd GV = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(X, sampling_size, 3);
//     Eigen::VectorXd S = Eigen::Map<Eigen::VectorXd>(FX, sampling_size);
//     Eigen::MatrixXd V, F;
//     igl::marching_cubes(S, GV, NX, NY, NZ, 0.0, V, F);
//     igl::writeOFF(res_name, V, F);
// }

int main(int argc, char * argv[])
{
    if (argc < 2)
    {
        printf("usage: ./mls mesh.off [result.off]\n");
        exit(1);
    }
    vector<double> vertices, normals;
    double* CP, * CV;
    double* grid_mins, * grid_maxs;
    double3 bb_min, bb_max;
    int* grid_range;
    int num_of_vertices;
    if (!read_off(argv[1], vertices, normals))
    {
        exit(1);
    }
    // printf("#V: %d\n", num_of_vertices);
    int num_of_vertices = vertices.size();

    // to measure time taken by a specific part of the code 
    set_up_constraints(&CP, &CV, &grid_range, &grid_mins, &grid_maxs, &bb_min, &bb_max, vertices.data(), normals.data(), num_of_vertices);

    // printf("%lf %lf %lf\n", bb_min.x, bb_min.y, bb_min.z);
    // printf("%lf %lf %lf\n", bb_max.x, bb_max.y, bb_max.z);
    double* X, * FX;
    X = (double*)malloc(sizeof(double) * sampling_size * 3);
    FX = (double*)malloc(sizeof(double) * sampling_size);
    evaluate_with_mls(X, FX, CP, CV, grid_range, grid_mins, grid_maxs, bb_min, bb_max);
    // for (int i = 0; i < sampling_size; ++i)
    // {
    //     printf("%lf %lf %lf\n", X[i*3], X[i*3+1], X[i*3+2]);
    // } 
    // for (int x = 0; x < NX; ++x)
    // {
    //     for (int y = 0; y < NY; ++y)
    //     {
    //         for (int z = 0; z < NZ; ++z)
    //         {
    //             int i = x + y * NX + z * NX * NY;
    //             printf("(%lf %lf %lf): %lf\n", X[i*3], X[i*3+1], X[i*3+2], FX[i]);
    //         }
    //     }
    // }
    
    // if (argc == 3)
    //     export_result(X, FX, argv[2]);
    return 0;
}
