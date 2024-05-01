#define _USE_MATH_DEFINES
#include <omp.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <array>
#include <tuple>
#include <cassert>

#include "parallel_cuda_sift.hpp"
#include "parallel_cuda_image.hpp"

using namespace std;
using namespace parallel_cuda_image;


namespace parallel_cuda_sift {

ScaleSpacePyramid generate_gaussian_pyramid(const Image& img, float sigma_min,
                                            int num_octaves, int scales_per_octave)
{
    // assume initial sigma is 1.0 (after resizing) and smooth
    // the image with sigma_diff to reach requried base_sigma
    float base_sigma = sigma_min / MIN_PIX_DIST;
    Image base_img = img.resize(img.width*2, img.height*2, Interpolation::BILINEAR);
    float sigma_diff = std::sqrt(base_sigma*base_sigma - 1.0f);
    base_img = gaussian_blur(base_img, sigma_diff);

    int imgs_per_octave = scales_per_octave + 3;

    // determine sigma values for bluring
    float k = std::pow(2, 1.0/scales_per_octave);
    std::vector<float> sigma_vals {base_sigma};
    for (int i = 1; i < imgs_per_octave; i++) {
        float sigma_prev = base_sigma * std::pow(k, i-1);
        float sigma_total = k * sigma_prev;
        sigma_vals.push_back(std::sqrt(sigma_total*sigma_total - sigma_prev*sigma_prev));
    }

    // create a scale space pyramid of gaussian images
    // images in each octave are half the size of images in the previous one
    ScaleSpacePyramid pyramid = {
        num_octaves,
        imgs_per_octave,
        std::vector<std::vector<Image>>(num_octaves)
    };
    
    // can't do parallelization here, since the current octave
    // depends on the previous octave (resize)
    for (int i = 0; i < num_octaves; i++) {
        pyramid.octaves[i].reserve(imgs_per_octave);
        pyramid.octaves[i].push_back(std::move(base_img));
        // can't do parallelization here, since the current image
        // depends on the previous image (gaussian blur)
        for (int j = 1; j < sigma_vals.size(); j++) {
            const Image& prev_img = pyramid.octaves[i].back();
            pyramid.octaves[i].push_back(gaussian_blur(prev_img, sigma_vals[j]));
        }
        // prepare base image for next octave
        const Image& next_base_img = pyramid.octaves[i][imgs_per_octave-3];
        base_img = next_base_img.resize(next_base_img.width/2, next_base_img.height/2,
                                        Interpolation::NEAREST);
    }
    return pyramid;
}

// generate pyramid of difference of gaussians (DoG) images
ScaleSpacePyramid generate_dog_pyramid(const ScaleSpacePyramid& img_pyramid)
{
    ScaleSpacePyramid dog_pyramid = {
        img_pyramid.num_octaves,
        img_pyramid.imgs_per_octave - 1,
        std::vector<std::vector<Image>>(img_pyramid.num_octaves)
    };
    for (int i = 0; i < dog_pyramid.num_octaves; i++) {
        dog_pyramid.octaves[i].reserve(dog_pyramid.imgs_per_octave);
        // can't do parallelization here, since the current image
        // depends on the previous image
        for (int j = 1; j < img_pyramid.imgs_per_octave; j++) {
            Image diff = img_pyramid.octaves[i][j];
            // #pragma omp parallel for num_threads(16) 
            // increasing the processing time, cost of parallelization is high
            for (int pix_idx = 0; pix_idx < diff.size; pix_idx++) {
                diff.data[pix_idx] -= img_pyramid.octaves[i][j-1].data[pix_idx];
            }
            dog_pyramid.octaves[i].push_back(diff);
        }
    }

    return dog_pyramid;
}

/* bool point_is_extremum(const std::vector<Image>& octave, int scale, int x, int y)
{
    const Image& img = octave[scale];
    const Image& prev = octave[scale-1];
    const Image& next = octave[scale+1];

    bool is_min = true, is_max = true;
    float val = img.get_pixel(x, y, 0), neighbor;

    for (int dx : {-1,0,1}) {
        for (int dy : {-1,0,1}) {
            neighbor = prev.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            neighbor = next.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            neighbor = img.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            if (!is_min && !is_max) return false;
        }
    }
    return true;
}
 */

__device__ float dev_get_pixel(const Image& img, int x, int y, int c) {
    if (x < 0)
        x = 0;
    if (x >= img.width)
        x = img.width - 1;
    if (y < 0)
        y = 0;
    if (y >= img.height)
        y = img.height - 1;
        return img.data[c*img.width*img.height + y*img.width + x];
}

__device__ bool point_is_extremum(const Image* octave, int scale, int x, int y)
{
    const Image& img = octave[scale];
    const Image& prev = octave[scale-1];
    const Image& next = octave[scale+1];

    bool is_min = true, is_max = true;
    float val = dev_get_pixel(img, x, y, 0), neighbor;

    for (int dx : {-1,0,1}) {
        for (int dy : {-1,0,1}) {
            neighbor = dev_get_pixel(prev, x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            neighbor = dev_get_pixel(next, x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            neighbor = dev_get_pixel(img, x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            if (!is_min && !is_max) return false;
        }
    }
    return true;
}


// fit a quadratic near the discrete extremum,
// update the keypoint (interpolated) extremum value
// and return offsets of the interpolated extremum from the discrete extremum

/* std::tuple<float, float, float> fit_quadratic(Keypoint& kp,
                                              const std::vector<Image>& octave,
                                              int scale)
{
    const Image& img = octave[scale];
    const Image& prev = octave[scale-1];
    const Image& next = octave[scale+1];

    float g1, g2, g3;
    float h11, h12, h13, h22, h23, h33;
    int x = kp.i, y = kp.j;

    // gradient 
    g1 = (next.get_pixel(x, y, 0) - prev.get_pixel(x, y, 0)) * 0.5;
    g2 = (img.get_pixel(x+1, y, 0) - img.get_pixel(x-1, y, 0)) * 0.5;
    g3 = (img.get_pixel(x, y+1, 0) - img.get_pixel(x, y-1, 0)) * 0.5;

    // hessian
    h11 = next.get_pixel(x, y, 0) + prev.get_pixel(x, y, 0) - 2*img.get_pixel(x, y, 0);
    h22 = img.get_pixel(x+1, y, 0) + img.get_pixel(x-1, y, 0) - 2*img.get_pixel(x, y, 0);
    h33 = img.get_pixel(x, y+1, 0) + img.get_pixel(x, y-1, 0) - 2*img.get_pixel(x, y, 0);
    h12 = (next.get_pixel(x+1, y, 0) - next.get_pixel(x-1, y, 0)
          -prev.get_pixel(x+1, y, 0) + prev.get_pixel(x-1, y, 0)) * 0.25;
    h13 = (next.get_pixel(x, y+1, 0) - next.get_pixel(x, y-1, 0)
          -prev.get_pixel(x, y+1, 0) + prev.get_pixel(x, y-1, 0)) * 0.25;
    h23 = (img.get_pixel(x+1, y+1, 0) - img.get_pixel(x+1, y-1, 0)
          -img.get_pixel(x-1, y+1, 0) + img.get_pixel(x-1, y-1, 0)) * 0.25;
    
    // invert hessian
    float hinv11, hinv12, hinv13, hinv22, hinv23, hinv33;
    float det = h11*h22*h33 - h11*h23*h23 - h12*h12*h33 + 2*h12*h13*h23 - h13*h13*h22;
    hinv11 = (h22*h33 - h23*h23) / det;
    hinv12 = (h13*h23 - h12*h33) / det;
    hinv13 = (h12*h23 - h13*h22) / det;
    hinv22 = (h11*h33 - h13*h13) / det;
    hinv23 = (h12*h13 - h11*h23) / det;
    hinv33 = (h11*h22 - h12*h12) / det;

    // find offsets of the interpolated extremum from the discrete extremum
    float offset_s = -hinv11*g1 - hinv12*g2 - hinv13*g3;
    float offset_x = -hinv12*g1 - hinv22*g2 - hinv23*g3;
    float offset_y = -hinv13*g1 - hinv23*g3 - hinv33*g3;

    float interpolated_extrema_val = img.get_pixel(x, y, 0)
                                   + 0.5*(g1*offset_s + g2*offset_x + g3*offset_y);
    kp.extremum_val = interpolated_extrema_val;
    return std::make_tuple(offset_s, offset_x, offset_y);
} */

__device__ Offsets fit_quadratic(Keypoint& kp, const Image* octave, int scale)
{
    Offsets kp_offsets;
    const Image& img = octave[scale];
    const Image& prev = octave[scale-1];
    const Image& next = octave[scale+1];

    float g1, g2, g3;
    float h11, h12, h13, h22, h23, h33;
    int x = kp.i, y = kp.j;

    // Gradient computation
    g1 = (dev_get_pixel(next, x, y, 0) - dev_get_pixel(prev, x, y, 0)) * 0.5f;
    g2 = (dev_get_pixel(img, x + 1, y, 0) - dev_get_pixel(img, x - 1, y, 0)) * 0.5f;
    g3 = (dev_get_pixel(img, x, y + 1, 0) - dev_get_pixel(img, x, y - 1, 0)) * 0.5f;

    // Hessian matrix computation
    h11 = dev_get_pixel(next, x, y, 0) + dev_get_pixel(prev, x, y, 0) - 2 * dev_get_pixel(img, x, y, 0);
    h22 = dev_get_pixel(img, x + 1, y, 0) + dev_get_pixel(img, x - 1, y, 0) - 2 * dev_get_pixel(img, x, y, 0);
    h33 = dev_get_pixel(img, x, y + 1, 0) + dev_get_pixel(img, x, y - 1, 0) - 2 * dev_get_pixel(img, x, y, 0);
    h12 = (dev_get_pixel(next, x + 1, y, 0) - dev_get_pixel(next, x - 1, y, 0)
          - dev_get_pixel(prev, x + 1, y, 0) + dev_get_pixel(prev, x - 1, y, 0)) * 0.25f;
    h13 = (dev_get_pixel(next, x, y + 1, 0) - dev_get_pixel(next, x, y - 1, 0)
          - dev_get_pixel(prev, x, y + 1, 0) + dev_get_pixel(prev, x, y - 1, 0)) * 0.25f;
    h23 = (dev_get_pixel(img, x + 1, y + 1, 0) - dev_get_pixel(img, x + 1, y - 1, 0)
          - dev_get_pixel(img, x - 1, y + 1, 0) + dev_get_pixel(img, x - 1, y - 1, 0)) * 0.25f;
    
    // Inverse Hessian computation
    float det = h11 * h22 * h33 - h11 * h23 * h23 - h12 * h12 * h33 + 2 * h12 * h13 * h23 - h13 * h13 * h22;
    float hinv11 = (h22 * h33 - h23 * h23) / det;
    float hinv12 = (h13 * h23 - h12 * h33) / det;
    float hinv13 = (h12 * h23 - h13 * h22) / det;
    float hinv22 = (h11 * h33 - h13 * h13) / det;
    float hinv23 = (h12 * h13 - h11 * h23) / det;
    float hinv33 = (h11 * h22 - h12 * h12) / det;

    // Calculate offset from the discrete extremum
    kp_offsets.s = -hinv11 * g1 - hinv12 * g2 - hinv13 * g3;
    kp_offsets.x = -hinv12 * g1 - hinv22 * g2 - hinv23 * g3;
    kp_offsets.y = -hinv13 * g1 - hinv23 * g2 - hinv33 * g3;

    float interpolated_extrema_val = dev_get_pixel(img, x, y, 0)
                                   + 0.5f * (g1 * kp_offsets.s + g2 * kp_offsets.x+ g3 * kp_offsets.y);
    kp.extremum_val = interpolated_extrema_val;

    return kp_offsets;
}

/* bool point_is_on_edge(const Keypoint& kp, const std::vector<Image>& octave, float edge_thresh=C_EDGE)
{
    const Image& img = octave[kp.scale];
    float h11, h12, h22;
    int x = kp.i, y = kp.j;
    h11 = img.get_pixel(x+1, y, 0) + img.get_pixel(x-1, y, 0) - 2*img.get_pixel(x, y, 0);
    h22 = img.get_pixel(x, y+1, 0) + img.get_pixel(x, y-1, 0) - 2*img.get_pixel(x, y, 0);
    h12 = (img.get_pixel(x+1, y+1, 0) - img.get_pixel(x+1, y-1, 0)
          -img.get_pixel(x-1, y+1, 0) + img.get_pixel(x-1, y-1, 0)) * 0.25;

    float det_hessian = h11*h22 - h12*h12;
    float tr_hessian = h11 + h22;
    float edgeness = tr_hessian*tr_hessian / det_hessian;
    if (edgeness > std::pow(edge_thresh+1, 2)/edge_thresh)
        return true;
    else
        return false;
} */

__device__ bool point_is_on_edge(const Keypoint& kp, const Image* octave, float edge_thresh = C_EDGE)
{
    const Image& img = octave[kp.scale];
    float h11, h12, h22;
    int x = kp.i, y = kp.j;

    // Second derivative computation using device-specific pixel access
    h11 = dev_get_pixel(img, x+1, y, 0) + dev_get_pixel(img, x-1, y, 0) - 2 * dev_get_pixel(img, x, y, 0);
    h22 = dev_get_pixel(img, x, y+1, 0) + dev_get_pixel(img, x, y-1, 0) - 2 * dev_get_pixel(img, x, y, 0);
    h12 = (dev_get_pixel(img, x+1, y+1, 0) - dev_get_pixel(img, x+1, y-1, 0)
          - dev_get_pixel(img, x-1, y+1, 0) + dev_get_pixel(img, x-1, y-1, 0)) * 0.25;

    // Hessian determinant and trace calculation for edgeness check
    float det_hessian = h11 * h22 - h12 * h12;
    float tr_hessian = h11 + h22;
    float edgeness = tr_hessian * tr_hessian / det_hessian;

    // Edge response check against threshold
    return edgeness > (pow(edge_thresh + 1, 2) / edge_thresh);
}

/* void find_input_img_coords(Keypoint& kp, float offset_s, float offset_x, float offset_y,
                                   float sigma_min=SIGMA_MIN,
                                   float min_pix_dist=MIN_PIX_DIST, int n_spo=N_SPO)
{
    kp.sigma = std::pow(2, kp.octave) * sigma_min * std::pow(2, (offset_s+kp.scale)/n_spo);
    kp.x = min_pix_dist * std::pow(2, kp.octave) * (offset_x+kp.i);
    kp.y = min_pix_dist * std::pow(2, kp.octave) * (offset_y+kp.j);
} */

__device__ void find_input_img_coords(Keypoint& kp, float offset_s, float offset_x, float offset_y,
                                      float sigma_min = SIGMA_MIN,
                                      float min_pix_dist = MIN_PIX_DIST, int n_spo = N_SPO)
{
    kp.sigma = powf(2, kp.octave) * sigma_min * powf(2, (offset_s + kp.scale) / n_spo);
    kp.x = min_pix_dist * powf(2, kp.octave) * (offset_x + kp.i);
    kp.y = min_pix_dist * powf(2, kp.octave) * (offset_y + kp.j);
}

/* bool refine_or_discard_keypoint(Keypoint& kp, const std::vector<Image>& octave,
                                 float contrast_thresh, float edge_thresh)
{
    int k = 0;
    bool kp_is_valid = false; 
    while (k++ < MAX_REFINEMENT_ITERS) {
        std::tuple<float, float, float> result = fit_quadratic(kp, octave, kp.scale);
        float offset_s = std::get<0>(result);
        float offset_x = std::get<1>(result);
        float offset_y = std::get<2>(result);

        float max_offset = std::max({std::abs(offset_s),
                                     std::abs(offset_x),
                                     std::abs(offset_y)});
        // find nearest discrete coordinates
        kp.scale += std::round(offset_s);
        kp.i += std::round(offset_x);
        kp.j += std::round(offset_y);
        if (kp.scale >= octave.size()-1 || kp.scale < 1)
            break;

        bool valid_contrast = std::abs(kp.extremum_val) > contrast_thresh;
        if (max_offset < 0.6 && valid_contrast && !point_is_on_edge(kp, octave, edge_thresh)) {
            find_input_img_coords(kp, offset_s, offset_x, offset_y);
            kp_is_valid = true;
            break;
        }
    }
    return kp_is_valid;
} */

__device__ float device_abs(float x) {
    return x < 0 ? -x : x;
}

__device__ float device_max(float a, float b, float c) {
    return fmaxf(a, fmaxf(b, c));
}

__device__ bool refine_or_discard_keypoint(Keypoint& kp, const Image* octave, int num_octaves,
                                           float contrast_thresh, float edge_thresh)
{
    int k = 0;
    bool kp_is_valid = false;
    Offsets kp_offsets;
    while (k++ < MAX_REFINEMENT_ITERS) {
        kp_offsets = fit_quadratic(kp, octave, kp.scale);

        float max_offset = device_max(device_abs(kp_offsets.s), device_abs(kp_offsets.x), device_abs(kp_offsets.y));

        // Update kp values
        int new_scale = kp.scale + roundf(kp_offsets.s);
        int new_i = kp.i + roundf(kp_offsets.x);
        int new_j = kp.j + roundf(kp_offsets.y);
        
        if (new_scale >= num_octaves - 1 || new_scale < 1)
            break;

        kp.scale = new_scale;
        kp.i = new_i;
        kp.j = new_j;

        if (device_abs(kp.extremum_val) > contrast_thresh && max_offset < 0.6 && !point_is_on_edge(kp, octave, edge_thresh)) {
            find_input_img_coords(kp, kp_offsets.s, kp_offsets.x, kp_offsets.y);
            kp_is_valid = true;
            break;
        }
    }
    return kp_is_valid;
}

/* std::vector<Keypoint> find_keypoints(const ScaleSpacePyramid& dog_pyramid, float contrast_thresh,
                                     float edge_thresh)
{
    std::vector<Keypoint> keypoints;
    // #pragma omp parallel for num_threads(16) 
    // increasing the processing time, cost of parallelization is high
    for (int i = 0; i < dog_pyramid.num_octaves; i++) {
        const std::vector<Image>& octave = dog_pyramid.octaves[i];
        // #pragma omp parallel for num_threads(16)
        for (int j = 1; j < dog_pyramid.imgs_per_octave-1; j++) {
            const Image& img = octave[j];
            #pragma omp parallel for collapse(2) num_threads(16)
            for (int x = 1; x < img.width-1; x++) {
                for (int y = 1; y < img.height-1; y++) {
                    if (std::abs(img.get_pixel(x, y, 0)) < 0.8*contrast_thresh) {
                        continue;
                    }
                    if (point_is_extremum(octave, j, x, y)) {
                        Keypoint kp = {x, y, i, j, -1, -1, -1, -1};
                        bool kp_is_valid = refine_or_discard_keypoint(kp, octave, contrast_thresh,
                                                                      edge_thresh);
                        if (kp_is_valid) {
                            #pragma omp critical 
                            {
                                keypoints.push_back(kp);
                            }
                        }
                    }
                }
            }
        }
    }
    return keypoints;
} */

__global__ void detect_keypoints(Image* octave, int octave_index, int img_index, int imgs_per_octave, float contrast_thresh, float edge_thresh, Keypoint* keypoints, int* keypoint_count, int max_keypoints) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < octave[img_index].width - 1 && y < octave[img_index].height - 1) {
        float pixelValue = dev_get_pixel(octave[img_index], x, y, 0);
        if (fabs(pixelValue) < 0.8 * contrast_thresh) {
            return;
        }

        if (point_is_extremum(octave, img_index, x, y)) {
            Keypoint kp = {x, y, octave_index, img_index, -1, -1, -1, -1};
            if (refine_or_discard_keypoint(kp, octave, imgs_per_octave, contrast_thresh, edge_thresh)) {
                int index = atomicAdd(keypoint_count, 1);
                if (index < max_keypoints) {
                    keypoints[index] = kp;
                }
            }
        }
    }
}

__global__ void image_data_ref_copy(Image* dev_octave, const int img_index, float *dev_img)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        dev_octave[img_index].data = dev_img;
    }
}

__global__ void set_device_image_size(Image* dev_octave, const int img_index, const int w, const int h, const int c){
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        dev_octave[img_index].width = w;
        dev_octave[img_index].height = h;
        dev_octave[img_index].channels = c;
        dev_octave[img_index].size = w*h*c;
    }
}

std::vector<Keypoint> find_keypoints(const ScaleSpacePyramid& dog_pyramid, float contrast_thresh,
                                     float edge_thresh)
{
    int* dev_kp_count;
    cudaMalloc(&dev_kp_count, sizeof(int));
    cudaMemset(dev_kp_count, 0, sizeof(int));

    Keypoint* dev_keypoints;
    int max_key_points = 10000;
    cudaMalloc(&dev_keypoints, sizeof(Keypoint) * max_key_points);

    std::vector<Keypoint> keypoints;

    for (int i = 0; i < dog_pyramid.num_octaves; i++) 
    {
        const std::vector<Image>& octave = dog_pyramid.octaves[i];

        Image* dev_octave_img[dog_pyramid.imgs_per_octave];
        float *dev_octave_img_data[dog_pyramid.imgs_per_octave];
        cudaError_t err;

        err = cudaMalloc((void**)&dev_octave_img, dog_pyramid.imgs_per_octave * sizeof(Image));
        if (err != cudaSuccess){
            std::cout<<cudaGetErrorString(err)<<std::endl;
            exit(-1);
        }

        // copy images in an octave to GPU
        for(int img_oct_index = 0; img_oct_index < dog_pyramid.imgs_per_octave; img_oct_index++)
        {
            set_device_image_size<<<1,1>>>(*dev_octave_img, img_oct_index, octave[img_oct_index].width, octave[img_oct_index].height, octave[img_oct_index].channels);
            cudaDeviceSynchronize();
            err = cudaMalloc((void**)&dev_octave_img_data[img_oct_index], octave[img_oct_index].size * sizeof(float));
            if (err != cudaSuccess){
                std::cout<<cudaGetErrorString(err)<<std::endl;
                exit(-1);
            }

            err = cudaMemcpy(dev_octave_img_data[img_oct_index], octave[img_oct_index].data, octave[img_oct_index].size*sizeof(float), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                std::cerr << "cudaMemcpy failed!" << std::endl;
            }
            image_data_ref_copy<<<1,1>>>(*dev_octave_img, img_oct_index, dev_octave_img_data[img_oct_index]);
            cudaDeviceSynchronize();
        }
        for (int j = 1; j < dog_pyramid.imgs_per_octave-1; j++) {
            const Image& img = octave[j];
            dim3 threads_per_block(16, 16);
            dim3 num_blocks((img.width + threads_per_block.x - 1) / threads_per_block.x, 
                    (img.height + threads_per_block.y - 1) / threads_per_block.y);

            detect_keypoints<<<num_blocks, threads_per_block>>>(*dev_octave_img, i, j, dog_pyramid.imgs_per_octave, contrast_thresh, edge_thresh, dev_keypoints, dev_kp_count, max_key_points);
            cudaDeviceSynchronize();
        }
        // freeing GPU space    
        for(int img_oct_index = 0; img_oct_index < dog_pyramid.imgs_per_octave; img_oct_index++) {
            cudaFree(dev_octave_img_data[img_oct_index]);
        }
        cudaFree(dev_octave_img);

    }
    
    int key_point_count = 0;

    cudaError_t cudaStatus = cudaMemcpy(&key_point_count, dev_kp_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed!" << std::endl;
    }
    keypoints.resize(key_point_count);

    cudaStatus = cudaMemcpy(keypoints.data(), dev_keypoints, sizeof(Keypoint) * key_point_count, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed!" << std::endl;
    }

    cudaFree(dev_keypoints);
    cudaFree(dev_kp_count);

    return keypoints;
}

// calculate x and y derivatives for all images in the input pyramid
ScaleSpacePyramid generate_gradient_pyramid(const ScaleSpacePyramid& pyramid)
{
    ScaleSpacePyramid grad_pyramid = {
        pyramid.num_octaves,
        pyramid.imgs_per_octave,
        std::vector<std::vector<Image>>(pyramid.num_octaves)
    };
    // #pragma omp parallel for num_threads(16) 
    // increasing the processing time, cost of parallelization is high
    for (int i = 0; i < pyramid.num_octaves; i++) {
        grad_pyramid.octaves[i].reserve(grad_pyramid.imgs_per_octave);
        int width = pyramid.octaves[i][0].width;
        int height = pyramid.octaves[i][0].height;
        // #pragma omp parallel for num_threads(16)
        for (int j = 0; j < pyramid.imgs_per_octave; j++) {
            Image grad(width, height, 2);
            float gx, gy;
            // #pragma omp parallel for collapse(2) num_threads(16) 
            // increasing the processing time, cost of parallelization is high
            for (int x = 1; x < grad.width-1; x++) {
                for (int y = 1; y < grad.height-1; y++) {
                    gx = (pyramid.octaves[i][j].get_pixel(x+1, y, 0)
                         -pyramid.octaves[i][j].get_pixel(x-1, y, 0)) * 0.5;
                    grad.set_pixel(x, y, 0, gx);
                    gy = (pyramid.octaves[i][j].get_pixel(x, y+1, 0)
                         -pyramid.octaves[i][j].get_pixel(x, y-1, 0)) * 0.5;
                    grad.set_pixel(x, y, 1, gy);
                }
            }
            // #pragma omp critical
            {
                grad_pyramid.octaves[i].push_back(grad);
            }
        }
    }
    return grad_pyramid;
}

// convolve 6x with box filter
__device__ void smooth_histogram(float hist[N_BINS])
{
    float tmp_hist[N_BINS];
    // can't do parallelization here, small number of iterations
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < N_BINS; j++) {
            int prev_idx = (j-1+N_BINS)%N_BINS;
            int next_idx = (j+1)%N_BINS;
            tmp_hist[j] = (hist[prev_idx] + hist[j] + hist[next_idx]) / 3;
        }
        for (int j = 0; j < N_BINS; j++) {
            hist[j] = tmp_hist[j];
        }
    }
}

__device__ int find_keypoint_orientations(Keypoint& kp, 
                                          const Image& img_grad,
                                          float lambda_ori, float lambda_desc,
                                          float* orientations, int max_orientations)
{
    float pix_dist = MIN_PIX_DIST * powf(2, kp.octave);
    // discard kp if too close to image borders 
    float min_dist_from_border = min(min(min(kp.x, kp.y), pix_dist * img_grad.width - kp.x), pix_dist * img_grad.height - kp.y);
    if (min_dist_from_border <= sqrtf(2.0f) * lambda_desc * kp.sigma) {
        return 0;
    }

    float hist[N_BINS] = {0};
    int bin;
    float gx, gy, grad_norm, weight, theta;
    float patch_sigma = lambda_ori * kp.sigma;
    float patch_radius = 3 * patch_sigma;
    int x_start = rintf((kp.x - patch_radius) / pix_dist);
    int x_end = rintf((kp.x + patch_radius) / pix_dist);
    int y_start = rintf((kp.y - patch_radius) / pix_dist);
    int y_end = rintf((kp.y + patch_radius) / pix_dist);

    // accumulate gradients in orientation histogram
    
    // can't do parallelization here, small number of iterations
    for (int x = x_start; x <= x_end; x++) {
        for (int y = y_start; y <= y_end; y++) {
            gx = dev_get_pixel(img_grad, x, y, 0);
            gy = dev_get_pixel(img_grad, x, y, 1);
            grad_norm = sqrtf(gx*gx + gy*gy);
            weight = expf(-((x*pix_dist - kp.x) * (x*pix_dist - kp.x) + (y*pix_dist - kp.y) * (y*pix_dist - kp.y))
                          / (2.0f * patch_sigma * patch_sigma));
            theta = fmodf(atan2f(gy, gx) + 2 * M_PI, 2 * M_PI);
            bin = (int)(rintf(N_BINS / (2 * M_PI) * theta)) % N_BINS;
            hist[bin] += weight * grad_norm;
        }
    }

    smooth_histogram(hist);

    // extract reference orientations
    float ori_thresh = 0.8f, ori_max = 0.0f;
    int num_orientations = 0;
    for (int j = 0; j < N_BINS; j++) {
        if (hist[j] > ori_max) {
            ori_max = hist[j];
        }
    }
    for (int j = 0; j < N_BINS; j++) {
        if (hist[j] >= ori_thresh * ori_max) {
            float prev = hist[(j - 1 + N_BINS) % N_BINS];
            float next = hist[(j + 1) % N_BINS];
            if (prev > hist[j] || next > hist[j])
                continue;
            float theta = 2 * M_PI * (j + 1) / N_BINS + M_PI / N_BINS * (prev - next) / (prev - 2 * hist[j] + next);
            if (num_orientations < max_orientations) {
                orientations[num_orientations++] = theta;
            }
        }
    }
    return num_orientations;
}

__device__ void update_histograms(float hist[N_HIST][N_HIST][N_ORI], float x, float y,
                       float contrib, float theta_mn, float lambda_desc)
{
    float x_i, y_j;
    // can't do parallelization here, small number of iterations
    for (int i = 1; i <= N_HIST; i++) {
        x_i = (i - (1 + (float)N_HIST) / 2) * 2 * lambda_desc / N_HIST;
        if (fabs(x_i - x) > 2 * lambda_desc / N_HIST)
            continue;
        for (int j = 1; j <= N_HIST; j++) {
            y_j = (j - (1 + (float)N_HIST) / 2) * 2 * lambda_desc / N_HIST;
            if (fabs(y_j - y) > 2 * lambda_desc / N_HIST)
                continue;

            float hist_weight = (1 - N_HIST * 0.5 / lambda_desc * fabs(x_i - x))
                                * (1 - N_HIST * 0.5 / lambda_desc * fabs(y_j - y));

            for (int k = 1; k <= N_ORI; k++) {
                float theta_k = 2 * M_PI * (k - 1) / N_ORI;
                float theta_diff = fmodf(theta_k - theta_mn + 2 * M_PI, 2 * M_PI);
                if (fabs(theta_diff) >= 2 * M_PI / N_ORI)
                    continue;
                float bin_weight = 1 - N_ORI * 0.5 / M_PI * fabs(theta_diff);
                hist[i-1][j-1][k-1] += hist_weight * bin_weight * contrib;
            }
        }
    }
}

__device__ void hists_to_vec(float histograms[N_HIST][N_HIST][N_ORI], int* feature_vec)
{
    int size = N_HIST * N_HIST * N_ORI;
    float *hist = reinterpret_cast<float *>(histograms);

    float norm = 0;
    for (int i = 0; i < size; i++) {
        norm += hist[i] * hist[i];
    }
    norm = sqrtf(norm);

    float norm2 = 0;
    for (int i = 0; i < size; i++) {
        hist[i] = min(hist[i], 0.2f * norm);
        norm2 += hist[i] * hist[i];
    }
    norm2 = sqrtf(norm2);

    for (int i = 0; i < size; i++) {
        float val = floorf(512 * hist[i] / norm2);
        feature_vec[i] = min((int)val, 255);
    }
}


__device__ void compute_keypoint_descriptor(Keypoint& kp, float theta,
                                 const Image& img_grad,
                                 float lambda_desc)
{
    float pix_dist = MIN_PIX_DIST * powf(2, kp.octave);
    float histograms[N_HIST][N_HIST][N_ORI] = {{{0}}};

    //find start and end coords for loops over image patch
    float half_size = sqrtf(2) * lambda_desc * kp.sigma * (N_HIST + 1) / N_HIST;
    int x_start = rintf((kp.x - half_size) / pix_dist);
    int x_end = rintf((kp.x + half_size) / pix_dist);
    int y_start = rintf((kp.y - half_size) / pix_dist);
    int y_end = rintf((kp.y + half_size) / pix_dist);

    float cos_t = cosf(theta), sin_t = sinf(theta);
    float patch_sigma = lambda_desc * kp.sigma;
    // accumulate samples into histograms
    // can't do parallelization here, small number of iterations
    for (int m = x_start; m <= x_end; m++) {
        for (int n = y_start; n <= y_end; n++) {
            // find normalized coords w.r.t. kp position and reference orientation
            float x = ((m * pix_dist - kp.x) * cos_t + (n * pix_dist - kp.y) * sin_t) / kp.sigma;
            float y = (-(m * pix_dist - kp.x) * sin_t + (n * pix_dist - kp.y) * cos_t) / kp.sigma;

            if (max(fabs(x), fabs(y)) > lambda_desc * (N_HIST + 1.) / N_HIST)
                continue;

            float gx = dev_get_pixel(img_grad, m, n, 0), gy = dev_get_pixel(img_grad, m, n, 1);
            float theta_mn = fmodf(atan2f(gy, gx) - theta + 4 * M_PI, 2 * M_PI);
            float grad_norm = sqrtf(gx * gx + gy * gy);
            float weight = expf(-(powf(m * pix_dist - kp.x, 2) + powf(n * pix_dist - kp.y, 2))
                                    / (2 * patch_sigma * patch_sigma));
            float contribution = weight * grad_norm;

            update_histograms(histograms, x, y, contribution, theta_mn, lambda_desc);
        }
    }

    // build feature vector (descriptor) from histograms
    hists_to_vec(histograms, kp.descriptor);
}

__global__ void set_device_image_size(Image* img, int width, int height, int channels) {
    img->width = width;
    img->height = height;
    img->channels = channels;
    img->size = width * height * channels;
}

__global__ void image_data_ref_copy(Image* img, float* data) {
    img->data = data;
}

__global__ void process_keypoints(Keypoint *tmp_kps, int num_kps, Image** grad_pyramid, int imgs_per_octave,
                                  float lambda_ori, float lambda_desc, Keypoint *output_kps, int *output_index) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_kps) return;

    const Image& img_grad = *grad_pyramid[tmp_kps[i].octave * imgs_per_octave + tmp_kps[i].scale];

    const int max_orientations = 10;
    float orientations[max_orientations];
    int num_orientations = find_keypoint_orientations(tmp_kps[i], img_grad,
                                                      lambda_ori, lambda_desc,
                                                      orientations, max_orientations);
    // if(i%256 == 0) {
    //     printf("Printing Orientations\n");
    //     for (int j = 0; j < num_orientations; j++) {
    //         printf("%f\n", orientations[j]);
    //     }
    // }
    
    for (int j = 0; j < num_orientations; j++) {
        Keypoint kp = tmp_kps[i];
        compute_keypoint_descriptor(kp, orientations[j], img_grad, lambda_desc);
        // if(i%256 == 0) {
        //     printf("Printing Descriptors\n");
        //     for (int j = 0; j < 128; j++) {
        //         printf("%d\n", kp.descriptor[j]);
        //     }
        // }
        int idx = atomicAdd(output_index, 1);
        output_kps[idx] = kp;
    }
}

void find_and_compute_kp_descriptors(vector<Keypoint>& tmp_kps, vector<Keypoint>& kps, ScaleSpacePyramid& grad_pyramid, float lambda_ori, float lambda_desc) {
    int num_kps = tmp_kps.size();
    Keypoint *device_tmp_kps;
    cudaMalloc(&device_tmp_kps, sizeof(Keypoint) * num_kps);
    cudaMemcpy(device_tmp_kps, tmp_kps.data(), sizeof(Keypoint) * num_kps, cudaMemcpyHostToDevice);

    int num_octaves = grad_pyramid.num_octaves;
    int imgs_per_octave = grad_pyramid.imgs_per_octave;

    Image** dev_grad_pyramid;
    cudaMalloc(&dev_grad_pyramid, num_octaves * imgs_per_octave * sizeof(Image*));

    float** dev_img_data;
    cudaMalloc(&dev_img_data, num_octaves * imgs_per_octave * sizeof(float*));

    for (int i = 0; i < num_octaves; i++) {
        for (int j = 0; j < imgs_per_octave; j++) {
            Image* dev_img;
            cudaMalloc(&dev_img, sizeof(Image));
            float* img_data;
            cudaMalloc(&img_data, grad_pyramid.octaves[i][j].size * sizeof(float));
            cudaMemcpy(img_data, grad_pyramid.octaves[i][j].data, grad_pyramid.octaves[i][j].size * sizeof(float), cudaMemcpyHostToDevice);

            set_device_image_size<<<1, 1>>>(dev_img, grad_pyramid.octaves[i][j].width, grad_pyramid.octaves[i][j].height, grad_pyramid.octaves[i][j].channels);
            cudaDeviceSynchronize();

            image_data_ref_copy<<<1, 1>>>(dev_img, img_data);
            cudaDeviceSynchronize();

            cudaMemcpy(&dev_grad_pyramid[i * imgs_per_octave + j], &dev_img, sizeof(Image*), cudaMemcpyHostToDevice);
            cudaMemcpy(&dev_img_data[i * imgs_per_octave + j], &img_data, sizeof(float*), cudaMemcpyHostToDevice);
        }
    }
    
    int max_key_points = 10000;
    Keypoint *dev_output_kps;
    cudaMalloc(&dev_output_kps, sizeof(Keypoint) * max_key_points);
    int *dev_key_point_count;
    cudaMalloc(&dev_key_point_count, sizeof(int));
    cudaMemset(dev_key_point_count, 0, sizeof(int));

    const int threads_per_block = 256;
    const int num_blocks = (num_kps + threads_per_block - 1) / threads_per_block;
    process_keypoints<<<num_blocks, threads_per_block>>>(device_tmp_kps, num_kps, dev_grad_pyramid, imgs_per_octave,
                                                         lambda_ori, lambda_desc, dev_output_kps, dev_key_point_count);
    cudaDeviceSynchronize();

    // after kernel execution, copy the keypoints back to host
    int key_point_count = 0;
    cudaMemcpy(&key_point_count, dev_key_point_count, sizeof(int), cudaMemcpyDeviceToHost);

    kps.resize(key_point_count);
    
    cudaMemcpy(kps.data(), dev_output_kps, sizeof(Keypoint) * key_point_count, cudaMemcpyDeviceToHost);

    cudaFree(device_tmp_kps);
    cudaFree(dev_output_kps);
    cudaFree(dev_key_point_count);
    for (int i = 0; i < num_octaves; i++) {
        for (int j = 0; j < imgs_per_octave; j++) {
            float* img_data;
            cudaMemcpy(&img_data, &dev_img_data[i * imgs_per_octave + j], sizeof(float*), cudaMemcpyDeviceToHost);
            cudaFree(img_data);
            Image* img;
            cudaMemcpy(&img, &dev_grad_pyramid[i * imgs_per_octave + j], sizeof(Image*), cudaMemcpyDeviceToHost);
            cudaFree(img);
        }
    }
    cudaFree(dev_grad_pyramid);
    cudaFree(dev_img_data);
}

std::vector<Keypoint> find_keypoints_and_descriptors(const Image& img, float sigma_min,
                                                     int num_octaves, int scales_per_octave, 
                                                     float contrast_thresh, float edge_thresh, 
                                                     float lambda_ori, float lambda_desc)
{
    assert(img.channels == 1 || img.channels == 3);

    const Image& input = img.channels == 1 ? img : rgb_to_grayscale(img);
    auto start = std::chrono::high_resolution_clock::now();
    ScaleSpacePyramid gaussian_pyramid = generate_gaussian_pyramid(input, sigma_min, num_octaves,
                                                                   scales_per_octave);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time to generate gaussian pyramid: " << elapsed.count() << "s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    ScaleSpacePyramid dog_pyramid = generate_dog_pyramid(gaussian_pyramid);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time to generate difference of gaussian pyramid: " << elapsed.count() << "s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    std::vector<Keypoint> tmp_kps = find_keypoints(dog_pyramid, contrast_thresh, edge_thresh);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time to find valid keypoints: " << elapsed.count() << "s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid(gaussian_pyramid);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time to generate gradient pyramid: " << elapsed.count() << "s" << std::endl;
    
    std::vector<Keypoint> kps;

    start = std::chrono::high_resolution_clock::now();

    // for (int i = 0; i< tmp_kps.size(); i++) {
    //     std::vector<float> orientations = find_keypoint_orientations(tmp_kps[i], grad_pyramid,
    //                                                                  lambda_ori, lambda_desc);
    //     for (float theta : orientations) {
    //         Keypoint kp = tmp_kps[i];
    //         compute_keypoint_descriptor(kp, theta, grad_pyramid, lambda_desc);
    //         kps.push_back(kp);
    //     }
    // }

    find_and_compute_kp_descriptors(tmp_kps, kps, grad_pyramid, lambda_ori, lambda_desc);

    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time to find key points orientation and compute descriptor: " << elapsed.count() << "s" << std::endl;

    return kps;
}

// float euclidean_dist(std::array<uint8_t, 128>& a, std::array<uint8_t, 128>& b)
// {
//     float dist = 0;
//     for (int i = 0; i < 128; i++) {
//         int di = (int)a[i] - b[i];
//         dist += di * di;
//     }
//     return std::sqrt(dist);
// }

// std::vector<std::pair<int, int>> find_keypoint_matches(std::vector<Keypoint>& a,
//                                                        std::vector<Keypoint>& b,
//                                                        float thresh_relative,
//                                                        float thresh_absolute)
// {
//     assert(a.size() >= 2 && b.size() >= 2);

//     std::vector<std::pair<int, int>> matches;

//     #pragma omp parallel for num_threads(16)
//     for (int i = 0; i < a.size(); i++) {
//         // find two nearest neighbours in b for current keypoint from a
//         int nn1_idx = -1;
//         float nn1_dist = 100000000, nn2_dist = 100000000;
//         // can't do parallelization here, because we are trying to 
//         // find minimum distance across iterations
//         for (int j = 0; j < b.size(); j++) {
//             float dist = euclidean_dist(a[i].descriptor, b[j].descriptor);
//             if (dist < nn1_dist) {
//                 nn2_dist = nn1_dist;
//                 nn1_dist = dist;
//                 nn1_idx = j;
//             } else if (nn1_dist <= dist && dist < nn2_dist) {
//                 nn2_dist = dist;
//             }
//         }
//         if (nn1_dist < thresh_relative*nn2_dist && nn1_dist < thresh_absolute) {
//             #pragma omp critical
//             {
//                 matches.push_back({i, nn1_idx});
//             }
//         }
//     }
//     return matches;
// }

__device__ float euclidean_dist(uint8_t* a, uint8_t* b)
{
    float dist = 0;
    for (int i = 0; i < 128; i++) {
        int di = (int)a[i] - b[i];
        dist += di * di;
    }
    return sqrt(dist);
}

__global__ void find_matches(uint8_t* a_descriptors, uint8_t* b_descriptors, int* matches, int a_size, int b_size, int desc_length, float thresh_relative, float thresh_absolute) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < a_size) {
        int nn1_idx = -1;
        float nn1_dist = 100000000, nn2_dist = 100000000;
        for (int j = 0; j < b_size; j++) {
            float dist = euclidean_dist(&a_descriptors[i * desc_length], &b_descriptors[j * desc_length]);
            if (dist < nn1_dist) {
                nn2_dist = nn1_dist;
                nn1_dist = dist;
                nn1_idx = j;
            } else if (nn1_dist <= dist && dist < nn2_dist) {
                nn2_dist = dist;
            }
        }
        if (nn1_dist < thresh_relative * nn2_dist && nn1_dist < thresh_absolute) {
            int idx = atomicAdd(matches, 1);
            matches[2 * idx + 1] = i;
            matches[2 * idx + 2] = nn1_idx;
        }
    }
}

std::vector<std::pair<int, int>> find_keypoint_matches(std::vector<Keypoint>& a,
                                                       std::vector<Keypoint>& b,
                                                       float thresh_relative,
                                                       float thresh_absolute)
{
    assert(a.size() >= 2 && b.size() >= 2);

    std::vector<std::pair<int, int>> matches;

    int *d_matches;
    int max_size_matches = (a.size() * 2 + 1);
    cudaMalloc(&d_matches, max_size_matches * sizeof(int));
    cudaMemset(d_matches, 0, max_size_matches * sizeof(int));

    uint8_t *d_a_descriptors;
    uint8_t *d_b_descriptors;
    int desc_size = 128;

    cudaMalloc((void**)&d_a_descriptors, a.size() * desc_size * sizeof(uint8_t));
    cudaMalloc((void**)&d_b_descriptors, b.size() * desc_size * sizeof(uint8_t));

    cudaError_t err;
    for(int i=0;i<a.size();i++) {
        err = cudaMemcpy(&d_a_descriptors[i*desc_size], &a[i].descriptor[0], desc_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy failed!" << std::endl;
        }
    }
    
    for(int i=0;i<b.size();i++) {
        err = cudaMemcpy(&d_b_descriptors[i*desc_size], &b[i].descriptor[0], desc_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy failed!" << std::endl;
        }
    }

    int threads_per_block = 256;
    int blocks_per_grid = (a.size() + threads_per_block - 1) / threads_per_block;
    find_matches<<<blocks_per_grid, threads_per_block>>>(d_a_descriptors, d_b_descriptors, d_matches, a.size(), b.size(), desc_size, thresh_relative, thresh_absolute);
    cudaDeviceSynchronize();

    int host_matches[max_size_matches];
    cudaMemcpy(host_matches, d_matches, max_size_matches*sizeof(int), cudaMemcpyDeviceToHost);
    
    // host_matches[0] holds the count 
    for(int i=1; i<host_matches[0]; i+=2)
        matches.emplace_back(host_matches[i], host_matches[i+1]);

    cudaFree(d_matches);
    cudaFree(d_a_descriptors);
    cudaFree(d_b_descriptors);

    return matches;
}

Image draw_keypoints(const Image& img, const std::vector<Keypoint>& kps)
{
    Image res(img);
    if (img.channels == 1) {
        res = grayscale_to_rgb(res);
    }
    for (auto& kp : kps) {
        draw_point(res, kp.x, kp.y, 5);
    }
    return res;
}

Image draw_matches(const Image& a, const Image& b, std::vector<Keypoint>& kps_a,
                   std::vector<Keypoint>& kps_b, std::vector<std::pair<int, int>> matches)
{
    Image res(a.width+b.width, std::max(a.height, b.height), 3);

    for (int i = 0; i < a.width; i++) {
        for (int j = 0; j < a.height; j++) {
            res.set_pixel(i, j, 0, a.get_pixel(i, j, 0));
            res.set_pixel(i, j, 1, a.get_pixel(i, j, a.channels == 3 ? 1 : 0));
            res.set_pixel(i, j, 2, a.get_pixel(i, j, a.channels == 3 ? 2 : 0));
        }
    }
    for (int i = 0; i < b.width; i++) {
        for (int j = 0; j < b.height; j++) {
            res.set_pixel(a.width+i, j, 0, b.get_pixel(i, j, 0));
            res.set_pixel(a.width+i, j, 1, b.get_pixel(i, j, b.channels == 3 ? 1 : 0));
            res.set_pixel(a.width+i, j, 2, b.get_pixel(i, j, b.channels == 3 ? 2 : 0));
        }
    }

    for (auto& m : matches) {
        Keypoint& kp_a = kps_a[m.first];
        Keypoint& kp_b = kps_b[m.second];
        draw_line(res, kp_a.x, kp_a.y, a.width+kp_b.x, kp_b.y);
    }
    return res;
}

} // namespace parallel_cuda_sift
