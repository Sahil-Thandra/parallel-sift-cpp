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
    // #pragma omp parallel for num_threads(8) 
    // increasing the processing time, cost of parallelization is high
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

__global__ void detectKeypoints(Image* octave, int octaveIndex, int imgIndex, int imgsPerOctave, float contrastThresh, float edgeThresh, Keypoint* keypoints, int* keypointCount, int maxKeypoints) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < octave[imgIndex].width - 1 && y < octave[imgIndex].height - 1) {
        float pixelValue = dev_get_pixel(octave[imgIndex], x, y, 0);
        if (fabs(pixelValue) < 0.8 * contrastThresh) {
            return;
        }

        if (point_is_extremum(octave, imgIndex, x, y)) {
            Keypoint kp = {x, y, octaveIndex, imgIndex, -1, -1, -1, -1};
            if (refine_or_discard_keypoint(kp, octave, imgsPerOctave, contrastThresh, edgeThresh)) {
                int index = atomicAdd(keypointCount, 1);
                if (index < maxKeypoints) {
                    keypoints[index] = kp;
                }
            }
        }
    }
}

__global__ void imageDataRefCopy(Image* dev_octave, const int img_index, float *dev_img)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        dev_octave[img_index].data = dev_img;
    }
}

__global__ void setDeviceImageSize(Image* dev_octave, const int img_index, const int w, const int h, const int c){
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
    int maxKeypoints = 10000;
    cudaMalloc(&dev_keypoints, sizeof(Keypoint) * maxKeypoints);

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
        for(int imgOctIndex = 0; imgOctIndex< dog_pyramid.imgs_per_octave; imgOctIndex++)
        {

            setDeviceImageSize<<<1,1>>>(*dev_octave_img, imgOctIndex, octave[imgOctIndex].width, octave[imgOctIndex].height, octave[imgOctIndex].channels);
            cudaDeviceSynchronize();

            err = cudaMalloc((void**)&dev_octave_img_data[imgOctIndex], octave[imgOctIndex].size * sizeof(float));
            if (err != cudaSuccess){
                std::cout<<cudaGetErrorString(err)<<std::endl;
                exit(-1);
            }

            cudaMemcpy(dev_octave_img_data[imgOctIndex], octave[imgOctIndex].data, octave[imgOctIndex].size*sizeof(float), cudaMemcpyHostToDevice);
            imageDataRefCopy<<<1,1>>>(*dev_octave_img, imgOctIndex, dev_octave_img_data[imgOctIndex]);
            cudaDeviceSynchronize();
        }
        for (int j = 1; j < dog_pyramid.imgs_per_octave-1; j++) {
            const Image& img = octave[j];
            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((img.width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                    (img.height + threadsPerBlock.y - 1) / threadsPerBlock.y);

            detectKeypoints<<<numBlocks, threadsPerBlock>>>(*dev_octave_img, i, j, dog_pyramid.imgs_per_octave, contrast_thresh, edge_thresh, dev_keypoints, dev_kp_count, maxKeypoints);
            cudaDeviceSynchronize();
        }
        // freeing GPU space    
        for(int imgOctIndex = 0; imgOctIndex< dog_pyramid.imgs_per_octave; imgOctIndex++) {
            cudaFree(dev_octave_img_data[imgOctIndex]);
        }
        cudaFree(dev_octave_img);

    }
    
    int keyPointCount = 0;

    cudaMemcpy(&keyPointCount, dev_kp_count, sizeof(int), cudaMemcpyDeviceToHost);
    keypoints.resize(keyPointCount);

    cudaError_t cudaStatus = cudaMemcpy(keypoints.data(), dev_keypoints, sizeof(Keypoint) * keyPointCount, cudaMemcpyDeviceToHost);
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
void smooth_histogram(float hist[N_BINS])
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

std::vector<float> find_keypoint_orientations(Keypoint& kp, 
                                              const ScaleSpacePyramid& grad_pyramid,
                                              float lambda_ori, float lambda_desc)
{
    float pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
    const Image& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];

    // discard kp if too close to image borders 
    float min_dist_from_border = std::min({kp.x, kp.y, pix_dist*img_grad.width-kp.x,
                                           pix_dist*img_grad.height-kp.y});
    if (min_dist_from_border <= std::sqrt(2)*lambda_desc*kp.sigma) {
        return {};
    }

    float hist[N_BINS] = {};
    int bin;
    float gx, gy, grad_norm, weight, theta;
    float patch_sigma = lambda_ori * kp.sigma;
    float patch_radius = 3 * patch_sigma;
    int x_start = std::round((kp.x - patch_radius)/pix_dist);
    int x_end = std::round((kp.x + patch_radius)/pix_dist);
    int y_start = std::round((kp.y - patch_radius)/pix_dist);
    int y_end = std::round((kp.y + patch_radius)/pix_dist);

    // accumulate gradients in orientation histogram
    
    // #pragma omp parallel for collapse(2) num_threads(16) 
    // increasing the processing time due to critical region
    for (int x = x_start; x <= x_end; x++) {
        for (int y = y_start; y <= y_end; y++) {
            gx = img_grad.get_pixel(x, y, 0);
            gy = img_grad.get_pixel(x, y, 1);
            grad_norm = std::sqrt(gx*gx + gy*gy);
            weight = std::exp(-(std::pow(x*pix_dist-kp.x, 2)+std::pow(y*pix_dist-kp.y, 2))
                              /(2*patch_sigma*patch_sigma));
            theta = std::fmod(std::atan2(gy, gx)+2*M_PI, 2*M_PI);
            bin = (int)std::round(N_BINS/(2*M_PI)*theta) % N_BINS;
            // #pragma omp critical 
            {
                hist[bin] += weight * grad_norm;
            }
        }
    }

    smooth_histogram(hist);

    // extract reference orientations
    float ori_thresh = 0.8, ori_max = 0;
    std::vector<float> orientations;
    // can't do parallelization here, small number of iterations
    for (int j = 0; j < N_BINS; j++) {
        if (hist[j] > ori_max) {
            ori_max = hist[j];
        }
    }
    // can't do parallelization here, small number of iterations
    for (int j = 0; j < N_BINS; j++) {
        if (hist[j] >= ori_thresh * ori_max) {
            float prev = hist[(j-1+N_BINS)%N_BINS], next = hist[(j+1)%N_BINS];
            if (prev > hist[j] || next > hist[j])
                continue;
            float theta = 2*M_PI*(j+1)/N_BINS + M_PI/N_BINS*(prev-next)/(prev-2*hist[j]+next);
            orientations.push_back(theta);
        }
    }
    return orientations;
}

void update_histograms(float hist[N_HIST][N_HIST][N_ORI], float x, float y,
                       float contrib, float theta_mn, float lambda_desc)
{
    float x_i, y_j;
    // can't do parallelization here, small number of iterations
    for (int i = 1; i <= N_HIST; i++) {
        x_i = (i-(1+(float)N_HIST)/2) * 2*lambda_desc/N_HIST;
        if (std::abs(x_i-x) > 2*lambda_desc/N_HIST)
            continue;
        for (int j = 1; j <= N_HIST; j++) {
            y_j = (j-(1+(float)N_HIST)/2) * 2*lambda_desc/N_HIST;
            if (std::abs(y_j-y) > 2*lambda_desc/N_HIST)
                continue;
            
            float hist_weight = (1 - N_HIST*0.5/lambda_desc*std::abs(x_i-x))
                               *(1 - N_HIST*0.5/lambda_desc*std::abs(y_j-y));

            for (int k = 1; k <= N_ORI; k++) {
                float theta_k = 2*M_PI*(k-1)/N_ORI;
                float theta_diff = std::fmod(theta_k-theta_mn+2*M_PI, 2*M_PI);
                if (std::abs(theta_diff) >= 2*M_PI/N_ORI)
                    continue;
                float bin_weight = 1 - N_ORI*0.5/M_PI*std::abs(theta_diff);
                hist[i-1][j-1][k-1] += hist_weight*bin_weight*contrib;
            }
        }
    }
}

void hists_to_vec(float histograms[N_HIST][N_HIST][N_ORI], std::array<uint8_t, 128>& feature_vec)
{
    int size = N_HIST*N_HIST*N_ORI;
    float *hist = reinterpret_cast<float *>(histograms);

    float norm = 0;
    for (int i = 0; i < size; i++) {
        norm += hist[i] * hist[i];
    }
    norm = std::sqrt(norm);
    float norm2 = 0;
    for (int i = 0; i < size; i++) {
        hist[i] = std::min(hist[i], 0.2f*norm);
        norm2 += hist[i] * hist[i];
    }
    norm2 = std::sqrt(norm2);
    for (int i = 0; i < size; i++) {
        float val = std::floor(512*hist[i]/norm2);
        feature_vec[i] = std::min((int)val, 255);
    }
}

void compute_keypoint_descriptor(Keypoint& kp, float theta,
                                 const ScaleSpacePyramid& grad_pyramid,
                                 float lambda_desc)
{
    float pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
    const Image& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];
    float histograms[N_HIST][N_HIST][N_ORI] = {{{0}}};

    //find start and end coords for loops over image patch
    float half_size = std::sqrt(2)*lambda_desc*kp.sigma*(N_HIST+1.)/N_HIST;
    int x_start = std::round((kp.x-half_size) / pix_dist);
    int x_end = std::round((kp.x+half_size) / pix_dist);
    int y_start = std::round((kp.y-half_size) / pix_dist);
    int y_end = std::round((kp.y+half_size) / pix_dist);

    float cos_t = std::cos(theta), sin_t = std::sin(theta);
    float patch_sigma = lambda_desc * kp.sigma;
    //accumulate samples into histograms
    #pragma omp parallel for collapse(2) num_threads(16)
    for (int m = x_start; m <= x_end; m++) {
        for (int n = y_start; n <= y_end; n++) {
            // find normalized coords w.r.t. kp position and reference orientation
            float x = ((m*pix_dist - kp.x)*cos_t
                      +(n*pix_dist - kp.y)*sin_t) / kp.sigma;
            float y = (-(m*pix_dist - kp.x)*sin_t
                       +(n*pix_dist - kp.y)*cos_t) / kp.sigma;

            // verify (x, y) is inside the description patch
            if (std::max(std::abs(x), std::abs(y)) > lambda_desc*(N_HIST+1.)/N_HIST)
                continue;

            float gx = img_grad.get_pixel(m, n, 0), gy = img_grad.get_pixel(m, n, 1);
            float theta_mn = std::fmod(std::atan2(gy, gx)-theta+4*M_PI, 2*M_PI);
            float grad_norm = std::sqrt(gx*gx + gy*gy);
            float weight = std::exp(-(std::pow(m*pix_dist-kp.x, 2)+std::pow(n*pix_dist-kp.y, 2))
                                    /(2*patch_sigma*patch_sigma));
            float contribution = weight * grad_norm;

            update_histograms(histograms, x, y, contribution, theta_mn, lambda_desc);
        }
    }

    // build feature vector (descriptor) from histograms
    hists_to_vec(histograms, kp.descriptor);
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
    #pragma omp parallel for num_threads(16)
    for (int i = 0; i< tmp_kps.size(); i++) {
        std::vector<float> orientations = find_keypoint_orientations(tmp_kps[i], grad_pyramid,
                                                                     lambda_ori, lambda_desc);
        for (float theta : orientations) {
            Keypoint kp = tmp_kps[i];
            compute_keypoint_descriptor(kp, theta, grad_pyramid, lambda_desc);
            #pragma omp critical
            {
                kps.push_back(kp);
            }
        }
    }
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
            // printf("matches[0]=%d\n", *matches);
            // printf("idx=%d, match %d,%d\n", idx, i, nn1_idx);
            matches[2 * idx + 1] = i;
            matches[2 * idx + 2] = nn1_idx;
        }
    }
}
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

__global__ void imageDesc(uint8_t* d_a_descriptors, int size)
{
    
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("Here at CUDA dec print\n");
        for (int k = 0; k < size; k++) {
            if((k<10) || k>(size-10))
            {
                printf("%d\n", d_a_descriptors[k*128]);
            }       
        }
    } 
}
__global__ void matchPrint(int* match)
{
    printf("Here at CUDA match print\n");
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        for (int k = 0; k < match[0]; k++) {
                printf("aID=%d , bID=%d\n", match[k], match[k+1]);  
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
    int maxSizeMatches = (a.size() * 2 + 1);
    cudaMalloc(&d_matches, maxSizeMatches * sizeof(int));
    cudaMemset(d_matches, 0, maxSizeMatches * sizeof(int));

    uint8_t *d_a_descriptors;
    uint8_t *d_b_descriptors;
    int desc_size = 128;

    cudaMalloc((void**)&d_a_descriptors, a.size() * desc_size * sizeof(uint8_t));
    cudaMalloc((void**)&d_b_descriptors, b.size() * desc_size * sizeof(uint8_t));

    for(int i=0;i<a.size();i++)
        cudaMemcpy(&d_a_descriptors[i*desc_size], &a[i].descriptor[0], desc_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
    
    for(int i=0;i<b.size();i++)
        cudaMemcpy(&d_b_descriptors[i*desc_size], &b[i].descriptor[0], desc_size * sizeof(uint8_t), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (a.size() + threadsPerBlock - 1) / threadsPerBlock;
    find_matches<<<blocksPerGrid, threadsPerBlock>>>(d_a_descriptors, d_b_descriptors, d_matches, a.size(), b.size(), desc_size, thresh_relative, thresh_absolute);
    cudaDeviceSynchronize();

    int hostMatches[maxSizeMatches];
    cudaMemcpy(hostMatches, d_matches, maxSizeMatches*sizeof(int), cudaMemcpyDeviceToHost);
    
    // hostMatches[0] holds the count 
    for(int i=1;i<hostMatches[0];i+=2)
        matches.emplace_back(hostMatches[i], hostMatches[i+1]);

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
