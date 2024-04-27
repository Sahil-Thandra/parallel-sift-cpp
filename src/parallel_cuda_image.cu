#include <omp.h>
#include <cmath>
#include <iostream>
#include <cassert>
#include <utility>

#include "parallel_cuda_image.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace parallel_cuda_image {

Image::Image(std::string file_path)
{
    unsigned char *img_data = stbi_load(file_path.c_str(), &width, &height, &channels, 0);
    if (img_data == nullptr) {
        const char *error_msg = stbi_failure_reason();
        std::cerr << "Failed to load image: " << file_path.c_str() << "\n";
        std::cerr << "Error msg (stb_image): " << error_msg << "\n";
        std::exit(1);
    }

    size = width * height * channels;
    data = new float[size]; 
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < channels; c++) {
                int src_idx = y*width*channels + x*channels + c;
                int dst_idx = c*height*width + y*width + x;
                data[dst_idx] = img_data[src_idx] / 255.;
            }
        }
    }
    if (channels == 4)
        channels = 3; //ignore alpha channel
    stbi_image_free(img_data);
}

Image::Image(int w, int h, int c)
    :width {w},
     height {h},
     channels {c},
     size {w*h*c},
     data {new float[w*h*c]()}
{
}

Image::Image()
    :width {0},
     height {0},
     channels {0},
     size {0},
     data {nullptr} 
{
}

Image::~Image()
{
    delete[] this->data;
}

Image::Image(const Image& other)
    :width {other.width},
     height {other.height},
     channels {other.channels},
     size {other.size},
     data {new float[other.size]}
{
    //std::cout << "copy constructor\n";
    for (int i = 0; i < size; i++)
        data[i] = other.data[i];
}

Image& Image::operator=(const Image& other)
{
    if (this != &other) {
        delete[] data;
        //std::cout << "copy assignment\n";
        width = other.width;
        height = other.height;
        channels = other.channels;
        size = other.size;
        data = new float[other.size];
        for (int i = 0; i < other.size; i++)
            data[i] = other.data[i];
    }
    return *this;
}

Image::Image(Image&& other)
    :width {other.width},
     height {other.height},
     channels {other.channels},
     size {other.size},
     data {other.data}
{
    //std::cout << "move constructor\n";
    other.data = nullptr;
    other.size = 0;
}

Image& Image::operator=(Image&& other)
{
    //std::cout << "move assignment\n";
    delete[] data;
    data = other.data;
    width = other.width;
    height = other.height;
    channels = other.channels;
    size = other.size;

    other.data = nullptr;
    other.size = 0;
    return *this;
}

//save image as jpg file
bool Image::save(std::string file_path)
{
    unsigned char *out_data = new unsigned char[width*height*channels]; 
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < channels; c++) {
                int dst_idx = y*width*channels + x*channels + c;
                int src_idx = c*height*width + y*width + x;
                out_data[dst_idx] = std::roundf(data[src_idx] * 255.);
            }
        }
    }
    bool success = stbi_write_jpg(file_path.c_str(), width, height, channels, out_data, 100);
    if (!success)
        std::cerr << "Failed to save image: " << file_path << "\n";

    delete[] out_data;
    return true;
}

void Image::set_pixel(int x, int y, int c, float val)
{
    if (x >= width || x < 0 || y >= height || y < 0 || c >= channels || c < 0) {
        std::cerr << "set_pixel() error: Index out of bounds.\n";
        std::exit(1);
    }
    data[c*width*height + y*width + x] = val;
}

float Image::get_pixel(int x, int y, int c) const
{
    if (x < 0)
        x = 0;
    if (x >= width)
        x = width - 1;
    if (y < 0)
        y = 0;
    if (y >= height)
        y = height - 1;
    return data[c*width*height + y*width + x];
}

void Image::clamp()
{
    int size = width * height * channels;
    for (int i = 0; i < size; i++) {
        float val = data[i];
        val = (val > 1.0) ? 1.0 : val;
        val = (val < 0.0) ? 0.0 : val;
        data[i] = val;
    }
}

//map coordinate from 0-current_max range to 0-new_max range
float map_coordinate(float new_max, float current_max, float coord)
{
    float a = new_max / current_max;
    float b = -0.5 + a*0.5;
    return a*coord + b;
}

Image Image::resize(int new_w, int new_h, Interpolation method) const
{
    Image resized(new_w, new_h, this->channels);
    float value = 0;
    // #pragma omp parallel for collapse(3) num_threads(16) 
    // increasing the processing time, cost of parallelization is high
    for (int x = 0; x < new_w; x++) {
        for (int y = 0; y < new_h; y++) {
            for (int c = 0; c < resized.channels; c++) {
                float old_x = map_coordinate(this->width, new_w, x);
                float old_y = map_coordinate(this->height, new_h, y);
                if (method == Interpolation::BILINEAR)
                    value = bilinear_interpolate(*this, old_x, old_y, c);
                else if (method == Interpolation::NEAREST)
                    value = nn_interpolate(*this, old_x, old_y, c);
                resized.set_pixel(x, y, c, value);
            }
        }
    }
    return resized;
}

float bilinear_interpolate(const Image& img, float x, float y, int c)
{
    float p1, p2, p3, p4, q1, q2;
    float x_floor = std::floor(x), y_floor = std::floor(y);
    float x_ceil = x_floor + 1, y_ceil = y_floor + 1;
    p1 = img.get_pixel(x_floor, y_floor, c);
    p2 = img.get_pixel(x_ceil, y_floor, c);
    p3 = img.get_pixel(x_floor, y_ceil, c);
    p4 = img.get_pixel(x_ceil, y_ceil, c);
    q1 = (y_ceil-y)*p1 + (y-y_floor)*p3;
    q2 = (y_ceil-y)*p2 + (y-y_floor)*p4;
    return (x_ceil-x)*q1 + (x-x_floor)*q2;
}

float nn_interpolate(const Image& img, float x, float y, int c)
{
    return img.get_pixel(std::round(x), std::round(y), c);
}

Image rgb_to_grayscale(const Image& img)
{
    assert(img.channels == 3);
    Image gray(img.width, img.height, 1);
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float red, green, blue;
            red = img.get_pixel(x, y, 0);
            green = img.get_pixel(x, y, 1);
            blue = img.get_pixel(x, y, 2);
            gray.set_pixel(x, y, 0, 0.299*red + 0.587*green + 0.114*blue);
        }
    }
    return gray;
}

Image grayscale_to_rgb(const Image& img)
{
    assert(img.channels == 1);
    Image rgb(img.width, img.height, 3);
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float gray_val = img.get_pixel(x, y, 0);
            rgb.set_pixel(x, y, 0, gray_val);
            rgb.set_pixel(x, y, 1, gray_val);
            rgb.set_pixel(x, y, 2, gray_val);
        }
    }
    return rgb;
}

__device__ void dev_set_pixel(Image& img, int x, int y, int c, float val)
{
    if (x >= img.width || x < 0 || y >= img.height || y < 0 || c >= img.channels || c < 0) {
        return;
    }
    img.data[c*img.width*img.height + y*img.width + x] = val;
}

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

__global__ void convolve_vertical(const Image& img, const Image& kernel, const int size, const int center, Image& out_img) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= img.width || y >= img.height) return;
    
    float sum = 0.0;
    for (int k = 0; k < size; k++) {
        int dy = -center + k;
        sum += dev_get_pixel(img, x, y + dy, 0) * kernel.data[k];
    }
    dev_set_pixel(out_img, x, y, 0, sum);
}

__global__ void convolve_horizontal(const Image& img, const Image& kernel, const int size, const int center, Image& out_img) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= img.width || y >= img.height) return;
    
    float sum = 0.0;
    for (int k = 0; k < size; k++) {
         int dx = -center + k;
        sum += dev_get_pixel(img, x+dx, y, 0) * kernel.data[k];
    }
    dev_set_pixel(out_img, x, y, 0, sum);
}

__global__ void setDeviceImageSize(Image* devimg, const int w, const int h, const int c){
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        devimg->width = w;
        devimg->height = h;
        devimg->channels = c;
        devimg->size = w*h*c;
    }
}

__global__ void setPixelKernel(Image* img, int x, int y, int c, float val) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        dev_set_pixel(*img, x, y, c, val);;
}

__global__ void normPixelKernel(Image* img, float sum) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        for (int k = 0; k < img->size; k++)
            img->data[k] /= sum;
    }
}

__global__ void imagePrint(Image* devimg)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        for (int k = 0; k < devimg->size; k++) {
            printf("%f\n", devimg->data[k]);
        }
    } 
}
__global__ void imageDataRefCopy(Image* devImg, float *srcDevImg)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        devImg->data = srcDevImg;
    }
}

// separable 2D gaussian blur for 1 channel image
Image gaussian_blur(const Image& img, float sigma)
{
    assert(img.channels == 1);

    Image *dev_img = nullptr;
    float *dev_img_data = nullptr;
    // need to do deep copy, fix sizeof struct being copied
    // https://forums.developer.nvidia.com/t/gpu-struct-allocation/42638
    // https://stackoverflow.com/questions/14284964/cuda-how-to-allocate-memory-for-data-member-of-a-class%5B/url%5D
    cudaError_t err = cudaMalloc((void**)&dev_img, sizeof(Image));
    if (err != cudaSuccess){
      std::cout<<cudaGetErrorString(err)<<std::endl;
      exit(-1);
    }
    setDeviceImageSize<<<1,1>>>(dev_img, img.width, img.height, img.channels);
    cudaDeviceSynchronize();

    err = cudaMalloc((void**)&dev_img_data, img.size*sizeof(float));
    if (err != cudaSuccess){
      std::cout<<cudaGetErrorString(err)<<std::endl;
      exit(-1);
    }

    cudaMemcpy(dev_img_data, img.data, img.size*sizeof(float), cudaMemcpyHostToDevice);
    imageDataRefCopy<<<1,1>>>(dev_img, dev_img_data);
    cudaDeviceSynchronize();
  
    int size = std::ceil(6 * sigma);
    if (size % 2 == 0)
        size++;
    int center = size / 2;

    Image *dev_kernel;
    float *dev_kernel_data;

    cudaMalloc((void**)&dev_kernel, sizeof(Image));
    cudaMalloc((void**)&dev_kernel_data, size * sizeof(float));
    imageDataRefCopy<<<1,1>>>(dev_kernel, dev_kernel_data);
    cudaDeviceSynchronize();
    setDeviceImageSize<<<1, 1>>>(dev_kernel, size, 1, 1);
    cudaDeviceSynchronize();

    float sum = 0;
    for (int k = -size/2; k <= size/2; k++) {
        float val = std::exp(-(k*k) / (2*sigma*sigma));
        setPixelKernel<<<1,1>>>(dev_kernel, center+k, 0, 0, val);
        cudaDeviceSynchronize();
        sum += val;
    }

    normPixelKernel<<<1,1>>>(dev_kernel, sum);
    cudaDeviceSynchronize();

    Image filtered(img.width, img.height, 1);
    Image *dev_temp;
    float *dev_temp_data;

    cudaMalloc((void**)&dev_temp, sizeof(Image));
    cudaMalloc((void**)&dev_temp_data, (img.width*img.height*1)*sizeof(float));
    imageDataRefCopy<<<1,1>>>(dev_temp, dev_temp_data);
    cudaDeviceSynchronize();

    setDeviceImageSize<<<1, 1>>>(dev_temp, img.width, img.height, 1);
    cudaDeviceSynchronize();

    Image *dev_filtered;
    float *dev_filtered_data;

    cudaMalloc((void**)&dev_filtered, sizeof(Image));
    cudaMalloc((void**)&dev_filtered_data, (img.width*img.height*1)*sizeof(float));
    imageDataRefCopy<<<1,1>>>(dev_filtered, dev_filtered_data);
    cudaDeviceSynchronize();

    setDeviceImageSize<<<1, 1>>>(dev_filtered, img.width, img.height, 1);
    cudaDeviceSynchronize();

    // convolve vertical
    dim3 threadsPerBlock(16, 16); // 16x16 threads per block
    dim3 numBlocks((img.width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (img.height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    convolve_vertical<<<numBlocks, threadsPerBlock>>>(*dev_img, *dev_kernel, size, center, *dev_temp);
    cudaDeviceSynchronize();

    // convolve horizontal
    convolve_horizontal<<<numBlocks, threadsPerBlock>>>(*dev_temp, *dev_kernel, size, center, *dev_filtered);
    cudaDeviceSynchronize();

    cudaMemcpy(filtered.data, dev_filtered_data, filtered.size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_img_data);
    cudaFree(dev_img);
    cudaFree(dev_kernel_data);
    cudaFree(dev_kernel);
    cudaFree(dev_temp_data);
    cudaFree(dev_temp);
    cudaFree(dev_filtered_data);
    cudaFree(dev_filtered);

    return filtered;
}

void draw_point(Image& img, int x, int y, int size)
{
    for (int i = x-size/2; i <= x+size/2; i++) {
        for (int j = y-size/2; j <= y+size/2; j++) {
            if (i < 0 || i >= img.width) continue;
            if (j < 0 || j >= img.height) continue;
            if (std::abs(i-x) + std::abs(j-y) > size/2) continue;
            if (img.channels == 3) {
                img.set_pixel(i, j, 0, 1.f);
                img.set_pixel(i, j, 1, 0.f);
                img.set_pixel(i, j, 2, 0.f);
            } else {
                img.set_pixel(i, j, 0, 1.f);
            }
        }
    }
}

void draw_line(Image& img, int x1, int y1, int x2, int y2)
{
    if (x2 < x1) {
        std::swap(x1, x2);
        std::swap(y1, y2);
    }
    int dx = x2 - x1, dy = y2 - y1;
    for (int x = x1; x < x2; x++) {
        int y = y1 + dy*(x-x1)/dx;
        if (img.channels == 3) {
            img.set_pixel(x, y, 0, 0.f);
            img.set_pixel(x, y, 1, 1.f);
            img.set_pixel(x, y, 2, 0.f);
        } else {
            img.set_pixel(x, y, 0, 1.f);
        }
    }
}

} // namespace parallel_cuda_image

