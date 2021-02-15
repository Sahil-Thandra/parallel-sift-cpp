#include <string>
#include <cassert>
#ifndef IMAGE_H
#define IMAGE_H

enum Interpolation {BILINEAR, NEAREST};

struct Image {
    explicit Image(std::string file_path);
    Image(int w, int h, int c);
    ~Image();
    Image(const Image& other);
    Image& operator=(const Image& other);
    Image(Image&& other);
    Image& operator=(Image&& other);
    int width;
    int height;
    int channels;
    int size;
    float *data;
    bool load();
    bool save(std::string file_path);
    void set_pixel(int x, int y, int c, float val);
    float get_pixel(int x, int y, int c) const;
    void clamp();
    Image resize(int new_w, int new_h, Interpolation interp = BILINEAR) const;
};



//map coordinate from 0-current_max range to 0-new_max range
inline float map_coordinate(float new_max, float current_max, float coord);
inline float bilinear_interpolate(const Image& img, float x, float y, int c);
inline float nn_interpolate(const Image& img, float x, float y, int c);
Image rgb_to_grayscale(const Image& img);
Image grayscale_to_rgb(const Image& img);

// drawing
void draw_point(Image& img, int x, int y);
void draw_line(Image& img, int x1, int y1, int x2, int y2);

// functions related to filtering
Image convolve(const Image& img, const Image& filter, bool preserve);
Image make_gx_filter();
Image make_gy_filter();
Image make_gaussian_filter(float sigma, bool normalize = true);
Image gaussian_blur(const Image& img, float sigma);
#endif