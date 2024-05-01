#include <chrono>
#include <iostream> 
#include <string>

#include "parallel_cuda_image.hpp"
#include "parallel_cuda_sift.hpp"

using namespace parallel_cuda_image;
using namespace parallel_cuda_sift;

int main(int argc, char *argv[])
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc != 3) {
        std::cerr << "Usage: ./find_keypoints input.jpg (or .png) output.jpg (or .png)\n";
        return 0;
    }
    Image img(argv[1]);
    img =  img.channels == 1 ? img : rgb_to_grayscale(img);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Keypoint> kps = find_keypoints_and_descriptors(img);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Total Time for finding keypoint descriptors: " << elapsed.count() << "s" << std::endl;

    Image result = draw_keypoints(img, kps);
    result.save(argv[2]);

    std::cout << "Output image is saved\n";
    return 0;
}
