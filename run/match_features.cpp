#include <iostream> 
#include <string>

#include "image.hpp"
#include "sift.hpp"

int main(int argc, char *argv[])
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc != 4) {
        std::cerr << "Usage: ./match_features a.jpg b.jpg (or .png) output.jpg (or .png)\n";
        return 0;
    }
    Image a(argv[1]), b(argv[2]);
    a = a.channels == 1 ? a : rgb_to_grayscale(a);
    b = b.channels == 1 ? b : rgb_to_grayscale(b);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<sift::Keypoint> kps_a = sift::find_keypoints_and_descriptors(a);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Total Time for finding keypoint descriptors for image 1: " << elapsed.count() << "s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    std::vector<sift::Keypoint> kps_b = sift::find_keypoints_and_descriptors(b);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Total Time for finding keypoint descriptors for image 2: " << elapsed.count() << "s" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<int, int>> matches = sift::find_keypoint_matches(kps_a, kps_b);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Total Time to match keypoint descriptors: " << elapsed.count() << "s" << std::endl;

    Image result = sift::draw_matches(a, b, kps_a, kps_b, matches);
    result.save(argv[3]);
    
    std::cout << "Found " << matches.size() << " feature matches. Output image is saved\n";
    return 0;
}