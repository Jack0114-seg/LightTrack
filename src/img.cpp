#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

int main() {
    cv::Mat img = cv::imread("input.jpg");

    if (img.empty()) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    int height = img.rows;
    int width = img.cols;
    int channels = img.channels();

    std::cout << "image information : [height,width,channels]-->" << height << width << channels << std::endl;

    std::ofstream outFile("imput.txt");

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);
            outFile << (int)pixel[2] << " " << (int)pixel[1] << " " << (int)pixel[0] << " \n";
        }
        outFile << std::endl;
    }

    outFile.close();

    std::cout << "Pixel data saved to pixel_data.txt" << std::endl;

    return 0;
}
