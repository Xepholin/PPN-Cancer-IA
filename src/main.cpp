#include <iostream>
#include <fstream>
#include <sstream>
#include <bitset>
#include <cstring>

#include "image.h"
#include "convolution.h"

int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    generateAllPBM("../assets/breast/train/0", "../assets/PBM/breast/train/0");

    auto stop = std::chrono::high_resolution_clock::now();

    // en millisecondes
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}
