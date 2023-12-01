#include <iostream>
#include <fstream>
#include <sstream>
#include <bitset>
#include <cstring>

#include "image.h"
#include "convolution.h"
#include "network.h"

int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    generateAllPBM2("../assets/breast", "../assets/PBM");

    auto stop = std::chrono::high_resolution_clock::now();

    // en millisecondes
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();

    auto tensor = importAllPBM("../assets/PBM", 3500);

    stop = std::chrono::high_resolution_clock::now();

    // en millisecondes
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}
