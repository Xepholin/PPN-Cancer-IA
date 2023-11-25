#include <iostream>
#include <fstream>
#include <sstream>
#include <bitset>
#include <cstring>

#include "image.h"
#include "convolution.h"

int main()
{
    std::unique_ptr<Image> result = pngData("../assets/input.png");

    if (result)
    {
        // Image a = *result;

        // // Convert the image to grayscale
        // auto grayMatrice = toGrayScale(a);

        // saveGrayToPNG("../assets/output.png", grayMatrice);
        // std::cout << "Grayscale Image:" << std::endl;
        // std::cout << grayMatrice << std::endl; // Now all channels will be the same

        // saveGrayToPNG("../assets/output.png", grayMatrice);

        // xt::xarray<bool> boolG = toSobel(grayMatrice);


        // std::cout << "\nEdge Image: \n" << std::endl;
        // std::cout << boolG << std::endl; // Now all channels will be the same

        // saveEdgetoPBM("../assets/outputBit.pbm",boolG);
        // std::cout << boolG.shape()[0] << std::endl; // Now all channels will be the same
        // std::cout << boolG.shape()[1] << std::endl; // Now all channels will be the same

        saveToCSV2("../assets/output.csv", "../assets/PBM");
    }


    else
    {
        std::cerr << "Error reading PNG file." << std::endl;
    }

    return 0;
}
