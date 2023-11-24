#include "../include/image.h"
#include "../include/convolution.h"

int main()
{
    std::unique_ptr<Image> result = pngData("../assets/input.png");

    if (result)
    {
        Image a = *result;

        // Convert the image to grayscale
        auto grayMatrice = toGrayScale(a);

        saveGrayToPNG("../assets/output.png", grayMatrice);
        std::cout << "Grayscale Image:" << std::endl;
        std::cout << grayMatrice << std::endl; // Now all channels will be the same

        saveGrayToPNG("../assets/output.png", grayMatrice);

        xt::xarray<float> sobX{{-1, 0, 1},
                               {-2, 0, 2},
                               {-1, 0, 1}};

        xt::xarray<float> sobY{{-1, -2, -1},
                               {0, 0, 0},
                               {1, 2, 1}};



        auto gx = matrixConvolution(grayMatrice, sobX);
        auto gy = matrixConvolution(grayMatrice, sobY);

        xt::xarray<float> g = xt::sqrt(gx * gx + gy * gy);

        xt::xarray<bool> boolG = xt::where(g < 128 , false, true);


        std::cout << "\nEdge Image: \n" << std::endl;
        std::cout << boolG << std::endl; // Now all channels will be the same

        saveEdgetoPBM("../assets/outputBit.pbm",boolG);
        std::cout << boolG.shape()[0] << std::endl; // Now all channels will be the same
        std::cout << boolG.shape()[1] << std::endl; // Now all channels will be the same

    }


    else
    {
        std::cerr << "Error reading PNG file." << std::endl;
    }

    return 0;
}
