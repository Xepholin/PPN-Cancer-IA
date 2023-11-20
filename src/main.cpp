#include "image.h"

int main()
{
    std::unique_ptr<Image> result = pngData("../assets/input.png");

    if (result)
    {
        Image a = *result;

        // Convert the image to grayscale
        a.toGrayscale();
        a.saveToPNG("../assets/output.png");
        std::cout << "Grayscale Image:" << std::endl;
        std::cout << a.r << std::endl; // Now all channels will be the same
    }
    else
    {
        std::cerr << "Error reading PNG file." << std::endl;
    }

    return 0;
}
