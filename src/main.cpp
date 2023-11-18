#include "image.h"

int main()
{
    std::unique_ptr<Image> result = pngData("../assets/input.png");

    if (result)
    {
        Image a = *result;
        
        std::cout << a.r << std::endl;
        std::cout << a.g << std::endl;
        std::cout << a.b << std::endl;
    }
    else
    {
        std::cerr << "Error reading PNG file." << std::endl;
    }

    return 0;
}
