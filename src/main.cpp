#include "matrice.h"

int main()
{
    Image a1{};

    const char* filename = "../assets/input.ppm";
    a1 = readByteFile(filename, a1);
    
    std::cout << a1.r << std::endl;


    return 0;
}
