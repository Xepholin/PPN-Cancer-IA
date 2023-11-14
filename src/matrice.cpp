#include <iostream>
#include <png.h>

#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

#include <vector>
#include <array>

#include <iostream>
#include <fstream>
#include <iomanip>

#include "../include/matrice.h"


Image readByteFile(const char * filename, Image a)
{

    // Open the file in binary mode
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return a; // Exit with an error code
    }

    // Set up to output in hexadecimal
    std::cout << std::hex << std::setw(2) << std::setfill('0');

    // Read and output each byte in hexadecimal
    char byte;
    int count = 0;
    int loopRow = 0;
    int loopColumn = 0;
    int mod3 = 0;

    while (file.read(&byte, 1))
    {
        
        // la premier ligne est inutile.
        if(count < 13)
        {    
            ++count; 
            continue;   
        }
        else{
            if(loopColumn == 50)
            {
                loopColumn = 0;
                ++loopRow;
            }
                
            if( mod3 %3 == 0)
            {
                a.r(loopRow,loopColumn) = static_cast<uint8_t>(static_cast<unsigned char>(byte));
                ++mod3;
            }

            else if( mod3 %3 == 1)
            {
                a.g(loopRow , loopColumn ) = static_cast<uint8_t>(static_cast<unsigned char>(byte));
                ++mod3;
            }
            
            else if(mod3 %3 == 2)
            {
                a.b(loopRow , loopColumn ) = static_cast<uint8_t>(static_cast<unsigned char>(byte));
                ++loopColumn;
                ++mod3;
            }
        }        

        ++count;
    }

    // Close the file
    file.close();
    return a;
}
