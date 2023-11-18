#include <iostream>
#include <png.h>

#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

#include <vector>
#include <array>

#include <iostream>
#include <fstream>
#include <iomanip>

#include "image.h"

std::unique_ptr<Image> pngData(const char* filename)
{
    try {
        std::ifstream fd(filename, std::ios::binary);
        if (!fd.is_open()) {
            throw std::runtime_error(std::string("Error opening PNG file: ") + filename);
        }

        std::string filenameStr(filename);

        if (!(filenameStr.length() >= 4 && filenameStr.substr(filenameStr.length() - 4) == ".png")) {
            throw std::runtime_error(std::string("Error: Not a PNG file: ") + filename);
        }

        png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
        if (!png) {
            throw std::runtime_error("Error creating PNG read struct.");
        }

        png_infop info = png_create_info_struct(png);
        if (!info) {
            png_destroy_read_struct(&png, nullptr, nullptr);
            throw std::runtime_error("Error creating PNG info struct.");
        }

        if (setjmp(png_jmpbuf(png))) {
            png_destroy_read_struct(&png, &info, nullptr);
            throw std::runtime_error("Error during PNG file reading.");
        }

        // Custom read function for libpng using std::ifstream
        png_set_read_fn(png, static_cast<png_voidp>(&fd), [](png_structp png_ptr, png_bytep data, png_size_t length) {
            auto file = static_cast<std::ifstream*>(png_get_io_ptr(png_ptr));
            file->read(reinterpret_cast<char*>(data), length);
        });

        png_read_info(png, info);

        Image image{};

        int width = png_get_image_width(png, info);
        int height = png_get_image_height(png, info);
        const int channels = 3; // RGB channels

        if (png_get_color_type(png, info) != PNG_COLOR_TYPE_RGB) {
            png_destroy_read_struct(&png, &info, nullptr);
            throw std::runtime_error("The image is not in RGB format");
        }

        std::vector<std::vector<uint8_t>> pixelData(height, std::vector<uint8_t>(width * channels, 0));

        png_bytep* pointers = new png_bytep[height];

        for (int i = 0; i < height; ++i) {
            pointers[i] = pixelData[i].data();
        }
        png_read_image(png, pointers);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width * channels; x += channels) {
                uint8_t red = pointers[y][x];
                uint8_t green = pointers[y][x + 1];
                uint8_t blue = pointers[y][x + 2];

                // Access RGB values
                image.r(y, x / channels) = red;
                image.g(y, x / channels) = green;
                image.b(y, x / channels) = blue;
            }
        }

        delete[] pointers;
        png_destroy_read_struct(&png, &info, nullptr);

        std::unique_ptr<Image> result = std::make_unique<Image>(image);

        return result;
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return nullptr;
    }
}


Image readByteFile(const char * filename, Image a)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return a;
    }

    std::cout << std::hex << std::setw(2) << std::setfill('0');

    char byte;
    int count = 0;
    int loopRow = 0;
    int loopColumn = 0;
    int mod3 = 0;

    while (file.read(&byte, 1))
    {

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

    file.close();
    return a;
}
