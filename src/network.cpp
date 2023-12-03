#include <iostream>
#include <fstream>

#include <dirent.h>
#include <stack>
#include <sys/stat.h>
#include <sys/types.h>
#include <filesystem>

#include "network.h"

xt::xarray<bool> importPBM(const char *path)
{
    const int width = 48;
    const int height = 48;
    const int rowSize = 6;
    const int headerSize = 9;
    int pixelValue = 0;

    std::ifstream image(path, std::ios::binary);

    if (!image.is_open()) {
        perror("Erreur lors de l'ouverture du fichier.");
    }

    xt::xarray<bool> PBM{xt::empty<bool>({POST_SIZE_MATRIX, POST_SIZE_MATRIX})};

    // saute 9 caractères (header .pbm)
    image.seekg(headerSize);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < rowSize; ++j) {
            unsigned char byte;
            image.read(reinterpret_cast<char*>(&byte), sizeof(byte));

            int count = 0;
            for (int k = 7; k >= 0 && j * 8 + 7 - k < width; --k) {
                pixelValue = (byte >> k) & 1;

                PBM(i, j * 8 + 7 - k) = pixelValue;
                count++;
            }
        }
    }

    image.close();

    return PBM;
}

xt::xarray<bool> importAllPBM(const char *path, int nbPBM)
{
    std::stack<std::string> directory;
    directory.push({path});

    // Define a placeholder for your result (modify as needed)
    xt::xarray<bool> result{xt::empty<bool>({nbPBM, POST_SIZE_MATRIX, POST_SIZE_MATRIX})};

    int position = 0;

    while (!directory.empty())
    {
        auto currentDir = directory.top();
        directory.pop();

        for (const auto &entry : std::filesystem::directory_iterator(currentDir))
        {
            std::string inputFullPath = entry.path().string();

            // ignore "." and ".." to avoid infinite loops
            if (entry.is_directory())
            {
                directory.push(inputFullPath);
            }
            // Check for a .png file
            else if (entry.is_regular_file() && entry.path().extension() == ".pbm")
            {
                xt::xarray<bool> newPBM = importPBM(inputFullPath.c_str());
                xt::view(result, xt::range(position, position + 1), xt::all(), xt::all()) = newPBM;

                position++;
            }
        }
    }

    return result;
}

