#include <iostream>
#include <png.h>

#include <vector>
#include <array>

#include <fstream>
#include <iomanip>

#include <dirent.h>
#include <stack>
#include <sys/stat.h>
#include <sys/types.h>
#include <filesystem>

#include "image.h"
#include "convolution.h"

void Image::saveToPNG(const char *outputPath)
{

    int width = 50;
    int height = 50;

    FILE *fp = fopen(outputPath, "wb");
    if (!fp)
        throw std::runtime_error("Error opening output file");

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png)
        throw std::runtime_error("Error creating PNG write struct");

    png_infop info = png_create_info_struct(png);
    if (!info)
    {
        png_destroy_write_struct(&png, (png_infopp)NULL);
        throw std::runtime_error("Error creating PNG info struct");
    }

    if (setjmp(png_jmpbuf(png)))
    {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        throw std::runtime_error("Error during PNG creation");
    }

    png_init_io(png, fp);

    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_GRAY,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    std::vector<png_bytep> row_pointers(height);
    for (int y = 0; y < PRE_SIZE_MATRIX; y++)
    {
        row_pointers[y] = reinterpret_cast<png_bytep>(&r(y, 0));
    }

    png_set_rows(png, info, row_pointers.data());
    png_write_png(png, info, PNG_TRANSFORM_IDENTITY, NULL);

    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

std::unique_ptr<Image> pngData(const char *filename)
{
    try
    {
        std::ifstream fd(filename, std::ios::binary);
        if (!fd.is_open())
        {
            throw std::runtime_error(std::string("Error opening PNG file: ") + filename);
        }

        std::string filenameStr(filename);

        if (!(filenameStr.length() >= 4 && filenameStr.substr(filenameStr.length() - 4) == ".png"))
        {
            throw std::runtime_error(std::string("Error: Not a PNG file: ") + filename);
        }

        png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
        if (!png)
        {
            throw std::runtime_error("Error creating PNG read struct.");
        }

        png_infop info = png_create_info_struct(png);
        if (!info)
        {
            png_destroy_read_struct(&png, nullptr, nullptr);
            throw std::runtime_error("Error creating PNG info struct.");
        }

        if (setjmp(png_jmpbuf(png)))
        {
            png_destroy_read_struct(&png, &info, nullptr);
            throw std::runtime_error("Error during PNG file reading.");
        }

        // Custom read function for libpng using std::ifstream
        png_set_read_fn(png, static_cast<png_voidp>(&fd), [](png_structp png_ptr, png_bytep data, png_size_t length)
                        {
            auto file = static_cast<std::ifstream*>(png_get_io_ptr(png_ptr));
            file->read(reinterpret_cast<char*>(data), length); });

        png_read_info(png, info);

        Image image{};

        int width = png_get_image_width(png, info);
        int height = png_get_image_height(png, info);
        const int channels = 3; // RGB channels

        if (png_get_color_type(png, info) != PNG_COLOR_TYPE_RGB)
        {
            png_destroy_read_struct(&png, &info, nullptr);
            throw std::runtime_error("The image is not in RGB format");
        }

        std::vector<std::vector<uint8_t>> pixelData(height, std::vector<uint8_t>(width * channels, 0));

        png_bytep *pointers = new png_bytep[height];

        for (int i = 0; i < height; ++i)
        {
            pointers[i] = pixelData[i].data();
        }
        png_read_image(png, pointers);

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width * channels; x += channels)
            {
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
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return nullptr;
    }
}

Image readByteFile(const char *filename, Image a)
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

        if (count < 13)
        {
            ++count;
            continue;
        }
        else
        {
            if (loopColumn == 50)
            {
                loopColumn = 0;
                ++loopRow;
            }

            if (mod3 % 3 == 0)
            {
                a.r(loopRow, loopColumn) = static_cast<uint8_t>(static_cast<unsigned char>(byte));
                ++mod3;
            }

            else if (mod3 % 3 == 1)
            {
                a.g(loopRow, loopColumn) = static_cast<uint8_t>(static_cast<unsigned char>(byte));
                ++mod3;
            }

            else if (mod3 % 3 == 2)
            {
                a.b(loopRow, loopColumn) = static_cast<uint8_t>(static_cast<unsigned char>(byte));
                ++loopColumn;
                ++mod3;
            }
        }

        ++count;
    }

    file.close();
    return a;
}

xt::xarray<float> toGrayScale(Image a)
{
    xt::xarray<float> grayMatrice{xt::empty<uint8_t>({PRE_SIZE_MATRIX, PRE_SIZE_MATRIX})};

    for (int y = 0; y < PRE_SIZE_MATRIX; ++y)
    {
        for (int x = 0; x < PRE_SIZE_MATRIX; ++x)
        {
            uint8_t red = a.r(y, x);
            uint8_t green = a.g(y, x);
            uint8_t blue = a.b(y, x);

            // NTSC formula
            //  uint8_t gray = static_cast<uint8_t>(0.299 * red + 0.587 * green + 0.114 * blue);

            grayMatrice(y, x) = (red + green + blue) / 3;
        }
    }
    return grayMatrice;
}

xt::xarray<bool> toSobel(xt::xarray<float> grayMatrice) {
    xt::xarray<float> sobX{{-1, 0, 1},
                               {-2, 0, 2},
                               {-1, 0, 1}};

        xt::xarray<float> sobY{{-1, -2, -1},
                               {0, 0, 0},
                               {1, 2, 1}};



        auto gx = matrixConvolution(grayMatrice, sobX, 0, 1);
        auto gy = matrixConvolution(grayMatrice, sobY, 0, 1);

        xt::xarray<float> g = xt::sqrt(gx * gx + gy * gy);

        xt::xarray<bool> boolG = xt::where(g < 128 , false, true);

        return boolG;
}

void saveGrayToPNG(const char *outputPath, xt::xarray<uint8_t> grayMatrice)
{

    int width = grayMatrice.shape()[1];
    int height = grayMatrice.shape()[0];

    FILE *fp = fopen(outputPath, "wb");
    if (!fp)
        throw std::runtime_error("Error opening output file");

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png)
        throw std::runtime_error("Error creating PNG write struct");

    png_infop info = png_create_info_struct(png);
    if (!info)
    {
        png_destroy_write_struct(&png, (png_infopp)NULL);
        throw std::runtime_error("Error creating PNG info struct");
    }

    if (setjmp(png_jmpbuf(png)))
    {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        throw std::runtime_error("Error during PNG creation");
    }

    png_init_io(png, fp);

    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_GRAY,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    std::vector<png_bytep> row_pointers(height);
    for (int y = 0; y < height; y++)
    {
        row_pointers[y] = &grayMatrice(y, 0);
    }

    png_set_rows(png, info, row_pointers.data());
    png_write_png(png, info, PNG_TRANSFORM_IDENTITY, NULL);

    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

void saveEdgetoPBM(const char *outputPath, const xt::xarray<bool> boolMatrix)
{
    int width = boolMatrix.shape()[1];
    int height = boolMatrix.shape()[0];

    std::ofstream outputFile(outputPath);

    if (!outputFile.is_open())
    {
        std::cerr << "Erreur lors de l'ouverture du fichier." << std::endl;
        return;
    }

    // Écrire l'en-tête du fichier PBM
    outputFile << "P4 "
               << width << " " << height << " ";

    // on prend par bloc de 8 bits(1octet)
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; j += 8)
        {
            char byte = 0;
            for (int k = 0; (k < 8) && ((j + k) < width); ++k)
            {
                byte += boolMatrix(i, j + k) << (7 - k);
            }
            outputFile.write(&byte, sizeof(char));
        }
    }

    outputFile.close();
}

void generateAllPBM(const char *folderConvPath, const char *folderOutput)
{
    std::stack<std::pair<std::string, std::string>> directories;    //stockage des dossiers entrés / sortis
    directories.push({folderConvPath, folderOutput});

    while (!directories.empty())
    {
        auto [currentInputDir, currentOutputDir] = directories.top();
        directories.pop();

        if (mkdir(currentOutputDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0 && errno != EEXIST) {
            std::cerr << "Erreur lors de la création du dossier : " << folderConvPath << std::endl;
            return;
        }

        DIR *dir;
        struct dirent *entry;

        if ((dir = opendir(currentInputDir.c_str())) == nullptr)
        {
            std::cerr << "Erreur lors de l'ouverture du répertoire : " << folderConvPath << std::endl;
            return;
        }
        else
        {
            while ((entry = readdir(dir)) != nullptr) {
                std::string inputFullPath = currentInputDir + "/" + entry->d_name;
                std::string outputFullPath = currentOutputDir + "/" + entry->d_name;

                // ignore "." et ".." pour éviter les boucle infinies
                if (entry->d_type == DT_DIR && std::string(entry->d_name) != "." && std::string(entry->d_name) != "..")
                {
                    directories.push({inputFullPath, outputFullPath});
                }

                // Vérif pour un fichier .png
                else if (entry->d_type == DT_REG && std::strstr(entry->d_name, ".png") != nullptr)
                {
                    std::ifstream inputFile(inputFullPath, std::ios::binary);
                    
                    if (inputFile.is_open())
                    {
                        // Pass the char* to the pngData function
                        char* charFilePath = const_cast<char*>(inputFullPath.c_str());
                        std::unique_ptr<Image> result = pngData(charFilePath);

                        if (result)
                        {
                            Image image = *result;

                            auto grayMatrice = toGrayScale(image);
                            xt::xarray<bool> boolG = toSobel(grayMatrice);

                            std::string outputFilePath = outputFullPath;
                            outputFilePath.replace(outputFilePath.rfind(".png"), 4, ".pbm");

                            saveEdgetoPBM(outputFilePath.c_str(), boolG);
                        }
                    }
                    else
                    {
                        std::cerr << "Erreur lors de l'ouverture du fichier : " << inputFullPath << std::endl;
                    }
                }
            }

            closedir(dir); // Close the directory after processing
        }
    }
}

void generateAllPBM2(const char *folderConvPath, const char *folderOutput)
{
    std::stack<std::pair<std::string, std::string>> directories;
    directories.push({folderConvPath, folderOutput});

    while (!directories.empty())
    {
        auto [currentInputDir, currentOutputDir] = directories.top();
        directories.pop();

        try
        {
            std::filesystem::create_directories(currentOutputDir);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error creating directory: " << folderConvPath << std::endl;
            return;
        }

        for (const auto &entry : std::filesystem::directory_iterator(currentInputDir))
        {
            std::string inputFullPath = entry.path().string();
            std::string outputFullPath = currentOutputDir + "/" + entry.path().filename().string();

            // ignore "." and ".." to avoid infinite loops
            if (entry.is_directory())
            {
                directories.push({inputFullPath, outputFullPath});
            }
            // Check for a .png file
            else if (entry.is_regular_file() && entry.path().extension() == ".png")
            {
                std::ifstream inputFile(inputFullPath, std::ios::binary);

                if (inputFile.is_open())
                {
                    // Pass the char* to the pngData function
                    char *charFilePath = const_cast<char *>(inputFullPath.c_str());
                    auto result = pngData(charFilePath);

                    if (result)
                    {
                        Image image = *result;

                        auto grayMatrice = toGrayScale(image);
                        xt::xarray<bool> boolG = toSobel(grayMatrice);

                        // Remove the ".png" extension and append ".pbm"
                        std::string outputFilePath = outputFullPath;
                        outputFilePath.replace(outputFilePath.rfind(".png"), 4, ".pbm");

                        saveEdgetoPBM(outputFilePath.c_str(), boolG);
                    }
                }
                else
                {
                    std::cerr << "Error opening file: " << inputFullPath << std::endl;
                }
            }
        }
    }
}