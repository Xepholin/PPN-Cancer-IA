#include <png.h>

#include <vector>

#include <fstream>
#include <iomanip>

#include <stack>
#include <sys/stat.h>
#include <sys/types.h>
#include <filesystem>

#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>

#include "image.h"
#include "conv_op.h"
#include "tools.h"
#include "const.h"

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
    for (int y = 0; y < PNGDim; y++)
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

		uint8_t red = 0;
		uint8_t green = 0;
		uint8_t blue = 0;

		int size = width * channels;

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < size; x += channels)
            {
                red = pointers[y][x];
                green = pointers[y][x + 1];
                blue = pointers[y][x + 2];

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
    xt::xarray<float> grayMatrice{xt::empty<uint8_t>({PNGDim, PNGDim})};

	uint8_t red = 0;
	uint8_t green = 0;
	uint8_t blue = 0;

    for (int y = 0; y < PNGDim; ++y)
    {
        for (int x = 0; x < PNGDim; ++x)
        {
            red = a.r(y, x);
            green = a.g(y, x);
            blue = a.b(y, x);

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

	xt::xarray<float> gx = matrixConvolution(grayMatrice, sobX, 1, 0);
	xt::xarray<float> gy = matrixConvolution(grayMatrice, sobY, 1, 0);

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
	char byte = 0;
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; j += 8)
        {
            byte = 0;
			
            for (int k = 0; (k < 8) && ((j + k) < width); ++k)
            {
                byte += boolMatrix(i, j + k) << (7 - k);
            }
            outputFile.write(&byte, sizeof(char));
        }
    }

    outputFile.close();
}

void generatePBM(const char *src, const char *dest)	{
	std::ifstream inputFile(src, std::ios::binary);

	if (inputFile.is_open())
	{
		// Pass the char* to the pngData function
		auto result = pngData(src);

		if (result)
		{
			Image image = *result;

			xt::xarray<float> grayMatrice = toGrayScale(image);

			xt::xarray<bool> boolG = toSobel(grayMatrice);

			// Remove the ".png" extension and append ".pbm"
			std::string outputFilePath = dest;
			outputFilePath.replace(outputFilePath.rfind(".png"), 4, ".pbm");

			saveEdgetoPBM(outputFilePath.c_str(), boolG);
		}
	}
	else
	{
		std::cerr << "Error opening file: " << src << std::endl;
	}
}


void generateAllPBM(const char *src, const char *dest)
{
    std::stack<std::pair<std::string, std::string>> directories;
    directories.push({src, dest});

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
            std::cerr << "Error creating directory: " << src << std::endl;
            return;
        }

        for (const auto &entry : std::filesystem::directory_iterator(currentInputDir))
        {
            std::string inputPath = entry.path().string();
            std::string outputPath = currentOutputDir + "/" + entry.path().filename().string();

            // ignore "." and ".." to avoid infinite loops
            if (entry.is_directory())
            {
                directories.push({inputPath, outputPath});
            }
            // Check for a .png file
            else if (entry.is_regular_file() && entry.path().extension() == ".png")
            {
				// std::cout << "here" << std::endl;
                generatePBM(inputPath.c_str(), outputPath.c_str());
            }
        }
    }
}

xt::xarray<bool> importPBM(const char *path)
{
    const int width = PBMDim;
    const int height = PBMDim;
    const int rowSize = 6;
    const int headerSize = 9;
    int pixelValue = 0;

    std::ifstream image(path, std::ios::binary);

    if (!image.is_open()) {
        perror("Erreur lors de l'ouverture du fichier.");
    }

    xt::xarray<bool> PBM{xt::empty<bool>({PBMDim, PBMDim})};

    // saute 9 caractères (header .pbm)
    image.seekg(headerSize);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < rowSize; ++j) {
            unsigned char byte;
            image.read(reinterpret_cast<char*>(&byte), sizeof(byte));

            int count = 0;
            for (int k = 7; k >= 0 && j * 8 + 7 - k < width; --k) {
                pixelValue = (byte >> k) & 1;

                PBM(i, j * 8 + 7 - k) = !pixelValue;
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
    xt::xarray<bool> result{xt::empty<bool>({nbPBM, PBMDim, PBMDim})};

    int position = 0;

    while (!directory.empty())
    {
        auto currentDir = directory.top();
        directory.pop();

        for (const auto &entry : std::filesystem::directory_iterator(currentDir))
        {
            std::string inputPath = entry.path().string();

            // ignore "." and ".." to avoid infinite loops
            if (entry.is_directory())
            {
                directory.push(inputPath);
            }
            // Check for a .png file
            else if (entry.is_regular_file() && entry.path().extension() == ".pbm")
            {
                xt::xarray<bool> newPBM = importPBM(inputPath.c_str());
                xt::view(result, xt::range(position, position + 1), xt::all(), xt::all()) = newPBM;

                position++;
            }
        }
    }

    return result;
}

xt::xarray<float> importAllPNG(const char *path, int nbPNG)	{
	std::stack<std::string> directory;
    directory.push({path});

    // Define a placeholder for your result (modify as needed)
    xt::xarray<float> result{xt::empty<float>({nbPNG, 3, PNGDim, PNGDim})};

    int position = 0;

    while (!directory.empty())
    {
        auto currentDir = directory.top();
        directory.pop();

        for (const auto &entry : std::filesystem::directory_iterator(currentDir))
        {
            std::string inputPath = entry.path().string();

            // ignore "." and ".." to avoid infinite loops
            if (entry.is_directory())
            {
                directory.push(inputPath);
            }
            // Check for a .png file
            else if (entry.is_regular_file() && entry.path().extension() == ".png")
            {
				std::unique_ptr<Image> temp = pngData(inputPath.c_str());
				Image image = *temp;
				
				xt::view(result, xt::range(position, position + 1), 0, xt::all(), xt::all()) = xt::abs(image.r - 255.0);
				xt::view(result, xt::range(position, position + 1), 1, xt::all(), xt::all()) = xt::abs(image.g - 255.0);
				xt::view(result, xt::range(position, position + 1), 2, xt::all(), xt::all()) = xt::abs(image.b - 255.0);

                position++;
            }
        }
    }

    return normalized(result);
}