#include "data.h"
#include "lodepng.h"
#include "utils.h"
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <array>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xstrided_view.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xrandom.hpp>

using nlohmann::json;
using std::pair;
using std::string;
using std::vector;

using namespace xt::placeholders; // required for `_` to work

/**
 * @brief Load precomputed model weights from an external txt file.
 *
 * @param filename
 * @return std::vector<float>
 */
std::vector<float> load_weights(const std::string &filename)
{
  std::cout << "Loading weights" << std::endl;
  std::vector<float> weights_buffer;
  std::ifstream in(filename);

  for (std::string f; getline(in, f, '\n');)
    weights_buffer.push_back(std::stof(f));

  // for (auto f : weights_buffer)
  // 	std::cout << f << ", ";
  // std::cout << std::endl;
  return weights_buffer;
}

/**
 * @brief Decode a PNG image to its RGB values, represented as a 1D vector of unsigned 8 bit integers (0-255).
 *
 * @param filename
 * @return A pair containing (dimensions, RGB values)
 */
pair<pair<int, int>, vector<uint8_t>> decodePNG(string filename)
{
  vector<uint8_t> image; // the raw pixels
  unsigned width, height;

  // decode to 3-channel RGB (removes alpha channel) with 8 bit color depth
  unsigned error = lodepng::decode(image, width, height, filename, LCT_RGB, 8U);

  // if there's an error, display it
  if (error)
    std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

  pair<int, int> dimensions(width, height);

  return {dimensions, image};
}

void encodePNG(string filename, const unsigned char *image, int width, int height)
{
  unsigned error = lodepng_encode24_file(filename.data(), image, (unsigned)width, (unsigned)height);
  if (error)
    std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
}

json read_json(string path)
{
  std::ifstream f{path};
  json jsonData = json::parse(f, nullptr, true, /*skip_comments=*/true);
  return jsonData;
}

pair<vector<string>, vector<xt::xtensor<float, 2, xt::layout_type::row_major>>> get_image_c2w(json jsonData, string datasetPath)
{
  vector<string> imagePaths;
  vector<xt::xtensor<float, 2, xt::layout_type::row_major>> c2ws;
  json frames = jsonData["frames"];

  for (const auto &it : frames.items())
  {
    const auto frame = it.value();
    string imagePath = frame["file_path"];
    imagePath.replace(imagePath.find("."), 1, datasetPath);
    imagePath.append(".png");
    imagePaths.push_back(imagePath);

    const json tMatrix = frame["transform_matrix"];
    // TODO: clean this up
    vector<vector<float>> c2w_vec;
    for (const auto &it2 : tMatrix.items())
    {
      const vector<float> tm = it2.value();
      c2w_vec.push_back(tm);
    }

    vector<float> c2w_1d = two_d_to_one_d<float>(c2w_vec);
    xt::xtensor<float, 2, xt::layout_type::row_major> c2w = xt::adapt(c2w_1d, {4, 4});

    c2ws.push_back(c2w);
  }

  return {imagePaths, c2ws};
}

vector<vector<vector<float>>> GetImages::load_image(string path)
{
  auto decoded = decodePNG(path);
  imageWidth = decoded.first.first;
  imageHeight = decoded.first.second;

  vector<uint8_t> img8 = decoded.second;

  return reshape_image(img8);
}

vector<vector<vector<float>>> GetImages::reshape_image(vector<uint8_t> img)
{
  vector<vector<vector<float>>> img3d;

  img3d.resize(imageHeight);
  for (int i = 0; i < imageHeight; ++i)
  {
    img3d[i].resize(imageWidth);

    for (int j = 0; j < imageWidth; ++j)
      img3d[i][j].resize(3);
  }

  for (int i = 0; i < imageHeight; i++)
  {
    for (int j = 0; j < imageWidth; j++)
    {
      for (int k = 0; k < 3; k++)
      {
        img3d[i][j][k] = (float)img[(i * imageWidth + j) * 3 + k] / (float)255;
      }
    }
  }

  return img3d;
}
