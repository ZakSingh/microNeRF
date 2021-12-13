#include "data.h"
#include "lodepng.h"
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>

using nlohmann::json;
using std::pair;
using std::string;
using std::vector;

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

json read_json(string path)
{
  std::ifstream f{path};
  json jsonData = json::parse(f, nullptr, true, /*skip_comments=*/true);
  return jsonData;
}

pair<vector<string>, vector<vector<vector<float>>>> get_image_c2w(json jsonData, string datasetPath)
{
  vector<string> imagePaths(100);
  vector<vector<vector<float>>> c2ws(100);
  json frames = jsonData["frames"];

  for (const auto &it : frames.items())
  {
    const auto frame = it.value();
    string imagePath = frame["file_path"];
    imagePath.replace(imagePath.find("."), 1, datasetPath);
    imagePath.append(".png");
    imagePaths.push_back(imagePath);

    const json tMatrix = frame["transform_matrix"];

    vector<vector<float>> c2w(4);
    for (const auto &it2 : tMatrix.items())
    {
      const vector<float> tm = it2.value();
      c2w.push_back(tm);
    }

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