#pragma once
#include <nlohmann/json.hpp>

#include <string>
#include <utility>
#include <vector>
#include <tuple>
#include <xtensor/xtensor.hpp>

using nlohmann::json;
using std::pair;
using std::string;
using std::vector;

std::vector<float> load_weights(const std::string &filename);

json read_json(string jsonPath);

pair<vector<string>, vector<xt::xtensor<float, 2, xt::layout_type::row_major>>> get_image_c2w(json jsonData, string datasetPath);

class GetImages
{
public:
  int imageWidth;
  int imageHeight;

  vector<vector<vector<float>>> load_image(string path);

private:
  vector<vector<vector<float>>> reshape_image(vector<uint8_t> img);
};

// class GetRays
// {
// public:
//   float focalLength;
//   int imageWidth;
//   int imageHeight;
//   float near;
//   float far;
//   // Number of samples for coarse model
//   int nC;

//   GetRays(float focalLength,
//           int imageWidth,
//           int imageHeight,
//           float near,
//           float far,
//           int nC)
//   {
//     this->focalLength = focalLength;
//     this->imageHeight = imageHeight;
//     this->imageWidth = imageWidth;
//     this->near = near;
//     this->far = far;
//     this->nC = nC;
//   }

//   std::tuple<vector<float>, vector<float>, vector<float>> computeRays(vector<vector<float>> camera2world);
// };