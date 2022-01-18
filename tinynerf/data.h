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
void encodePNG(string filename, const unsigned char *image, int width, int height);

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