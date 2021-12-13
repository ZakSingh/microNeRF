#pragma once
#include <nlohmann/json.hpp>

#include <string>
#include <utility>
#include <vector>

using nlohmann::json;
using std::pair;
using std::string;
using std::vector;

json read_json(string jsonPath);
pair<vector<string>, vector<vector<vector<float>>>> get_image_c2w(json jsonData, string datasetPath);

class GetImages
{
public:
  int imageWidth;
  int imageHeight;

  vector<vector<vector<float>>> load_image(string path);

private:
  vector<vector<vector<float>>> reshape_image(vector<uint8_t> img);
};