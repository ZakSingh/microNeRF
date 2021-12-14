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
  vector<string> imagePaths;
  vector<vector<vector<float>>> c2ws;
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

std::tuple<vector<float>, vector<float>, vector<float>> GetRays::computeRays(vector<vector<float>> camera2world)
{
  std::cout << "Computing rays..." << std::endl;
  auto [i, j] = xt::meshgrid(xt::arange<float>(imageWidth), xt::arange<float>(imageHeight));
  // Take transposes to convert from ij indexing to xy
  auto x = xt::transpose(i);
  auto y = xt::transpose(j);

  // Define the camera coordinates
  std::array<size_t, 2> xCameraShape = {800, 800};
  xt::xtensor<float, 2, xt::layout_type::row_major> xCamera(xCameraShape);
  xCamera = (x - (float)imageWidth * 0.5) / focalLength;

  std::array<size_t, 2> yCameraShape = {800, 800};
  xt::xtensor<float, 2, xt::layout_type::row_major> yCamera(yCameraShape);
  yCamera = (y - (float)imageWidth * 0.5) / focalLength;

  // Define the camera vector
  auto xCyCzC = xt::stack(xt::xtuple(xCamera, -yCamera, -xt::ones_like(x)), 2);

  // Slice the camera2world matrix to get rotation and translation matrices
  // Need to convert camera2world matrix to 1D row-stacked representation
  vector<float> c2w_1d = two_d_to_one_d<float>(camera2world);
  std::vector<std::size_t> shape = {4, 4};
  auto c2w = xt::adapt(c2w_1d, shape);

  auto rotation = xt::view(c2w, xt::range(0, 3), xt::range(0, 3));
  auto translation = xt::view(c2w, xt::range(0, 3), -1);

  auto xCyCzC_2 = xt::strided_view(xCyCzC, {xt::ellipsis(), xt::newaxis(), xt::all()});
  auto xWyWzW = xCyCzC_2 * rotation;

  // Calculate direction vector of the ray
  auto rayD = xt::sum(xWyWzW, {-1});
  auto rayD_2 = rayD / xt::reshape_view(xt::norm_l2(rayD, {2}), {800, 800, 1});

  // Calculate the origin vector of the ray
  auto rayO = xt::broadcast(translation, rayD_2.shape());

  // Get sample points from the ray.
  // TODO: Implement noise
  auto zVals = xt::linspace<float>(near, far, nC);
  auto pts = xt::strided_view(rayO, {xt::ellipsis(), xt::newaxis(), xt::all()}) +
             xt::strided_view(rayD, {xt::ellipsis(), xt::newaxis(), xt::all()}) *
                 xt::strided_view(zVals, {xt::ellipsis(), xt::all(), xt::newaxis()});

  auto pts_flat = xt::reshape_view(pts, {-1, 3});

  auto dists = xt::hstack(xtuple(
      xt::strided_view(zVals, {xt::ellipsis(), xt::range(1, _)}) - xt::strided_view(zVals, {xt::ellipsis(), xt::range(_, -1)}),
      xt::broadcast(xt::xarray<float>({1e10}), xt::strided_view(zVals, {xt::ellipsis(), xt::range(_, 1)}).shape())));

  // Convert xtensors to 1D vectors
  std::vector<float> pts_vec(pts.begin(), pts.end());
  std::vector<float> pts_flat_vec(pts_flat.begin(), pts_flat.end());
  std::vector<float> dists_vec(dists.begin(), dists.end());

  // OK to return xtensor datastructures instead of vectors?
  return {pts_vec, pts_flat_vec, dists_vec};
}

// Dists needs to be injected into the loss fn.
// pts_flat is the input of the network, so its already there in the 'predictions' var.
// Within loss fn: calculate RGB map of the network output and use L2 loss between it and target img
// best hope: CUDA has functions for reductions, piecewise operators, etc.

/*
1. Convert xtensor views to C arrays, return them from computeRays
2. Create new loss function

Unresolved: how to deal w/ residual?



*/