#pragma once

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
#include <xtensor/xnpy.hpp>

using nlohmann::json;
using std::pair;
using std::string;
using std::vector;

using namespace xt::placeholders; // required for `_` to work

/**
 * @brief Compute the bundle of rays passing through all pixels of an image. (There is one ray per pixel.)
 *
 * @param height Height of the image in pixels
 * @param width Width of the image in pixels
 * @param focalLength Focal length of the camera (number of pixels)
 * @param cam2world A 6-DoF rigid-body transform (4x4 matrix) that transforms a 3D point from the camera frame to the world frame.
 * @return A pair containing the following:
 *    ray_origins: A tensor of shape (width, height, 3) denoting the centers of each ray.
 *                 ray_origins[i][j] is the origin of the ray passing through pixel at row index j and column index i.
 *    ray_directions: A tensor of shape (width, height, 3) denoting the direction of each ray (a unit vector).
 */
std::pair<xt::xtensor<float, 3, xt::layout_type::row_major>, xt::xtensor<float, 3, xt::layout_type::row_major>>
get_ray_bundle(size_t imageWidth, size_t imageHeight, float focalLength, xt::xtensor<float, 2, xt::layout_type::row_major> &cam2world)
{
  auto [i, j] = xt::meshgrid(xt::arange<float>(imageWidth), xt::arange<float>(imageHeight));
  // Take transposes to convert from ij indexing to xy
  auto x = xt::transpose(i);
  auto y = xt::transpose(j);

  // Define the camera coordinates
  xt::xtensor<float, 2, xt::layout_type::row_major> xCamera({imageWidth, imageHeight});
  xCamera = (x - (float)imageWidth * 0.5) / focalLength;

  xt::xtensor<float, 2, xt::layout_type::row_major> yCamera({imageWidth, imageHeight});
  yCamera = (y - (float)imageWidth * 0.5) / focalLength;

  // Define the camera vector

  auto rotation = xt::view(cam2world, xt::range(_, 3), xt::range(_, 3));
  auto translation = xt::view(cam2world, xt::range(_, 3), -1);

  auto xCyCzC = xt::strided_view(xt::stack(xt::xtuple(xCamera, -yCamera, -xt::ones_like(x)), 2), {xt::ellipsis(), xt::newaxis(), xt::all()});
  // auto xCyCzC_2 = xt::strided_view(xCyCzC, {xt::ellipsis(), xt::newaxis(), xt::all()});
  auto xWyWzW = xCyCzC * rotation;

  auto ray_directions = xt::sum(xWyWzW, {-1});
  auto ray_origins = xt::broadcast(translation, ray_directions.shape());

  return {ray_origins, ray_directions};
}

/**
 * @brief Compute 3D query points given a "bundle" of rays. The near and far arguments indicate the bounds from within which to sample.
 *
 * @param ray_origins
 * @param ray_directions
 * @param near
 * @param far
 * @param n_samples
 * @param randomize
 * @return std::pair<xt::xtensor<float, 4, xt::layout_type::row_major>, xt::xtensor<float, 1, xt::layout_type::row_major>>
 */
std::pair<xt::xtensor<float, 4, xt::layout_type::row_major>, xt::xtensor<float, 1, xt::layout_type::row_major>>
compute_query_points_from_rays(xt::xtensor<float, 3, xt::layout_type::row_major> &ray_origins, xt::xtensor<float, 3, xt::layout_type::row_major> &ray_directions, float near, float far, int n_samples, bool randomize = false)
{
  auto depth_values = xt::linspace<float>(near, far, n_samples);
  // TODO: Implement randomize functionality
  auto query_pts = xt::strided_view(ray_origins, {xt::ellipsis(), xt::newaxis(), xt::all()}) +
                   xt::strided_view(ray_directions, {xt::ellipsis(), xt::newaxis(), xt::all()}) *
                       xt::strided_view(depth_values, {xt::ellipsis(), xt::all(), xt::newaxis()});

  return {query_pts, depth_values};
}

/**
 * @brief Perform volume rendering to produce RGB image from a radiance field, given the origin of each ray in the bundle, and the
 * sampled depth values along them.
 *
 * @param radiance_field
 * @param ray_origins
 * @param depth_values
 * @return 3D tensor of RGB values for each pixel
 */
xt::xtensor<float, 3, xt::layout_type::row_major>
render_rays(xt::xtensor<float, 4> &radiance_field, xt::xtensor<float, 1, xt::layout_type::row_major> &depth_values)
{
  auto densities_raw = xt::strided_view(radiance_field, {xt::ellipsis(), 3});
  // Calculate ReLU of each density
  auto sigma_a = xt::fmax(densities_raw, xt::zeros_like(densities_raw));

  auto rgb_raw = xt::strided_view(radiance_field, {xt::ellipsis(), xt::range(_, 3)});

  // Calculate sigmoid of each raw rgb output to get 0-1 val for rendering
  auto rgb = 1. / (1. + xt::exp(-rgb_raw));

  auto dists = xt::hstack(xtuple(
      xt::strided_view(depth_values, {xt::ellipsis(), xt::range(1, _)}) - xt::strided_view(depth_values, {xt::ellipsis(), xt::range(_, -1)}),
      xt::broadcast(xt::xarray<float>({1e10}), xt::strided_view(depth_values, {xt::ellipsis(), xt::range(_, 1)}).shape())));

  auto alpha = 1. - xt::exp(-sigma_a * dists);

  // Replicate 'exclusive cumprod' behaviour from tensorflow
  auto cumprod = xt::cumprod(1. - alpha + 1e-10, -1);
  cumprod = xt::roll(cumprod, 1, 2);
  xt::strided_view(cumprod, {xt::ellipsis(), 0}) = 1;
  auto weights = alpha * cumprod;
  auto rgb_map = xt::sum(xt::strided_view(weights, {xt::ellipsis(), xt::newaxis()}) * rgb, -2);
  return rgb_map;
}

xt::xtensor<float, 2> pos_encode(xt::xtensor<float, 2> x)
{
  int L_embed = 6;
  std::vector<xt::xtensor<float, 2>> rets;
  rets.push_back(x);
  for (int i = 0; i < L_embed; i++)
  {
    rets.push_back(xt::sin(pow(2., i) * x));
    rets.push_back(xt::cos(pow(2., i) * x));
  }
  // Each element of rets is { 640000, 3 }. rets is 13 elements long 1 + (6 x 2)
  auto xrets = xt::adapt(rets);
  auto concat = xt::concatenate(xt::xtuple(xrets[0], xrets[1], xrets[2], xrets[3], xrets[4], xrets[5], xrets[6], xrets[7], xrets[8], xrets[9], xrets[10], xrets[11], xrets[12]), 1);
  return concat;
}

xt::xtensor<float, 2> flatten_query_pts(xt::xtensor<float, 4> &query_pts)
{
  auto pts_shape = xt::adapt(query_pts.shape());
  int l = pts_shape[0] * pts_shape[1] * pts_shape[2];
  auto pts_flat = xt::reshape_view(query_pts, {l, 3});
  auto encoded_pts = pos_encode(pts_flat);
  return encoded_pts;
}
