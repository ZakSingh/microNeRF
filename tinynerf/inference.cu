#include <tiny-cuda-nn/misc_kernels.h>
#include <tiny-cuda-nn/networks/fully_fused_mlp.h>

#include <tiny-cuda-nn/config.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "data.h"
#include "utils.h"
#include "nerf.h"
#include <xtensor/xio.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xoperation.hpp>
#include <algorithm>

using namespace tcnn;
using std::string;
using std::vector;
using precision_t = network_precision_t;

std::ostream &operator<<(std::ostream &o, tcnn::MatrixLayout c)
{
  std::cout << static_cast<int>(c);
  return o;
}

template <typename T>
void render_output(const T *output, int n_coords, int image_height, int image_width, int n_samples, int n_output_dims, xt::xtensor<float, 1UL, xt::layout_type::row_major> depth_values)
{
  std::vector<T> host_data(n_coords * n_output_dims);
  CUDA_CHECK_THROW(cudaMemcpy(host_data.data(), output, host_data.size() * sizeof(T), cudaMemcpyDeviceToHost));

  std::vector<float> float_host_data(host_data.size());
  for (size_t i = 0; i < host_data.size(); ++i)
  {
    float_host_data[i] = (float)host_data[i];
  }

  std::vector<std::size_t> rf_shape = {(size_t)image_height, (size_t)image_width, (size_t)n_samples, (size_t)n_output_dims};
  auto radiance_field = xt::adapt(float_host_data, rf_shape);
  auto rgb = render_rays(radiance_field, depth_values);
}

int main(int argc, char *argv[])
{

  // auto n_radiance_field = xt::load_npy<float>("../nerfdata/tinylego/n_rf.npy");
  // auto n_ray_origins = xt::load_npy<float>("../nerfdata/tinylego/n_rayso.npy");
  // auto n_depth_values = xt::load_npy<float>("../nerfdata/tinylego/n_dv.npy");
  // auto rgb = render_rays(n_radiance_field, n_ray_origins, n_depth_values);
  // return EXIT_SUCCESS;

  // 0. Load precomputed weights into network
  // 1. Load a known pose
  // 2. Run inference with the pose
  // 3. Perform volume rendering to output RGB image
  // 4. Write image to file

  const uint32_t image_width = 100;
  const uint32_t image_height = 100;
  const uint8_t n_frequencies = 6;
  const uint8_t n_samples = 64;
  const float near = 2.;
  const float far = 6.;
  // TinyNeRF only uses (xyz) -- no direction
  const uint32_t n_input_dims = 39;
  const uint32_t n_output_dims = 4; // RGB + density

  // Number of (x, y, z) coordinates to compute
  uint32_t n_coords = image_width * image_height * n_samples;

  // try
  // {
  json config = {
      {"loss", {{"otype", "L2"}}},
      {"optimizer", {
                        {"otype", "Adam"},
                        {"learning_rate", 1e-5},
                        {"beta1", 0.9f},
                        {"beta2", 0.99f},
                    }},
      {"encoding", {
                       {"otype", "Identity"},
                   }},
      {"network", {
                      {"otype", "FullyFusedMLP"},
                      {"n_neurons", 64},
                      {"n_hidden_layers", 8},
                      {"activation", "ReLU"},
                      {"output_activation", "None"},
                  }},
  };

  json encoding_opts = config.value("encoding", json::object());
  json loss_opts = config.value("loss", json::object());
  json optimizer_opts = config.value("optimizer", json::object());
  json network_opts = config.value("network", json::object());

  std::shared_ptr<Loss<precision_t>> loss{create_loss<precision_t>(loss_opts)};
  std::shared_ptr<Optimizer<precision_t>> optimizer{create_optimizer<precision_t>(optimizer_opts)};
  std::shared_ptr<NetworkWithInputEncoding<precision_t>> network = std::make_shared<NetworkWithInputEncoding<precision_t>>(n_input_dims, n_output_dims, encoding_opts, network_opts);

  auto trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(network, optimizer, loss);

  // 0. Load precomputed network weights
  vector<float> weights = load_weights("../nerfdata/tinylego/weights.txt");
  // std::cout << "weights: " << weights.size() << std::endl;

  // for (auto &p : network->layer_sizes())
  // {
  //   std::cout << p.second << ", " << p.first << std::endl;
  // }
  trainer->set_params_full_precision(weights.data(), weights.size());

  // 1. Load a known pose

  json transform_data = read_json("../nerfdata/tinylego/transforms.json");
  string dataset_path = "../nerfdata/tinylego";
  // In TinyNeRF dataset, camer_angle_x is the focal length (no need to calculate)
  float focal_length = transform_data["camera_angle_x"];
  auto [image_paths, c2ws] = get_image_c2w(transform_data, dataset_path);
  auto pose = c2ws[20];

  auto [ray_origins, ray_directions] = get_ray_bundle(image_width, image_height, focal_length, pose);
  auto [query_pts, depth_values] = compute_query_points_from_rays(ray_origins, ray_directions, near, far, n_samples);

  auto pts_flat = flatten_query_pts(query_pts);
  std::vector<float> host_pts_vec(pts_flat.begin(), pts_flat.end());
  GPUMemory<float> pts_vec(host_pts_vec.size());
  pts_vec.copy_from_host(host_pts_vec);
  std::cout << "Total number of (x, y, z) points for inference: " << host_pts_vec.size() / n_input_dims << std::endl;

  cudaStream_t inference_stream;
  CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));

  // Note: These are column-major
  GPUMatrix<float> inference_batch(pts_vec.data(), n_input_dims, n_coords);
  GPUMatrix<float> prediction(n_output_dims, n_coords);

  // 2. Run inference
  network->inference(inference_stream, inference_batch, prediction);

  render_output(prediction.data(),
                n_coords,
                image_height,
                image_width,
                n_samples,
                n_output_dims,
                depth_values);

  return EXIT_SUCCESS;
}
