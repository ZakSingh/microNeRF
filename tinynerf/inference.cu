#include <tiny-cuda-nn/misc_kernels.h>

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

using namespace tcnn;
using std::string;
using std::vector;
using precision_t = network_precision_t;

int main(int argc, char *argv[])
{
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

  const uint32_t n_input_dims = 4;  // 3 + 3 * 2 * n_frequencies;
  const uint32_t n_output_dims = 4; // RGB + density
  const uint32_t batch_size = 1 << 16;

  try
  {
    json config = {
        {"loss", {{"otype", "L2"}}},
        {"optimizer", {
                          {"otype", "Adam"},
                          {"learning_rate", 1e-5},
                          {"beta1", 0.9f},
                          {"beta2", 0.99f},
                      }},
        // {"encoding", {{"otype", "Identity"}, {"scale", 1.0}, {"offset", 0.0}}},
        {"encoding", {
                         {"otype", "Frequency"},
                         {"n_frequencies", n_frequencies},
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

    cudaStream_t inference_stream;
    CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));

    GPUMatrix<float> prediction(n_output_dims, image_width * image_height);

    auto trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(network, optimizer, loss);

    // 0. Load precomputed network weights
    vector<float> weights = load_weights("../nerfdata/tinylego/weights.txt");
    std::cout << "weights: " << weights.size() << std::endl;

    for (auto &p : network->layer_sizes())
    {
      std::cout << p.first << ", " << p.second << std::endl;
    }
    trainer->set_params_full_precision(weights.data(), weights.size());

    // 1. Load a known pose

    json transform_data = read_json("../nerfdata/tinylego/transforms.json");
    string dataset_path = "../nerfdata/tinylego";
    float focal_length = transform_data["camera_angle_x"];

    auto [image_paths, c2ws] = get_image_c2w(transform_data, dataset_path);

    auto pose = c2ws[0];
    auto [ray_origins, ray_directions] = get_ray_bundle(100, 100, focal_length, pose);
    auto [query_pts, depth_values] = compute_query_points_from_rays(ray_origins, ray_directions, near, far, n_samples);
    auto pts_flat = flatten_query_pts(query_pts);
    std::vector<float> pts_vec(pts_flat.begin(), pts_flat.end());
    GPUMatrix<float> inference_batch(pts_vec.data(), n_input_dims, 1024);

    // // 2. Run inference
    network->inference(inference_stream, inference_batch, prediction);

    // // 3. Volume rendering
    // auto radiance_field = xt::adapt(prediction);
    // auto rgb = render_rays(radiance_field, ray_origins, depth_values);
    // std::cout << rgb << std::endl;
  }
  catch (std::exception &e)
  {
    std::cout << "Uncaught exception: " << e.what() << std::endl;
  }

  return EXIT_SUCCESS;
}

// 1. Get TinyNERF to output weights in the exact format needed
// 2. Run the code here and debug as needed

// 30,720
// 16,384
// 1,024