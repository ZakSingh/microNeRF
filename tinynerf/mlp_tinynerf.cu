// /*
//  * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
//  *
//  * Redistribution and use in source and binary forms, with or without modification, are permitted
//  * provided that the following conditions are met:
//  *     * Redistributions of source code must retain the above copyright notice, this list of
//  *       conditions and the following disclaimer.
//  *     * Redistributions in binary form must reproduce the above copyright notice, this list of
//  *       conditions and the following disclaimer in the documentation and/or other materials
//  *       provided with the distribution.
//  *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
//  *       to endorse or promote products derived from this software without specific prior written
//  *       permission.
//  *
//  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
//  * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
//  * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
//  * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
//  * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
//  * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
//  * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//  * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//  */
// /*
//  */

// /** @file   mlp-learning-an-image.cu
//  *  @author Thomas Müller, NVIDIA
//  *  @brief  Sample application that uses the tiny cuda nn framework to learn a
// 						2D function that represents an image.
//  */
// #include <tiny-cuda-nn/misc_kernels.h>

// #include <tiny-cuda-nn/config.h>

// #include <chrono>
// #include <cstdlib>
// #include <fstream>
// #include <iostream>
// #include <random>
// #include <stdexcept>
// #include <string>
// #include <thread>
// #include <vector>

// #include "data.h"
// #include "utils.h"
// #include "nerf.h"
// #include <xtensor/xio.hpp>

// using namespace tcnn;
// using precision_t = network_precision_t;

// void train()
// {
// 	std::string datasetPath = "../tinynerfdata/tinylego";
// 	int imageWidth = 100;
// 	int imageHeight = 100;
// 	float near = 2.0;
// 	float far = 6.0;
// 	int n_samples = 64;

// 	json jsonTrainData = read_json("../tinynerfdata/tinylego/transforms.json");

// 	// float focalLength = get_focal_from_fov(jsonTrainData["camera_angle_x"], imageWidth);
// 	float focal_length = jsonTrainData["camera_angle_x"];
// 	std::cout << "focalLength: " << focal_length << std::endl;

// 	auto [trainImagePaths, train_c2ws] = get_image_c2w(jsonTrainData, datasetPath);

// 	// Load the image dataset into memory
// 	auto getImages = GetImages();
// 	vector<vector<vector<vector<float>>>> images;
// 	// for (auto it : trainImagePaths)
// 	// 	// TODO: this should ideally be parallelized
// 	// 	images.push_back(getImages.load_image(it));

// 	auto [ray_origins, ray_directions] = get_ray_bundle(100, 100, focal_length, train_c2ws[0]);
// 	auto [query_pts, depth_values] = compute_query_points_from_rays(ray_origins, ray_directions, near, far, n_samples);
// 	auto pts_flat = flatten_query_pts(query_pts);

// 	std::cout << pts_flat << std::endl;

// 	// Compute rays
// 	// auto getRays = GetRays(focalLength, imageWidth, imageHeight, near, far, 64);
// 	// vector<std::tuple<vector<float>, vector<float>, vector<float>>> rays;
// 	// for (auto c2w : trainC2Ws)
// 	// {
// 	// 	rays.push_back(getRays.computeRays(c2w));
// 	// 	return;
// 	// }
// }

// // int main(int argc, char *argv[])
// // {
// // 	train();

// // 	if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2)))
// // 	{
// // 		std::cout << "Turing Tensor Core operations must be compiled with CUDA 10.2 Toolkit or later." << std::endl;
// // 		return -1;
// // 	}

// // 	cudaDeviceProp props;

// // 	cudaError_t error = cudaGetDeviceProperties(&props, 0);
// // 	if (error != cudaSuccess)
// // 	{
// // 		std::cout << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
// // 		return -1;
// // 	}

// // 	if (!((props.major * 10 + props.minor) >= 75))
// // 	{
// // 		std::cout << "Turing Tensor Core operations must be run on a machine with compute capability at least 75."
// // 							<< std::endl;
// // 		return -1;
// // 	}

// // 	if (argc < 2)
// // 	{
// // 		std::cout << "USAGE: " << argv[0] << " "
// // 							<< "path-to-image.exr [path-to-optional-config.json]" << std::endl;
// // 		std::cout << "Sample EXR files are provided in 'data/images'." << std::endl;
// // 		return 0;
// // 	}

// // 	try
// // 	{
// // 		json config = {
// // 				{"loss", {{"otype", "photometric"}}},
// // 				{"optimizer", {
// // 													{"otype", "Adam"},
// // 													{"learning_rate", 1e-5},
// // 													{"beta1", 0.9f},
// // 													{"beta2", 0.99f},
// // 											}},
// // 				{"encoding", {
// // 												 {"otype", "Frequency"},
// // 												 {"n_frequencies", 6},
// // 										 }},
// // 				{"network", {
// // 												{"otype", "FullyFusedMLP"},
// // 												{"n_neurons", 64},
// // 												{"n_layers", 8},
// // 												{"activation", "ReLU"},
// // 												{"output_activation", "None"},
// // 										}},
// // 		};

// // 		const uint32_t batch_size = 1 << 16;
// // 		const uint32_t n_training_steps = 1000;
// // 		const uint32_t n_frequencies = 6;
// // 		const uint32_t n_input_dims = 3 + 3 * 2 * n_frequencies;
// // 		const uint32_t n_output_dims = 4; // RGB + density

// // 		const uint32_t image_width = 100;
// // 		const uint32_t image_height = 100;
// // 		const uint32_t n_coords = image_width * image_height;

// // 		// Holds 2d pixel coordinates for all pixels in image
// // 		GPUMemory<float> xs_and_ys(n_coords * 2);

// // 		cudaStream_t inference_stream;
// // 		CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));
// // 		cudaStream_t training_stream = inference_stream;

// // 		default_rng_t rng{1337};

// // 		// Our training target is photometric (only RGB -- no density)
// // 		GPUMatrix<float> training_target(n_output_dims - 1, batch_size);
// // 		GPUMatrix<float> training_batch(n_input_dims, batch_size);

// // 		// A 'prediction' constitutes an RGB + density for each pixel in the input
// // 		GPUMatrix<float> prediction(n_output_dims, n_coords);
// // 		// An inference batch is a 3D matrix
// // 		// [pixel_loc * inputs * total_pixels]
// // 		// i.e. for each 2d pixel location, it has a 4D vector with values for all coordinates
// // 		GPUMatrix<float> inference_batch(xs_and_ys.data(), n_input_dims, n_coords);

// // 		json encoding_opts = config.value("encoding", json::object());
// // 		json loss_opts = config.value("loss", json::object());
// // 		json optimizer_opts = config.value("optimizer", json::object());
// // 		json network_opts = config.value("network", json::object());

// // 		std::shared_ptr<Loss<precision_t>> loss{create_loss<precision_t>(loss_opts)};
// // 		std::shared_ptr<Optimizer<precision_t>> optimizer{create_optimizer<precision_t>(optimizer_opts)};
// // 		std::shared_ptr<NetworkWithInputEncoding<precision_t>> network = std::make_shared<NetworkWithInputEncoding<precision_t>>(n_input_dims, n_output_dims, encoding_opts, network_opts);

// // 		auto trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(network, optimizer, loss);

// // 		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

// // 		float tmp_loss = 0;
// // 		uint32_t tmp_loss_counter = 0;

// // 		std::cout << "Beginning optimization with " << n_training_steps << " training steps." << std::endl;

// // 		for (uint32_t i = 0; i < n_training_steps; ++i)
// // 		{
// // 			bool print_loss = i % 1000 == 0;
// // 			bool visualize_learned_func = i % 100 == 0;

// // 			// (here)
// // 			// Sample a random image from the dataset
// // 			// Compute rays for all pixels in that image

// // 			// (inside training step in tinycudann)
// // 			// Perform volume rendering to get RGB values of network output
// // 			// Compare to target image to calculate loss

// // 			// If I implement volume rendering w/ xtensor, I could use tinycudann for inference only
// // 			float loss_value;

// // 			trainer->training_step(training_stream, training_batch, training_target, &loss_value);
// // 			tmp_loss += loss_value;
// // 			++tmp_loss_counter;
// // 		}
// // 	}
// // 	catch (std::exception &e)
// // 	{
// // 		std::cout << "Uncaught exception: " << e.what() << std::endl;
// // 	}

// // 	return EXIT_SUCCESS;
// // }

// // 	// First step: Load precomputed weights from .npy file into CPU memory
// // 	std::vector<float> weights = load_weights(argv[1]);
// // 	// Various constants for the network and optimization

// // 	const uint32_t n_input_dims = 3 + (3 * 2 * 6); //
// // 	const uint32_t n_output_dims = 4;							 // RGB color + density

// // 	cudaStream_t inference_stream;
// // 	CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));

// // Auxiliary matrices for evaluation
// // TODO: Figure out params here
// // GPUMatrix<float> prediction(n_output_dims, n_coords_padded);
// // GPUMatrix<float> inference_batch(xs_and_ys.data(), n_input_dims, n_coords_padded);

// // json encoding_opts = config.value("encoding", json::object());
// // json loss_opts = config.value("loss", json::object());
// // json optimizer_opts = config.value("optimizer", json::object());
// // json network_opts = config.value("network", json::object());

// // std::shared_ptr<Loss<precision_t>> loss{create_loss<precision_t>(loss_opts)};
// // std::shared_ptr<Optimizer<precision_t>> optimizer{create_optimizer<precision_t>(optimizer_opts)};
// // std::shared_ptr<NetworkWithInputEncoding<precision_t>> network = std::make_shared<NetworkWithInputEncoding<precision_t>>(n_input_dims, n_output_dims, encoding_opts, network_opts);

// // auto trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(network, optimizer, loss);
// // trainer.set_params_full_precision(weights.data());

// // const float focal = 138.88887889922103;
// // const uint32_t n_samples = 64 // The number of samples along each ray

// // 		// Dump final image if a name was specified
// // 		if (argc >= 5)
// // {
// // 	network->inference(inference_stream, inference_batch, prediction);
// // 	// save_image(prediction.data(), sampling_width, sampling_height, 3, n_output_dims, argv[4]);
// // }
// // }
// // catch (std::exception &e)
// // {
// // 	std::cout << "Uncaught exception: " << e.what() << std::endl;
// // }