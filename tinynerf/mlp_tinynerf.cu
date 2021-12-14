/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/*
 */

/** @file   mlp-learning-an-image.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  Sample application that uses the tiny cuda nn framework to learn a
						2D function that represents an image.
 */
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
#include <xtensor/xio.hpp>

using namespace tcnn;
using precision_t = network_precision_t;

std::vector<float> load_weights(const std::string &filename)
{
	std::cout << "Loading weights" << std::endl;
	std::vector<float> weightsBuffer;
	std::ifstream in(filename);

	for (std::string f; getline(in, f, '\n');)
		weightsBuffer.push_back(std::stof(f));

	// for (auto f : ReplayBuffer)
	// 	std::cout << f << ", ";
	// std::cout << std::endl;
	return weightsBuffer;
}

void train()
{
	std::string datasetPath = "../tinynerfdata/lego";
	int imageWidth = 800;
	int imageHeight = 800;
	float near = 2.0;
	float far = 6.0;

	int n_C = 32;
	int n_F = 64;

	json jsonTrainData = read_json("../tinynerfdata/lego/transforms_train.json");
	json jsonValData = read_json("../tinynerfdata/lego/transforms_val.json");
	json jsonTestData = read_json("../tinynerfdata/lego/transforms_test.json");

	float focalLength = get_focal_from_fov(jsonTrainData["camera_angle_x"], imageWidth);
	std::cout << "focalLength: " << focalLength << std::endl;

	auto [trainImagePaths, trainC2Ws] = get_image_c2w(jsonTrainData, datasetPath);

	// Load the image dataset into memory
	auto getImages = GetImages();
	vector<vector<vector<vector<float>>>> images;
	// for (auto it : trainImagePaths)
	// 	// TODO: this should ideally be parallelized
	// 	images.push_back(getImages.load_image(it));

	// Compute rays
	auto getRays = GetRays(focalLength, imageWidth, imageHeight, near, far, n_C);
	vector<std::tuple<vector<float>, vector<float>, vector<float>>> rays;
	for (auto c2w : trainC2Ws)
	{
		rays.push_back(getRays.computeRays(c2w));
		return;
	}
}

int main(int argc, char *argv[])
{
	train();

	// if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2)))
	// {
	// 	std::cout << "Turing Tensor Core operations must be compiled with CUDA 10.2 Toolkit or later." << std::endl;
	// 	return -1;
	// }

	// cudaDeviceProp props;

	// cudaError_t error = cudaGetDeviceProperties(&props, 0);
	// if (error != cudaSuccess)
	// {
	// 	std::cout << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
	// 	return -1;
	// }

	// if (!((props.major * 10 + props.minor) >= 75))
	// {
	// 	std::cout << "Turing Tensor Core operations must be run on a machine with compute capability at least 75."
	// 						<< std::endl;
	// 	return -1;
	// }

	// if (argc < 2)
	// {
	// 	std::cout << "USAGE: " << argv[0] << " "
	// 						<< "path-to-image.exr [path-to-optional-config.json]" << std::endl;
	// 	std::cout << "Sample EXR files are provided in 'data/images'." << std::endl;
	// 	return 0;
	// }

	// try
	// {
	// 	json config = {
	// 			{"loss", {{"otype", "L2"}}},
	// 			{"optimizer", {
	// 												{"otype", "Adam"},
	// 												{"learning_rate", 1e-5},
	// 												{"beta1", 0.9f},
	// 												{"beta2", 0.99f},
	// 										}},
	// 			{"encoding", {
	// 											 {"otype", "Frequency"},
	// 											 {"n_frequencies", 6},
	// 									 }},
	// 			{"network", {
	// 											{"otype", "FullyFusedMLP"},
	// 											{"n_neurons", 64},
	// 											{"n_layers", 8},
	// 											{"activation", "ReLU"},
	// 											{"output_activation", "None"},
	// 									}},
	// 	};

	// 	if (argc >= 3)
	// 	{
	// 		std::cout << "Loading custom json config '" << argv[2] << "'." << std::endl;
	// 		std::ifstream f{argv[2]};
	// 		config = json::parse(f, nullptr, true, /*skip_comments=*/true);
	// 	}

	// 	// First step: Load precomputed weights from .npy file into CPU memory
	// 	std::vector<float> weights = load_weights(argv[1]);
	// 	// Various constants for the network and optimization

	// 	const uint32_t n_input_dims = 3 + (3 * 2 * 6); //
	// 	const uint32_t n_output_dims = 4;							 // RGB color + density

	// 	cudaStream_t inference_stream;
	// 	CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));

	// Auxiliary matrices for evaluation
	// TODO: Figure out params here
	// GPUMatrix<float> prediction(n_output_dims, n_coords_padded);
	// GPUMatrix<float> inference_batch(xs_and_ys.data(), n_input_dims, n_coords_padded);

	// json encoding_opts = config.value("encoding", json::object());
	// json loss_opts = config.value("loss", json::object());
	// json optimizer_opts = config.value("optimizer", json::object());
	// json network_opts = config.value("network", json::object());

	// std::shared_ptr<Loss<precision_t>> loss{create_loss<precision_t>(loss_opts)};
	// std::shared_ptr<Optimizer<precision_t>> optimizer{create_optimizer<precision_t>(optimizer_opts)};
	// std::shared_ptr<NetworkWithInputEncoding<precision_t>> network = std::make_shared<NetworkWithInputEncoding<precision_t>>(n_input_dims, n_output_dims, encoding_opts, network_opts);

	// auto trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(network, optimizer, loss);
	// trainer.set_params_full_precision(weights.data());

	// const float focal = 138.88887889922103;
	// const uint32_t n_samples = 64 // The number of samples along each ray

	// 		// Dump final image if a name was specified
	// 		if (argc >= 5)
	// {
	// 	network->inference(inference_stream, inference_batch, prediction);
	// 	// save_image(prediction.data(), sampling_width, sampling_height, 3, n_output_dims, argv[4]);
	// }
	// }
	// catch (std::exception &e)
	// {
	// 	std::cout << "Uncaught exception: " << e.what() << std::endl;
	// }

	return EXIT_SUCCESS;
}
