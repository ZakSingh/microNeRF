/*
Custom photometric loss function replicating that of NeRF.
Defined as the L2 loss b/w the RGB values of all pixels in the image.


*/

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/misc_kernels.h>
#include <tiny-cuda-nn/loss.h>

TCNN_NAMESPACE_BEGIN

template <typename T>
__global__ void photometric_loss(
    const uint32_t n_elements,
    const uint32_t stride,
    const uint32_t dims,
    const float loss_scale,
    const T *__restrict__ predictions,
    const float *__restrict__ targets,
    float *__restrict__ values,
    T *__restrict__ gradients,
    const float *__restrict__ data_pdf = nullptr)
{
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n_elements)
    return;

  const uint32_t intra_elem_idx = i % stride;
  const uint32_t inter_elem_idx = i / stride;
  if (intra_elem_idx >= dims)
  {
    values[i] = 0;
    gradients[i] = 0;
    return;
  }

  const uint32_t target_idx = inter_elem_idx * dims + intra_elem_idx;

  const uint32_t n_total = n_elements / stride * dims;

  const float prediction = (float)predictions[i];

  const float pdf = data_pdf ? data_pdf[target_idx] : 1;
  const float difference = prediction - targets[target_idx] / pdf;

  values[i] = difference * difference / n_total;

  float gradient = 2 * difference;
  gradients[i] = (T)(loss_scale * gradient / n_total);
}

template <typename T>
class PhotometricLoss : public Loss<T>
{
public:
  void evaluate(
      cudaStream_t stream,
      const uint32_t stride,
      const uint32_t dims,
      const float loss_scale,
      const GPUMatrix<T> &prediction,
      const GPUMatrix<float> &target,
      GPUMatrix<float> &values,
      GPUMatrix<T> &gradients,
      const GPUMatrix<float> *data_pdf = nullptr) const override
  {
    if (prediction.n() != target.n())
    {
      throw std::runtime_error(std::string("Prediction and target don't have matching batch size ") + std::to_string(prediction.n()) + "!=" + std::to_string(target.n()));
    }

    if (prediction.m() != stride)
    {
      throw std::runtime_error(std::string("Prediction does not have appropriate dimensions ") + std::to_string(prediction.m()) + "!=" + std::to_string(stride));
    }

    if (target.m() != dims)
    {
      throw std::runtime_error(std::string("Target does not have appropriate dimensions ") + std::to_string(target.m()) + "!=" + std::to_string(dims));
    }

    linear_kernel(photometric_loss<T>, 0, stream,
                  prediction.n_elements(),
                  stride,
                  dims,
                  loss_scale,
                  prediction.data(),
                  target.data(),
                  values.data(),
                  gradients.data(),
                  data_pdf ? data_pdf->data() : nullptr);
  }

  void update_hyperparams(const json &params) override {}
};

TCNN_NAMESPACE_END