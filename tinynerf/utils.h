#pragma once
#include <vector>
using std::vector;

float get_focal_from_fov(float fieldOfView, int width);

template <typename T>
vector<T> two_d_to_one_d(vector<vector<T>> matrix)
{
  vector<T> flatVec;

  for (int i = 0; i < matrix.size(); i++)
  {
    for (int j = 0; j < matrix[i].size(); j++)
    {
      flatVec.push_back(matrix[i][j]);
    }
  }

  return flatVec;
}