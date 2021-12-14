#include <math.h>
#include <vector>
#include "utils.h"
using std::vector;

float get_focal_from_fov(float fieldOfView, int width)
{
  float focalLength = 0.5 * (float)width / tan(0.5 * fieldOfView);
  return focalLength;
}
