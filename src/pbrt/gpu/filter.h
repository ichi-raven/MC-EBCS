#include <pbrt/pbrt.h>

#include <optional>

#ifndef PBRT_GPU_FILTER_H
#define PBRT_GPU_FILTER_H

namespace pbrt {

void invokeTestKernel(int val);

Image meanFilter(const Image &target, const Bounds2i bounds);

Image medianFilter(const Image &target, const Bounds2i bounds);

Image NLMeansFilter(const Image &target, const Image &weightTarget, const Bounds2i bounds,
                    const double sigma = 0.2, const double h = 0.2,
                    const Image* pVarImage = nullptr);

}  // namespace pbrt
#endif
