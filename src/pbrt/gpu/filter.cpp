
#include <pbrt/gpu/filter.h>

#include <pbrt/gpu/util.h>

#include <pbrt/util/image.h>

#include <vector>
#include <optional>

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

#ifndef MIN
#define MIN(a, b) (a > b ? b : a)
#endif

namespace pbrt {

void invokeTestKernel(int val) {  // TEST CUDA

    GPUParallelFor("test", 10000, [=] PBRT_GPU(int i) { printf("%d\n", i); });
    GPUWait();

    cudaDeviceSynchronize();
    exit(42);
}

Image meanFilter(const Image &target, const Bounds2i bounds) {
    constexpr int kKernelSize = 9;
    constexpr int kw = kKernelSize / 2;
    constexpr int kh = kKernelSize / 2;

    Image filtered(PixelFormat::Float, bounds.pMax, {"R", "G", "B"});

    ParallelFor2D(bounds, [&](const Bounds2i tileBounds) {
        float rgb[3] = {};

        for (const auto pPixel : tileBounds) {
            for (int color = 0; color < 3; ++color) {
                float sum = 0;
                const auto xfrom = std::max(0, pPixel.x - kw);
                const auto xto = std::min(tileBounds.pMax.x, pPixel.x + kw);
                const auto yfrom = std::max(0, pPixel.y - kh);
                const auto yto = std::min(tileBounds.pMax.y, pPixel.y + kh);
                const auto area = (yto - yfrom) * (xto - xfrom);

                for (int i = xfrom; i <= xto; ++i) {
                    for (int j = yfrom; j <= yto; ++j) {
                        const Point2i pnew(i, j);
                        sum += target.GetChannels(pnew)[color];
                    }
                }

                rgb[color] = sum / static_cast<float>(area);
            }

            filtered.SetChannels(pPixel, rgb);
        }
    });

    return filtered;
}

Image medianFilter(const Image &target, const Bounds2i bounds) {
    constexpr int kKernelSize = 7;
    constexpr int kw = kKernelSize / 2;
    constexpr int kh = kKernelSize / 2;

    Image filtered(PixelFormat::Float, bounds.pMax, {"R", "G", "B"});

    ParallelFor2D(bounds, [&](const Bounds2i tileBounds) {
        std::vector<float> vals(kKernelSize * kKernelSize, 0);
        float rgb[3] = {};

        for (const auto pPixel : tileBounds) {
            for (int color = 0; color < 3; ++color) {
                const auto xfrom = std::max(0, pPixel.x - kw);
                const auto xto = std::min(tileBounds.pMax.x, pPixel.x + kw);
                const auto yfrom = std::max(0, pPixel.y - kh);
                const auto yto = std::min(tileBounds.pMax.y, pPixel.y + kh);

                vals.clear();

                for (int i = xfrom; i <= xto; ++i) {
                    for (int j = yfrom; j <= yto; ++j) {
                        const Point2i pnew(i, j);
                        vals.emplace_back(target.GetChannels(pnew)[color]);
                    }
                }

                std::sort(vals.begin(), vals.end());
                rgb[color] = vals[vals.size() / 2];
            }

            filtered.SetChannels(pPixel, rgb);
        }
    });

    return filtered;
}

Image NLMeansFilter(const Image &target, const Image &weightTarget, const Bounds2i bounds,
                    const double sigma, const double h,
                    const Image* pVarImage) {
#define USE_GPU  // use GPU

#ifdef USE_GPU
#define MODE PBRT_GPU
#else
#define MODE
#endif

    constexpr int kKernelSize = 5;
    constexpr int kSupportSize = 13;
    constexpr int kw = kKernelSize / 2;
    constexpr int kh = kKernelSize / 2;
    constexpr int ksw = kSupportSize / 2;
    constexpr int ksh = kSupportSize / 2;
    constexpr auto kKernelArea = kKernelSize * kKernelSize;

    Image filtered(PixelFormat::Float, bounds.pMax, {"R", "G", "B"});

    const auto width = bounds.pMax.x - bounds.pMin.x;
    const auto height = bounds.pMax.y - bounds.pMin.y;
    const auto defaultVar = sigma * sigma;

    RGB *pTarget = nullptr;
    RGB *pWeightTarget = nullptr;
    float *pVariance = nullptr;
    RGB *pResult = nullptr;
    RGB *pHostTarget = nullptr;
    RGB *pHostWeightTarget = nullptr;
    float *pHostVariance = nullptr;
    RGB *pHostResult = nullptr;
    const size_t pointsSize = kKernelArea * sizeof(Point2i);
    const size_t imageSize = height * width * sizeof(RGB);
    const size_t varianceSize = height * width * sizeof(float);

    // allcate host memory
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&pHostTarget), imageSize));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&pHostWeightTarget), imageSize));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&pHostVariance), varianceSize));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&pHostResult), imageSize));

    // copy to device memory if use GPU
#ifdef USE_GPU

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&pTarget), imageSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&pWeightTarget), imageSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&pVariance), varianceSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&pResult), imageSize));

    // copy images to device memory
    ParallelFor2D(bounds, [&](const Bounds2i tileBounds) {
        for (const auto pPixel : tileBounds) {
            pHostTarget[pPixel.y * height + pPixel.x].r = target.GetChannels(pPixel)[0];
            pHostTarget[pPixel.y * height + pPixel.x].g = target.GetChannels(pPixel)[1];
            pHostTarget[pPixel.y * height + pPixel.x].b = target.GetChannels(pPixel)[2];

            pHostWeightTarget[pPixel.y * height + pPixel.x].r =
                weightTarget.GetChannels(pPixel)[0];
            pHostWeightTarget[pPixel.y * height + pPixel.x].g =
                weightTarget.GetChannels(pPixel)[1];
            pHostWeightTarget[pPixel.y * height + pPixel.x].b =
                weightTarget.GetChannels(pPixel)[2];

            if (pVarImage) {
                pHostVariance[pPixel.y * height + pPixel.x] =
                    pVarImage->GetChannel(pPixel, 0);
            }
            else
            {
                pHostVariance[pPixel.y * height + pPixel.x] = defaultVar;
            }
        }
    });
    // copy host to device
    cudaMemcpyAsync(pTarget, pHostTarget, imageSize, cudaMemcpyDefault);
    cudaMemcpyAsync(pWeightTarget, pHostWeightTarget, imageSize, cudaMemcpyDefault);
    cudaMemcpyAsync(pVariance, pHostVariance, varianceSize, cudaMemcpyDefault);
    GPUWait();

#else  // otherwise, use host memory
    pTarget = pHostTarget;
    pWeightTarget = pHostWeightTarget;
    pVariance = pHostVariance;
    pResult = pHostResult;
#endif

    // calc distance between target kernel and neighborhood kernel
    const auto calcKernelDistSq = [=] MODE(const Point2i ps[],
                                           const Point2i qs[]) -> double {
        double distSq = 0;

        for (int i = 0; i < kKernelArea; ++i) {
            const double rd = pWeightTarget[ps[i].y * height + ps[i].x].r -
                              pWeightTarget[qs[i].y * height + qs[i].x].r;
            const double gd = pWeightTarget[ps[i].y * height + ps[i].x].g -
                              pWeightTarget[qs[i].y * height + qs[i].x].g;
            const double bd = pWeightTarget[ps[i].y * height + ps[i].x].b -
                              pWeightTarget[qs[i].y * height + qs[i].x].b;

            distSq += rd * rd + gd * gd + bd * bd;
        }

        return distSq;
    };

    // calc weight between target kernel and neighborhood kernel
    const auto calcWeight = [=] MODE(const Point2i ps[], const Point2i qs[],
                                     const Point2i targetPixel) -> double {
        const double var = pVariance[targetPixel.y * height + targetPixel.x];
      
        const auto arg = -std::max(0., calcKernelDistSq(ps, qs) - 2. * var);

        return exp(arg / (h * h));
    };

    // get kernel area pixels
    const auto sampleKernel = [=] MODE(Point2i p, Bounds2i bounds, Point2i * pKernel) {
        size_t now = 0;

        for (int i = p.x - kw; i <= p.x + kw; ++i) {
            for (int j = p.y - kh; j <= p.y + kh; ++j) {
                const auto x = std::min(bounds.pMax.x - 1, std::max(bounds.pMin.x, i));
                const auto y = std::min(bounds.pMax.y - 1, std::max(bounds.pMin.y, j));
                assert(now < kKernelArea);
                pKernel[now++] = Point2i(x, y);
            }
        }
    };

    const auto runFilter = [=] MODE(size_t threadIdx) {
        const Point2i pPixel(threadIdx % width, threadIdx / width);
        RGB rgb{};
        double weightSum = 0;

        const auto xsfrom = std::max(0, pPixel.x - ksw);
        const auto xsto = std::min(bounds.pMax.x - 1, pPixel.x + ksw);
        const auto ysfrom = std::max(0, pPixel.y - ksh);
        const auto ysto = std::min(bounds.pMax.y - 1, pPixel.y + ksh);

        Point2i pKernelPixels[kKernelArea];
        Point2i pFocusPixels[kKernelArea];

        // set kernel targets to pTargets
        sampleKernel(pPixel, bounds, pFocusPixels);

        for (int i1 = xsfrom; i1 <= xsto; ++i1) {
            for (int j1 = ysfrom; j1 <= ysto; ++j1) {
                const Point2i pnew(i1, j1);
                sampleKernel(pnew, bounds, pKernelPixels);
                const Float weight = calcWeight(pFocusPixels, pKernelPixels, pPixel);
                rgb += weight * pTarget[pnew.y * height + pnew.x];

                weightSum += weight;
            }
        }
        assert(weightSum != 0 || !"weightsum should be non zero!");

        // normalize
        rgb /= weightSum;

        // set final value
        pResult[pPixel.y * height + pPixel.x] = rgb;
    };

#ifdef USE_GPU
    GPUParallelFor("test", bounds.Area(), runFilter);

    {  // copy all results to host memory
        cudaMemcpyAsync(pHostResult, pResult, imageSize, cudaMemcpyDefault);
        GPUWait();
    }

    CUDA_CHECK(cudaFree(pTarget));
    CUDA_CHECK(cudaFree(pWeightTarget));
    CUDA_CHECK(cudaFree(pVariance));
    CUDA_CHECK(cudaFree(pResult));

    CUDA_CHECK(cudaFreeHost(pHostResult));
#else
    ParallelFor(0, bounds.Area(), runFilter);
#endif

    // create result image
    ParallelFor2D(bounds, [&](const Bounds2i tileBounds) {
        float res[3];
        for (const auto pPixel : tileBounds) {
            res[0] = pHostResult[pPixel.y * height + pPixel.x].r;
            res[1] = pHostResult[pPixel.y * height + pPixel.x].g;
            res[2] = pHostResult[pPixel.y * height + pPixel.x].b;

            filtered.SetChannels(pPixel, res);
        }
    });

    CUDA_CHECK(cudaFreeHost(pHostTarget));
    CUDA_CHECK(cudaFreeHost(pHostWeightTarget));
    CUDA_CHECK(cudaFreeHost(pHostVariance));

    return filtered;
}

}  // namespace pbrt