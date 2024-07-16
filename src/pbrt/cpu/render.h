// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_CPU_RENDER_H
#define PBRT_CPU_RENDER_H

#include <pbrt/pbrt.h>

#include <vector>

namespace pbrt {

//class BasicScene;
struct BuiltScene;
struct EAdaptOption;

void RenderCPU(BuiltScene &scene);
void RenderMultipleScenesCPU(std::vector<BuiltScene> &scenes, const EAdaptOption& eAdaptOptions);

}  // namespace pbrt

#endif  // PBRT_CPU_RENDER_H
