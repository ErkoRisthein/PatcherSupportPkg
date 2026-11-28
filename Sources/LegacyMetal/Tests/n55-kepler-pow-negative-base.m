// n55-kepler-pow-negative-base: pow(x, y) and powr(x, y) under fast math (Metal's default; SPIRV-Cross writes GLSL's pow
// as powr) with x < 0 must not be NaN: other Macs' GPUs give |x|^y (Intel: for every exponent, odd ones included), and
// Kepler's fast pow and powr are exp2(y * log2(x)), NaN for every negative x, so RetroArch's bicubic shaders (MoltenVK:
// weights from powr(x, 2.0) of negative distances) and yee64 drew black. Fix: GeForceAIRCompat's fast pow / powr calls
// take |x| (NVAIR_POW_SIGN_FIX). Checks pow(x, 2.0), pow(x, 3.0), pow(x, -2.0), pow(x, 2.2), powr(x, 2.0), a float3 pow
// with exponents (2, 3, 4) and pow(x, y) with y = 2 read at run time, of negative and positive x read from a buffer, each
// against |x|^y, in a compute kernel and in a fragment function. Argument: GPU name (default: NVIDIA).
#import "repro.h"

static NSString *const src = @"#include <metal_stdlib>\nusing namespace metal;\n"
    "static float4 powers(float x, float y) { float3 v = pow(float3(x), float3(2.0, 3.0, 4.0));\n"
    "    return float4(pow(x, 2.0) + 10.0 * powr(x, 2.0), pow(x, 3.0), pow(x, -2.0) + pow(x, y), v.x + v.y + v.z + pow(x, 2.2)); }\n"
    "kernel void k(device const float *in [[buffer(0)]], device float4 *out [[buffer(1)]], uint i [[thread_position_in_grid]]) { out[i] = powers(in[i], in[6]); }\n"
    "struct V { float4 p [[position]]; };\n"
    "vertex V vs(uint id [[vertex_id]]) { V o; o.p = float4(id == 1 ? 3.0 : -1.0, id == 2 ? 3.0 : -1.0, 0, 1); return o; }\n"
    "fragment float4 fs(V v [[stage_in]], device const float *in [[buffer(0)]]) { return powers(in[uint(v.p.x)], in[6]); }\n";

int main(int argc, char **argv) { @autoreleasepool {
    const char *bug = "n55-kepler-pow-negative-base";
    id<MTLDevice> dev = reproDevice(argc, argv, "NVIDIA");
    if (!dev) return SKIP(bug, "no such GPU");
    const char *name = dev.name.UTF8String;
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[reproUniqueComment() stringByAppendingString:src] options:nil error:&err];  // fast math: the default
    if (!lib) return FAIL(bug, "compile: %s", err.localizedDescription.UTF8String);
    id<MTLComputePipelineState> cps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k"] error:&err];
    if (!cps) return FAIL(bug, "%s: compute pipeline: %s", name, err.localizedDescription.UTF8String);
    MTLRenderPipelineDescriptor *rpd = [MTLRenderPipelineDescriptor new];
    rpd.vertexFunction = [lib newFunctionWithName:@"vs"];
    rpd.fragmentFunction = [lib newFunctionWithName:@"fs"];
    rpd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
    id<MTLRenderPipelineState> rps = [dev newRenderPipelineStateWithDescriptor:rpd error:&err];
    if (!rps) return FAIL(bug, "%s: render pipeline: %s", name, err.localizedDescription.UTF8String);
    enum { kCount = 6 };
    const float xs[kCount + 1] = {-1.5f, -0.5f, -2.0f, -3.0f, 0.5f, 2.0f, 2.0f};  // the last: the run-time exponent
    id<MTLBuffer> in = [dev newBufferWithBytes:xs length:sizeof xs options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [dev newBufferWithLength:kCount * 16 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    if (!reproDispatch(queue, cps, @[in, out], MTLSizeMake(1, 1, 1), MTLSizeMake(kCount, 1, 1)))
        return FAIL(bug, "%s: compute command buffer failed", name);
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:kCount height:1 mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget;
    td.storageMode = MTLStorageModeManaged;
    id<MTLTexture> target = [dev newTextureWithDescriptor:td];
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:pass];
    [re setRenderPipelineState:rps];
    [re setFragmentBuffer:in offset:0 atIndex:0];
    [re drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [re endEncoding];
    id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
    [be synchronizeResource:target];
    [be endEncoding];
    if (!reproRun(cb)) return FAIL(bug, "%s: render command buffer failed", name);
    float drawn[kCount * 4];
    [target getBytes:drawn bytesPerRow:sizeof drawn fromRegion:MTLRegionMake2D(0, 0, kCount, 1) mipmapLevel:0];
    for (int stage = 0; stage < 2; stage++) {
        const float *got = stage ? drawn : (const float *)out.contents;
        for (int i = 0; i < kCount; i++) {
            float a = fabsf(xs[i]);  // |x|^y, as Intel's fast pow / powr give it
            float want[4] = {11 * a * a, a * a * a, 1 / (a * a) + a * a, a * a + a * a * a + a * a * a * a + powf(a, 2.2f)};
            for (int c = 0; c < 4; c++)
                if (!(fabsf(got[i * 4 + c] - want[c]) <= 1e-3f * fmaxf(1, fabsf(want[c]))))
                    return FAIL(bug, "%s: %s: x = %g: %s read %g, want %g (|x|^y)", name, stage ? "fragment function" : "compute kernel", xs[i],
                                (const char *[]){"pow(x, 2.0) + 10 powr(x, 2.0)", "pow(x, 3.0)", "pow(x, -2.0) + pow(x, y = 2 at run time)",
                                                 "the sum of pow(float3(x), (2, 3, 4)) and pow(x, 2.2)"}[c], got[i * 4 + c], want[c]);
        }
    }
    return PASS(bug, "%s: fast pow / powr of negative x is |x|^y (exponents 2, 3, -2, 2.2, a float3 of 2, 3, 4, one read at run time), compute and fragment", name);
}}
