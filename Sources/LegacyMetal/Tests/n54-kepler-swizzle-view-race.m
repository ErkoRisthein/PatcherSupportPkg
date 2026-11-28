// n54-kepler-swizzle-view-race: textures made while another thread makes a swizzled view must not get its swizzle. Item 88
// makes a swizzled view on Kepler by putting the swizzle into the driver's per-format header table for the moment the
// view is made; a plain texture or view of the same format that another thread made in that moment took the swizzle
// too, for good (an R8 texture read (r, r, r, 1) instead of (r, 0, 0, 1)). Fix: MetalOldCompat item 88 applies the
// swizzle only inside the driver's header build for that view, under a lock every header build from the table takes.
// One thread makes swizzled R8 views (RRR1, as MoltenVK for an L8 image) without pause; the main thread makes plain R8
// textures and plain views in batches and samples each, for up to 5 s or until one reads swizzled; the swizzled views
// must still read swizzled. SKIP where swizzled views are nil (item 88 off, Intel). Argument: GPU name (default: NVIDIA).
#import "repro.h"
#import <stdatomic.h>

static NSString *const src = @"#include <metal_stdlib>\nusing namespace metal;\n"
    "kernel void readone(texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]], device float4 *out [[buffer(0)]], constant uint &i [[buffer(1)]]) {\n"
    "    out[i] = t.sample(s, float2(0.5), level(0));\n}\n";

int main(int argc, char **argv) { @autoreleasepool {
    const char *bug = "n54-kepler-swizzle-view-race";
    id<MTLDevice> dev = reproDevice(argc, argv, "NVIDIA");
    if (!dev) return SKIP(bug, "no such GPU");
    const char *name = dev.name.UTF8String;
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[reproUniqueComment() stringByAppendingString:src] options:nil error:&err];
    if (!lib) return FAIL(bug, "compile: %s", err.localizedDescription.UTF8String);
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"readone"] error:&err];
    if (!ps) return FAIL(bug, "pipeline: %s", err.localizedDescription.UTF8String);
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    id<MTLSamplerState> sampler = [dev newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]];
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm width:4 height:4 mipmapped:NO];
    td.storageMode = MTLStorageModeManaged;
    uint8_t texels[16];
    memset(texels, 200, sizeof texels);
    id<MTLTexture> base = [dev newTextureWithDescriptor:td];
    [base replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:texels bytesPerRow:4];
    const MTLTextureSwizzleChannels rrr1 = {MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleOne};
    SEL swizzled = @selector(newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:);
    if (![base respondsToSelector:swizzled]) return SKIP(bug, "%s: no swizzled views", name);
    id<MTLTexture> first = [base newTextureViewWithPixelFormat:MTLPixelFormatR8Unorm textureType:MTLTextureType2D levels:NSMakeRange(0, 1)
                                                        slices:NSMakeRange(0, 1) swizzle:rrr1];
    if (!first) return SKIP(bug, "%s: swizzled views are nil (item 88 off, or not Kepler)", name);

    enum { kBatch = 32 };
    id<MTLBuffer> out = [dev newBufferWithLength:kBatch * 16 options:MTLResourceStorageModeShared];
    // reads each texture of a batch (and slot kBatch - 1 from the extra one, if any) into out
    BOOL (^readAll)(NSArray<id<MTLTexture>> *) = ^BOOL(NSArray<id<MTLTexture>> *textures) {
        memset(out.contents, 0xff, kBatch * 16);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        [ce setComputePipelineState:ps];
        [ce setSamplerState:sampler atIndex:0];
        [ce setBuffer:out offset:0 atIndex:0];
        for (uint32_t i = 0; i < textures.count; i++) {
            [ce setTexture:textures[i] atIndex:0];
            [ce setBytes:&i length:4 atIndex:1];
            [ce dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        }
        [ce endEncoding];
        return reproRun(cb);
    };

    __block atomic_bool stop = false;
    __block atomic_long views = 0;
    dispatch_group_t group = dispatch_group_create();
    dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        while (!atomic_load(&stop)) @autoreleasepool {
            id<MTLTexture> v = [base newTextureViewWithPixelFormat:MTLPixelFormatR8Unorm textureType:MTLTextureType2D levels:NSMakeRange(0, 1)
                                                            slices:NSMakeRange(0, 1) swizzle:rrr1];
            if (v) atomic_fetch_add(&views, 1);
        }
    });
    long made = 0, wrong = 0;
    char detail[200] = "";
    double t0 = CFAbsoluteTimeGetCurrent();
    while (CFAbsoluteTimeGetCurrent() - t0 < 5.0 && !wrong) @autoreleasepool {
        NSMutableArray<id<MTLTexture>> *batch = [NSMutableArray array];
        for (int i = 0; i < kBatch; i++) {
            id<MTLTexture> t;
            if (i & 1) t = [base newTextureViewWithPixelFormat:MTLPixelFormatR8Unorm textureType:MTLTextureType2D levels:NSMakeRange(0, 1) slices:NSMakeRange(0, 1)];
            else {
                t = [dev newTextureWithDescriptor:td];
                [t replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:texels bytesPerRow:4];
            }
            [batch addObject:t];
        }
        if (!readAll(batch)) { atomic_store(&stop, true); dispatch_group_wait(group, DISPATCH_TIME_FOREVER); return FAIL(bug, "%s: command buffer failed", name); }
        const float *o = out.contents;
        for (int i = 0; i < kBatch; i++, made++)
            if (fabsf(o[i * 4] - 200 / 255.0f) > 0.01f || fabsf(o[i * 4 + 1]) > 0.01f || fabsf(o[i * 4 + 3] - 1) > 0.01f) {
                if (!wrong++)
                    snprintf(detail, sizeof detail, "a plain %s read (%.3f, %.3f, %.3f, %.3f), want (0.784, 0, 0, 1)", i & 1 ? "view" : "texture",
                             o[i * 4], o[i * 4 + 1], o[i * 4 + 2], o[i * 4 + 3]);
            }
    }
    atomic_store(&stop, true);
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    long swizzledViews = atomic_load(&views);
    if (wrong) return FAIL(bug, "%s: %ld of %ld plain R8 textures and views made while another thread made %ld swizzled views took the swizzle: %s",
                           name, wrong, made, swizzledViews, detail);
    // and the swizzled views still read swizzled
    id<MTLTexture> last = [base newTextureViewWithPixelFormat:MTLPixelFormatR8Unorm textureType:MTLTextureType2D levels:NSMakeRange(0, 1)
                                                       slices:NSMakeRange(0, 1) swizzle:rrr1];
    if (!readAll(@[first, last])) return FAIL(bug, "%s: command buffer failed", name);
    const float *o = out.contents;
    for (int i = 0; i < 2; i++)
        if (fabsf(o[i * 4 + 1] - 200 / 255.0f) > 0.01f || fabsf(o[i * 4 + 2] - 200 / 255.0f) > 0.01f || fabsf(o[i * 4 + 3] - 1) > 0.01f)
            return FAIL(bug, "%s: a swizzled RRR1 view read (%.3f, %.3f, %.3f, %.3f)", name, o[i * 4], o[i * 4 + 1], o[i * 4 + 2], o[i * 4 + 3]);
    if (swizzledViews < 100) return FAIL(bug, "%s: only %ld swizzled views made in 5 s: the race wasn't exercised", name, swizzledViews);
    return PASS(bug, "%s: %ld plain R8 textures and views made beside %ld swizzled ones read plain; the swizzled ones read RRR1", name, made, swizzledViews);
}}
