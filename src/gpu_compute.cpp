#include "oil/gpu_compute.h"

#ifdef OIL_HAS_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
// Forward declarations of CUDA kernels
void launch_cuda_softmax(float* y, const float* x, int rows, int cols);
void launch_cuda_layernorm(float* y, const float* x, const float* gamma,
                            const float* beta, float eps, int n, int d);
void launch_cuda_rmsnorm(float* y, const float* x, const float* gamma,
                          float eps, int n, int d);
void launch_cuda_relu(float* y, const float* x, int n);
void launch_cuda_gelu(float* y, const float* x, int n);
void launch_cuda_silu(float* y, const float* x, int n);
void launch_cuda_add(float* c, const float* a, const float* b, int n);
void launch_cuda_mul(float* c, const float* a, const float* b, int n);
void launch_cuda_scale(float* y, const float* x, float s, int n);
void launch_cuda_embedding(float* out, const float* table,
                            const int* indices, int B, int S, int D);
void launch_cuda_moe_gather(float* out, const float* x,
                             const int64_t* indices, const float* weights,
                             int T, int K, int D);
void launch_cuda_gemm(float alpha, const float* A, const float* B,
                       float beta, float* C, int M, int N, int K);

// cuBLAS convenience wrapper
static inline cublasStatus_t cublas_gemm(cublasHandle_t handle, int M, int N, int K,
                                          float alpha, const float* A, const float* B,
                                          float beta, float* C) {
    return cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K, &alpha, B, CUDA_R_32F, N,
                        A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
#endif

#if defined(OIL_USE_DIRECTX)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#else
// Stub headers for when DirectX SDK is not available
#include <windows.h>
#endif

#include <cstring>
#include <vector>
#include <unordered_map>
#include <cassert>

namespace oil {
namespace gpu {

// ========================================================================
// HLSL compute shader sources (embedded as strings)
// ========================================================================

static const char* HLSL_GEMM = R"(
RWStructuredBuffer<float> C : register(u0);
StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
cbuffer Params : register(b0) {
    uint M, N, K;
    float alpha, beta;
};
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= N || tid.y >= M) return;
    float sum = 0.0f;
    for (uint i = 0; i < K; ++i)
        sum += A[tid.y * K + i] * B[i * N + tid.x];
    uint idx = tid.y * N + tid.x;
    C[idx] = alpha * sum + beta * C[idx];
}
)";

static const char* HLSL_GEMV = R"(
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> x : register(t1);
cbuffer Params : register(b0) {
    uint M, N;
    float alpha, beta;
};
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= M) return;
    float sum = 0.0f;
    for (uint i = 0; i < N; ++i)
        sum += A[tid.x * N + i] * x[i];
    y[tid.x] = alpha * sum + beta * y[tid.x];
}
)";

static const char* HLSL_RELU = R"(
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float> x : register(t0);
cbuffer Params : register(b0) {
    uint N;
};
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= N) return;
    y[tid.x] = max(x[tid.x], 0.0f);
}
)";

static const char* HLSL_GELU = R"(
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float> x : register(t0);
cbuffer Params : register(b0) {
    uint N;
};
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= N) return;
    float v = x[tid.x];
    y[tid.x] = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
}
)";

static const char* HLSL_SILU = R"(
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float> x : register(t0);
cbuffer Params : register(b0) {
    uint N;
};
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= N) return;
    float v = x[tid.x];
    y[tid.x] = v / (1.0f + exp(-v));
}
)";

static const char* HLSL_ADD = R"(
RWStructuredBuffer<float> c : register(u0);
StructuredBuffer<float> a : register(t0);
StructuredBuffer<float> b : register(t1);
cbuffer Params : register(b0) { uint N; };
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= N) return;
    c[tid.x] = a[tid.x] + b[tid.x];
}
)";

static const char* HLSL_MUL = R"(
RWStructuredBuffer<float> c : register(u0);
StructuredBuffer<float> a : register(t0);
StructuredBuffer<float> b : register(t1);
cbuffer Params : register(b0) { uint N; };
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= N) return;
    c[tid.x] = a[tid.x] * b[tid.x];
}
)";

static const char* HLSL_SCALE = R"(
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float> x : register(t0);
cbuffer Params : register(b0) { uint N; float s; };
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= N) return;
    y[tid.x] = s * x[tid.x];
}
)";

static const char* HLSL_SOFTMAX = R"(
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float> x : register(t1);
cbuffer Params : register(b0) { uint rows, cols; };
groupshared float smem[256];
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= rows) return;
    uint r = tid.x;
    float maxv = x[r * cols];
    for (uint c = 1; c < cols; ++c)
        maxv = max(maxv, x[r * cols + c]);
    float sum = 0.0f;
    for (uint c = 0; c < cols; ++c) {
        float e = exp(x[r * cols + c] - maxv);
        y[r * cols + c] = e;
        sum += e;
    }
    float inv = 1.0f / sum;
    for (uint c = 0; c < cols; ++c)
        y[r * cols + c] *= inv;
}
)";

static const char* HLSL_RMS_NORM = R"(
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float> x : register(t0);
StructuredBuffer<float> gamma : register(t1);
cbuffer Params : register(b0) { uint N, D; float eps; };
groupshared float sq[256];
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= N) return;
    uint r = tid.x;
    float ss = 0.0f;
    for (uint d = 0; d < D; ++d)
        ss += x[r * D + d] * x[r * D + d];
    float rms = rsqrt(ss / (float)D + eps);
    for (uint d = 0; d < D; ++d)
        y[r * D + d] = rms * x[r * D + d] * gamma[d];
}
)";

static const char* HLSL_LAYER_NORM = R"(
RWStructuredBuffer<float> y : register(u0);
StructuredBuffer<float> x : register(t0);
StructuredBuffer<float> gamma : register(t1);
StructuredBuffer<float> beta : register(t2);
cbuffer Params : register(b0) { uint N, D; float eps; };
groupshared float buf[256];
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= N) return;
    uint r = tid.x;
    float sum = 0.0f;
    for (uint d = 0; d < D; ++d)
        sum += x[r * D + d];
    float mean = sum / (float)D;
    float var = 0.0f;
    for (uint d = 0; d < D; ++d) {
        float diff = x[r * D + d] - mean;
        var += diff * diff;
    }
    float inv_std = rsqrt(var / (float)D + eps);
    for (uint d = 0; d < D; ++d)
        y[r * D + d] = (x[r * D + d] - mean) * inv_std * gamma[d] + beta[d];
}
)";

static const char* HLSL_MOE_GATHER = R"(
RWStructuredBuffer<float> out_buf : register(u0);
StructuredBuffer<float> expert_out : register(t0);
StructuredBuffer<int64_t> indices : register(t1);
StructuredBuffer<float> weights : register(t2);
cbuffer Params : register(b0) { uint T, K, D; };
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint t = tid.x / D;
    uint d = tid.x % D;
    if (t >= T || d >= D) return;
    float sum = 0.0f;
    for (uint k = 0; k < K; ++k) {
        int64_t e = indices[t * K + k];
        float w = weights[t * K + k];
        sum += w * expert_out[(uint)e * T * D + t * D + d];
    }
    out_buf[t * D + d] = sum;
}
)";

static const char* HLSL_MOE_SCATTER_ADD = R"(
RWStructuredBuffer<float> expert_grad : register(u0);
StructuredBuffer<float> grad_in : register(t0);
StructuredBuffer<int64_t> indices : register(t1);
StructuredBuffer<float> weights : register(t2);
cbuffer Params : register(b0) { uint T, K, D, E; };
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint t = tid.x / D;
    uint d = tid.x % D;
    if (t >= T || d >= D) return;
    float g = grad_in[t * D + d];
    for (uint k = 0; k < K; ++k) {
        int64_t e = indices[t * K + k];
        float w = weights[t * K + k];
        expert_grad[(uint)e * D + d] += g * w;
    }
}
)";

// ========================================================================
// DirectX 12 Compute Implementation
// ========================================================================

struct DirectXCompute::Impl {
#if defined(OIL_USE_DIRECTX) || !defined(OIL_NO_DIRECTX_FALLBACK)
    ID3D12Device* device = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* cmd_alloc = nullptr;
    ID3D12GraphicsCommandList* cmd_list = nullptr;
    ID3D12DescriptorHeap* uav_heap = nullptr;
    ID3D12DescriptorHeap* srv_heap = nullptr;
    ID3D12RootSignature* root_sig = nullptr;
    ID3D12Fence* fence = nullptr;
    HANDLE fence_event = nullptr;
    UINT64 fence_value = 0;
    bool initialized = false;

    struct Shader {
        ID3D12PipelineState* pso = nullptr;
        ID3D12RootSignature* rs = nullptr;
    };
    std::unordered_map<std::string, Shader> shaders;

    struct Buffer {
        ID3D12Resource* resource = nullptr;
        size_t size = 0;
    };
    std::vector<Buffer> allocations;

    int64_t total_memory = 0;
    int64_t used_memory = 0;

    bool init_device(int64_t device_id) {
        HRESULT hr;

        // Create DXGI factory
        IDXGIFactory6* factory = nullptr;
        hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
        if (FAILED(hr)) return false;

        // Find best adapter
        IDXGIAdapter1* adapter = nullptr;
        int64_t current_id = 0;
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc;
            adapter->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            if (current_id == device_id) {
                hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
                if (SUCCEEDED(hr)) {
                    total_memory = (int64_t)(desc.DedicatedVideoMemory + desc.SharedSystemMemory);
                }
                adapter->Release();
                break;
            }
            current_id++;
            adapter->Release();
        }
        factory->Release();

        if (!device) return false;

        // Create command queue
        D3D12_COMMAND_QUEUE_DESC qdesc = {};
        qdesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        qdesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_HIGH;
        hr = device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&queue));
        if (FAILED(hr)) return false;

        // Create command allocator and list
        hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&cmd_alloc));
        if (FAILED(hr)) return false;

        hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, cmd_alloc,
                                       nullptr, IID_PPV_ARGS(&cmd_list));
        if (FAILED(hr)) return false;

        // Create root signature
        D3D12_ROOT_PARAMETER params[3] = {};
        // t0-t1: SRV (A, B, x)
        D3D12_DESCRIPTOR_RANGE srv_range = {
            D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0, 0,
            D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND
        };
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[0].DescriptorTable.NumDescriptorRanges = 1;
        params[0].DescriptorTable.pDescriptorRanges = &srv_range;
        params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        // u0: UAV (C, y)
        D3D12_DESCRIPTOR_RANGE uav_range = {
            D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0,
            D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND
        };
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[1].DescriptorTable.NumDescriptorRanges = 1;
        params[1].DescriptorTable.pDescriptorRanges = &uav_range;
        params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        // b0: constant buffer
        params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[2].Constants.Num32BitValues = 8;
        params[2].Constants.ShaderRegister = 0;
        params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_ROOT_SIGNATURE_DESC rs_desc = {};
        rs_desc.NumParameters = 3;
        rs_desc.pParameters = params;
        rs_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

        ID3DBlob* rs_blob = nullptr;
        ID3DBlob* error = nullptr;
        hr = D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1, &rs_blob, &error);
        if (FAILED(hr)) { if (error) error->Release(); return false; }

        hr = device->CreateRootSignature(0, rs_blob->GetBufferPointer(), rs_blob->GetBufferSize(),
                                         IID_PPV_ARGS(&root_sig));
        rs_blob->Release();
        if (error) error->Release();
        if (FAILED(hr)) return false;

        // Create descriptor heaps
        D3D12_DESCRIPTOR_HEAP_DESC heap_desc = {};
        heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heap_desc.NumDescriptors = 64;
        heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        hr = device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&srv_heap));
        if (FAILED(hr)) return false;

        // Create fence
        hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        if (FAILED(hr)) return false;

        fence_event = CreateEventA(nullptr, FALSE, FALSE, nullptr);
        if (!fence_event) return false;

        cmd_list->Close();
        initialized = true;
        return true;
    }

    ID3D12Resource* create_buffer(size_t size, D3D12_RESOURCE_FLAGS flags) {
        D3D12_HEAP_PROPERTIES heap = {
            D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            D3D12_MEMORY_POOL_UNKNOWN, 0, 0
        };
        D3D12_RESOURCE_DESC desc = {
            D3D12_RESOURCE_DIMENSION_BUFFER, 0, (UINT64)size, 1, 1, 1,
            DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, flags
        };
        ID3D12Resource* res = nullptr;
        HRESULT hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
                                                     &desc, D3D12_RESOURCE_STATE_COMMON,
                                                     nullptr, IID_PPV_ARGS(&res));
        if (SUCCEEDED(hr)) return res;
        return nullptr;
    }

    ID3D12Resource* create_upload_buffer(size_t size) {
        D3D12_HEAP_PROPERTIES heap = {
            D3D12_HEAP_TYPE_UPLOAD, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            D3D12_MEMORY_POOL_UNKNOWN, 0, 0
        };
        D3D12_RESOURCE_DESC desc = {
            D3D12_RESOURCE_DIMENSION_BUFFER, 0, (UINT64)size, 1, 1, 1,
            DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE
        };
        ID3D12Resource* res = nullptr;
        HRESULT hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
                                                     &desc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                     nullptr, IID_PPV_ARGS(&res));
        if (SUCCEEDED(hr)) return res;
        return nullptr;
    }

    Shader compile_shader(const char* name, const char* hlsl, const char* entry) {
        Shader s = {};

        ID3DBlob* shader_blob = nullptr;
        ID3DBlob* error_blob = nullptr;
        UINT flags = D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_SKIP_VALIDATION;
        HRESULT hr = D3DCompile(hlsl, strlen(hlsl), name, nullptr, nullptr,
                                entry, "cs_5_0", flags, 0, &shader_blob, &error_blob);
        if (FAILED(hr)) {
            if (error_blob) error_blob->Release();
            return s;
        }
        if (error_blob) error_blob->Release();

        D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc = {};
        pso_desc.pRootSignature = root_sig;
        pso_desc.CS = { shader_blob->GetBufferPointer(), shader_blob->GetBufferSize() };

        hr = device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&s.pso));
        shader_blob->Release();
        if (FAILED(hr)) return s;

        return s;
    }

    Shader get_shader(const std::string& name, const char* hlsl, const char* entry) {
        auto it = shaders.find(name);
        if (it != shaders.end()) return it->second;
        Shader s = compile_shader(name.c_str(), hlsl, entry);
        shaders[name] = s;
        return s;
    }

    void execute(const std::string& shader_name, const char* hlsl, const char* entry,
                 void* uav, const std::vector<void*>& srvs,
                 const std::vector<uint32_t>& constants,
                 uint32_t x, uint32_t y = 1, uint32_t z = 1) {
        if (!initialized) return;

        Shader s = get_shader(shader_name, hlsl, entry);
        if (!s.pso) return;

        ID3D12Resource* uav_res = (ID3D12Resource*)uav;

        cmd_alloc->Reset();
        cmd_list->Reset(cmd_alloc, s.pso);

        cmd_list->SetComputeRootSignature(root_sig);

        // Transition SRVs to NON_PIXEL_SHADER_RESOURCE and UAV to UNORDERED_ACCESS
        int num_resources = (int)srvs.size() + 1; // SRVs + UAV
        std::vector<D3D12_RESOURCE_BARRIER> barriers(num_resources);
        for (int i = 0; i < (int)srvs.size(); ++i) {
            barriers[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barriers[i].Transition.pResource = (ID3D12Resource*)srvs[i];
            barriers[i].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
            barriers[i].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            barriers[i].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        }
        // Last barrier is for UAV
        barriers[num_resources - 1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barriers[num_resources - 1].Transition.pResource = uav_res;
        barriers[num_resources - 1].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        barriers[num_resources - 1].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        barriers[num_resources - 1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmd_list->ResourceBarrier((UINT)barriers.size(), barriers.data());

        // Set SRV table
        UINT srv_size = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_CPU_DESCRIPTOR_HANDLE srv_handle = srv_heap->GetCPUDescriptorHandleForHeapStart();

        int num_srvs = (int)srvs.size();
        for (int i = 0; i < num_srvs && i < 2; ++i) {
            ID3D12Resource* res = (ID3D12Resource*)srvs[i];
            D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
            srv_desc.Format = DXGI_FORMAT_R32_FLOAT;
            srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srv_desc.Buffer.FirstElement = 0;
            srv_desc.Buffer.NumElements = (UINT)(res->GetDesc().Width / 4);
            srv_desc.Buffer.StructureByteStride = 0;
            srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            device->CreateShaderResourceView(res, &srv_desc, srv_handle);
            srv_handle.ptr += srv_size;
        }

        // Set UAV
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
        uav_desc.Format = DXGI_FORMAT_R32_FLOAT;
        uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav_desc.Buffer.FirstElement = 0;
        uav_desc.Buffer.NumElements = (UINT)(uav_res->GetDesc().Width / 4);
        device->CreateUnorderedAccessView(uav_res, nullptr, &uav_desc, srv_handle);

        ID3D12DescriptorHeap* heaps[] = { srv_heap };
        cmd_list->SetDescriptorHeaps(1, heaps);

        D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = srv_heap->GetGPUDescriptorHandleForHeapStart();
        D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = { srv_gpu.ptr + srv_size * num_srvs };

        cmd_list->SetComputeRootDescriptorTable(0, srv_gpu);
        cmd_list->SetComputeRootDescriptorTable(1, uav_gpu);
        cmd_list->SetComputeRoot32BitConstants(2, (UINT)constants.size(), constants.data(), 0);

        cmd_list->Dispatch(x, y, z);

        // Transition resources back to COMMON
        for (int i = 0; i < num_resources; ++i) {
            auto old_state = (i == num_resources - 1)
                ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            barriers[i].Transition.StateBefore = old_state;
            barriers[i].Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
        }
        cmd_list->ResourceBarrier((UINT)barriers.size(), barriers.data());

        cmd_list->Close();

        ID3D12CommandList* lists[] = { cmd_list };
        queue->ExecuteCommandLists(1, lists);

        // Wait for completion
        fence_value++;
        queue->Signal(fence, fence_value);
        if (fence->GetCompletedValue() < fence_value) {
            fence->SetEventOnCompletion(fence_value, fence_event);
            WaitForSingleObject(fence_event, INFINITE);
        }
    }
#else
    bool initialized = false;
    bool init_device(int64_t) { return false; }
#endif
};

DirectXCompute::DirectXCompute() : impl_(new Impl) {}
DirectXCompute::~DirectXCompute() { shutdown(); }

bool DirectXCompute::init(int64_t device_id) {
    return impl_->init_device(device_id);
}

bool DirectXCompute::is_initialized() const {
    return impl_->initialized;
}

void DirectXCompute::shutdown() {
#if defined(OIL_USE_DIRECTX) || !defined(OIL_NO_DIRECTX_FALLBACK)
    if (!impl_->initialized) return;
    synchronize();
    for (auto& s : impl_->shaders) {
        if (s.second.pso) s.second.pso->Release();
    }
    impl_->shaders.clear();
    for (auto& b : impl_->allocations) {
        if (b.resource) b.resource->Release();
    }
    impl_->allocations.clear();
    if (impl_->cmd_list) impl_->cmd_list->Release();
    if (impl_->cmd_alloc) impl_->cmd_alloc->Release();
    if (impl_->queue) impl_->queue->Release();
    if (impl_->srv_heap) impl_->srv_heap->Release();
    if (impl_->root_sig) impl_->root_sig->Release();
    if (impl_->fence) impl_->fence->Release();
    if (impl_->fence_event) CloseHandle(impl_->fence_event);
    if (impl_->device) impl_->device->Release();
#endif
    impl_->initialized = false;
}

void* DirectXCompute::allocate(size_t bytes) {
#if defined(OIL_USE_DIRECTX) || !defined(OIL_NO_DIRECTX_FALLBACK)
    if (!impl_->initialized) return nullptr;
    bytes = (bytes + 255) & ~255;
    ID3D12Resource* res = impl_->create_buffer(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    if (!res) return nullptr;
    impl_->allocations.push_back({res, bytes});
    impl_->used_memory += bytes;
    return (void*)res;
#else
    (void)bytes;
    return nullptr;
#endif
}

void DirectXCompute::free(void* ptr) {
#if defined(OIL_USE_DIRECTX) || !defined(OIL_NO_DIRECTX_FALLBACK)
    if (!ptr) return;
    for (size_t i = 0; i < impl_->allocations.size(); ++i) {
        if (impl_->allocations[i].resource == ptr) {
            impl_->allocations[i].resource->Release();
            impl_->used_memory -= impl_->allocations[i].size;
            impl_->allocations.erase(impl_->allocations.begin() + i);
            return;
        }
    }
#endif
}

void DirectXCompute::upload(const Tensor& src, void* dst) {
#if defined(OIL_USE_DIRECTX) || !defined(OIL_NO_DIRECTX_FALLBACK)
    if (!impl_->initialized || !dst) return;
    size_t bytes = src.numel() * sizeof(float);
    ID3D12Resource* upload = impl_->create_upload_buffer(bytes);
    if (!upload) return;

    void* mapped;
    upload->Map(0, nullptr, &mapped);
    std::memcpy(mapped, src.data<float>(), bytes);
    upload->Unmap(0, nullptr);

    impl_->cmd_alloc->Reset();
    impl_->cmd_list->Reset(impl_->cmd_alloc, nullptr);

    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = (ID3D12Resource*)dst;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    impl_->cmd_list->ResourceBarrier(1, &barrier);

    impl_->cmd_list->CopyResource((ID3D12Resource*)dst, upload);

    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    impl_->cmd_list->ResourceBarrier(1, &barrier);

    impl_->cmd_list->Close();

    ID3D12CommandList* lists[] = { impl_->cmd_list };
    impl_->queue->ExecuteCommandLists(1, lists);

    impl_->fence_value++;
    impl_->queue->Signal(impl_->fence, impl_->fence_value);
    if (impl_->fence->GetCompletedValue() < impl_->fence_value) {
        impl_->fence->SetEventOnCompletion(impl_->fence_value, impl_->fence_event);
        WaitForSingleObject(impl_->fence_event, INFINITE);
    }

    upload->Release();
#else
    (void)src; (void)dst;
#endif
}

void DirectXCompute::download(void* src, Tensor& dst) {
#if defined(OIL_USE_DIRECTX) || !defined(OIL_NO_DIRECTX_FALLBACK)
    if (!impl_->initialized || !src) return;
    size_t bytes = dst.numel() * sizeof(float);
    ID3D12Resource* readback = nullptr;
    D3D12_HEAP_PROPERTIES heap = {
        D3D12_HEAP_TYPE_READBACK, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN, 0, 0
    };
    D3D12_RESOURCE_DESC desc = {
        D3D12_RESOURCE_DIMENSION_BUFFER, 0, (UINT64)bytes, 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE
    };
    impl_->device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
                                           &desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                           nullptr, IID_PPV_ARGS(&readback));
    if (!readback) return;

    impl_->cmd_alloc->Reset();
    impl_->cmd_list->Reset(impl_->cmd_alloc, nullptr);

    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = (ID3D12Resource*)src;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    impl_->cmd_list->ResourceBarrier(1, &barrier);

    impl_->cmd_list->CopyResource(readback, (ID3D12Resource*)src);

    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    impl_->cmd_list->ResourceBarrier(1, &barrier);

    impl_->cmd_list->Close();

    ID3D12CommandList* lists[] = { impl_->cmd_list };
    impl_->queue->ExecuteCommandLists(1, lists);

    impl_->fence_value++;
    impl_->queue->Signal(impl_->fence, impl_->fence_value);
    if (impl_->fence->GetCompletedValue() < impl_->fence_value) {
        impl_->fence->SetEventOnCompletion(impl_->fence_value, impl_->fence_event);
        WaitForSingleObject(impl_->fence_event, INFINITE);
    }

    void* mapped;
    readback->Map(0, nullptr, &mapped);
    std::memcpy(dst.data<float>(), mapped, bytes);
    readback->Unmap(0, nullptr);

    readback->Release();
#else
    (void)src; (void)dst;
#endif
}

void DirectXCompute::copy(void* dst, const void* src, size_t bytes) {
#if defined(OIL_USE_DIRECTX) || !defined(OIL_NO_DIRECTX_FALLBACK)
    if (!impl_->initialized) return;
    (void)bytes;

    impl_->cmd_alloc->Reset();
    impl_->cmd_list->Reset(impl_->cmd_alloc, nullptr);

    // Transition both resources: src→COPY_SOURCE, dst→COPY_DEST
    D3D12_RESOURCE_BARRIER barriers[2] = {};
    barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[0].Transition.pResource = (ID3D12Resource*)src;
    barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[1].Transition.pResource = (ID3D12Resource*)dst;
    barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    impl_->cmd_list->ResourceBarrier(2, barriers);

    // GPU-to-GPU copy directly (no CPU staging needed)
    impl_->cmd_list->CopyResource((ID3D12Resource*)dst, (ID3D12Resource*)src);

    // Transition back to COMMON
    barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    impl_->cmd_list->ResourceBarrier(2, barriers);
    impl_->cmd_list->Close();

    ID3D12CommandList* lists[] = { impl_->cmd_list };
    impl_->queue->ExecuteCommandLists(1, lists);
    impl_->fence_value++;
    impl_->queue->Signal(impl_->fence, impl_->fence_value);
    if (impl_->fence->GetCompletedValue() < impl_->fence_value) {
        impl_->fence->SetEventOnCompletion(impl_->fence_value, impl_->fence_event);
        WaitForSingleObject(impl_->fence_event, INFINITE);
    }
#else
    (void)dst; (void)src; (void)bytes;
#endif
}

void DirectXCompute::gemm(float alpha, const void* A, const void* B, float beta, void* C,
                           int64_t M, int64_t N, int64_t K) {
#if defined(OIL_USE_DIRECTX) || !defined(OIL_NO_DIRECTX_FALLBACK)
    std::vector<uint32_t> constants = {
        (uint32_t)M, (uint32_t)N, (uint32_t)K,
        0, 0, 0, 0, 0
    };
    std::memcpy(&constants[3], &alpha, 4);
    std::memcpy(&constants[4], &beta, 4);

    std::vector<void*> srvs = { (void*)A, (void*)B };
    uint32_t groups_x = (uint32_t)((N + 15) / 16);
    uint32_t groups_y = (uint32_t)((M + 15) / 16);
    impl_->execute("gemm", HLSL_GEMM, "main", (void*)C, srvs, constants, groups_x, groups_y);
#else
    (void)alpha; (void)A; (void)B; (void)beta; (void)C; (void)M; (void)N; (void)K;
#endif
}

void DirectXCompute::gemv(float alpha, const void* A, const void* x, float beta, void* y,
                           int64_t M, int64_t N) {
#if defined(OIL_USE_DIRECTX) || !defined(OIL_NO_DIRECTX_FALLBACK)
    std::vector<uint32_t> constants = {
        (uint32_t)M, (uint32_t)N, 0, 0, 0, 0, 0, 0
    };
    std::memcpy(&constants[2], &alpha, 4);
    std::memcpy(&constants[3], &beta, 4);

    std::vector<void*> srvs = { (void*)A, (void*)x };
    uint32_t groups = (uint32_t)((M + 255) / 256);
    impl_->execute("gemv", HLSL_GEMV, "main", (void*)y, srvs, constants, groups);
#else
    (void)alpha; (void)A; (void)x; (void)beta; (void)y; (void)M; (void)N;
#endif
}

void DirectXCompute::relu(const void* x, void* y, int64_t n) {
#if defined(OIL_USE_DIRECTX) || !defined(OIL_NO_DIRECTX_FALLBACK)
    std::vector<uint32_t> constants = { (uint32_t)n, 0,0,0,0,0,0,0 };
    uint32_t groups = (uint32_t)((n + 255) / 256);
    impl_->execute("relu", HLSL_RELU, "main", (void*)y, {(void*)x}, constants, groups);
#else
    (void)x; (void)y; (void)n;
#endif
}

void DirectXCompute::gelu(const void* x, void* y, int64_t n) {
#if defined(OIL_USE_DIRECTX) || !defined(OIL_NO_DIRECTX_FALLBACK)
    std::vector<uint32_t> constants = { (uint32_t)n, 0,0,0,0,0,0,0 };
    uint32_t groups = (uint32_t)((n + 255) / 256);
    impl_->execute("gelu", HLSL_GELU, "main", (void*)y, {(void*)x}, constants, groups);
#else
    (void)x; (void)y; (void)n;
#endif
}

void DirectXCompute::silu(const void* x, void* y, int64_t n) {
#if defined(OIL_USE_DIRECTX) || !defined(OIL_NO_DIRECTX_FALLBACK)
    std::vector<uint32_t> constants = { (uint32_t)n, 0,0,0,0,0,0,0 };
    uint32_t groups = (uint32_t)((n + 255) / 256);
    impl_->execute("silu", HLSL_SILU, "main", (void*)y, {(void*)x}, constants, groups);
#else
    (void)x; (void)y; (void)n;
#endif
}

void DirectXCompute::add(const void* a, const void* b, void* c, int64_t n) {
#if defined(OIL_USE_DIRECTX) || !defined(OIL_NO_DIRECTX_FALLBACK)
    std::vector<uint32_t> constants = { (uint32_t)n, 0,0,0,0,0,0,0 };
    uint32_t groups = (uint32_t)((n + 255) / 256);
    impl_->execute("add", HLSL_ADD, "main", (void*)c, {(void*)a, (void*)b}, constants, groups);
#else
    (void)a; (void)b; (void)c; (void)n;
#endif
}

void DirectXCompute::mul(const void* a, const void* b, void* c, int64_t n) {
#if defined(OIL_USE_DIRECTX) || !defined(OIL_NO_DIRECTX_FALLBACK)
    std::vector<uint32_t> constants = { (uint32_t)n, 0,0,0,0,0,0,0 };
    uint32_t groups = (uint32_t)((n + 255) / 256);
    impl_->execute("mul", HLSL_MUL, "main", (void*)c, {(void*)a, (void*)b}, constants, groups);
#else
    (void)a; (void)b; (void)c; (void)n;
#endif
}

void DirectXCompute::scale(float s, const void* x, void* y, int64_t n) {
#if defined(OIL_USE_DIRECTX) || !defined(OIL_NO_DIRECTX_FALLBACK)
    uint32_t s_bits;
    std::memcpy(&s_bits, &s, 4);
    std::vector<uint32_t> constants = { (uint32_t)n, s_bits, 0,0,0,0,0,0 };
    uint32_t groups = (uint32_t)((n + 255) / 256);
    impl_->execute("scale", HLSL_SCALE, "main", (void*)y, {(void*)x}, constants, groups);
#else
    (void)s; (void)x; (void)y; (void)n;
#endif
}

void DirectXCompute::softmax(const void* x, void* y, int64_t rows, int64_t cols) {
#if defined(OIL_USE_DIRECTX) || !defined(OIL_NO_DIRECTX_FALLBACK)
    std::vector<uint32_t> constants = { (uint32_t)rows, (uint32_t)cols, 0,0,0,0,0,0 };
    uint32_t groups = (uint32_t)((rows + 255) / 256);
    impl_->execute("softmax", HLSL_SOFTMAX, "main", (void*)y, {(void*)x}, constants, groups);
#else
    (void)x; (void)y; (void)rows; (void)cols;
#endif
}

void DirectXCompute::rms_norm(const void* x, const void* gamma, void* y, float eps, int64_t n, int64_t d) {
#if defined(OIL_USE_DIRECTX) || !defined(OIL_NO_DIRECTX_FALLBACK)
    uint32_t eps_bits;
    std::memcpy(&eps_bits, &eps, 4);
    std::vector<uint32_t> constants = { (uint32_t)n, (uint32_t)d, eps_bits, 0,0,0,0,0 };
    uint32_t groups = (uint32_t)((n + 255) / 256);
    impl_->execute("rms_norm", HLSL_RMS_NORM, "main", (void*)y, {(void*)x, (void*)gamma}, constants, groups);
#else
    (void)x; (void)gamma; (void)y; (void)eps; (void)n; (void)d;
#endif
}

void DirectXCompute::layer_norm(const void* x, const void* gamma, const void* beta, void* y, float eps, int64_t n, int64_t d) {
#if defined(OIL_USE_DIRECTX) || !defined(OIL_NO_DIRECTX_FALLBACK)
    uint32_t eps_bits;
    std::memcpy(&eps_bits, &eps, 4);
    std::vector<uint32_t> constants = { (uint32_t)n, (uint32_t)d, eps_bits, 0,0,0,0,0 };
    uint32_t groups = (uint32_t)((n + 255) / 256);
    impl_->execute("layer_norm", HLSL_LAYER_NORM, "main", (void*)y,
                   {(void*)x, (void*)gamma, (void*)beta}, constants, groups);
#else
    (void)x; (void)gamma; (void)beta; (void)y; (void)eps; (void)n; (void)d;
#endif
}

void DirectXCompute::moe_gather(const void* x, const int64_t* indices, const float* weights,
                                 void* out, int64_t T, int64_t K, int64_t D) {
#if defined(OIL_USE_DIRECTX) || !defined(OIL_NO_DIRECTX_FALLBACK)
    std::vector<uint32_t> constants = { (uint32_t)T, (uint32_t)K, (uint32_t)D, 0,0,0,0,0 };
    uint32_t groups = (uint32_t)((T * D + 255) / 256);
    impl_->execute("moe_gather", HLSL_MOE_GATHER, "main", (void*)out,
                   {(void*)x, (void*)indices, (void*)weights}, constants, groups);
#else
    (void)x; (void)indices; (void)weights; (void)out; (void)T; (void)K; (void)D;
#endif
}

void DirectXCompute::moe_scatter_add(void* out, const int64_t* indices, const float* weights,
                                       const void* expert_out, int64_t T, int64_t K, int64_t D) {
#if defined(OIL_USE_DIRECTX) || !defined(OIL_NO_DIRECTX_FALLBACK)
    std::vector<uint32_t> constants = { (uint32_t)T, (uint32_t)K, (uint32_t)D, 0,0,0,0,0 };
    uint32_t groups = (uint32_t)((T * D + 255) / 256);
    impl_->execute("moe_scatter_add", HLSL_MOE_SCATTER_ADD, "main", (void*)out,
                   {(void*)expert_out, (void*)indices, (void*)weights}, constants, groups);
#else
    (void)out; (void)indices; (void)weights; (void)expert_out; (void)T; (void)K; (void)D;
#endif
}

void DirectXCompute::synchronize() {
#if defined(OIL_USE_DIRECTX) || !defined(OIL_NO_DIRECTX_FALLBACK)
    if (!impl_->initialized) return;
    impl_->fence_value++;
    impl_->queue->Signal(impl_->fence, impl_->fence_value);
    if (impl_->fence->GetCompletedValue() < impl_->fence_value) {
        impl_->fence->SetEventOnCompletion(impl_->fence_value, impl_->fence_event);
        WaitForSingleObject(impl_->fence_event, INFINITE);
    }
#endif
}

int64_t DirectXCompute::memory_free() const {
    return impl_->initialized ? (impl_->total_memory - impl_->used_memory) : 0;
}

int64_t DirectXCompute::memory_total() const {
    return impl_->initialized ? impl_->total_memory : 0;
}

// ========================================================================
// CUDA Backend — proper implementation with conditional compilation
// ========================================================================

struct CUDABackend::Impl {
    bool initialized = false;
#ifdef OIL_HAS_CUDA
    cudaDeviceProp device_prop_;
    void* cublas_handle_ = nullptr;
    std::unordered_map<void*, size_t> allocations_;
#endif
};

CUDABackend::CUDABackend() : impl_(new Impl) {}
CUDABackend::~CUDABackend() { shutdown(); }

bool CUDABackend::init(int64_t device_id) {
#ifdef OIL_HAS_CUDA
    cudaError_t err = cudaSetDevice((int)device_id);
    if (err != cudaSuccess) return false;
    err = cudaGetDeviceProperties(&impl_->device_prop_, (int)device_id);
    if (err != cudaSuccess) return false;
    cublasCreate((cublasHandle_t*)&impl_->cublas_handle_);
    impl_->initialized = true;
    return true;
#else
    return false;
#endif
}

bool CUDABackend::is_initialized() const { return impl_->initialized; }

void CUDABackend::shutdown() {
#ifdef OIL_HAS_CUDA
    if (impl_->cublas_handle_) {
        cublasDestroy((cublasHandle_t)impl_->cublas_handle_);
        impl_->cublas_handle_ = nullptr;
    }
    for (auto& [ptr, sz] : impl_->allocations_)
        cudaFree(ptr);
    impl_->allocations_.clear();
#endif
    impl_->initialized = false;
}

void* CUDABackend::allocate(size_t bytes) {
#ifdef OIL_HAS_CUDA
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) return nullptr;
    impl_->allocations_[ptr] = bytes;
    return ptr;
#else
    return nullptr;
#endif
}

void CUDABackend::free(void* ptr) {
#ifdef OIL_HAS_CUDA
    impl_->allocations_.erase(ptr);
    cudaFree(ptr);
#endif
}

void CUDABackend::upload(const Tensor& src, void* dst) {
#ifdef OIL_HAS_CUDA
    cudaMemcpy(dst, src.data(), src.size_bytes(), cudaMemcpyHostToDevice);
#endif
}

void CUDABackend::download(void* src, Tensor& dst) {
#ifdef OIL_HAS_CUDA
    cudaMemcpy(dst.data(), src, dst.size_bytes(), cudaMemcpyDeviceToHost);
#endif
}

void CUDABackend::gemm(float alpha, const void* A, const void* B, float beta,
                        void* C, int64_t M, int64_t N, int64_t K) {
#ifdef OIL_HAS_CUDA
    cublasHandle_t handle = (cublasHandle_t)impl_->cublas_handle_;
    float alpha_f = alpha, beta_f = beta;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 (int)N, (int)M, (int)K,
                 &alpha_f, B, CUDA_R_32F, (int)N,
                 A, CUDA_R_32F, (int)K,
                 &beta_f, C, CUDA_R_32F, (int)N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
#endif
}

void CUDABackend::softmax(const void* x, void* y, int64_t rows, int64_t cols) {
#ifdef OIL_HAS_CUDA
    launch_cuda_softmax((float*)y, (const float*)x, (int)rows, (int)cols);
#endif
}

void CUDABackend::gemv(float alpha, const void* A, const void* x, float beta,
                        void* y, int64_t M, int64_t N) {
#ifdef OIL_HAS_CUDA
    cublasHandle_t handle = (cublasHandle_t)impl_->cublas_handle_;
    float alpha_f = alpha, beta_f = beta;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                 1, (int)M, (int)N,
                 &alpha_f, x, CUDA_R_32F, (int)N,
                 A, CUDA_R_32F, (int)N,
                 &beta_f, y, CUDA_R_32F, 1);
#endif
}

void CUDABackend::relu(const void* x, void* y, int64_t n) {
#ifdef OIL_HAS_CUDA
    launch_cuda_relu((float*)y, (const float*)x, (int)n);
#endif
}

void CUDABackend::gelu(const void* x, void* y, int64_t n) {
#ifdef OIL_HAS_CUDA
    launch_cuda_gelu((float*)y, (const float*)x, (int)n);
#endif
}

void CUDABackend::silu(const void* x, void* y, int64_t n) {
#ifdef OIL_HAS_CUDA
    launch_cuda_silu((float*)y, (const float*)x, (int)n);
#endif
}

void CUDABackend::add(const void* a, const void* b, void* c, int64_t n) {
#ifdef OIL_HAS_CUDA
    launch_cuda_add((float*)c, (const float*)a, (const float*)b, (int)n);
#endif
}

void CUDABackend::mul(const void* a, const void* b, void* c, int64_t n) {
#ifdef OIL_HAS_CUDA
    launch_cuda_mul((float*)c, (const float*)a, (const float*)b, (int)n);
#endif
}

void CUDABackend::scale(float s, const void* x, void* y, int64_t n) {
#ifdef OIL_HAS_CUDA
    launch_cuda_scale((float*)y, (const float*)x, s, (int)n);
#endif
}

void CUDABackend::rms_norm(const void* x, const void* gamma, void* y,
                            float eps, int64_t n, int64_t d) {
#ifdef OIL_HAS_CUDA
    launch_cuda_rmsnorm((float*)y, (const float*)x, (const float*)gamma,
                         eps, (int)n, (int)d);
#endif
}

void CUDABackend::layer_norm(const void* x, const void* gamma, const void* beta,
                              void* y, float eps, int64_t n, int64_t d) {
#ifdef OIL_HAS_CUDA
    launch_cuda_layernorm((float*)y, (const float*)x, (const float*)gamma,
                           (const float*)beta, eps, (int)n, (int)d);
#endif
}

void CUDABackend::moe_gather(const void* x, const int64_t* indices,
                              const float* weights, void* out,
                              int64_t T, int64_t K, int64_t D) {
#ifdef OIL_HAS_CUDA
    launch_cuda_moe_gather((float*)out, (const float*)x, indices, weights,
                            (int)T, (int)K, (int)D);
#endif
}

void CUDABackend::moe_scatter_add(void* out, const int64_t* indices,
                                   const float* weights, const void* expert_out,
                                   int64_t T, int64_t K, int64_t D) {
#ifdef OIL_HAS_CUDA
    const float* eo = (const float*)expert_out;
    float* o = (float*)out;
    for (int64_t t = 0; t < T; t++) {
        for (int64_t k = 0; k < K; k++) {
            int64_t expert = indices[t * K + k];
            float w = weights[t * K + k];
            for (int64_t d = 0; d < D; d++) {
                o[expert * D + d] += w * eo[t * D + d];
            }
        }
    }
#endif
}

int64_t CUDABackend::memory_free() const {
#ifdef OIL_HAS_CUDA
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    return (int64_t)free_mem;
#else
    return 0;
#endif
}

int64_t CUDABackend::memory_total() const {
#ifdef OIL_HAS_CUDA
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    return (int64_t)total_mem;
#else
    return 0;
#endif
}

void CUDABackend::synchronize() {
#ifdef OIL_HAS_CUDA
    cudaDeviceSynchronize();
#endif
}

// E13: GPU profiler
class GPUProfiler {
public:
    GPUProfiler() {
#ifdef OIL_HAS_CUDA
        cudaEventCreate(&start_);
        cudaEventCreate(&end_);
#endif
    }
    ~GPUProfiler() {
#ifdef OIL_HAS_CUDA
        cudaEventDestroy(start_);
        cudaEventDestroy(end_);
#endif
    }
    void begin() {
#ifdef OIL_HAS_CUDA
        cudaEventRecord(start_);
#endif
    }
    float end() {
#ifdef OIL_HAS_CUDA
        cudaEventRecord(end_);
        cudaEventSynchronize(end_);
        float ms = 0;
        cudaEventElapsedTime(&ms, start_, end_);
        return ms;
#else
        return 0;
#endif
    }
private:
#ifdef OIL_HAS_CUDA
    cudaEvent_t start_, end_;
#endif
};

// E12: Multi-GPU support
static std::vector<CUDABackend*> g_multi_gpu;

void init_multi_gpu(int count) {
#ifdef OIL_HAS_CUDA
    for (int i = 0; i < count; i++) {
        auto* bk = new CUDABackend();
        if (bk->init(i)) g_multi_gpu.push_back(bk);
        else delete bk;
    }
#endif
}

CUDABackend* get_device(int idx) {
    if (idx >= 0 && idx < (int)g_multi_gpu.size()) return g_multi_gpu[idx];
    return nullptr;
}

// ========================================================================
// GPU autodetection and factory
// ========================================================================

GPUType detect_best_gpu() {
    return GPUType::DIRECTX12;
}

static DirectXCompute* g_dx = nullptr;
static CUDABackend* g_cuda = nullptr;

DirectXCompute& get_dx_compute() {
    if (!g_dx) {
        g_dx = new DirectXCompute();
        if (!g_dx->init(0)) {
            delete g_dx;
            g_dx = nullptr;
        }
    }
    return *g_dx;
}

CUDABackend& get_cuda_backend() {
    if (!g_cuda) g_cuda = new CUDABackend();
    return *g_cuda;
}

bool gpu_available() {
    return get_dx_compute().is_initialized();
}

void init_gpu(GPUType type, int64_t device) {
    if (type == GPUType::DIRECTX12) {
        get_dx_compute().init(device);
    }
}

void shutdown_gpu() {
    if (g_dx) { g_dx->shutdown(); delete g_dx; g_dx = nullptr; }
    if (g_cuda) { g_cuda->shutdown(); delete g_cuda; g_cuda = nullptr; }
}

} // namespace gpu
} // namespace oil
