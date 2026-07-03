#pragma once

#include "rtp_llm/cpp/devices/NativeGraphRunnerBase.h"
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

namespace rtp_llm {

class CudaDevice;

class CudaGraphExecutor: public ExecutorBase {
public:
    CudaGraphExecutor(CudaDevice* device, std::shared_ptr<at::cuda::CUDAStream> stream):
        ExecutorBase(), device_(device), graph_(std::make_shared<at::cuda::CUDAGraph>()), capture_stream_(stream) {}
    void replay() override {
        graph_->replay();
    }
    void captureBegin() override;
    void captureEnd() override;

private:
    CudaDevice*                           device_         = nullptr;
    std::shared_ptr<at::cuda::CUDAGraph>  graph_          = nullptr;
    std::shared_ptr<at::cuda::CUDAStream> capture_stream_ = nullptr;
    std::shared_ptr<at::cuda::CUDAStream> origin_stream_  = nullptr;
};

template<typename Input, typename Output>
class NativeCudaGraphRunner: public NativeGraphRunnerBase<Input, Output> {
public:
    NativeCudaGraphRunner(DeviceBase* device):
        NativeGraphRunnerBase<Input, Output>(device),
        capture_stream_(std::make_shared<at::cuda::CUDAStream>(at::cuda::getStreamFromPool(true))) {}
    std::shared_ptr<ExecutorBase> makeExecutor() override {
        return std::make_shared<CudaGraphExecutor>(dynamic_cast<CudaDevice*>(this->device_), capture_stream_);
    }

private:
    std::shared_ptr<at::cuda::CUDAStream> capture_stream_ = nullptr;
};

#define INSTANTIATE_CUDAGRAPH_RUNNER(I, O) template class NativeCudaGraphRunner<I, O>;

}  // namespace rtp_llm
