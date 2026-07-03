#include "rtp_llm/cpp/devices/cuda_impl/NativeCudaGraphRunner.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"

namespace rtp_llm {

void CudaGraphExecutor::captureBegin() {
    origin_stream_ = std::make_shared<at::cuda::CUDAStream>(at::cuda::getCurrentCUDAStream(at::cuda::current_device()));
    // Route both torch tensor allocations and RTP-LLM BufferManager allocations to the
    // graph-safe private mempool for the whole captured forward. PyTorch's CUDAGraph will
    // additionally flip TorchCudaAllocator into pool mode via beginAllocateToPool, but we set
    // this flag first so that device_->allocateBuffer() calls inside forward() also stay private.
    device_->nativeGraphBeginCapture();
    device_->setStream(capture_stream_->stream());
    at::cuda::setCurrentCUDAStream(*capture_stream_);
    device_->syncAndCheck();
    graph_->capture_begin();
    CaptureCheck::in_cuda_graph_capture = true;
}

void CudaGraphExecutor::captureEnd() {
    graph_->capture_end();
    CaptureCheck::in_cuda_graph_capture = false;
    device_->syncAndCheck();
    device_->setStream(origin_stream_->stream());
    at::cuda::setCurrentCUDAStream(*origin_stream_);
    device_->nativeGraphEndCapture();
    device_->registerARGraphBuffers();
}

INSTANTIATE_CUDAGRAPH_RUNNER(GptModelInputs, GptModelOutputs)

std::shared_ptr<NativeGraphRunner> CudaDevice::getNativeGraphRunner() {
    return std::make_shared<NativeCudaGraphRunner<GptModelInputs, GptModelOutputs>>(this);
}

}  // namespace rtp_llm
