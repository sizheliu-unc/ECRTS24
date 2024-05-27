#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <time.h>
#include <unistd.h>
#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"

#include "logger.h"
#include "util.h"
std::string IMGFILE = "../src/data/goldfish.ppm";

typedef std::chrono::high_resolution_clock Clock;
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {

            std::cout << msg << std::endl;

    }
};

constexpr long long operator"" _MiB(long long unsigned val)
{
    return val * (1 << 20);
}

class InferEngine {

public:
    InferEngine(const std::string filename, std::vector<std::string> bindings);
    ~InferEngine() noexcept;
    bool infer(bool record_stats);
    bool inferGraph(bool record_stats);
    void resetStats();
    void stats();
    
private:
    std::string mEngineFilename;                    //!< Filename of the serialized engine.
    nvinfer1::Dims mInputDims;                      //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims;                     //!< The dimensions of the output to the network.
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    cudaStream_t stream;
    size_t getMemorySize(const nvinfer1::Dims& dims, const int32_t elem_size);
    void* input_mem_h{nullptr};
    void* input_mem{nullptr};
    size_t input_size;
    std::vector<void*> output_mems;
    std::vector<void*> output_mems_h;
    std::vector<size_t> output_sizes;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    cudaGraph_t cudaGraph;
    cudaGraphExec_t cudaGraphInstance;
    bool graphCreated;

    std::chrono::microseconds min_dur;
    std::chrono::microseconds max_dur;
    std::chrono::microseconds total_dur;
    int count;
};

size_t
InferEngine::getMemorySize(const nvinfer1::Dims& dims, const int32_t elem_size) {
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>()) * elem_size;
}

InferEngine::InferEngine(const std::string filename, std::vector<std::string> bindings) {
    graphCreated = false;
    // De-serialize engine from file
    std::ifstream engineFile(filename, std::ios::binary);
    if (engineFile.fail())
    {
        return;
    }

    engineFile.seekg(0, std::ifstream::end);
    auto fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    runtime.reset(nvinfer1::createInferRuntime(gLogger));
    mEngine.reset(runtime->deserializeCudaEngine(engineData.data(), fsize));
    auto result = cudaStreamCreate(&stream);
    if (result != cudaSuccess)
    {
        std::cout << "ERROR: cuda stream creation failed. " << result << std::endl;
        return;
    }
    context.reset(mEngine->createExecutionContext());
    if (!context) {
        return;
    }
    
    {
        auto input_dims = context->getTensorShape(bindings[0].c_str());
        input_size = getMemorySize(input_dims, sizeof(float));
        // cudaMallocHost(&input_mem_h, input_size);
        if (cudaMalloc(&input_mem, input_size) != cudaSuccess) {
            std::cout << "ERROR: input cuda memory allocation failed, size = " << input_size << " bytes" << std::endl;
            return;
        }
        context->setInputTensorAddress(bindings[0].c_str(), input_mem);


        const std::vector<float> mean{0.485f, 0.456f, 0.406f};
        const std::vector<float> stddev{0.229f, 0.224f, 0.225f};
        std::cout << "Input size: " << input_size << "bytes" << std::endl;
        auto input_image{util::RGBImageReader(IMGFILE, input_dims, mean, stddev)};
        input_image.read();
        input_mem_h = input_image.process();
    }
    for (int i = 1; i < bindings.size(); i++) {
        void* output_mem{nullptr};
        void* output_mem_h{nullptr};
        auto output_dims = context->getTensorShape(bindings[i].c_str());
        auto output_size = getMemorySize(output_dims, sizeof(float));
        auto err = cudaMallocHost(&output_mem_h, output_size);
        if (err != cudaSuccess || (cudaMalloc(&output_mem, output_size) != cudaSuccess)) {
            std::cout << "ERROR: output cuda memory allocation failed, size = " << output_size << " bytes" << std::endl;
            return;
        }
        context->setTensorAddress(bindings[i].c_str(), output_mem);
        output_mems_h.emplace_back(output_mem_h);
        output_mems.emplace_back(output_mem);
        output_sizes.emplace_back(output_size);
    }
    this->resetStats();
}

InferEngine::~InferEngine() noexcept {
    cudaFree(input_mem);
    free(input_mem_h);
    for (void* mem: output_mems) {
        cudaFree(mem);
    }
    for (void* mem: output_mems_h) {
        free(mem);
    }
    cudaStreamDestroy(stream);
    if (graphCreated) {
        cudaGraphDestroy(cudaGraph);
        cudaGraphExecDestroy(cudaGraphInstance);
    }
}

void
InferEngine::resetStats() {
    min_dur = std::chrono::microseconds::max();
    max_dur = std::chrono::microseconds::zero();
    total_dur = std::chrono::microseconds::zero();
    count = 0;
}

bool
InferEngine::infer(bool record_stats) {
    auto start = Clock::now();
    auto err = cudaMemcpyAsync(input_mem, input_mem_h, input_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::cout << "CUDA host to device failed! exiting" << std::endl;
        return false;
    }
    bool status = context->enqueueV3(stream);
    if (!status)
    {
        std::cout << "ERROR: TensorRT inference failed" << std::endl;
        return false;
    }
    for (int i = 0; i < output_mems.size(); i++) {
        err = cudaMemcpyAsync(output_mems_h[i], output_mems[i], output_sizes[i], cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
            std::cout << "CUDA device to host failed! exiting" << std::endl;
            return false;
        }
    }
    
    cudaStreamSynchronize(stream);
    if (record_stats) {
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start);
        min_dur = min(min_dur, dur);
        max_dur = max(max_dur, dur);
        total_dur = dur + total_dur;
        count++;        
    }
    return true;
}

bool
InferEngine::inferGraph(bool record_stats) {

    if (!graphCreated) {
        context->enqueueV3(stream);
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        auto err = cudaMemcpyAsync(input_mem, input_mem_h, input_size, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            std::cout << "CUDA host to device failed! exiting" << std::endl;
            return false;
        }
        bool status = context->enqueueV3(stream);
        if (!status)
        {
            std::cout << "ERROR: TensorRT inference failed" << std::endl;
            return false;
        }
        for (int i = 0; i < output_mems.size(); i++) {
            err = cudaMemcpyAsync(output_mems_h[i], output_mems[i], output_sizes[i], cudaMemcpyDeviceToHost, stream);
            if (err != cudaSuccess) {
                std::cout << "CUDA device to host failed! exiting" << std::endl;
                return false;
            }
        }
        cudaStreamEndCapture(stream, &cudaGraph);
        cudaGraphInstantiate(&cudaGraphInstance, cudaGraph, NULL, NULL, 0);
        graphCreated = true;
    }
    auto start = Clock::now();
    if (cudaGraphLaunch(cudaGraphInstance, stream) != cudaSuccess) {
        std::cout << "ERROR: TensorRT inference failed" << std::endl;
        return false;
    }
    cudaStreamSynchronize(stream);
    if (record_stats) {
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start);
        min_dur = min(min_dur, dur);
        max_dur = max(max_dur, dur);
        total_dur = dur + total_dur;
        count++;        
    }
    return true;
}
void
InferEngine::stats() {
    std::cout << "Number of inference performed: " << count << std::endl;
    std::cout << "Max inference time: " << (float) max_dur.count() / 1000 << "ms" << std::endl;
    std::cout << "Min inference time: " << (float) min_dur.count() / 1000 << "ms" << std::endl;
    std::cout << "Total inference time: " <<  (float) total_dur.count() / 1000 << "ms" << std::endl;
    std::cout << "Avg inference time: " << (float) (total_dur).count() / (count * 1000) << "ms" << std::endl;
}

int
main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Inference type not entered! Use \'small\' or \'large\'. Exiting without running." << std::endl;
        return 0;        
    }
    std::string arg(argv[1]);
    InferEngine* engine;
    
    if (arg == "regnet") {
        std::vector<std::string> bindings{"x", "730"};
        engine = new InferEngine("../models/regnet.engine", bindings);
    } else if (arg == "segformer") {
        IMGFILE = "../src/data/cityscape.ppm";
        std::vector<std::string> bindings{"img", "logits"};
        engine = new InferEngine("../models/segformer.engine", bindings);        
    } else if (arg == "detr") {
        IMGFILE = "../src/data/cityscape.ppm";
        std::vector<std::string> bindings{"samples", "4616", "4617"};
            // "samples", "6636", "6637", "6663", "6665", "6667", "6669", "6671", "6679", "6681", "6683", "6685", "6687"};
        engine = new InferEngine("../models/detr.engine", bindings);
    } else if (arg == "contrived") {
        std::vector<std::string> bindings{"input", "3"};
        engine = new InferEngine("../models/conv.engine", bindings);
    } else if (arg == "vit") {
        std::vector<std::string> bindings{"x", "1339"};
        engine = new InferEngine("../models/vit.engine", bindings);
    } else if (arg == "deit") {
        std::vector<std::string> bindings{"x", "1137"};
        engine = new InferEngine("../models/deit.engine", bindings);
    } else {
        std::cerr << "Inference type: " << arg << " not supported!" << std::endl;
        return 0;
    }
    bool useGraph = false;
    if (argc >= 3) {
        std::string arg2(argv[2]);
        if (arg2 == "graph") {
            std::cout << "\"graph\" specified, using cudaGraph" << std::endl;
            useGraph = true;
        }
    }
    std::cout << "Warming up! (5s)" << std::endl;
    auto start = Clock::now();
    if(useGraph) {
        while (Clock::now() - start <= std::chrono::seconds(5) && engine->inferGraph(false));
    } else {
        while (Clock::now() - start <= std::chrono::seconds(5) && engine->infer(false));
    }
    
    start = Clock::now();
    std::cout << "Warmup complete. Starting inference. (30s)" << std::endl;
    if (useGraph) {
        while (Clock::now() - start <= std::chrono::seconds(30) && engine->inferGraph(true));
    } else {
        while (Clock::now() - start <= std::chrono::seconds(30) && engine->infer(true));
    }
    std::cout << "inference complete. Dumping statistics:" << std::endl;
    engine->stats();

    return 0;
}
