#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// ========================================
// Kernel 1: Scale
// ========================================
__global__ void scaleKernel(float *data, int n, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * factor;
    }
}

// ========================================
// Kernel 2: Add
// ========================================
__global__ void addKernel(float *data, int n, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] + value;
    }
}

// ========================================
// Kernel 3: Square
// ========================================
__global__ void squareKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * data[idx];
    }
}

// ========================================
// CPU Reference Implementation
// ========================================
void processVectorCPU(float *data, int n) {
    // Scale
    for (int i = 0; i < n; i++) {
        data[i] = data[i] * 2.0f;
    }
    
    // Add
    for (int i = 0; i < n; i++) {
        data[i] = data[i] + 10.0f;
    }
    
    // Square
    for (int i = 0; i < n; i++) {
        data[i] = data[i] * data[i];
    }
}

// ========================================
// PART A: Sequential GPU Version (No Streams)
// ========================================
float processSequential(float **h_input, float **h_output, int vectorCount, int vectorSize) 
{
    printf("\n=== SEQUENTIAL (NO STREAMS) ===\n");
    
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    
    // Allocate device memory for ONE vector
    float *d_data;
    size_t bytes = vectorSize * sizeof(float);
    CHECK(cudaMalloc(&d_data, bytes));
    
    // Kernel configuration
    int blockSize = 256;
    int gridSize = (vectorSize + blockSize - 1) / blockSize;
    
    CHECK(cudaEventRecord(start));
    
    // Process each vector sequentially
    for (int v = 0; v < vectorCount; v++) {
        // 1. Copy input vector to device
        CHECK(cudaMemcpy(d_data, h_input[v], bytes, cudaMemcpyHostToDevice));
        
        // 2. Launch scale kernel
        scaleKernel<<<gridSize, blockSize>>>(d_data, vectorSize, 2.0f);
        
        // 3. Launch add kernel
        addKernel<<<gridSize, blockSize>>>(d_data, vectorSize, 10.0f);
        
        // 4. Launch square kernel
        squareKernel<<<gridSize, blockSize>>>(d_data, vectorSize);
        
        // 5. Copy result back to host
        CHECK(cudaMemcpy(h_output[v], d_data, bytes, cudaMemcpyDeviceToHost));
        
        // 6. Synchronize
        CHECK(cudaDeviceSynchronize());
    }
    
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    printf("Time: %.2f ms\n", milliseconds);
    
    // Cleanup
    CHECK(cudaFree(d_data));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    
    return milliseconds;
}

// ========================================
// PART B: Breadth-First with Streams
// ========================================
float processBreadthFirst(float **h_input, float **h_output, int vectorCount, int vectorSize) {
    printf("\n=== BREADTH-FIRST PIPELINE ===\n");
    
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    
    // Create streams
    cudaStream_t *streams = (cudaStream_t*)malloc(vectorCount * sizeof(cudaStream_t));
    for (int i = 0; i < vectorCount; i++) {
        CHECK(cudaStreamCreate(&streams[i]));
    }
    
    // Allocate device memory for each stream
    float **d_data = (float**)malloc(vectorCount * sizeof(float*));
    size_t bytes = vectorSize * sizeof(float);
    for (int i = 0; i < vectorCount; i++) {
        CHECK(cudaMalloc(&d_data[i], bytes));
    }
    
    // Kernel configuration
    int blockSize = 256;
    int gridSize = (vectorSize + blockSize - 1) / blockSize;
    
    CHECK(cudaEventRecord(start));
    
    // Phase 1: All H2D transfers
    for (int v = 0; v < vectorCount; v++) {
        CHECK(cudaMemcpyAsync(d_data[v], h_input[v], bytes, 
                              cudaMemcpyHostToDevice, streams[v]));
    }
    
    // Phase 2: All scale operations
    for (int v = 0; v < vectorCount; v++) {
        scaleKernel<<<gridSize, blockSize, 0, streams[v]>>>(d_data[v], vectorSize, 2.0f);
    }
    
    // Phase 3: All add operations
    for (int v = 0; v < vectorCount; v++) {
        addKernel<<<gridSize, blockSize, 0, streams[v]>>>(d_data[v], vectorSize, 10.0f);
    }
    
    // Phase 4: All square operations
    for (int v = 0; v < vectorCount; v++) {
        squareKernel<<<gridSize, blockSize, 0, streams[v]>>>(d_data[v], vectorSize);
    }
    
    // Phase 5: All D2H transfers
    for (int v = 0; v < vectorCount; v++) {
        CHECK(cudaMemcpyAsync(h_output[v], d_data[v], bytes, 
                              cudaMemcpyDeviceToHost, streams[v]));
    }
    
    // Synchronize all streams
    for (int i = 0; i < vectorCount; i++) {
        CHECK(cudaStreamSynchronize(streams[i]));
    }
    
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    printf("Time: %.2f ms\n", milliseconds);
    
    // Cleanup
    for (int i = 0; i < vectorCount; i++) {
        CHECK(cudaFree(d_data[i]));
        CHECK(cudaStreamDestroy(streams[i]));
    }
    free(d_data);
    free(streams);
    
    return milliseconds;
}

// ========================================
// PART C: Depth-First with Streams
// ========================================
float processDepthFirst(float **h_input, float **h_output, int vectorCount, int vectorSize) {
    printf("\n=== DEPTH-FIRST PIPELINE ===\n");
    
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    
    // Create streams
    cudaStream_t *streams = (cudaStream_t*)malloc(vectorCount * sizeof(cudaStream_t));
    for (int i = 0; i < vectorCount; i++) {
        CHECK(cudaStreamCreate(&streams[i]));
    }
    
    // Allocate device memory
    float **d_data = (float**)malloc(vectorCount * sizeof(float*));
    size_t bytes = vectorSize * sizeof(float);
    for (int i = 0; i < vectorCount; i++) {
        CHECK(cudaMalloc(&d_data[i], bytes));
    }
    
    // Kernel configuration
    int blockSize = 256;
    int gridSize = (vectorSize + blockSize - 1) / blockSize;
   
    CHECK(cudaEventRecord(start));
    
    // Process each vector completely in its own stream
    for (int v = 0; v < vectorCount; v++) {
        // 1. H2D transfer
        CHECK(cudaMemcpyAsync(d_data[v], h_input[v], bytes, 
                              cudaMemcpyHostToDevice, streams[v]));
        
        // 2. Scale kernel
        scaleKernel<<<gridSize, blockSize, 0, streams[v]>>>(d_data[v], vectorSize, 2.0f);
        
        // 3. Add kernel
        addKernel<<<gridSize, blockSize, 0, streams[v]>>>(d_data[v], vectorSize, 10.0f);
        
        // 4. Square kernel
        squareKernel<<<gridSize, blockSize, 0, streams[v]>>>(d_data[v], vectorSize);
        
        // 5. D2H transfer
        CHECK(cudaMemcpyAsync(h_output[v], d_data[v], bytes, 
                              cudaMemcpyDeviceToHost, streams[v]));
    }
    
    // Wait for all streams to complete
    for (int i = 0; i < vectorCount; i++) {
        CHECK(cudaStreamSynchronize(streams[i]));
    }
    
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    printf("Time: %.2f ms\n", milliseconds);
    
    // Cleanup
    for (int i = 0; i < vectorCount; i++) {
        CHECK(cudaFree(d_data[i]));
        CHECK(cudaStreamDestroy(streams[i]));
    }
    free(d_data);
    free(streams);
    
    return milliseconds;
}

// ========================================
// Verification Function
// ========================================
bool verifyResults(float **result1, float **result2, int vectorCount, int vectorSize) {
    for (int v = 0; v < vectorCount; v++) {
        for (int i = 0; i < vectorSize; i++) {
            if (fabs(result1[v][i] - result2[v][i]) > 1e-3) {
                printf("Mismatch at vector %d, index %d: %.2f vs %.2f\n",
                       v, i, result1[v][i], result2[v][i]);
                return false;
            }
        }
    }
    return true;
}

// ========================================
// Main Function
// ========================================
int main(int argc, char **argv) {
    // Configuration
    int vectorCount = 8;
    int vectorSize = 1 << 22;  // 4M elements = 16 MB per vector
    
    if (argc > 1) vectorCount = atoi(argv[1]);
    if (argc > 2) vectorSize = atoi(argv[2]);
    
    printf("╔════════════════════════════════════════╗\n");
    printf("║  VECTOR OPERATIONS PIPELINE EXERCISE   ║\n");
    printf("╚════════════════════════════════════════╝\n\n");
    
    printf("Configuration:\n");
    printf("  Number of vectors: %d\n", vectorCount);
    printf("  Elements per vector: %d (%.2f MB)\n", 
           vectorSize, vectorSize * sizeof(float) / 1024.0 / 1024.0);
    printf("  Total data: %.2f MB\n\n",
           vectorCount * vectorSize * sizeof(float) / 1024.0 / 1024.0);
    
    // Print GPU info
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Concurrent kernels supported: %s\n", 
           prop.concurrentKernels ? "Yes" : "No");
    printf("Number of copy engines: %d\n\n", prop.asyncEngineCount);
    
    // Allocate host memory (pinned for async transfers)
    float **h_input = (float**)malloc(vectorCount * sizeof(float*));
    float **h_output_seq = (float**)malloc(vectorCount * sizeof(float*));
    float **h_output_bf = (float**)malloc(vectorCount * sizeof(float*));
    float **h_output_df = (float**)malloc(vectorCount * sizeof(float*));
    
    for (int v = 0; v < vectorCount; v++) {
        // Allocate pinned memory for better transfer performance
        CHECK(cudaMallocHost(&h_input[v], vectorSize * sizeof(float)));
        CHECK(cudaMallocHost(&h_output_seq[v], vectorSize * sizeof(float)));
        CHECK(cudaMallocHost(&h_output_bf[v], vectorSize * sizeof(float)));
        CHECK(cudaMallocHost(&h_output_df[v], vectorSize * sizeof(float)));
        
        // Initialize input
        for (int i = 0; i < vectorSize; i++) {
            h_input[v][i] = (float)(i % 100) / 10.0f;
        }
    }
    
    // Run all implementations
    float timeSeq = processSequential(h_input, h_output_seq, vectorCount, vectorSize);
    float timeBF = processBreadthFirst(h_input, h_output_bf, vectorCount, vectorSize);
    float timeDF = processDepthFirst(h_input, h_output_df, vectorCount, vectorSize);
    
    // Verify results
    printf("\n=== VERIFICATION ===\n");
    bool seqVsBF = verifyResults(h_output_seq, h_output_bf, vectorCount, vectorSize);
    bool seqVsDF = verifyResults(h_output_seq, h_output_df, vectorCount, vectorSize);
    
    if (seqVsBF && seqVsDF) {
        printf("✓ All results match!\n");
    } else {
        printf("✗ Results don't match!\n");
    }
    
    // Performance summary
    printf("\n╔════════════════════════════════════════╗\n");
    printf("║         PERFORMANCE SUMMARY            ║\n");
    printf("╚════════════════════════════════════════╝\n");
    printf("Sequential:     %.2f ms\n", timeSeq);
    printf("Breadth-First:  %.2f ms  (%.2fx speedup)\n", 
           timeBF, timeSeq / timeBF);
    printf("Depth-First:    %.2f ms  (%.2fx speedup)\n", 
           timeDF, timeSeq / timeDF);
    printf("\nDepth-First vs Breadth-First: %.2fx\n", timeBF / timeDF);
    
    // Cleanup all memory that has been allocated
    for (int v = 0; v < vectorCount; v++) {
        CHECK(cudaFreeHost(h_input[v]));
        CHECK(cudaFreeHost(h_output_seq[v]));
        CHECK(cudaFreeHost(h_output_bf[v]));
        CHECK(cudaFreeHost(h_output_df[v]));
    }
    free(h_input);
    free(h_output_seq);
    free(h_output_bf);
    free(h_output_df);
    
    return 0;
}
