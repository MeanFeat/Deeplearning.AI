// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
//// CUDA and CUBLAS functions
#include <helper_cuda.h>
#include <Eigen/dense>
#include <iostream>
#include <chrono>

#define TIMING
#ifdef TIMING
#define INIT_TIMER auto start = std::chrono::high_resolution_clock::now();
#define START_TIMER  start = std::chrono::high_resolution_clock::now();
#define STOP_TIMER(name)  std::cout << "RUNTIME of " << name << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-start).count() << " ms " << std::endl; 
#else
#define INIT_TIMER
#define START_TIMER
#define STOP_TIMER(name)
#endif

static cublasHandle_t cuda_handle;
const float alpha = 1.0f;
const float beta = 0.0f;
namespace Eigen {
	MatrixXf cuda_mult( MatrixXf a, MatrixXf b) {
		MatrixXf outMat = MatrixXf::Zero(a.rows(), b.cols());
		float *d_A, *d_B, *d_C;
		unsigned int mem_size_A = sizeof(float) * b.size();
		unsigned int mem_size_B = sizeof(float) * a.size();
		unsigned int mem_size_C = sizeof(float) * outMat.size();
		checkCudaErrors(cudaMalloc((void **)&d_A, mem_size_A));
		checkCudaErrors(cudaMalloc((void **)&d_B, mem_size_B));
		checkCudaErrors(cudaMemcpy(d_A, b.data(), mem_size_A, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_B, a.data(), mem_size_B, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc((void **)&d_C, mem_size_C));
		INIT_TIMER
		START_TIMER
		checkCudaErrors(cublasSgemm(cuda_handle, CUBLAS_OP_N, CUBLAS_OP_N, a.rows(), b.cols(), b.rows(), &alpha, d_B, a.rows(), d_A, b.rows(), &beta, d_C, a.rows()));
		checkCudaErrors(cudaMemcpy(outMat.data(), d_C, mem_size_C, cudaMemcpyDeviceToHost));
		STOP_TIMER("GPU-CUDA")
		checkCudaErrors(cudaFree(d_A));
		checkCudaErrors(cudaFree(d_B));
		checkCudaErrors(cudaFree(d_C));
		return outMat;
	}
}

using namespace Eigen;

void TestMatrixMut() {
	MatrixXf A = MatrixXf::Random(10001, 1000);
	MatrixXf B = MatrixXf::Random(1000, 10001);

	checkCudaErrors(cublasCreate(&cuda_handle));
	printf("Starting test...\n");
	MatrixXf testCuda = cuda_mult(A, B);
	checkCudaErrors(cublasDestroy(cuda_handle));
	MatrixXf control;
	INIT_TIMER
	START_TIMER
	control = A*B;
	STOP_TIMER("CPU")

		if(testCuda.isApprox(control)) {
			printf("Pass!\n");
		} else {
			printf("Fail!\n");
		}
}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
	TestMatrixMut();
	return(0);
}
