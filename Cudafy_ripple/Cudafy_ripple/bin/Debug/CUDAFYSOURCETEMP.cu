
// CudafyByExample.ripple_gpu
extern "C" __global__  void thekernel( unsigned char* ptr, int ptrLen0, int ticks);

// CudafyByExample.ripple_gpu
extern "C" __global__  void thekernel( unsigned char* ptr, int ptrLen0, int ticks)
{
	int num = threadIdx.x + blockIdx.x * blockDim.x;
	int num2 = threadIdx.y + blockIdx.y * blockDim.y;
	int num3 = num + num2 * blockDim.x * gridDim.x;
	float num4 = (float)(num - 512);
	float num5 = (float)(num2 - 512);
	float num6 = sqrtf(num4 * num4 + num5 * num5);
	unsigned char b = (unsigned char)(128.0f + 127.0f * cosf(num6 / 10.0f - (float)ticks / 7.0f) / (num6 / 10.0f + 1.0f));
	ptr[(num3 * 4)] = b;
	ptr[(num3 * 4 + 1)] = b;
	ptr[(num3 * 4 + 2)] = b;
	ptr[(num3 * 4 + 3)] = 255;
}
