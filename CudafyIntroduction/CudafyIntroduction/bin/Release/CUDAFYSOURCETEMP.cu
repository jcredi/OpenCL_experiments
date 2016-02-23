
// CudafyIntroduction.Program
extern "C" __global__  void kernel();
// CudafyIntroduction.Program
extern "C" __global__  void add(int a, int b,  int* c, int cLen0);
// CudafyIntroduction.Program
extern "C" __global__  void WriteHelloWorldOnGPU( unsigned short* c, int cLen0);
// CudafyIntroduction.Program
extern "C" __global__  void addVector( int* a, int aLen0,  int* b, int bLen0,  int* c, int cLen0);

// CudafyIntroduction.Program
extern "C" __global__  void kernel()
{
}
// CudafyIntroduction.Program
extern "C" __global__  void add(int a, int b,  int* c, int cLen0)
{
	c[(0)] = a + b;
}
// CudafyIntroduction.Program
extern "C" __global__  void WriteHelloWorldOnGPU( unsigned short* c, int cLen0)
{
	c[(0)] = 72;
	c[(1)] = 101;
	c[(2)] = 108;
	c[(3)] = 108;
	c[(4)] = 111;
	c[(5)] = 44;
	c[(6)] = 32;
	c[(7)] = 119;
	c[(8)] = 111;
	c[(9)] = 114;
	c[(10)] = 108;
	c[(11)] = 100;
}
// CudafyIntroduction.Program
extern "C" __global__  void addVector( int* a, int aLen0,  int* b, int bLen0,  int* c, int cLen0)
{
	int x = blockIdx.x;
	if (x < aLen0)
	{
		c[(x)] = a[(x)] + b[(x)];
	}
}
