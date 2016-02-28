using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Cudafy;
using Cudafy.Host;
using Cudafy.Maths.BLAS;
using Cudafy.Translator;

namespace test_blas
{
    class Program
    {
        static void Main(string[] args)
        {
            // Get GPU device
            CudafyModes.DeviceId = ChooseGPU();
            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);

            // Create GPGPUBLAS (CUBLAS Wrapper)
            GPGPUBLAS blas = GPGPUBLAS.Create(gpu);

            // Prepare sample data
            Random rand = new Random();
            int n = 500;
            double[] cpuVectorX = new double[n];
            double[] cpuVectorY = new double[n];
            double[] cpuMatrixA = new double[n * n];

            for (int i = 0; i < n; i++)
            {
                cpuVectorX[i] = rand.Next(100);
                cpuVectorY[i] = rand.Next(100);
            }

            for (int i = 0; i < n * n; i++)
            {
                cpuMatrixA[i] = rand.Next(100);
            }

            // Copy CPU to GPU memory
            // Before using GPGPUBLAS, You have to copy data from cpu to gpu.
            double[] gpuVectorX = gpu.CopyToDevice(cpuVectorX);
            double[] gpuVectorY = gpu.CopyToDevice(cpuVectorY);
            double[] gpuMatrixA = gpu.CopyToDevice(cpuMatrixA);

            // BLAS1 sample : y = x + y
            blas.AXPY(1.0, gpuVectorX, gpuVectorY);

            // BLAS2 sample : y = Ax + y
            blas.GEMV(n, n, 1.0, gpuMatrixA, gpuVectorX, 1.0, gpuVectorY);

            // Get result from GPU
            gpu.CopyFromDevice<double>(gpuVectorY, cpuVectorY);

            // And you can use result cpuVectorY for any other purpose.

            Exit();
        }

        static int ChooseGPU()
        {
            // Specify CUDAfy target and language
            CudafyModes.Target = eGPUType.Cuda;
            CudafyTranslator.Language = eLanguage.Cuda;

            // Look for suitable devices
            int deviceCount = CudafyHost.GetDeviceCount(CudafyModes.Target);
            if (deviceCount == 0)
            {
                Console.WriteLine("No suitable {0} devices found!", CudafyModes.Target);
                Exit();
            }

            // List devices and allow user to choose device to use
            Console.WriteLine("Listing OpenCL capable devices found:\n");
            int i = 0;
            foreach (GPGPUProperties prop in CudafyHost.GetDeviceProperties(eGPUType.OpenCL, false))
            {
                Console.WriteLine("   --- General Information for device {0} ---", i);
                Console.WriteLine("Device name:  {0}", prop.Name);
                Console.WriteLine("Platform Name:  {0}\n", prop.PlatformName);
                i++;
            }
            Console.WriteLine("Enter ID of device to use: ");
            int input = Int32.Parse(Console.ReadLine());
            if (input > i)
            {
                Console.WriteLine("Input {0} is invalid. Program will close.", input);
                Exit();
            }
            CudafyModes.DeviceId = input;
            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            Console.WriteLine("\nYou chose to use device {0}: {1}", CudafyModes.DeviceId, gpu.GetDeviceProperties(false).Name);
            Console.WriteLine("Retreiving device properties...\n");
            System.Threading.Thread.Sleep(1000);

            GPGPUProperties gpuProperties = gpu.GetDeviceProperties(false);
            Console.WriteLine("   --- General Information for device {0} ---", CudafyModes.DeviceId);
            Console.WriteLine("Name:  {0}", gpuProperties.Name);
            Console.WriteLine("Platform Name:  {0}", gpuProperties.PlatformName);
            Console.WriteLine("Architecture:  {0}", gpu.GetArchitecture());
            //Console.WriteLine("Compute capability:  {0}.{1}", gpuProperties.Capability.Major, gpuProperties.Capability.Minor); // for CUDA
            Console.WriteLine("Clock rate: {0}", gpuProperties.ClockRate);
            Console.WriteLine("Simulated: {0}", gpuProperties.IsSimulated);
            Console.WriteLine();

            Console.WriteLine("   --- Memory Information for device {0} ---", CudafyModes.DeviceId);
            Console.WriteLine("Total global mem:  {0}", gpuProperties.TotalMemory);
            Console.WriteLine("Total constant Mem:  {0}", gpuProperties.TotalConstantMemory);
            Console.WriteLine("Max mem pitch:  {0}", gpuProperties.MemoryPitch);
            Console.WriteLine("Texture Alignment:  {0}", gpuProperties.TextureAlignment);
            Console.WriteLine();

            Console.WriteLine("   --- MP Information for device {0} ---", CudafyModes.DeviceId);
            Console.WriteLine("Shared mem per mp: {0}", gpuProperties.SharedMemoryPerBlock);
            Console.WriteLine("Registers per mp:  {0}", gpuProperties.RegistersPerBlock);
            Console.WriteLine("Threads in warp:  {0}", gpuProperties.WarpSize);
            Console.WriteLine("Max threads per block:  {0}", gpuProperties.MaxThreadsPerBlock);
            Console.WriteLine("Max thread dimensions:  ({0}, {1}, {2})", gpuProperties.MaxThreadsSize.x, gpuProperties.MaxThreadsSize.y, gpuProperties.MaxThreadsSize.z);
            Console.WriteLine("Max grid dimensions:  ({0}, {1}, {2})", gpuProperties.MaxGridSize.x, gpuProperties.MaxGridSize.y, gpuProperties.MaxGridSize.z);

            Console.WriteLine();


            return CudafyModes.DeviceId;
        }

        static void Exit()
        {
            Console.WriteLine();
            Console.WriteLine("The program will now close. Press a key to exit...");
            Console.ReadKey();
        }
    }
}
