using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text;
using System.Diagnostics;
using System.Runtime.InteropServices;
using MathNet.Numerics.LinearAlgebra;

using Cudafy;
using Cudafy.Host;
using Cudafy.Atomics;
using Cudafy.Translator;


namespace matrixMultiplication_cudafy
{
    class Program
    {
        const int nRowsA = 1024;
        const int nColsA = 1024;
        const int nRowsB = nColsA;
        const int nColsB = 1024;
        static readonly int[] dimensions = { nRowsA, nColsA, nColsB };



        static dim3 blockSize = new dim3(8, 16); // because my GPU has 128 cores per computing unit
        static dim3 gridSize = new dim3(((nRowsA - (nRowsA % (int)blockSize.x)) / blockSize.x) + 1,
                                        ((nColsB - (nColsB % blockSize.y)) / blockSize.y) + 1); 
                                        // good choice apparently...try to understand why!
        static Random rng = new Random();

        [STAThread]
        public static void Main(string[] args)
        {
            

            float[,] A = new float[nRowsA, nColsA];
            float[,] B = new float[nRowsB, nColsB];
            float[,] C = new float[nRowsA, nColsB];

            // Populate A and B with random values (on the CPU)
            for (int i = 0; i < nRowsA; i++)
            {
                for (int j = 0; j < nColsA; j++)
                A[i,j] = (float)rng.NextDouble();
            }
            for (int i = 0; i < nRowsB; i++)
            {
                for (int j = 0; j < nColsB; j++)
                    B[i, j] = (float)rng.NextDouble();
            }

            try
            {
                CudafyModes.DeviceId = ChooseGPU();
                CudafyModes.Target = eGPUType.OpenCL;
                GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
                CudafyModule km = CudafyTranslator.Cudafy();
                Console.WriteLine("Attempting to load module...");
                System.Threading.Thread.Sleep(300);
                gpu.LoadModule(km);
                Console.WriteLine("Module successfully loaded.\n");
                System.Threading.Thread.Sleep(500);

                Console.WriteLine("Multiplying a {0}x{1} matrix by a {1}x{2} matrix...\n", dimensions[0], dimensions[1], dimensions[2]);


                // First let's do it on the CPU
                Matrix<float> mathA = Matrix<float>.Build.Random(nRowsA, nColsA);
                Matrix<float> mathB = Matrix<float>.Build.Random(nRowsB, nColsB);

                // time this
                Stopwatch stopwatch = Stopwatch.StartNew();
                Matrix<float> mathC = mathA.Multiply(mathB);
                stopwatch.Stop();
                Console.WriteLine("CPU time: {0} ms", stopwatch.ElapsedMilliseconds);


                // start timer here
                gpu.StartTimer();

                // Allocate arrays on GPU
                float[,] gpuA = gpu.Allocate(A);
                float[,] gpuB = gpu.Allocate(B);
                float[,] gpuC = gpu.Allocate(C);
                

                gpu.CopyToDevice(A, gpuA);
                gpu.CopyToDevice(B, gpuB);
                int[] gpu_dimensions = gpu.CopyToDevice(dimensions);

                gpu.Launch(gridSize, blockSize, "MultiplyMatrices", gpuA, gpuB, gpu_dimensions, gpuC);

                gpu.Synchronize();
                gpu.CopyFromDevice(gpuC, C);

                // stop timer here
                float elapsedTime = gpu.StopTimer();
                Console.WriteLine("Total GPU time: {0} ms", elapsedTime);

                gpu.FreeAll();
            }
            catch (CudafyLanguageException cle)
            {
                HandleException(cle);
            }
            catch (CudafyCompileException cce)
            {
                HandleException(cce);
            }
            catch (CudafyHostException che)
            {
                HandleException(che);
            }


            Exit();
        }

        static int ChooseGPU()
        {
            // Specify CUDAfy target and language
            CudafyModes.Target = eGPUType.OpenCL;
            CudafyTranslator.Language = eLanguage.OpenCL;

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

        static void HandleException(Exception ex)
        {
            Console.WriteLine(ex.Message);
        }

        static void Exit()
        {
            Console.WriteLine();
            Console.WriteLine("The program will now close. Press a key to exit...");
            Console.ReadKey();
        }

        [Cudafy]
        public static void MultiplyMatrices(GThread gthread, float[,] A, float[,] B, int[] dimensions, float[,] C)
        {
            int row = gthread.blockIdx.x * gthread.blockDim.x + gthread.threadIdx.x;
            int col = gthread.blockIdx.y * gthread.blockDim.y + gthread.threadIdx.y;

            if (row >= dimensions[0] || col >= dimensions[2])
            {
                return;
            }

            float sum = 0.0f;
            for (int k = 0; k < dimensions[1]; k++)
            {
                sum += A[row, k] * B[k, col];
            }

            C[row, col] = sum;
        }


    }
}
