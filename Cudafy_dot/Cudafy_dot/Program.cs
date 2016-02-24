/* 
 * This software is based upon the book CUDA By Example by Sanders and Kandrot
 * and source code provided by NVIDIA Corporation.
 * It is a good idea to read the book while studying the examples!
*/
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text;
using System.Diagnostics;
using System.Runtime.InteropServices;

using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace CudafyByExample
{
    class Program
    {
        const int threadsPerBlock = 256;
        const int blocksPerGrid = 32;

        [STAThread]
        public static void Main(string[] args)
        {
            // declarations

            Random rng = new Random();

            const int N = 2048 * 2048;
            float relative_tolerance = 0.000001f;

            float[] a = new float[N];
            float[] b = new float[N];
            double c;
            float[] partial_c = new float[blocksPerGrid];

            // fill the arrays 'a' and 'b' on the CPU
            for (int j = 0; j < N; j++)
            {
                a[j] = (float)rng.NextDouble();
                b[j] = (float)rng.NextDouble();
            }

            float elapsedTime;

            try
            {
                GPGPU gpu = InitializeGPU();
                gpu.FreeAll();

                // repeat all 10 times (to measure runtime)
                for (int t = 0; t < 10; t++)
                {
                    // allocation and copy
                    gpu.StartTimer();
                    float[] dev_a = gpu.Allocate<float>(N);
                    float[] dev_b = gpu.Allocate<float>(N);
                    float[] dev_partial_c = gpu.Allocate<float>(blocksPerGrid);
                    float[] dev_test = gpu.Allocate<float>(blocksPerGrid * blocksPerGrid);
                    gpu.CopyToDevice(a, dev_a);
                    gpu.CopyToDevice(b, dev_b);
                    elapsedTime = gpu.StopTimer();
                    Console.WriteLine("Allocation and copy to the GPU took {0} ms", elapsedTime);

                    // launch the kernel on the GPU!
                    gpu.StartTimer();
                    gpu.Launch(blocksPerGrid, threadsPerBlock).Dot(dev_a, dev_b, dev_partial_c);
                    elapsedTime = gpu.StopTimer();
                    Console.WriteLine("Running kernel took {0} ms.", elapsedTime);
                    
                    // copy the array 'c' back from the GPU to the CPU
                    gpu.StartTimer();
                    gpu.CopyFromDevice(dev_partial_c, partial_c);
                    elapsedTime = gpu.StopTimer();
                    Console.WriteLine("Copying data back took {0} ms.", elapsedTime);

                    // finish up on the CPU side
                    c = 0;
                    for (int i = 0; i < blocksPerGrid; i++)
                    {
                        c += (double)partial_c[i];
                    }

                    // compute dot product with LINQ
                    double c_check = a.Zip(b, (x1, x2) => x1 * x2).Sum();

                    if (Math.Abs(c - c_check) / c_check < relative_tolerance)
                        Console.WriteLine("Check passed.");
                    else
                    {
                        Console.WriteLine("Check FAILED!");
                        Console.WriteLine("Error = {0}", Math.Abs(c - c_check));
                        Console.WriteLine("c_GPU = {0}, but c_CPU = {1}", c, c_check);
                    }

                    gpu.FreeAll();
                }
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

        static GPGPU InitializeGPU()
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
            Console.WriteLine("Attempting to load module...");
            System.Threading.Thread.Sleep(1000);

            // Get GPU arcitecture and load corresponding module
            //eArchitecture arch = gpu.GetArchitecture();
            //Console.WriteLine("Module: {0}", arch);
            CudafyModule km = CudafyTranslator.Cudafy();
            gpu.LoadModule(km);
            Console.WriteLine("Module successfully loaded.\n");
            System.Threading.Thread.Sleep(1000);

            return gpu;
        }

        static float sum_squares(float x)
        {
            return (x * (x + 1) * (2 * x + 1) / 6);
        }

        
        [Cudafy]
        public static void Dot(GThread thread, float[] a, float[] b, float[] c)
        {
            float[] cache = thread.AllocateShared<float>("cache", threadsPerBlock);

            int threadIndex = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int cacheIndex = thread.threadIdx.x; // we are within a block!

            float temp = 0;
            while (threadIndex < a.Length)
            {
                temp += a[threadIndex] * b[threadIndex];
                threadIndex += thread.blockDim.x * thread.gridDim.x;
            }

            // set the cache values
            cache[cacheIndex] = temp;

            // synchronize threads in this block
            // this means that all threads within the block have to reach this checkpoint (barrier) before continuing
            thread.SyncThreads();

            // for reductions, threadsPerBlock must be a power of 2
            // because of the following code
            int i = thread.blockDim.x / 2;
            while (i != 0)
            {
                if (cacheIndex < i)
                    cache[cacheIndex] += cache[cacheIndex + i];
                thread.SyncThreads();
                i /= 2;
            }

            if (cacheIndex == 0)
                c[thread.blockIdx.x] = cache[0];
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
    }
}