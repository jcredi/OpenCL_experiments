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

        #region Main method
        [STAThread]
        static void Main(string[] args)
        {
            try
            {
                GPGPU gpu = InitializeGPU();

                const int N = 1024 * 4096;

                // declare some arrays and allocate corresponding memory on GPU
                int[] a = new int[N];
                int[] b = new int[N];
                int[] c = new int[N];

                int[] dev_a = gpu.Allocate<int>(a);
                int[] dev_b = gpu.Allocate<int>(b);
                int[] dev_c = gpu.Allocate<int>(c);

                // fill the arrays 'a' and 'b' on the CPU and copy them to the device
                for (int j = 0; j < N; j++)
                {
                    a[j] = -j;
                    b[j] = 2 * j;
                }
                gpu.CopyToDevice(a, dev_a);
                gpu.CopyToDevice(b, dev_b);

                // launch the kernel on the GPU!
                Console.WriteLine("Summing two vectors of {0} elements on the GPU...", N);
                gpu.StartTimer();
                gpu.Launch(128, 1).addVectors(dev_a, dev_b, dev_c);

                // copy the array 'c' back from the GPU to the CPU
                gpu.CopyFromDevice(dev_c, c);
                float elapsedTime = gpu.StopTimer();

                // Verify that the GPU did the work we requested
                bool success = true;
                for (int i = 0; i < N; i++)
                {
                    if ((a[i] + b[i]) != c[i])
                    {
                        Console.WriteLine("{0} + {1} != {2}", a[i], b[i], c[i]);
                        success = false;
                        break;
                    }
                }
                if (success)
                    Console.WriteLine("We did it! In {0} ms :)", elapsedTime);
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
        #endregion

        public static GPGPU InitializeGPU()
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


        [Cudafy]
        public static void addVectors(GThread thread, int[] a, int[] b, int[] c)
        {
            // Get the id of the thread.
            int threadID = thread.blockIdx.x;
            // Make sure that the id is less than the length of the vectors
            while (threadID < a.Length)
            {
                c[threadID] = a[threadID] + b[threadID];
                // increment the thread id by the number of blocks in the grid
                threadID += thread.gridDim.x;
            }
        }

        private static void HandleException(Exception ex)
        {
            Console.WriteLine(ex.Message);
        }

        
        public static void Exit()
        {
            Console.WriteLine();
            Console.WriteLine("The program will now close. Press a key to exit...");
            Console.ReadKey();
        }
    }
}