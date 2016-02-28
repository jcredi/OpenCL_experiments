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
using Cudafy.Atomics;
using Cudafy.Translator;

namespace CudafyByExample
{

    class Program
    {
        //const int threadsPerBlock = 256;
        //const int blocksPerGrid = 32;

        [STAThread]
        public static void Main(string[] args)
        {
            
            try
            {
                CudafyModes.DeviceId = ChooseGPU();
                hist_gpu_shmem_atomics.Execute(CudafyModes.DeviceId);
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
    }


    public class hist_gpu_shmem_atomics
    {
        public const int SIZE = 100 * 1024 * 1024;

        [Cudafy]
        public static void histo_kernel(GThread thread, byte[] buffer, int size, uint[] histo)
        {
            // clear out the accumulation buffer called temp
            // since we are launched with 256 threads, it is easy
            // to clear that memory with one write per thread
            uint[] temp = thread.AllocateShared<uint>("temp", 256);
            temp[thread.threadIdx.x] = 0;
            thread.SyncThreads();

            // calculate the starting index and the offset to the next
            // block that each thread will be processing
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int stride = thread.blockDim.x * thread.gridDim.x;
            while (i < size)
            {
                thread.atomicAdd(ref temp[buffer[i]], 1);
                i += stride;
            }
            // sync the data from the above writes to shared memory
            // then add the shared memory values to the values from
            // the other thread blocks using global memory
            // atomic adds
            // same as before, since we have 256 threads, updating the
            // global histogram is just one write per thread!
            thread.SyncThreads();

            thread.atomicAdd(ref (histo[thread.threadIdx.x]), temp[thread.threadIdx.x]);
        }

        static byte[] big_random_block(int size)
        {
            Random rand = new Random(DateTime.Now.Millisecond);
            byte[] data = new byte[size];
            for (int i = 0; i < size; i++)
                data[i] = (byte)rand.Next(Byte.MaxValue);

            return data;
        } // a random array of bytes (0, 255), of length <size>

        public static int Execute(int deviceID)
        {
            // set up GPU and load module
            CudafyModule km = CudafyTranslator.Cudafy();
            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);

            Console.WriteLine("Attempting to load module...");
            System.Threading.Thread.Sleep(300);
            gpu.LoadModule(km);
            Console.WriteLine("Module successfully loaded.\n");
            System.Threading.Thread.Sleep(500);

            // cudart.dll must be accessible!
            GPGPUProperties prop = null;
            try
            {
                prop = gpu.GetDeviceProperties(true);
            }
            catch (DllNotFoundException)
            {
                prop = gpu.GetDeviceProperties(false);
            }

            byte[] buffer = big_random_block(SIZE);

            // repeat all 10 times (to measure runtime)
            for (int t = 0; t < 10; t++)
            {
                Console.WriteLine("Computing histogram of {0} bytes elements.", SIZE);
                // capture the start time
                // starting the timer here so that we include the cost of
                // all of the operations on the GPU.  if the data were
                // already on the GPU and we just timed the kernel
                // the timing would drop from 74 ms to 15 ms.  Very fast.
                gpu.StartTimer();

                // allocate memory on the GPU for the file's data
                byte[] dev_buffer = gpu.CopyToDevice(buffer);
                uint[] dev_histo = gpu.Allocate<uint>(256);
                gpu.Set(dev_histo); // set device array to zero

                // kernel launch - 2x the number of mps gave best timing
                Console.WriteLine("Processors: {0}", prop.MultiProcessorCount);
                int blocks = Math.Min(2 * prop.MultiProcessorCount, 32);
                gpu.Launch(blocks, 256).histo_kernel(dev_buffer, SIZE, dev_histo);

                uint[] histo = new uint[256];
                gpu.CopyFromDevice(dev_histo, histo);

                // get stop time, and display the timing results
                float elapsedTime = gpu.StopTimer();
                Console.WriteLine("GPU time: {0} ms", elapsedTime);

                /*
                // verify that we have the same counts via CPU
                for (int i = 0; i < SIZE; i++)
                    histo[buffer[i]]--;
                for (int i = 0; i < 256; i++)
                {
                    if (histo[i] != 0)
                        Console.WriteLine("Failure at {0}!", i);
                }
                 * */

                // Now do the same on the CPU
                Stopwatch stopwatch = Stopwatch.StartNew();
                uint[] histo_CPU = new uint[256];
                //Array.Clear(integerArray, 0, integerArray.Length); // needed??
                for (int i = 0; i < SIZE; i++)
                    histo_CPU[buffer[i]]++;
                stopwatch.Stop();
                Console.WriteLine("CPU time: {0} ms\n", stopwatch.ElapsedMilliseconds);

                // Sanity check
                for (int i = 0; i < 256; i++)
                {
                    if (histo[i] != histo_CPU[i])
                        Console.WriteLine("Failure at {0}!", i);
                }

                gpu.FreeAll();
            }
            
            return 0;
        }

    }
}