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

        [STAThread]
        public static void Main(string[] args)
        {

            try
            {
                CudafyModes.DeviceId = ChooseGPU();
                basic_double_stream_correct.Execute();
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


    public class basic_double_stream_correct
    {
        public const int N = (1024*1024);
        public const int FULL_DATA_SIZE =  (N*20);

        [Cudafy]
        public static void thekernel(GThread thread, int[] a, int[] b, int[] c)
        {
            int idx = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            if (idx < N) 
            {
                int idx1 = (idx + 1) % 256;
                int idx2 = (idx + 2) % 256;
                float aS = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
                float bS = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
                c[idx] = (int)(aS + bS) / 2;
            }
        }


        public static void Execute()
        {
            CudafyModule km = CudafyTranslator.Cudafy();
            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            Console.WriteLine("Double-check: running with device: {0}", gpu.GetDeviceProperties(false).Name);
            gpu.LoadModule(km);
            

            int[] dev_a0, dev_b0, dev_c0;
            int[] dev_a1, dev_b1, dev_c1;

            // allocate the memory on the GPU
            dev_a0 = gpu.Allocate<int>(N);
            dev_b0 = gpu.Allocate<int>(N);
            dev_c0 = gpu.Allocate<int>(N);
            dev_a1 = gpu.Allocate<int>(N);
            dev_b1 = gpu.Allocate<int>(N);
            dev_c1 = gpu.Allocate<int>(N);

            // allocate host locked memory, used to stream
            IntPtr host_aPtr = gpu.HostAllocate<int>(FULL_DATA_SIZE);
            IntPtr host_bPtr = gpu.HostAllocate<int>(FULL_DATA_SIZE);
            IntPtr host_cPtr = gpu.HostAllocate<int>(FULL_DATA_SIZE);
            
            Random rand = new Random();
            for (int i = 0; i < FULL_DATA_SIZE; i++)
            {
                host_aPtr.Set(i, rand.Next(1024 * 1024));  // There will be differences between the .NET code and the GPU
                host_bPtr.Set(i, rand.Next(1024 * 1024));  // So let's keep these to a minimum by having a max random values.
            }

            // start timer
            gpu.StartTimer();
 
            // now loop over full data, in bite-sized chunks
            for (int i = 0; i < FULL_DATA_SIZE; i += N * 2)
            {
                gpu.CopyToDeviceAsync(host_aPtr, i, dev_a0, 0, N, 1);
                gpu.CopyToDeviceAsync(host_bPtr, i, dev_b0, 0, N, 2);
                gpu.CopyToDeviceAsync(host_aPtr, i + N, dev_a1, 0, N, 1);
                gpu.CopyToDeviceAsync(host_bPtr, i + N, dev_b1, 0, N, 2);
                gpu.LaunchAsync(N / 256, 256, 1, "thekernel", dev_a0, dev_b0, dev_c0);
                gpu.LaunchAsync(N / 256, 256, 2, "thekernel", dev_a1, dev_b1, dev_c1);
                gpu.CopyFromDeviceAsync(dev_c0, 0, host_cPtr, i, N, 1);
                gpu.CopyFromDeviceAsync(dev_c1, 0, host_cPtr, i + N, N, 2);
            }
            gpu.SynchronizeStream(1);
            gpu.SynchronizeStream(2);
            
            float elapsed = gpu.StopTimer();

            // verify
            int[] host_a = new int[FULL_DATA_SIZE];
            int[] host_b = new int[FULL_DATA_SIZE];
            int[] host_c = new int[FULL_DATA_SIZE];

            GPGPU.CopyOnHost(host_aPtr, 0, host_a, 0, FULL_DATA_SIZE);
            GPGPU.CopyOnHost(host_bPtr, 0, host_b, 0, FULL_DATA_SIZE);
            GPGPU.CopyOnHost(host_cPtr, 0, host_c, 0, FULL_DATA_SIZE);
            Console.WriteLine("Elapsed: {0} ms", elapsed);

            int[] host_d = new int[FULL_DATA_SIZE];
            int errors = 0;
            int id = 0;
            {
                for (int j = 0; j < N; j++, id++)
                {
                    control(id, j, host_a, host_b, host_d);
                    if (host_c[id] > host_d[id] + 1) // There will be differences between the .NET code and the GPU
                    {
                        Console.WriteLine("Mismatch at {0}: {1} != {2}", id, host_c[id], host_d[id]);
                        errors++;
                        if (errors > 8)
                            break;
                    }
                }
            }
            
            gpu.HostFree(host_aPtr);
            gpu.HostFree(host_bPtr);
            gpu.HostFree(host_cPtr);
            gpu.DestroyStream(1);
            gpu.DestroyStream(2);
        }

        /// <summary>
        ///  Controls calculations.
        /// </summary>
        public static void control(int idx, int jdx, int[] a, int[] b, int[] c)
        {
            int idx1 = idx/N + (jdx + 1) % 256;
            int idx2 = idx/N + (jdx + 2) % 256;
            float aS = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
            float bS = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
            c[idx] = (int)(aS + bS) / 2;
        }
    }
}