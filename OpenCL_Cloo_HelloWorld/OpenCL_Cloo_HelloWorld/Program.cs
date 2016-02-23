using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using System.IO;
using Cloo;

namespace test
{
    class Program
    {
        static void Main(string[] args)
        {
            // pick first platform
            ComputePlatform platform = ComputePlatform.Platforms[0];

            // create context with all gpu devices
            ComputeContext context = new ComputeContext(ComputeDeviceTypes.Gpu,
                new ComputeContextPropertyList(platform), null, IntPtr.Zero);

            // create a command queue with first gpu found
            ComputeCommandQueue queue = new ComputeCommandQueue(context,
                context.Devices[0], ComputeCommandQueueFlags.None);

            // load opencl source
            StreamReader streamReader = new StreamReader("../../kernels.cl");
            string clSource = streamReader.ReadToEnd();
            streamReader.Close();

            // create program with opencl source
            ComputeProgram program = new ComputeProgram(context, clSource);

            // compile opencl source
            program.Build(null, null, null, IntPtr.Zero);

            // load chosen kernel from program
            ComputeKernel kernel = program.CreateKernel("IncrementNumber");

            // create an integer array and save its length
            int[] myIntArray = new int[] { 1, 2, 3, 4, 5, 6, 7 };
            int myArraySize = myIntArray.Length;
            IntPtr[] outputPointerArray = new IntPtr[myArraySize];

            // allocate a memory buffer with the message (the int array)
            ComputeBuffer<int> myInputBuffer = new ComputeBuffer<int>(context,
                ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, myIntArray);
            ComputeBuffer<int> myOutputBuffer = new ComputeBuffer<int>(context,
                ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, myIntArray);

            kernel.SetMemoryArgument(0, myInputBuffer); // set the input integer array
            kernel.SetMemoryArgument(1, myOutputBuffer); // set the output integer array

            // execute kernel
            queue.ExecuteTask(kernel, null);

            // wait for completion
            queue.Finish();

            // read output buffer
            queue.Read(myOutputBuffer, true, 0, myArraySize, outputPointerArray[0], null);
        }
    }
}
