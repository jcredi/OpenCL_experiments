using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace CudafyIntroduction
{
    class Program
    {
        private static int N = 1024;

        private const int XSIZE = 4;
        private const int YSIZE = 8;
        private const int ZSIZE = 16;

        static void Main(string[] args)
        {
            try
            {
                // This 'smart' method will Cudafy all members with the Cudafy attribute in the calling type (i.e. Program)
                
                // If cudafying will not work for you (CUDA SDK + VS not set up right) then comment out above and
                // uncomment below. Remember to also comment out the Structs and 3D arrays region below.
                // CUDA 5.5 SDK must be installed and cl.exe (VC++ compiler) must be in path.
                //CudafyModule km = CudafyModule.Deserialize(typeof(Program).Name);

                // Get the first CUDA device and load our module
                //_gpu = CudafyHost.GetDevice(eGPUType.OpenCL);
                //_gpu.LoadModule(km);

                // MY EXPERIMENTS
                CudafyModes.Target = eGPUType.OpenCL;
                //CudafyModes.DeviceId = 0;
                CudafyTranslator.Language = eLanguage.OpenCL;
                _gpu = CudafyHost.GetDevice(CudafyModes.Target);
                eArchitecture arch = _gpu.GetArchitecture();
                CudafyModule km = CudafyTranslator.Cudafy(arch);
                _gpu.LoadModule(km);

                #region Simplest GPU function possible
                // Call the kernel method (which does nothing useful, but does it on the GPU)
                // We use .NET 4.0 Dynamics to resolve the method. We could also use _gpu.Launch(1, 1, "kernel");
                _gpu.Launch().myKernel();
                #endregion


                #region Add two numbers on GPU
                // Next we will add together some numbers. First we need to allocate memory on GPU for result (one int).
                // Then we launch our method and then read our results back again.
                int result;
                int[] dev_result = _gpu.Allocate<int>();
                _gpu.Launch().add(2, 7, dev_result); // or gpu.Launch(1, 1, "add", 2, 7, dev_c);
                _gpu.CopyFromDevice(dev_result, out result);
                Console.WriteLine("2 + 7 = {0}", result);
                Debug.Assert(result == 9);
                #endregion


                #region Hello, world
                // Write Hello, world on GPU
                string str = "Hello, world";
                char[] dev_str = _gpu.Allocate<char>(str.Length);
                char[] char_array = new char[str.Length];
                _gpu.Launch(1, 1, "WriteHelloWorldOnGPU", dev_str);
                _gpu.CopyFromDevice(dev_str, char_array);
                string host_str = new string(char_array);
                Console.WriteLine(host_str);
                Debug.Assert(str == host_str);
                #endregion

                #region Add vectors
                // Add vectors - GPUs are best at algorithms like working on matrices and large vectors
                // where lots of calculations can be done independently in parallel.
                int[] a = new int[N];
                int[] b = new int[N];
                int[] c = new int[N];
                // fill the arrays 'a' and 'b' on the CPU
                for (int i = 0; i < N; i++)
                {
                    a[i] = -i;
                    b[i] = i * i;
                }
                // copy the arrays 'a' and 'b' to the GPU - these overloads automatically allocate GPU memory
                int[] dev_a = _gpu.CopyToDevice(a);
                int[] dev_b = _gpu.CopyToDevice(b);
                // allocate memory on the GPU for the result - this allocate enough memory to hold a vector the
                // same length as vector c - it does not copy vector c (same as _gpu.Allocate<int>(c.Length);)
                int[] dev_c = _gpu.Allocate<int>(c);
                // Threads are grouped in Blocks. Blocks are grouped in a Grid. Here we launch N Blocks where
                // each block contains 1 thread. Note addVector contains a GThread arg - no need to pass this.
                // GThread is the Cudafy equivalent of the built-in CUDA variables. Use it to identify thread id.
                _gpu.Launch(N, 1).addVector(dev_a, dev_b, dev_c);
                // copy the array 'c' back from the GPU to the CPU
                _gpu.CopyFromDevice(dev_c, c);
                for (int i = 0; i < N; i++)
                    Debug.Assert(a[i] + b[i] == c[i]);
                Console.WriteLine("We just added {0} elements of our two vectors in {0} parallel threads.", N);
                // This used a bit more precious GPU memory than the earlier examples, so let's free it
                _gpu.FreeAll();
                #endregion

                #region Structs and 3D arrays
                // Here we will cudafy a .NET struct and use a 3D array - let's make a new module and this time
                // we will explicitly state what types to cudafy.
                km = CudafyTranslator.Cudafy(typeof(ComplexFloat), typeof(Struct3D)); // see Struct3D.cs
                _gpu.LoadModule(km, false); // don't unload existing loaded module so now there are two modules loaded
                Debug.Assert(_gpu.GetFunctionNames().Count() > 1);// prove it

                ComplexFloat[, ,] host_array3DS = new ComplexFloat[XSIZE, YSIZE, ZSIZE];
                ComplexFloat[, ,] result_array3DS = new ComplexFloat[XSIZE, YSIZE, ZSIZE];
                int ctr = 0;
                for (int x = 0; x < XSIZE; x++)
                    for (int y = 0; y < YSIZE; y++)
                        for (int z = 0; z < ZSIZE; z++, ctr++)
                        {
                            ComplexFloat entry = new ComplexFloat();
                            entry.Real = ctr * 2;
                            entry.Imag = ctr;
                            host_array3DS[x, y, z] = entry;
                        }
                ComplexFloat[, ,] dev_array3DS = _gpu.CopyToDevice(host_array3DS);

                // Let's launch old school sans dynamic. XSIZE*YSIZE blocks of 1 thread each.
                _gpu.Launch(new dim3(XSIZE, YSIZE), 1, "struct3D", dev_array3DS);
                _gpu.CopyFromDevice(dev_array3DS, result_array3DS);
                bool pass = true;
                ctr = 0;
                for (int x = 0; x < XSIZE; x++)
                {
                    for (int y = 0; y < YSIZE; y++)
                    {
                        for (int z = 0; z < ZSIZE && pass; z++, ctr++)
                        {
                            ComplexFloat expected = new ComplexFloat();
                            expected.Real = ctr * 4;
                            expected.Imag = ctr * 2;
                            //ComplexFloat expected = new ComplexFloat(ctr * 2, ctr).Add(new ComplexFloat(ctr * 2, ctr));
                            ComplexFloat res = result_array3DS[x, y, z];
                            pass = res.Real == expected.Real && res.Imag == expected.Imag;
                        }
                    }
                }
                Console.WriteLine(pass ? "Pass" : "Fail");
                #endregion

                Console.WriteLine("Done!");
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

            Console.ReadLine();
        }

        [Cudafy]
        public static void myKernel()
        {
        }

        [Cudafy]
        public static void add(int a, int b, int[] c)
        {
            c[0] = a + b;
        }

        [Cudafy]
        public static void WriteHelloWorldOnGPU(char[] c)
        {
            c[0] = 'H';
            c[1] = 'e';
            c[2] = 'l';
            c[3] = 'l';
            c[4] = 'o';
            c[5] = ',';
            c[6] = ' ';
            c[7] = 'w';
            c[8] = 'o';
            c[9] = 'r';
            c[10] = 'l';
            c[11] = 'd';
        }

        [Cudafy]
        public static void addVector(GThread thread, int[] a, int[] b, int[] c)
        {
            // Get the id of the thread. addVector is called N times in parallel, so we need 
            // to know which one we are dealing with.
            int tid = thread.blockIdx.x;
            // To prevent reading beyond the end of the array we check that the id is less than Length
            if (tid < a.Length)
                c[tid] = a[tid] + b[tid];
        }

        private static GPGPU _gpu;

        private static void HandleException(Exception ex)
        {
            Console.WriteLine(ex.Message);
        }
    }
}