using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Cudafy;
namespace CudafyIntroduction
{
    /// <summary>
    /// When the Cudafy attribute is placed on a struct all members are translated.
    /// </summary>
    [Cudafy]
    public struct ComplexFloat
    {
        
        public float Real;
        public float Imag;

        // CAN'T CUDAFY METHODS WITHIN STRUCTS WITH OPENCL
        /*
        public ComplexFloat Add(ComplexFloat c)
        {
            ComplexFloat result = new ComplexFloat();
            result.Real = Real + c.Real;
            result.Imag = Imag + c.Imag;
            return result;
        }
        */
    }
    
    public class Struct3D
    {
        [Cudafy]
        public static void struct3D(GThread thread, ComplexFloat[, ,] result)
        {
            int x = thread.blockIdx.x;
            int y = thread.blockIdx.y;
            int z = 0;
            while (z < result.GetLength(2)) 
            {
                result[x, y, z].Real = result[x, y, z].Real + result[x, y, z].Real;
                result[x, y, z].Imag = result[x, y, z].Imag + result[x, y, z].Imag;
                z++;
            }
        }
    }
}
