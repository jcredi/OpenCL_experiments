/* 
 * This software is based upon the book CUDA By Example by Sanders and Kandrot
 * and source code provided by NVIDIA Corporation.
 * It is a good idea to read the book while studying the examples!
*/
using System;
using System.Collections.Generic;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using System.Drawing.Imaging;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace CudafyByExample
{
    public class ripple : Form
    {
        private bool bDONE = false;

        public ripple()
        {
            InitializeComponent();
        }

        public void Execute()
        {
            Show();
            int loops = CudafyModes.Target == eGPUType.Emulator ? 2 : 200;
            int side = ripple_gpu.DIM;
            Bitmap bmp = new Bitmap(side, side, PixelFormat.Format32bppArgb);
            Rectangle rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            int bytes = side * side * 4;
            byte[] rgbValues = new byte[bytes];
            ripple_gpu ripple = new ripple_gpu();
            ripple.Initialize(bytes);
            for (int x = 0; x < loops && !bDONE; x++)
            {
                ripple.Execute(rgbValues, Environment.TickCount);
                BitmapData bmpData = bmp.LockBits(rect, ImageLockMode.ReadWrite, bmp.PixelFormat);
                IntPtr ptr = bmpData.Scan0;

                System.Runtime.InteropServices.Marshal.Copy(rgbValues, 0, ptr, bytes);
                bmp.UnlockBits(bmpData);
                Text = x.ToString();
                pictureBox.Image = bmp;
                Refresh();
            }
            ripple.ShutDown();
            if (CudafyModes.Target == eGPUType.Emulator)
                MessageBox.Show("Click to continue.", "Information", MessageBoxButtons.OK, MessageBoxIcon.Information);
            Close();
        }

        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.pictureBox = new System.Windows.Forms.PictureBox();
            this.timer1 = new System.Windows.Forms.Timer(this.components);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox)).BeginInit();
            this.SuspendLayout();
            // 
            // pictureBox
            // 
            this.pictureBox.Location = new System.Drawing.Point(0, 0);
            this.pictureBox.Name = "pictureBox";
            this.pictureBox.Size = new System.Drawing.Size(1024, 1024);
            this.pictureBox.TabIndex = 1;
            this.pictureBox.TabStop = false;

            // 
            // timer1
            // 
            this.timer1.Interval = 5000;

            // 
            // ripple
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1026, 1045);
            this.Controls.Add(this.pictureBox);
            this.Name = "ripple";
            this.Text = "ripple";
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.PictureBox pictureBox;
        private System.Windows.Forms.Timer timer1;

    }

    public class ripple_gpu
    {
        public ripple_gpu()
        {
        }

        public const int DIM = 1024;

        private byte[] _dev_bitmap;

        private GPGPU _gpu;

        private dim3 _blocks;

        private dim3 _threads;

        public void Initialize(int bytes)
        {
            _gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            Console.WriteLine("Using device {0}: {1}", CudafyModes.DeviceId, _gpu.GetDeviceProperties(false).Name);
            Console.WriteLine("Platform Name:  {0}", _gpu.GetDeviceProperties(false).PlatformName);
            Console.WriteLine("Architecture:  {0}", _gpu.GetArchitecture());

            CudafyModule km = CudafyTranslator.Cudafy();
            _gpu.LoadModule(km);

            _dev_bitmap = _gpu.Allocate<byte>(bytes);

            _blocks = new dim3(DIM / 16, DIM / 16);
            _threads = new dim3(16, 16);
        }

        public void Execute(byte[] resultBuffer, int ticks)
        {
            _gpu.Launch(_blocks, _threads).thekernel(_dev_bitmap, ticks);
            _gpu.CopyFromDevice(_dev_bitmap, resultBuffer);
        }

        [Cudafy]
        public static void thekernel(GThread thread, byte[] ptr, int ticks)
        {
            // map from threadIdx/BlockIdx to pixel position
            int x = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int y = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
            int offset = x + y * thread.blockDim.x * thread.gridDim.x;

            // now calculate the value at that position
            float fx = x - DIM / 2;
            float fy = y - DIM / 2;
            float d = GMath.Sqrt(fx * fx + fy * fy);
            //float d = thread.sqrtf(fx * fx + fy * fy);
            byte grey = (byte)(128.0f + 127.0f * GMath.Cos(d / 10.0f - ticks / 7.0f) /
                                                 (d / 10.0f + 1.0f));
            ptr[offset * 4 + 0] = grey;
            ptr[offset * 4 + 1] = grey;
            ptr[offset * 4 + 2] = grey;
            ptr[offset * 4 + 3] = 255;
        }

        public void ShutDown()
        {
            _gpu.FreeAll();
        }
    }

    class Program
    {

        [STAThread]
        public static void Main(string[] args)
        {
            CudafyModes.Target = eGPUType.OpenCL;
            CudafyModes.DeviceId = 0;
            CudafyTranslator.Language = eLanguage.OpenCL;

            try
            {
                ripple r = new ripple();
                r.Execute();
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

        /*
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
        */
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