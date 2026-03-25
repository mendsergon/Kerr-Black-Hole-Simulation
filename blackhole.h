#ifndef BLACKHOLE_H
#define BLACKHOLE_H

#include <glm/glm.hpp>
#include <vector>
#include <string>

// OpenCL for GPU-accelerated ray tracing
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

// ============================================================================
// Physical Constants (geometrized units: G = c = 1)
// ============================================================================

// Black hole mass — sets the scale of the simulation
// In geometrized units, M defines the Schwarzschild radius: r_s = 2M
const float BH_MASS = 1.0f;

// Black hole spin parameter: a = J / (M * c)
// Range [0, 1): 0 = Schwarzschild (non-spinning), ~1 = extremal Kerr
// Interstellar's Gargantua uses a ≈ 0.9999 — we use 0.998 for stability
const float BH_SPIN = 0.998f;

// Accretion disk inner and outer radii (in units of M)
// ISCO for a = 0.998 prograde is ~1.24M, we use a slightly larger value
const float DISK_INNER = 2.0f;
const float DISK_OUTER = 20.0f;

// Ray marching parameters
const int MAX_STEPS = 2000;        // Maximum integration steps per ray
const float STEP_SIZE = 0.04f;     // Base RK4 step size (adaptive near BH)
const float HORIZON_THRESHOLD = 0.01f; // Stop ray when this close to horizon

// Render resolution
const int RENDER_WIDTH = 1280;
const int RENDER_HEIGHT = 720;

// ============================================================================
// Camera State
// ============================================================================

struct Camera {
    float distance;     // Distance from black hole (in units of M)
    float theta;        // Polar angle (radians from +Y axis)
    float phi;          // Azimuthal angle (radians in XZ plane)
    float fov;          // Field of view in degrees

    // Mouse interaction state
    double lastMouseX;
    double lastMouseY;
    bool dragging;
};

// ============================================================================
// GPU Ray Tracer
// ============================================================================

class GPURayTracer {
public:
    GPURayTracer();
    ~GPURayTracer();

    // Initialize OpenCL with best available GPU
    bool initialize();

    // Clean up all OpenCL resources
    void cleanup();

    // Render a frame — traces all rays on GPU, writes to pixel buffer
    void renderFrame(const Camera& camera, int width, int height);

    // Get pointer to rendered pixel data (RGBA float4)
    const std::vector<float>& getPixels() const { return m_pixels; }

    // Check if GPU acceleration is available
    bool isAvailable() const { return m_initialized; }

    // Get device info string for display
    std::string getDeviceInfo() const { return m_deviceInfo; }

    // Get last frame render time in milliseconds
    double getLastFrameTimeMs() const { return m_lastFrameTimeMs; }

private:
    bool m_initialized;
    std::string m_deviceInfo;
    double m_lastFrameTimeMs;

    // Rendered pixel data (RGBA, width * height * 4 floats)
    std::vector<float> m_pixels;

    // OpenCL objects
    cl_platform_id m_platform;
    cl_device_id m_device;
    cl_context m_context;
    cl_command_queue m_queue;
    cl_program m_program;

    // Kernels
    cl_kernel m_raytraceKernel;

    // Buffers
    cl_mem m_pixelBuffer;  // Output: RGBA float4 per pixel
    int m_bufferWidth;
    int m_bufferHeight;

    // Helper methods
    bool selectBestDevice();
    bool createContext();
    bool buildProgram();
    bool createBuffers(int width, int height);
    std::string loadKernelSource(const std::string& filename);
};

// ============================================================================
// CPU Fallback Ray Tracer
// ============================================================================

// Trace all rays on CPU when GPU is unavailable
void renderFrameCPU(const Camera& camera, int width, int height, std::vector<float>& pixels);

// ============================================================================
// Global GPU Ray Tracer Instance
// ============================================================================

extern GPURayTracer g_tracer;

// ============================================================================
// Utility Functions
// ============================================================================

// Create default camera positioned to see the full lensing effect
Camera createDefaultCamera();

#endif // BLACKHOLE_H
