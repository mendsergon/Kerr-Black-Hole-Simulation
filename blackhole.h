#ifndef BLACKHOLE_H
#define BLACKHOLE_H

#include <glm/glm.hpp>
#include <vector>
#include <string>

// OpenCL for GPU-accelerated ray tracing
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

// ============================================================================
// Simulation Configuration (all overridable via command line)
// ============================================================================

struct SimConfig {
    float spin = 0.998f;          // Kerr spin parameter a/M
    float diskInner = 2.0f;       // Inner disk radius (units of M)
    float diskOuter = 20.0f;      // Outer disk radius
    int   maxSteps = 4000;        // Max RK4 steps per ray
    int   windowWidth = 1280;     // Initial window width
    int   windowHeight = 720;     // Initial window height
    float camDistance = 25.0f;    // Initial camera distance
    float camTheta = 1.3f;       // Initial camera polar angle (rad)
    float camFov = 60.0f;        // Initial FOV (degrees)
};

// Parse command-line arguments into config
SimConfig parseArgs(int argc, char** argv);

// Physical constants (not configurable)
const float BH_MASS = 1.0f;

// ============================================================================
// Camera State
// ============================================================================

struct Camera {
    float distance;     // Distance from black hole (in units of M)
    float theta;        // Polar angle (radians from +Y axis)
    float phi;          // Azimuthal angle (radians in XZ plane)
    float fov;          // Field of view in degrees

    // Smooth interpolation targets (inputs go here, actual values lerp toward them)
    float targetDistance;
    float targetTheta;
    float targetPhi;
    float targetFov;

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
    bool initialize(const SimConfig& config);

    // Clean up all OpenCL resources
    void cleanup();

    // Trace geodesics — writes geometry buffer (expensive, call on camera change only)
    void traceGeometry(const Camera& camera, int width, int height);

    // Shade from cached geometry — reads g-buffer, writes pixels (cheap, call every frame)
    // applyTonemap: true = output sRGB (fast path), false = output HDR (for bloom)
    void shadeFrame(int width, int height, float simTime, bool applyTonemap = true);

    // Get mutable reference to pixel data (avoid copying every frame)
    std::vector<float>& getPixels() { return m_pixels; }

    // Check if GPU acceleration is available
    bool isAvailable() const { return m_initialized; }

    // Check if g-buffer is valid for current dimensions
    bool hasGeometry() const { return m_gbufValid; }

    // Get device info string for display
    std::string getDeviceInfo() const { return m_deviceInfo; }

    // Get last frame render time in milliseconds
    double getLastTraceTimeMs() const { return m_lastTraceTimeMs; }
    double getLastShadeTimeMs() const { return m_lastShadeTimeMs; }

private:
    bool m_initialized;
    bool m_gbufValid;
    std::string m_deviceInfo;
    double m_lastTraceTimeMs;
    double m_lastShadeTimeMs;

    // Rendered pixel data (RGBA, width * height * 4 floats)
    std::vector<float> m_pixels;

    // OpenCL objects
    cl_platform_id m_platform;
    cl_device_id m_device;
    cl_context m_context;
    cl_command_queue m_queue;
    cl_program m_program;

    // Kernels — two-pass architecture
    cl_kernel m_traceKernel;   // Heavy: geodesic integration → g-buffer
    cl_kernel m_shadeKernel;   // Light: g-buffer → pixel colors

    // Buffers
    cl_mem m_gbufBuffer;   // Geometry buffer: 4 float4s per pixel
    cl_mem m_pixelBuffer;  // Output: RGBA float4 per pixel
    int m_bufferWidth;
    int m_bufferHeight;

    // Helper methods
    bool selectBestDevice();
    bool createContext();
    bool buildProgram(const SimConfig& config);
    bool createBuffers(int width, int height);
    std::string loadKernelSource(const std::string& filename);
};

// ============================================================================
// CPU Fallback Ray Tracer
// ============================================================================

// Trace all rays on CPU when GPU is unavailable
void renderFrameCPU(const Camera& camera, int width, int height, std::vector<float>& pixels,
                    float simTime, const SimConfig& config);

// ============================================================================
// Global Instances
// ============================================================================

extern GPURayTracer g_tracer;
extern SimConfig g_config;

// ============================================================================
// Utility Functions
// ============================================================================

// Create default camera from config
Camera createDefaultCamera(const SimConfig& config);

// Post-processing: bloom glow and gamma correction
void applyBloom(std::vector<float>& pixels, int width, int height);
void applyGammaCorrection(std::vector<float>& pixels);

// sRGB gamma LUT — precomputed, accessible for inline use in main loop
extern float g_srgbLUT[4097];
void initSRGBLUT();  // Call once before using g_srgbLUT

// Save screenshot as PPM (no library dependency)
void saveScreenshot(const std::vector<float>& pixels, int width, int height);

// Draw HUD text into pixel buffer (embedded bitmap font)
void drawHUD(std::vector<float>& pixels, int width, int height,
             const Camera& camera, double fps, double traceMs, double shadeMs,
             const SimConfig& config);

#endif // BLACKHOLE_H
