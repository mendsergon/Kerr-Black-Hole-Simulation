#include "blackhole.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <ctime>
#include <sys/stat.h>

// ============================================================================
// Globals
// ============================================================================

GPURayTracer g_tracer;
SimConfig g_config;

// Compatibility macros — CPU physics code reads from config without mass edits
#define BH_SPIN (g_config.spin)
#define DISK_INNER (g_config.diskInner)
#define DISK_OUTER (g_config.diskOuter)
#define MAX_STEPS (g_config.maxSteps)
#define HORIZON_THRESHOLD 0.01f

// ============================================================================
// GPURayTracer Implementation
// ============================================================================

GPURayTracer::GPURayTracer()
    : m_initialized(false)
    , m_gbufValid(false)
    , m_lastTraceTimeMs(0.0)
    , m_lastShadeTimeMs(0.0)
    , m_platform(nullptr)
    , m_device(nullptr)
    , m_context(nullptr)
    , m_queue(nullptr)
    , m_program(nullptr)
    , m_traceKernel(nullptr)
    , m_shadeKernel(nullptr)
    , m_gbufBuffer(nullptr)
    , m_pixelBuffer(nullptr)
    , m_bufferWidth(0)
    , m_bufferHeight(0)
{
}

GPURayTracer::~GPURayTracer() {
    cleanup();
}

bool GPURayTracer::initialize(const SimConfig& config) {
    std::cout << "Initializing OpenCL GPU ray tracer..." << std::endl;

    if (!selectBestDevice()) {
        std::cerr << "No suitable OpenCL device found. Falling back to CPU." << std::endl;
        return false;
    }

    if (!createContext()) {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return false;
    }

    if (!buildProgram(config)) {
        std::cerr << "Failed to build OpenCL program." << std::endl;
        cleanup();
        return false;
    }

    m_initialized = true;
    std::cout << "✓ GPU ray tracer enabled: " << m_deviceInfo << std::endl;
    return true;
}

void GPURayTracer::cleanup() {
    if (m_traceKernel) clReleaseKernel(m_traceKernel);
    if (m_shadeKernel) clReleaseKernel(m_shadeKernel);
    if (m_gbufBuffer) clReleaseMemObject(m_gbufBuffer);
    if (m_pixelBuffer) clReleaseMemObject(m_pixelBuffer);
    if (m_program) clReleaseProgram(m_program);
    if (m_queue) clReleaseCommandQueue(m_queue);
    if (m_context) clReleaseContext(m_context);

    m_traceKernel = nullptr;
    m_shadeKernel = nullptr;
    m_gbufBuffer = nullptr;
    m_pixelBuffer = nullptr;
    m_program = nullptr;
    m_queue = nullptr;
    m_context = nullptr;
    m_initialized = false;
    m_gbufValid = false;
}

bool GPURayTracer::selectBestDevice() {
    cl_uint numPlatforms;
    cl_int err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        std::cerr << "No OpenCL platforms found." << std::endl;
        return false;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    // Score devices: prefer discrete GPUs with more compute units
    cl_device_id bestDevice = nullptr;
    cl_platform_id bestPlatform = nullptr;
    int bestScore = -1;

    for (auto platform : platforms) {
        char platformName[256];
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr);

        cl_uint numDevices;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
        if (err != CL_SUCCESS || numDevices == 0) continue;

        std::vector<cl_device_id> devices(numDevices);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);

        for (auto device : devices) {
            char deviceName[256];
            char deviceVendor[256];
            cl_uint computeUnits;

            clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
            clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(deviceVendor), deviceVendor, nullptr);
            clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, nullptr);

            int score = computeUnits;
            std::string vendor(deviceVendor);

            // Prefer AMD and NVIDIA over Intel integrated
            if (vendor.find("AMD") != std::string::npos ||
                vendor.find("Advanced Micro") != std::string::npos) {
                score += 1000;
            } else if (vendor.find("NVIDIA") != std::string::npos) {
                score += 1000;
            } else if (vendor.find("Intel") != std::string::npos) {
                score += 100;
            }

            std::cout << "  Found: " << deviceName << " (" << deviceVendor << ") "
                      << "CUs: " << computeUnits << ", Score: " << score << std::endl;

            if (score > bestScore) {
                bestScore = score;
                bestDevice = device;
                bestPlatform = platform;
                m_deviceInfo = std::string(deviceName) + " (" + deviceVendor + ")";
            }
        }
    }

    if (bestDevice == nullptr) {
        std::cerr << "No GPU device found." << std::endl;
        return false;
    }

    m_platform = bestPlatform;
    m_device = bestDevice;
    return true;
}

bool GPURayTracer::createContext() {
    cl_int err;

    m_context = clCreateContext(nullptr, 1, &m_device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create context: " << err << std::endl;
        return false;
    }

#ifdef CL_VERSION_2_0
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    m_queue = clCreateCommandQueueWithProperties(m_context, m_device, props, &err);
#else
    m_queue = clCreateCommandQueue(m_context, m_device, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue: " << err << std::endl;
        return false;
    }

    return true;
}

std::string GPURayTracer::loadKernelSource(const std::string& filename) {
    std::vector<std::string> paths = {
        filename,
        "./" + filename,
        "../" + filename,
        "kernels/" + filename
    };

    for (const auto& path : paths) {
        std::ifstream file(path);
        if (file.is_open()) {
            std::stringstream ss;
            ss << file.rdbuf();
            return ss.str();
        }
    }

    std::cerr << "Could not find kernel file: " << filename << std::endl;
    return "";
}

bool GPURayTracer::buildProgram(const SimConfig& config) {
    cl_int err;

    std::string source = loadKernelSource("blackhole.cl");
    if (source.empty()) {
        return false;
    }

    const char* sourcePtr = source.c_str();
    size_t sourceLen = source.length();

    m_program = clCreateProgramWithSource(m_context, 1, &sourcePtr, &sourceLen, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create program: " << err << std::endl;
        return false;
    }

    // Build with config values injected as defines
    char options[512];
    snprintf(options, sizeof(options),
             "-cl-fast-relaxed-math -cl-mad-enable "
             "-DSPIN=%.6ff -DDISK_INNER=%.2ff -DDISK_OUTER=%.2ff -DMAX_STEPS=%d",
             config.spin, config.diskInner, config.diskOuter, config.maxSteps);
    err = clBuildProgram(m_program, 1, &m_device, options, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(m_program, m_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(m_program, m_device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Build failed:\n" << log.data() << std::endl;
        return false;
    }

    m_traceKernel = clCreateKernel(m_program, "raytrace", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create raytrace kernel: " << err << std::endl;
        return false;
    }

    m_shadeKernel = clCreateKernel(m_program, "shade", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create shade kernel: " << err << std::endl;
        return false;
    }

    return true;
}

bool GPURayTracer::createBuffers(int width, int height) {
    if (width == m_bufferWidth && height == m_bufferHeight && m_pixelBuffer != nullptr) {
        return true;
    }

    if (m_pixelBuffer) clReleaseMemObject(m_pixelBuffer);
    if (m_gbufBuffer) clReleaseMemObject(m_gbufBuffer);

    cl_int err;

    // G-buffer: 4 float4s per pixel (crossing geometry + min_r)
    size_t gbufSize = width * height * 4 * sizeof(cl_float4);
    m_gbufBuffer = clCreateBuffer(m_context, CL_MEM_READ_WRITE, gbufSize, nullptr, &err);
    if (err != CL_SUCCESS) return false;

    // Pixel buffer: 1 float4 per pixel (RGBA output)
    size_t pixelSize = width * height * sizeof(cl_float4);
    m_pixelBuffer = clCreateBuffer(m_context, CL_MEM_WRITE_ONLY, pixelSize, nullptr, &err);
    if (err != CL_SUCCESS) return false;

    m_bufferWidth = width;
    m_bufferHeight = height;
    m_gbufValid = false;  // New size invalidates cached geometry
    return true;
}

void GPURayTracer::traceGeometry(const Camera& camera, int width, int height) {
    if (!m_initialized) return;
    if (!createBuffers(width, height)) return;

    auto startTime = std::chrono::high_resolution_clock::now();

    float cam_fov_rad = camera.fov * 3.14159265f / 180.0f;

    // Set trace kernel arguments — writes g-buffer
    clSetKernelArg(m_traceKernel, 0, sizeof(cl_mem), &m_gbufBuffer);
    clSetKernelArg(m_traceKernel, 1, sizeof(int), &width);
    clSetKernelArg(m_traceKernel, 2, sizeof(int), &height);
    clSetKernelArg(m_traceKernel, 3, sizeof(float), &camera.distance);
    clSetKernelArg(m_traceKernel, 4, sizeof(float), &camera.theta);
    clSetKernelArg(m_traceKernel, 5, sizeof(float), &camera.phi);
    clSetKernelArg(m_traceKernel, 6, sizeof(float), &cam_fov_rad);

    size_t totalPixels = width * height;
    size_t localSize = 256;
    size_t globalSize = ((totalPixels + localSize - 1) / localSize) * localSize;

    cl_int err = clEnqueueNDRangeKernel(m_queue, m_traceKernel, 1, nullptr,
                                         &globalSize, &localSize, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to launch raytrace kernel: " << err << std::endl;
        return;
    }
    clFinish(m_queue);

    m_gbufValid = true;

    auto endTime = std::chrono::high_resolution_clock::now();
    m_lastTraceTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
}

void GPURayTracer::shadeFrame(int width, int height, float simTime, bool applyTonemap) {
    if (!m_initialized || !m_gbufValid) return;

    // If resolution changed since trace, g-buffer is stale — need re-trace
    if (width != m_bufferWidth || height != m_bufferHeight) {
        m_gbufValid = false;
        return;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    int tonemapFlag = applyTonemap ? 1 : 0;

    // Set shade kernel arguments — reads g-buffer, writes pixels
    clSetKernelArg(m_shadeKernel, 0, sizeof(cl_mem), &m_gbufBuffer);
    clSetKernelArg(m_shadeKernel, 1, sizeof(cl_mem), &m_pixelBuffer);
    clSetKernelArg(m_shadeKernel, 2, sizeof(int), &width);
    clSetKernelArg(m_shadeKernel, 3, sizeof(int), &height);
    clSetKernelArg(m_shadeKernel, 4, sizeof(float), &simTime);
    clSetKernelArg(m_shadeKernel, 5, sizeof(int), &tonemapFlag);

    size_t totalPixels = width * height;
    size_t localSize = 256;
    size_t globalSize = ((totalPixels + localSize - 1) / localSize) * localSize;

    cl_int err = clEnqueueNDRangeKernel(m_queue, m_shadeKernel, 1, nullptr,
                                         &globalSize, &localSize, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to launch shade kernel: " << err << std::endl;
        return;
    }
    clFinish(m_queue);

    // Read pixels back
    m_pixels.resize(width * height * 4);
    clEnqueueReadBuffer(m_queue, m_pixelBuffer, CL_TRUE, 0,
                        width * height * sizeof(cl_float4), m_pixels.data(),
                        0, nullptr, nullptr);

    auto endTime = std::chrono::high_resolution_clock::now();
    m_lastShadeTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
}

// ============================================================================
// CPU Fallback Ray Tracer
// ============================================================================

// Duplicates the kernel logic for systems without OpenCL.
// Much slower but produces identical results.

namespace cpu {

const float PI = 3.14159265359f;

float kerr_sigma(float r, float theta) {
    float ct = cosf(theta);
    return r * r + BH_SPIN * BH_SPIN * ct * ct;
}

float kerr_delta(float r) {
    return r * r - 2.0f * BH_MASS * r + BH_SPIN * BH_SPIN;
}

float kerr_A(float r, float theta) {
    float rr_aa = r * r + BH_SPIN * BH_SPIN;
    float st = sinf(theta);
    return rr_aa * rr_aa - BH_SPIN * BH_SPIN * kerr_delta(r) * st * st;
}

float hamiltonian(float r, float theta, float p_r, float p_theta,
                  float E, float L) {
    float a = BH_SPIN;
    float sigma = kerr_sigma(r, theta);
    float delta = kerr_delta(r);
    float bigA = kerr_A(r, theta);
    float st = sinf(theta);
    float st2 = st * st;

    float g_rr = delta / sigma;
    float g_thth = 1.0f / sigma;
    float g_tt = -bigA / (sigma * delta);
    float g_tph = -2.0f * BH_MASS * a * r / (sigma * delta);
    float g_phph = (st2 > 1e-10f) ? (delta - a * a * st2) / (sigma * delta * st2) : 0.0f;

    return 0.5f * (g_rr * p_r * p_r + g_thth * p_theta * p_theta
                 + g_tt * E * E - 2.0f * g_tph * E * L + g_phph * L * L);
}

struct GeoState { float r, theta, p_r, p_theta, phi; };
struct GeoDerivs { float dr, dtheta, dp_r, dp_theta, dphi; };

GeoDerivs geodesic_derivs(GeoState s, float E, float L) {
    float a = BH_SPIN;
    float sigma = kerr_sigma(s.r, s.theta);
    float delta = kerr_delta(s.r);
    float st = sinf(s.theta);
    float st2 = st * st;

    GeoDerivs d;
    d.dr = (delta / sigma) * s.p_r;
    d.dtheta = (1.0f / sigma) * s.p_theta;

    float eps = 1e-4f;
    float Hp = hamiltonian(s.r + eps, s.theta, s.p_r, s.p_theta, E, L);
    float Hm = hamiltonian(s.r - eps, s.theta, s.p_r, s.p_theta, E, L);
    d.dp_r = -(Hp - Hm) / (2.0f * eps);

    Hp = hamiltonian(s.r, s.theta + eps, s.p_r, s.p_theta, E, L);
    Hm = hamiltonian(s.r, s.theta - eps, s.p_r, s.p_theta, E, L);
    d.dp_theta = -(Hp - Hm) / (2.0f * eps);

    float g_tph = -2.0f * BH_MASS * a * s.r / (sigma * delta);
    float g_phph = (st2 > 1e-10f) ? (delta - a * a * st2) / (sigma * delta * st2) : 0.0f;
    d.dphi = g_phph * L - g_tph * E;

    return d;
}

GeoState rk4_step(GeoState s, float E, float L, float h) {
    GeoDerivs k1 = geodesic_derivs(s, E, L);

    GeoState s2 = { s.r + 0.5f*h*k1.dr, s.theta + 0.5f*h*k1.dtheta,
                    s.p_r + 0.5f*h*k1.dp_r, s.p_theta + 0.5f*h*k1.dp_theta,
                    s.phi + 0.5f*h*k1.dphi };
    GeoDerivs k2 = geodesic_derivs(s2, E, L);

    GeoState s3 = { s.r + 0.5f*h*k2.dr, s.theta + 0.5f*h*k2.dtheta,
                    s.p_r + 0.5f*h*k2.dp_r, s.p_theta + 0.5f*h*k2.dp_theta,
                    s.phi + 0.5f*h*k2.dphi };
    GeoDerivs k3 = geodesic_derivs(s3, E, L);

    GeoState s4 = { s.r + h*k3.dr, s.theta + h*k3.dtheta,
                    s.p_r + h*k3.dp_r, s.p_theta + h*k3.dp_theta,
                    s.phi + h*k3.dphi };
    GeoDerivs k4 = geodesic_derivs(s4, E, L);

    GeoState result;
    result.r       = s.r       + (h/6.0f) * (k1.dr       + 2*k2.dr       + 2*k3.dr       + k4.dr);
    result.theta   = s.theta   + (h/6.0f) * (k1.dtheta   + 2*k2.dtheta   + 2*k3.dtheta   + k4.dtheta);
    result.p_r     = s.p_r     + (h/6.0f) * (k1.dp_r     + 2*k2.dp_r     + 2*k3.dp_r     + k4.dp_r);
    result.p_theta = s.p_theta + (h/6.0f) * (k1.dp_theta + 2*k2.dp_theta + 2*k3.dp_theta + k4.dp_theta);
    result.phi     = s.phi     + (h/6.0f) * (k1.dphi     + 2*k2.dphi     + 2*k3.dphi     + k4.dphi);
    return result;
}

void camera_ray_to_momenta(float cam_r, float cam_theta,
                           float n_r, float n_th, float n_phi,
                           float& E, float& L, float& p_r, float& p_theta) {
    float a = BH_SPIN;
    float sigma = kerr_sigma(cam_r, cam_theta);
    float delta = kerr_delta(cam_r);
    float bigA = kerr_A(cam_r, cam_theta);
    float st = sinf(cam_theta);

    float alpha = sqrtf(sigma * delta / bigA);
    float omega = 2.0f * BH_MASS * a * cam_r / bigA;

    float pt_up = 1.0f / alpha;
    float pr_up = n_r * sqrtf(delta / sigma);
    float pth_up = n_th / sqrtf(sigma);
    float pphi_up = omega / alpha + n_phi / (st * sqrtf(bigA / sigma));

    float g_tt = -(delta - a * a * st * st) / sigma;
    float g_tph = -2.0f * BH_MASS * a * cam_r * st * st / sigma;
    float g_rr_cov = sigma / delta;
    float g_thth = sigma;
    float g_phph = st * st * bigA / sigma;

    E = -(g_tt * pt_up + g_tph * pphi_up);
    L = g_tph * pt_up + g_phph * pphi_up;
    p_r = g_rr_cov * pr_up;
    p_theta = g_thth * pth_up;
}

glm::vec3 temperature_to_color(float T) {
    if (T < 0.25f) return glm::vec3(0.4f + T * 2.4f, T * 0.4f, T * 0.1f);
    if (T < 0.5f) { float t = (T - 0.25f) * 4.0f; return glm::vec3(1.0f, 0.1f + t * 0.5f, t * 0.05f); }
    if (T < 0.75f) { float t = (T - 0.5f) * 4.0f; return glm::vec3(1.0f, 0.6f + t * 0.3f, 0.05f + t * 0.2f); }
    float t = (T - 0.75f) * 4.0f; return glm::vec3(1.0f, 0.9f + t * 0.1f, 0.25f + t * 0.75f);
}

// Procedural star field (matches GPU kernel)
float hash(float x, float y) {
    float h = x * 127.1f + y * 311.7f;
    float v = sinf(h) * 43758.5453123f;
    return v - floorf(v);
}

glm::vec3 starfield_color(float theta, float phi) {
    float u = phi / (2.0f * PI);
    float v = theta / PI;

    glm::vec3 color(0.003f, 0.003f, 0.005f);

    // Layer 1: dense dim stars — check 3x3 neighborhood
    float gx1 = u * 400.0f, gy1 = v * 200.0f;
    float cx1 = floorf(gx1), cy1 = floorf(gy1);
    float fx1 = gx1 - cx1, fy1 = gy1 - cy1;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            float ncx = cx1 + dx, ncy = cy1 + dy;
            if (hash(ncx, ncy) > 0.94f) {
                float sx = hash(ncx + 1.0f, ncy);
                float sy = hash(ncx, ncy + 1.0f);
                float ddx = fx1 - sx - dx, ddy = fy1 - sy - dy;
                float dist = sqrtf(ddx*ddx + ddy*ddy);
                float br = expf(-dist*dist*80.0f) * 0.3f * (0.3f + 0.7f * hash(ncx+3.0f, ncy+7.0f));
                float t = hash(ncx+5.0f, ncy+3.0f);
                color += glm::vec3(0.7f+0.3f*t, 0.75f+0.25f*t, 0.9f-0.2f*t) * br;
            }
        }
    }

    // Layer 2: sparse bright stars — check 3x3 neighborhood
    float gx2 = u * 120.0f, gy2 = v * 60.0f;
    float cx2 = floorf(gx2), cy2 = floorf(gy2);
    float fx2 = gx2 - cx2, fy2 = gy2 - cy2;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            float ncx = cx2 + dx, ncy = cy2 + dy;
            if (hash(ncx+77.0f, ncy+33.0f) > 0.96f) {
                float sx = hash(ncx+11.0f, ncy);
                float sy = hash(ncx, ncy+11.0f);
                float ddx = fx2 - sx - dx, ddy = fy2 - sy - dy;
                float dist = sqrtf(ddx*ddx + ddy*ddy);
                float br = expf(-dist*dist*40.0f) * (0.5f + 0.5f * hash(ncx+13.0f, ncy+17.0f));
                float t = hash(ncx+15.0f, ncy+13.0f);
                color += glm::vec3(0.9f+0.1f*t, 0.85f+0.15f*t, 1.0f-0.3f*t) * br;
            }
        }
    }

    return color;
}

} // namespace cpu

void renderFrameCPU(const Camera& camera, int width, int height, std::vector<float>& pixels,
                    float simTime, const SimConfig& /* config — read via g_config macros */) {
    pixels.resize(width * height * 4);
    float cam_fov_rad = camera.fov * cpu::PI / 180.0f;
    float aspect = (float)width / (float)height;
    float tan_hfov = tanf(cam_fov_rad * 0.5f);  // Hoisted out of parallel loop
    float r_horizon = BH_MASS + sqrtf(BH_MASS * BH_MASS - BH_SPIN * BH_SPIN);

    // Process each pixel (slow on CPU but works without GPU)
    #pragma omp parallel for schedule(dynamic, 16)
    for (int idx = 0; idx < width * height; idx++) {
        int px = idx % width;
        int py = idx / width;

        float ndc_x = (2.0f * (px + 0.5f) / width - 1.0f) * aspect;
        float ndc_y = (2.0f * (py + 0.5f) / height - 1.0f);
        float sx = ndc_x * tan_hfov;
        float sy = -ndc_y * tan_hfov;

        float ray_len = sqrtf(sx * sx + sy * sy + 1.0f);
        float n_r   =  1.0f / ray_len;
        float n_phi =  sx / ray_len;
        float n_th  =  sy / ray_len;

        float E, L, p_r_init, p_theta_init;
        cpu::camera_ray_to_momenta(camera.distance, camera.theta,
                                   n_r, n_th, n_phi,
                                   E, L, p_r_init, p_theta_init);

        cpu::GeoState state = { camera.distance, camera.theta, p_r_init, p_theta_init, camera.phi };

        bool hit_disk = false;
        float disk_alpha = 0.0f;
        glm::vec3 disk_col(0.0f);

        for (int step = 0; step < MAX_STEPS; step++) {
            float h = 0.04f;
            if (state.r < 10.0f) h = 0.02f;
            if (state.r < 5.0f) h = 0.01f;
            if (state.r < 3.0f) h = 0.005f;

            float prev_theta = state.theta;
            float prev_r = state.r;
            float prev_phi = state.phi;

            state = cpu::rk4_step(state, E, L, -h);

            if (state.theta < 0.01f) { state.theta = 0.01f; state.p_theta = fabsf(state.p_theta); }
            if (state.theta > cpu::PI - 0.01f) { state.theta = cpu::PI - 0.01f; state.p_theta = -fabsf(state.p_theta); }

            float half_pi = cpu::PI * 0.5f;
            if ((prev_theta - half_pi) * (state.theta - half_pi) < 0.0f) {
                float t_cross = (half_pi - prev_theta) / (state.theta - prev_theta);
                float cross_r = prev_r + t_cross * (state.r - prev_r);
                float cross_phi = prev_phi + t_cross * (state.phi - prev_phi);

                if (cross_r >= DISK_INNER && cross_r <= DISK_OUTER) {
                    float T_norm = powf(DISK_INNER / cross_r, 0.75f);
                    float a = BH_SPIN;
                    // Animate: offset phi by Keplerian angular velocity × time
                    float omega_k = 1.0f / (sqrtf(cross_r) + a);
                    float disk_phi = cross_phi - omega_k * simTime;
                    float v_orb = 1.0f / (sqrtf(cross_r) + a);
                    float gamma = 1.0f / sqrtf(1.0f - v_orb * v_orb);
                    float doppler = 1.0f / (gamma * (1.0f + v_orb * sinf(disk_phi)));
                    float T_boosted = std::clamp(T_norm * doppler, 0.0f, 1.0f);
                    float intensity = std::clamp(doppler * doppler * doppler * doppler * 1.5f, 0.0f, 5.0f);
                    glm::vec3 base = cpu::temperature_to_color(T_boosted);
                    float opacity = 0.85f;
                    disk_col += (1.0f - disk_alpha) * opacity * base * intensity;
                    disk_alpha += (1.0f - disk_alpha) * opacity;
                    hit_disk = true;
                }
            }

            if (state.r <= r_horizon + HORIZON_THRESHOLD) {
                pixels[idx * 4 + 0] = hit_disk ? disk_col.r : 0.0f;
                pixels[idx * 4 + 1] = hit_disk ? disk_col.g : 0.0f;
                pixels[idx * 4 + 2] = hit_disk ? disk_col.b : 0.0f;
                pixels[idx * 4 + 3] = 1.0f;
                goto next_pixel;
            }

            if (state.r > 50.0f && state.p_r < 0.0f) {
                float exit_phi = fmodf(state.phi + 100.0f * cpu::PI, 2.0f * cpu::PI);
                glm::vec3 bg = cpu::starfield_color(state.theta, exit_phi);
                pixels[idx * 4 + 0] = disk_col.r + (1.0f - disk_alpha) * bg.r;
                pixels[idx * 4 + 1] = disk_col.g + (1.0f - disk_alpha) * bg.g;
                pixels[idx * 4 + 2] = disk_col.b + (1.0f - disk_alpha) * bg.b;
                pixels[idx * 4 + 3] = 1.0f;
                goto next_pixel;
            }
        }

        // Max steps — treat as escaped at current angle (not black)
        {
            float exit_phi = fmodf(state.phi + 100.0f * cpu::PI, 2.0f * cpu::PI);
            glm::vec3 bg = cpu::starfield_color(state.theta, exit_phi);
            pixels[idx * 4 + 0] = disk_col.r + (1.0f - disk_alpha) * bg.r;
            pixels[idx * 4 + 1] = disk_col.g + (1.0f - disk_alpha) * bg.g;
            pixels[idx * 4 + 2] = disk_col.b + (1.0f - disk_alpha) * bg.b;
            pixels[idx * 4 + 3] = 1.0f;
        }

        next_pixel:;
    }
}

// ============================================================================
// Camera Factory
// ============================================================================

Camera createDefaultCamera(const SimConfig& config) {
    Camera cam;
    cam.distance = config.camDistance;
    cam.theta = config.camTheta;
    cam.phi = 0.0f;
    cam.fov = config.camFov;
    cam.targetDistance = cam.distance;
    cam.targetTheta = cam.theta;
    cam.targetPhi = cam.phi;
    cam.targetFov = cam.fov;
    cam.lastMouseX = 0.0;
    cam.lastMouseY = 0.0;
    cam.dragging = false;
    return cam;
}

// ============================================================================
// Post-Processing: Bloom
// ============================================================================

void applyBloom(std::vector<float>& pixels, int width, int height) {
    int size = width * height * 4;
    std::vector<float> bright(size, 0.0f);
    float threshold = 0.75f;

    for (int i = 0; i < size; i += 4) {
        float luminance = 0.2126f * pixels[i] + 0.7152f * pixels[i+1] + 0.0722f * pixels[i+2];
        if (luminance > threshold) {
            float factor = (luminance - threshold) / (1.0f - threshold);
            bright[i + 0] = pixels[i + 0] * factor;
            bright[i + 1] = pixels[i + 1] * factor;
            bright[i + 2] = pixels[i + 2] * factor;
        }
    }

    int radius = std::max(2, width / 80);
    float sigma = (float)radius / 2.0f;

    std::vector<float> weights(radius + 1);
    float weightSum = 0.0f;
    for (int i = 0; i <= radius; i++) {
        weights[i] = expf(-(float)(i * i) / (2.0f * sigma * sigma));
        weightSum += (i == 0) ? weights[i] : 2.0f * weights[i];
    }
    for (int i = 0; i <= radius; i++) weights[i] /= weightSum;

    // Horizontal blur — reuse bright as output after reading
    std::vector<float> temp(size, 0.0f);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r = 0, g = 0, b = 0;
            for (int k = -radius; k <= radius; k++) {
                int sx = std::max(0, std::min(width - 1, x + k));
                int idx = (y * width + sx) * 4;
                float w = weights[abs(k)];
                r += bright[idx + 0] * w;
                g += bright[idx + 1] * w;
                b += bright[idx + 2] * w;
            }
            int idx = (y * width + x) * 4;
            temp[idx + 0] = r;
            temp[idx + 1] = g;
            temp[idx + 2] = b;
        }
    }

    // Vertical blur — write directly into bright (reuse allocation)
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r = 0, g = 0, b = 0;
            for (int k = -radius; k <= radius; k++) {
                int sy = std::max(0, std::min(height - 1, y + k));
                int idx = (sy * width + x) * 4;
                float w = weights[abs(k)];
                r += temp[idx + 0] * w;
                g += temp[idx + 1] * w;
                b += temp[idx + 2] * w;
            }
            int idx = (y * width + x) * 4;
            bright[idx + 0] = r;
            bright[idx + 1] = g;
            bright[idx + 2] = b;
        }
    }

    // Composite
    float bloomStrength = 0.4f;
    for (int i = 0; i < size; i += 4) {
        pixels[i + 0] = std::min(1.0f, pixels[i + 0] + bright[i + 0] * bloomStrength);
        pixels[i + 1] = std::min(1.0f, pixels[i + 1] + bright[i + 1] * bloomStrength);
        pixels[i + 2] = std::min(1.0f, pixels[i + 2] + bright[i + 2] * bloomStrength);
    }
}

// ============================================================================
// Post-Processing: sRGB Gamma Correction (LUT-accelerated)
// ============================================================================

// Precomputed LUT: maps linear [0,1] to sRGB [0,1] in 4096 steps
float g_srgbLUT[4097];
static bool g_srgbLUTReady = false;

void initSRGBLUT() {
    if (g_srgbLUTReady) return;
    for (int i = 0; i <= 4096; i++) {
        float x = (float)i / 4096.0f;
        g_srgbLUT[i] = (x < 0.0031308f) ? (12.92f * x) : (1.055f * powf(x, 1.0f / 2.4f) - 0.055f);
    }
    g_srgbLUTReady = true;
}

static inline float linearToSRGB(float x) {
    if (x <= 0.0f) return 0.0f;
    if (x >= 1.0f) return 1.0f;
    return g_srgbLUT[(int)(x * 4096.0f)];
}

void applyGammaCorrection(std::vector<float>& pixels) {
    initSRGBLUT();
    for (size_t i = 0; i < pixels.size(); i += 4) {
        pixels[i + 0] = linearToSRGB(pixels[i + 0]);
        pixels[i + 1] = linearToSRGB(pixels[i + 1]);
        pixels[i + 2] = linearToSRGB(pixels[i + 2]);
    }
}

// ============================================================================
// Command-Line Argument Parser
// ============================================================================

SimConfig parseArgs(int argc, char** argv) {
    SimConfig cfg;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Kerr Black Hole Simulation v1.2.1\n\n"
                      << "Usage: ./blackhole [options]\n\n"
                      << "Options:\n"
                      << "  --spin <float>       Spin parameter a/M (default: 0.998)\n"
                      << "  --disk-inner <float> Inner disk radius in M (default: 2.0)\n"
                      << "  --disk-outer <float> Outer disk radius in M (default: 20.0)\n"
                      << "  --steps <int>        Max RK4 steps per ray (default: 2000)\n"
                      << "  --width <int>        Window width (default: 1280)\n"
                      << "  --height <int>       Window height (default: 720)\n"
                      << "  --distance <float>   Camera distance in M (default: 25.0)\n"
                      << "  --theta <float>      Camera polar angle in degrees (default: 74.5)\n"
                      << "  --fov <float>        Field of view in degrees (default: 60.0)\n"
                      << "  --help               Show this help\n";
            exit(0);
        }
        if (i + 1 < argc) {
            if (arg == "--spin")       cfg.spin = std::stof(argv[++i]);
            else if (arg == "--disk-inner") cfg.diskInner = std::stof(argv[++i]);
            else if (arg == "--disk-outer") cfg.diskOuter = std::stof(argv[++i]);
            else if (arg == "--steps")      cfg.maxSteps = std::stoi(argv[++i]);
            else if (arg == "--width")      cfg.windowWidth = std::stoi(argv[++i]);
            else if (arg == "--height")     cfg.windowHeight = std::stoi(argv[++i]);
            else if (arg == "--distance")   cfg.camDistance = std::stof(argv[++i]);
            else if (arg == "--theta")      cfg.camTheta = std::stof(argv[++i]) * 3.14159265f / 180.0f;
            else if (arg == "--fov")        cfg.camFov = std::stof(argv[++i]);
        }
    }
    // Clamp spin to valid range
    if (cfg.spin < 0.0f) cfg.spin = 0.0f;
    if (cfg.spin >= 1.0f) cfg.spin = 0.999f;
    return cfg;
}

// ============================================================================
// Screenshot — saves as PPM (no library dependency)
// ============================================================================

void saveScreenshot(const std::vector<float>& pixels, int width, int height) {
    // Create Screenshots directory if it doesn't exist
    std::string dir = "Screenshots";
    #ifdef _WIN32
    _mkdir(dir.c_str());
    #else
    mkdir(dir.c_str(), 0755);
    #endif

    time_t now = time(nullptr);
    struct tm* t = localtime(&now);
    char filename[256];
    snprintf(filename, sizeof(filename), "Screenshots/blackhole_%04d%02d%02d_%02d%02d%02d.ppm",
             t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
             t->tm_hour, t->tm_min, t->tm_sec);

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to save screenshot: " << filename << std::endl;
        return;
    }

    file << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; i++) {
        unsigned char r = (unsigned char)(std::min(1.0f, std::max(0.0f, pixels[i * 4 + 0])) * 255.0f);
        unsigned char g = (unsigned char)(std::min(1.0f, std::max(0.0f, pixels[i * 4 + 1])) * 255.0f);
        unsigned char b = (unsigned char)(std::min(1.0f, std::max(0.0f, pixels[i * 4 + 2])) * 255.0f);
        file.write((char*)&r, 1);
        file.write((char*)&g, 1);
        file.write((char*)&b, 1);
    }

    file.close();
    std::cout << "Screenshot saved: " << filename << std::endl;
}

// ============================================================================
// HUD — embedded 4x6 bitmap font, drawn into pixel buffer
// ============================================================================

// Tiny 4x6 font covering ASCII 32-127 (space through ~)
// Each character is 4 pixels wide, 6 tall, packed as 3 bytes (24 bits = 4*6)
static const unsigned char g_font4x6[][3] = {
    {0x00,0x00,0x00}, // 32 space
    {0x44,0x40,0x40}, // 33 !
    {0xAA,0x00,0x00}, // 34 "
    {0xAE,0xAE,0xA0}, // 35 #
    {0x6C,0x6C,0x40}, // 36 $
    {0xA2,0x48,0xA0}, // 37 %
    {0x4A,0x4A,0x50}, // 38 &
    {0x44,0x00,0x00}, // 39 '
    {0x24,0x44,0x20}, // 40 (
    {0x42,0x22,0x40}, // 41 )
    {0xA4,0xA0,0x00}, // 42 *
    {0x04,0xE4,0x00}, // 43 +
    {0x00,0x02,0x40}, // 44 ,
    {0x00,0xE0,0x00}, // 45 -
    {0x00,0x00,0x40}, // 46 .
    {0x22,0x48,0x80}, // 47 /
    {0x4A,0xAA,0x40}, // 48 0
    {0x4C,0x44,0xE0}, // 49 1
    {0xC2,0x48,0xE0}, // 50 2
    {0xC2,0x42,0xC0}, // 51 3
    {0xAA,0xE2,0x20}, // 52 4
    {0xE8,0xC2,0xC0}, // 53 5
    {0x68,0xEA,0xE0}, // 54 6
    {0xE2,0x44,0x40}, // 55 7
    {0xEA,0xEA,0xE0}, // 56 8
    {0xEA,0xE2,0xC0}, // 57 9
    {0x04,0x04,0x00}, // 58 :
    {0x04,0x02,0x40}, // 59 ;
    {0x24,0x82,0x00}, // 60 <
    {0x0E,0x0E,0x00}, // 61 =
    {0x82,0x48,0x00}, // 62 >
    {0xC2,0x40,0x40}, // 63 ?
    {0x4A,0xE8,0x60}, // 64 @
    {0x4A,0xEA,0xA0}, // 65 A
    {0xCA,0xCA,0xC0}, // 66 B
    {0x68,0x88,0x60}, // 67 C
    {0xCA,0xAA,0xC0}, // 68 D
    {0xE8,0xC8,0xE0}, // 69 E
    {0xE8,0xC8,0x80}, // 70 F
    {0x68,0xAA,0x60}, // 71 G
    {0xAA,0xEA,0xA0}, // 72 H
    {0xE4,0x44,0xE0}, // 73 I
    {0x22,0x2A,0x40}, // 74 J
    {0xAA,0xCA,0xA0}, // 75 K
    {0x88,0x88,0xE0}, // 76 L
    {0xAE,0xAA,0xA0}, // 77 M
    {0xCA,0xAA,0xA0}, // 78 N
    {0x4A,0xAA,0x40}, // 79 O
    {0xCA,0xC8,0x80}, // 80 P
    {0x4A,0xAC,0x60}, // 81 Q
    {0xCA,0xCA,0xA0}, // 82 R
    {0x68,0x42,0xC0}, // 83 S
    {0xE4,0x44,0x40}, // 84 T
    {0xAA,0xAA,0xE0}, // 85 U
    {0xAA,0xAA,0x40}, // 86 V
    {0xAA,0xAE,0xA0}, // 87 W
    {0xAA,0x4A,0xA0}, // 88 X
    {0xAA,0x44,0x40}, // 89 Y
    {0xE2,0x48,0xE0}, // 90 Z
    {0x64,0x44,0x60}, // 91 [
    {0x88,0x42,0x20}, // 92 backslash
    {0x62,0x22,0x60}, // 93 ]
    {0x4A,0x00,0x00}, // 94 ^
    {0x00,0x00,0xE0}, // 95 _
    {0x42,0x00,0x00}, // 96 `
    {0x06,0xAA,0x60}, // 97 a
    {0x8C,0xAA,0xC0}, // 98 b
    {0x06,0x88,0x60}, // 99 c
    {0x26,0xAA,0x60}, // 100 d
    {0x04,0xAC,0x60}, // 101 e
    {0x24,0xE4,0x40}, // 102 f
    {0x06,0xA6,0x2C}, // 103 g
    {0x8C,0xAA,0xA0}, // 104 h
    {0x40,0x44,0x40}, // 105 i
    {0x20,0x22,0xA4}, // 106 j
    {0x8A,0xCA,0xA0}, // 107 k
    {0xC4,0x44,0xE0}, // 108 l
    {0x0A,0xEA,0xA0}, // 109 m
    {0x0C,0xAA,0xA0}, // 110 n
    {0x04,0xAA,0x40}, // 111 o
    {0x0C,0xAC,0x80}, // 112 p
    {0x06,0xA6,0x20}, // 113 q
    {0x06,0x88,0x80}, // 114 r
    {0x06,0xC2,0xC0}, // 115 s
    {0x4E,0x44,0x20}, // 116 t
    {0x0A,0xAA,0x60}, // 117 u
    {0x0A,0xAA,0x40}, // 118 v
    {0x0A,0xAE,0xA0}, // 119 w
    {0x0A,0x4A,0xA0}, // 120 x
    {0x0A,0xA6,0x2C}, // 121 y
    {0x0E,0x24,0xE0}, // 122 z
    {0x24,0x84,0x20}, // 123 {
    {0x44,0x44,0x40}, // 124 |
    {0x84,0x24,0x80}, // 125 }
    {0x5A,0x00,0x00}, // 126 ~
};

// Get pixel bit for character at (cx, cy) within 4x6 glyph
static bool fontPixel(char ch, int cx, int cy) {
    if (ch < 32 || ch > 126) return false;
    int idx = ch - 32;
    // 24 bits packed into 3 bytes, row-major, 4 pixels per row
    int bit = cy * 4 + cx;
    int byteIdx = bit / 8;
    int bitIdx = 7 - (bit % 8);
    return (g_font4x6[idx][byteIdx] >> bitIdx) & 1;
}

// Draw a string into pixel buffer at (x, y) with scale factor
static void drawString(std::vector<float>& pixels, int width, int height,
                       int startX, int startY, const char* text, int scale,
                       float r, float g, float b) {
    int curX = startX;
    for (int i = 0; text[i] != '\0'; i++) {
        if (text[i] == '\n') {
            curX = startX;
            startY += 7 * scale;
            continue;
        }
        for (int cy = 0; cy < 6; cy++) {
            for (int cx = 0; cx < 4; cx++) {
                if (fontPixel(text[i], cx, cy)) {
                    for (int sy = 0; sy < scale; sy++) {
                        for (int sx = 0; sx < scale; sx++) {
                            int px = curX + cx * scale + sx;
                            int py = startY + cy * scale + sy;
                            if (px >= 0 && px < width && py >= 0 && py < height) {
                                int idx = (py * width + px) * 4;
                                pixels[idx + 0] = r;
                                pixels[idx + 1] = g;
                                pixels[idx + 2] = b;
                            }
                        }
                    }
                }
            }
        }
        curX += 5 * scale;
    }
}

void drawHUD(std::vector<float>& pixels, int width, int height,
             const Camera& camera, double fps, double traceMs, double shadeMs,
             const SimConfig& config) {
    char buf[512];
    snprintf(buf, sizeof(buf),
             "a=%.3f  r=%.1fM  th=%.1f  fov=%.0f\n"
             "%.0f FPS  trace:%.0fms  shade:%.1fms\n"
             "disk: %.0f-%.0fM  steps:%d",
             config.spin, camera.distance,
             camera.theta * 180.0f / 3.14159265f,
             camera.fov,
             fps, traceMs, shadeMs,
             config.diskInner, config.diskOuter, config.maxSteps);

    int scale = std::max(1, width / 640);  // Scale with resolution
    drawString(pixels, width, height, 4 * scale, 4 * scale, buf, scale,
               1.0f, 1.0f, 1.0f);
}
