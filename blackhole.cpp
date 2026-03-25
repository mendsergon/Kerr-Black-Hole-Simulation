#include "blackhole.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>

// ============================================================================
// Global GPU Ray Tracer Instance
// ============================================================================

GPURayTracer g_tracer;

// ============================================================================
// GPURayTracer Implementation
// ============================================================================

GPURayTracer::GPURayTracer()
    : m_initialized(false)
    , m_lastFrameTimeMs(0.0)
    , m_platform(nullptr)
    , m_device(nullptr)
    , m_context(nullptr)
    , m_queue(nullptr)
    , m_program(nullptr)
    , m_raytraceKernel(nullptr)
    , m_pixelBuffer(nullptr)
    , m_bufferWidth(0)
    , m_bufferHeight(0)
{
}

GPURayTracer::~GPURayTracer() {
    cleanup();
}

bool GPURayTracer::initialize() {
    std::cout << "Initializing OpenCL GPU ray tracer..." << std::endl;

    if (!selectBestDevice()) {
        std::cerr << "No suitable OpenCL device found. Falling back to CPU." << std::endl;
        return false;
    }

    if (!createContext()) {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return false;
    }

    if (!buildProgram()) {
        std::cerr << "Failed to build OpenCL program." << std::endl;
        cleanup();
        return false;
    }

    m_initialized = true;
    std::cout << "✓ GPU ray tracer enabled: " << m_deviceInfo << std::endl;
    return true;
}

void GPURayTracer::cleanup() {
    if (m_raytraceKernel) clReleaseKernel(m_raytraceKernel);
    if (m_pixelBuffer) clReleaseMemObject(m_pixelBuffer);
    if (m_program) clReleaseProgram(m_program);
    if (m_queue) clReleaseCommandQueue(m_queue);
    if (m_context) clReleaseContext(m_context);

    m_raytraceKernel = nullptr;
    m_pixelBuffer = nullptr;
    m_program = nullptr;
    m_queue = nullptr;
    m_context = nullptr;
    m_initialized = false;
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

bool GPURayTracer::buildProgram() {
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

    // Build with fast math for GPU performance
    const char* options = "-cl-fast-relaxed-math -cl-mad-enable";
    err = clBuildProgram(m_program, 1, &m_device, options, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(m_program, m_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(m_program, m_device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Build failed:\n" << log.data() << std::endl;
        return false;
    }

    m_raytraceKernel = clCreateKernel(m_program, "raytrace", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create raytrace kernel: " << err << std::endl;
        return false;
    }

    return true;
}

bool GPURayTracer::createBuffers(int width, int height) {
    if (width == m_bufferWidth && height == m_bufferHeight && m_pixelBuffer != nullptr) {
        return true;
    }

    if (m_pixelBuffer) clReleaseMemObject(m_pixelBuffer);

    cl_int err;
    size_t bufferSize = width * height * sizeof(cl_float4);

    m_pixelBuffer = clCreateBuffer(m_context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &err);
    if (err != CL_SUCCESS) return false;

    m_bufferWidth = width;
    m_bufferHeight = height;
    return true;
}

void GPURayTracer::renderFrame(const Camera& camera, int width, int height) {
    if (!m_initialized) return;

    if (!createBuffers(width, height)) {
        std::cerr << "Failed to create GPU buffers." << std::endl;
        return;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    // Convert camera spherical to kernel parameters
    float cam_fov_rad = camera.fov * 3.14159265f / 180.0f;

    // Set kernel arguments
    clSetKernelArg(m_raytraceKernel, 0, sizeof(cl_mem), &m_pixelBuffer);
    clSetKernelArg(m_raytraceKernel, 1, sizeof(int), &width);
    clSetKernelArg(m_raytraceKernel, 2, sizeof(int), &height);
    clSetKernelArg(m_raytraceKernel, 3, sizeof(float), &camera.distance);
    clSetKernelArg(m_raytraceKernel, 4, sizeof(float), &camera.theta);
    clSetKernelArg(m_raytraceKernel, 5, sizeof(float), &camera.phi);
    clSetKernelArg(m_raytraceKernel, 6, sizeof(float), &cam_fov_rad);

    // Launch kernel — one work item per pixel
    size_t totalPixels = width * height;
    size_t localSize = 256;
    size_t globalSize = ((totalPixels + localSize - 1) / localSize) * localSize;

    cl_int err = clEnqueueNDRangeKernel(m_queue, m_raytraceKernel, 1, nullptr,
                                         &globalSize, &localSize, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to launch raytrace kernel: " << err << std::endl;
        return;
    }

    clFinish(m_queue);

    // Read back pixels
    m_pixels.resize(width * height * 4);
    clEnqueueReadBuffer(m_queue, m_pixelBuffer, CL_TRUE, 0,
                        width * height * sizeof(cl_float4), m_pixels.data(),
                        0, nullptr, nullptr);

    auto endTime = std::chrono::high_resolution_clock::now();
    m_lastFrameTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
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
    float gx = u * 200.0f;
    float gy = v * 100.0f;
    float cx = floorf(gx);
    float cy = floorf(gy);
    float fx = gx - cx;
    float fy = gy - cy;

    float star_chance = hash(cx, cy);
    glm::vec3 color(0.005f, 0.005f, 0.015f);

    if (star_chance > 0.97f) {
        float sx = hash(cx + 1.0f, cy);
        float sy = hash(cx, cy + 1.0f);
        float dx = fx - sx;
        float dy = fy - sy;
        float dist = sqrtf(dx * dx + dy * dy);
        float brightness = expf(-dist * dist * 50.0f) * (0.5f + 0.5f * hash(cx + 3.0f, cy + 7.0f));
        float temp = hash(cx + 5.0f, cy + 3.0f);
        glm::vec3 star_col(0.8f + 0.2f * temp, 0.85f + 0.15f * temp, 1.0f - 0.3f * temp);
        color += star_col * brightness;
    }
    return color;
}

} // namespace cpu

void renderFrameCPU(const Camera& camera, int width, int height, std::vector<float>& pixels) {
    pixels.resize(width * height * 4);
    float cam_fov_rad = camera.fov * cpu::PI / 180.0f;
    float aspect = (float)width / (float)height;
    float half_fov = cam_fov_rad * 0.5f;
    float r_horizon = BH_MASS + sqrtf(BH_MASS * BH_MASS - BH_SPIN * BH_SPIN);

    // Process each pixel (slow on CPU but works without GPU)
    #pragma omp parallel for schedule(dynamic, 16)
    for (int idx = 0; idx < width * height; idx++) {
        int px = idx % width;
        int py = idx / width;

        float ndc_x = (2.0f * (px + 0.5f) / width - 1.0f) * aspect;
        float ndc_y = (2.0f * (py + 0.5f) / height - 1.0f);
        float sx = ndc_x * tanf(half_fov);
        float sy = -ndc_y * tanf(half_fov);

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
                    float v_orb = 1.0f / (sqrtf(cross_r) + a);
                    float gamma = 1.0f / sqrtf(1.0f - v_orb * v_orb);
                    float doppler = 1.0f / (gamma * (1.0f + v_orb * sinf(cross_phi)));
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

        pixels[idx * 4 + 0] = hit_disk ? disk_col.r : 0.02f;
        pixels[idx * 4 + 1] = hit_disk ? disk_col.g : 0.01f;
        pixels[idx * 4 + 2] = hit_disk ? disk_col.b : 0.03f;
        pixels[idx * 4 + 3] = 1.0f;

        next_pixel:;
    }
}

// ============================================================================
// Camera Factory
// ============================================================================

Camera createDefaultCamera() {
    Camera cam;
    cam.distance = 25.0f;                  // 25M from the black hole
    cam.theta = 1.3f;                      // Slightly above equatorial plane (~74°)
    cam.phi = 0.0f;                        // Initial azimuthal angle
    cam.fov = 60.0f;                       // 60 degree field of view
    cam.lastMouseX = 0.0;
    cam.lastMouseY = 0.0;
    cam.dragging = false;
    return cam;
}
