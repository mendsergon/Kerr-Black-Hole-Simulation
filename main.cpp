#include <iostream>
#include <vector>
#include <cmath>
#include "blackhole.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>

// ============================================================================
// Global State
// ============================================================================

Camera g_camera;
int g_renderWidth = RENDER_WIDTH;
int g_renderHeight = RENDER_HEIGHT;
std::vector<float> g_pixels;
GLuint g_texture = 0;

// Simulation time — accumulates every frame for disk rotation
float g_simTime = 0.0f;
float g_simSpeed = 15.0f;  // Time units per real second (fast enough to see rotation)

// Camera smoothing factor (0 = no smoothing, 1 = frozen)
const float CAM_SMOOTH = 0.12f;

// High-quality render state
float g_settledTime = 0.0f;       // How long camera has been still
bool g_hqRendered = false;         // Whether we've done a high-quality frame since last move

// ============================================================================
// Callbacks
// ============================================================================

// Window resize
void framebuffer_size_callback([[maybe_unused]] GLFWwindow* window, int width, int height) {
    if (width <= 0 || height <= 0) return;  // Minimized window
    glViewport(0, 0, width, height);
    g_renderWidth = width;
    g_renderHeight = height;
    g_hqRendered = false;   // Force retrace at new resolution
    g_settledTime = 0.0f;
}

// ESC to exit
void key_callback(GLFWwindow* window, int key, [[maybe_unused]] int scancode,
                  int action, [[maybe_unused]] int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    // R key to reset camera
    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        g_camera = createDefaultCamera();
    }

    // +/- to adjust FOV
    if (key == GLFW_KEY_EQUAL && action != GLFW_RELEASE) {
        g_camera.targetFov = std::max(10.0f, g_camera.targetFov - 5.0f);
    }
    if (key == GLFW_KEY_MINUS && action != GLFW_RELEASE) {
        g_camera.targetFov = std::min(120.0f, g_camera.targetFov + 5.0f);
    }
}

// Mouse button — start/stop drag
void mouse_button_callback([[maybe_unused]] GLFWwindow* window, int button,
                           int action, [[maybe_unused]] int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            g_camera.dragging = true;
            glfwGetCursorPos(window, &g_camera.lastMouseX, &g_camera.lastMouseY);
        } else if (action == GLFW_RELEASE) {
            g_camera.dragging = false;
        }
    }
}

// Mouse motion — orbit camera around black hole
void cursor_position_callback([[maybe_unused]] GLFWwindow* window,
                              double xpos, double ypos) {
    if (!g_camera.dragging) return;

    double dx = xpos - g_camera.lastMouseX;
    double dy = ypos - g_camera.lastMouseY;
    g_camera.lastMouseX = xpos;
    g_camera.lastMouseY = ypos;

    // Rotate camera: horizontal mouse -> phi, vertical mouse -> theta
    float sensitivity = 0.005f;
    g_camera.targetPhi -= (float)dx * sensitivity;
    g_camera.targetTheta += (float)dy * sensitivity;

    // Clamp theta to avoid poles (singularities at 0 and pi)
    g_camera.targetTheta = std::max(0.1f, std::min(3.04f, g_camera.targetTheta));
}

// Scroll — zoom (change camera distance)
void scroll_callback([[maybe_unused]] GLFWwindow* window,
                     [[maybe_unused]] double xoffset, double yoffset) {
    float zoomFactor = powf(0.9f, (float)yoffset);
    g_camera.targetDistance *= zoomFactor;

    // Clamp distance: don't go inside photon sphere or too far out
    float r_horizon = BH_MASS + sqrtf(BH_MASS * BH_MASS - BH_SPIN * BH_SPIN);
    g_camera.targetDistance = std::max(r_horizon + 1.5f, std::min(200.0f, g_camera.targetDistance));
}

// ============================================================================
// Texture Upload — Displays ray-traced image via fullscreen quad
// ============================================================================

void uploadPixelsToTexture(const std::vector<float>& pixels, int width, int height) {
    if (g_texture == 0) {
        glGenTextures(1, &g_texture);
    }

    glBindTexture(GL_TEXTURE_2D, g_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0,
                 GL_RGBA, GL_FLOAT, pixels.data());
}

void drawFullscreenQuad() {
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_texture);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1, 1, -1, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 1); glVertex2f(-1, -1);
    glTexCoord2f(1, 1); glVertex2f( 1, -1);
    glTexCoord2f(1, 0); glVertex2f( 1,  1);
    glTexCoord2f(0, 0); glVertex2f(-1,  1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
}

// ============================================================================
// Main
// ============================================================================

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
    std::cout << "============================================" << std::endl;
    std::cout << "  Kerr Black Hole Simulation v1.0" << std::endl;
    std::cout << "  Gravitational Lensing Ray Tracer" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Black hole parameters:" << std::endl;
    std::cout << "  Mass: " << BH_MASS << " (geometrized units)" << std::endl;
    std::cout << "  Spin: " << BH_SPIN << " (a/M, Interstellar-like)" << std::endl;
    std::cout << "  Accretion disk: " << DISK_INNER << "M to " << DISK_OUTER << "M" << std::endl;
    std::cout << std::endl;

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    GLFWwindow* window = glfwCreateWindow(RENDER_WIDTH, RENDER_HEIGHT,
                                           "Kerr Black Hole — GPU Ray Tracer", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Enable VSync — caps at monitor refresh rate, prevents wasted GPU cycles
    glfwSwapInterval(1);

    // ====================================================================
    // Initialize GPU Ray Tracer
    // ====================================================================

    std::cout << std::endl;
    if (!g_tracer.initialize()) {
        std::cout << "⚠ GPU acceleration unavailable — using CPU fallback" << std::endl;
        std::cout << "  (This will be significantly slower)" << std::endl;
    }
    std::cout << std::endl;

    // Create camera
    g_camera = createDefaultCamera();

    std::cout << "Controls:" << std::endl;
    std::cout << "  Left-click + Drag — Orbit camera" << std::endl;
    std::cout << "  Scroll Up/Down    — Zoom in/out" << std::endl;
    std::cout << "  +/-               — Adjust FOV" << std::endl;
    std::cout << "  R                 — Reset camera" << std::endl;
    std::cout << "  ESC               — Exit" << std::endl;
    std::cout << std::endl;
    std::cout << "Rendering first frame..." << std::endl;

    // Performance tracking
    double lastTime = glfwGetTime();
    int frameCount = 0;
    double fpsUpdateTime = 0.0;

    // Track camera movement for resolution scaling
    float prevCamDist = g_camera.distance;
    float prevCamTheta = g_camera.theta;
    float prevCamPhi = g_camera.phi;

    // ====================================================================
    // Main Loop — continuous rendering (disk always rotating)
    // ====================================================================

    while (!glfwWindowShouldClose(window)) {
        // Frame timing
        double currentTime = glfwGetTime();
        float dt = (float)(currentTime - lastTime);
        dt = std::min(dt, 0.1f);  // Clamp to avoid spiral after long frames
        frameCount++;
        fpsUpdateTime += dt;
        lastTime = currentTime;

        // Accumulate simulation time — disk rotates regardless of input
        g_simTime += dt * g_simSpeed;

        // Smooth camera interpolation — lerp actual values toward targets
        g_camera.distance += CAM_SMOOTH * (g_camera.targetDistance - g_camera.distance);
        g_camera.theta    += CAM_SMOOTH * (g_camera.targetTheta - g_camera.theta);
        g_camera.phi      += CAM_SMOOTH * (g_camera.targetPhi - g_camera.phi);
        g_camera.fov      += CAM_SMOOTH * (g_camera.targetFov - g_camera.fov);

        // Detect if camera is still settling (moved since last frame)
        float camDelta = fabsf(g_camera.distance - prevCamDist)
                       + fabsf(g_camera.theta - prevCamTheta)
                       + fabsf(g_camera.phi - prevCamPhi);
        bool cameraMoving = g_camera.dragging || camDelta > 1e-4f;
        prevCamDist = g_camera.distance;
        prevCamTheta = g_camera.theta;
        prevCamPhi = g_camera.phi;

        // FPS display
        if (fpsUpdateTime >= 1.0) {
            char title[256];
            const char* mode = g_tracer.isAvailable() ? "GPU" : "CPU";
            snprintf(title, sizeof(title),
                     "Kerr Black Hole [%s: %.1f FPS | Trace: %.0fms Shade: %.1fms | r=%.1fM θ=%.1f° a=%.3f]",
                     mode, frameCount / fpsUpdateTime,
                     g_tracer.isAvailable() ? g_tracer.getLastTraceTimeMs() : 0.0,
                     g_tracer.isAvailable() ? g_tracer.getLastShadeTimeMs() : 0.0,
                     g_camera.distance,
                     g_camera.theta * 180.0f / 3.14159265f,
                     BH_SPIN);
            glfwSetWindowTitle(window, title);
            frameCount = 0;
            fpsUpdateTime = 0.0;
        }

        // ================================================================
        // Trace/Shade split architecture:
        //   Camera moves → re-trace geodesics (expensive, ~50-200ms)
        //   Camera still → shade only from cached g-buffer (cheap, ~1-3ms)
        //   Disk animation runs in shade pass at full framerate
        // ================================================================

        // Skip frame if window is minimized or invalid
        if (g_renderWidth <= 0 || g_renderHeight <= 0) {
            glfwSwapBuffers(window);
            glfwPollEvents();
            continue;
        }

        // Determine render resolution
        float scale;
        bool doBloom = false;
        bool needTrace = false;

        if (g_camera.dragging) {
            scale = 0.5f;
            g_settledTime = 0.0f;
            g_hqRendered = false;
            needTrace = true;
        } else if (cameraMoving) {
            scale = 0.5f;
            g_settledTime = 0.0f;
            g_hqRendered = false;
            needTrace = true;
        } else if (!g_hqRendered) {
            g_settledTime += dt;
            if (g_settledTime > 0.3f) {
                scale = 1.0f;
                doBloom = true;
                needTrace = true;
                g_hqRendered = true;
            } else {
                scale = 0.7f;
                needTrace = !g_tracer.hasGeometry();
            }
        } else {
            scale = 1.0f;
            needTrace = false;
        }

        int rw = (int)(g_renderWidth * scale);
        int rh = (int)(g_renderHeight * scale);
        rw = std::max(rw, 128);
        rh = std::max(rh, 72);

        // GPU path: always force trace if g-buffer is stale or missing
        std::vector<float>* pixelPtr = nullptr;
        bool gotPixels = false;

        if (g_tracer.isAvailable()) {
            if (!g_tracer.hasGeometry()) needTrace = true;
            if (needTrace) {
                g_tracer.traceGeometry(g_camera, rw, rh);
            }
            if (g_tracer.hasGeometry()) {
                g_tracer.shadeFrame(rw, rh, g_simTime);
                std::vector<float>& tpix = g_tracer.getPixels();
                if (!tpix.empty() && (int)tpix.size() == rw * rh * 4) {
                    pixelPtr = &tpix;
                    gotPixels = true;
                }
            }
        } else {
            renderFrameCPU(g_camera, rw, rh, g_pixels, g_simTime);
            if (!g_pixels.empty()) {
                pixelPtr = &g_pixels;
                gotPixels = true;
            }
        }

        // If no valid pixel data, just display whatever is already in the texture
        if (!gotPixels) {
            glClear(GL_COLOR_BUFFER_BIT);
            drawFullscreenQuad();
            glfwSwapBuffers(window);
            glfwPollEvents();
            continue;
        }

        std::vector<float>& pixels = *pixelPtr;

        // Post-processing
        if (doBloom) {
            for (size_t i = 0; i < pixels.size(); i += 4) {
                pixels[i + 0] = pixels[i + 0] / (1.0f + pixels[i + 0]);
                pixels[i + 1] = pixels[i + 1] / (1.0f + pixels[i + 1]);
                pixels[i + 2] = pixels[i + 2] / (1.0f + pixels[i + 2]);
            }
            applyBloom(pixels, rw, rh);
            applyGammaCorrection(pixels);
        } else {
            for (size_t i = 0; i < pixels.size(); i += 4) {
                float r = pixels[i + 0] / (1.0f + pixels[i + 0]);
                float g = pixels[i + 1] / (1.0f + pixels[i + 1]);
                float b = pixels[i + 2] / (1.0f + pixels[i + 2]);
                pixels[i + 0] = (r < 0.0031308f) ? (12.92f * r) : (1.055f * powf(r, 1.0f/2.4f) - 0.055f);
                pixels[i + 1] = (g < 0.0031308f) ? (12.92f * g) : (1.055f * powf(g, 1.0f/2.4f) - 0.055f);
                pixels[i + 2] = (b < 0.0031308f) ? (12.92f * b) : (1.055f * powf(b, 1.0f/2.4f) - 0.055f);
            }
        }

        uploadPixelsToTexture(pixels, rw, rh);

        // Display the ray-traced image
        glClear(GL_COLOR_BUFFER_BIT);
        drawFullscreenQuad();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // ====================================================================
    // Cleanup
    // ====================================================================

    std::cout << "Shutting down..." << std::endl;
    if (g_texture) glDeleteTextures(1, &g_texture);
    g_tracer.cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();

    std::cout << "Done." << std::endl;
    return 0;
}
