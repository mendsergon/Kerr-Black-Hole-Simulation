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
bool g_needsRedraw = true;
int g_renderWidth = RENDER_WIDTH;
int g_renderHeight = RENDER_HEIGHT;
std::vector<float> g_pixels;
GLuint g_texture = 0;

// Render resolution scale (1.0 = full, 0.5 = half for interactive dragging)
float g_renderScale = 1.0f;
bool g_highQueuedRender = false;

// ============================================================================
// Callbacks
// ============================================================================

// Window resize
void framebuffer_size_callback([[maybe_unused]] GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    g_renderWidth = width;
    g_renderHeight = height;
    g_needsRedraw = true;
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
        g_needsRedraw = true;
    }

    // +/- to adjust FOV
    if (key == GLFW_KEY_EQUAL && action != GLFW_RELEASE) {
        g_camera.fov = std::max(10.0f, g_camera.fov - 5.0f);
        g_needsRedraw = true;
    }
    if (key == GLFW_KEY_MINUS && action != GLFW_RELEASE) {
        g_camera.fov = std::min(120.0f, g_camera.fov + 5.0f);
        g_needsRedraw = true;
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
            // Queue a high-res render after drag ends
            g_highQueuedRender = true;
            g_needsRedraw = true;
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
    g_camera.phi -= (float)dx * sensitivity;
    g_camera.theta += (float)dy * sensitivity;

    // Clamp theta to avoid poles (singularities at 0 and pi)
    g_camera.theta = std::max(0.1f, std::min(3.04f, g_camera.theta));

    g_needsRedraw = true;
}

// Scroll — zoom (change camera distance)
void scroll_callback([[maybe_unused]] GLFWwindow* window,
                     [[maybe_unused]] double xoffset, double yoffset) {
    float zoomFactor = powf(0.9f, (float)yoffset);
    g_camera.distance *= zoomFactor;

    // Clamp distance: don't go inside photon sphere or too far out
    float r_horizon = BH_MASS + sqrtf(BH_MASS * BH_MASS - BH_SPIN * BH_SPIN);
    g_camera.distance = std::max(r_horizon + 1.5f, std::min(200.0f, g_camera.distance));

    g_needsRedraw = true;
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

    // Disable VSync for maximum performance during rendering
    glfwSwapInterval(0);

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

    // ====================================================================
    // Main Loop
    // ====================================================================

    while (!glfwWindowShouldClose(window)) {
        // FPS tracking
        double currentTime = glfwGetTime();
        frameCount++;
        fpsUpdateTime += currentTime - lastTime;
        lastTime = currentTime;

        if (fpsUpdateTime >= 1.0) {
            char title[256];
            const char* mode = g_tracer.isAvailable() ? "GPU" : "CPU";
            snprintf(title, sizeof(title),
                     "Kerr Black Hole [%s: %.1f FPS | Render: %.0f ms | r=%.1fM θ=%.1f° a=%.3f]",
                     mode, frameCount / fpsUpdateTime,
                     g_tracer.isAvailable() ? g_tracer.getLastFrameTimeMs() : 0.0,
                     g_camera.distance,
                     g_camera.theta * 180.0f / 3.14159265f,
                     BH_SPIN);
            glfwSetWindowTitle(window, title);
            frameCount = 0;
            fpsUpdateTime = 0.0;
        }

        // Only re-render when camera changes
        if (g_needsRedraw) {
            // Use lower resolution during drag for interactivity
            float scale = g_camera.dragging ? 0.25f : (g_highQueuedRender ? 1.0f : 0.5f);
            int rw = (int)(g_renderWidth * scale);
            int rh = (int)(g_renderHeight * scale);
            rw = std::max(rw, 64);
            rh = std::max(rh, 36);

            if (g_tracer.isAvailable()) {
                g_tracer.renderFrame(g_camera, rw, rh);
                g_pixels = g_tracer.getPixels();
            } else {
                renderFrameCPU(g_camera, rw, rh, g_pixels);
            }

            // Apply simple tone mapping (clamp HDR to [0,1])
            for (size_t i = 0; i < g_pixels.size(); i += 4) {
                // Reinhard tone mapping
                g_pixels[i + 0] = g_pixels[i + 0] / (1.0f + g_pixels[i + 0]);
                g_pixels[i + 1] = g_pixels[i + 1] / (1.0f + g_pixels[i + 1]);
                g_pixels[i + 2] = g_pixels[i + 2] / (1.0f + g_pixels[i + 2]);
            }

            uploadPixelsToTexture(g_pixels, rw, rh);
            g_needsRedraw = g_camera.dragging;  // Keep rendering during drag
            g_highQueuedRender = false;

            // After first non-drag render at low res, queue a full-res render
            if (!g_camera.dragging && scale < 1.0f) {
                g_highQueuedRender = true;
                g_needsRedraw = true;
            }
        }

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
