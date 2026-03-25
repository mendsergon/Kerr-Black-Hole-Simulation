// blackhole.cl — OpenCL Kernel for Kerr Black Hole Gravitational Lensing
//
// Implements GPU-accelerated ray tracing through curved spacetime using the
// Kerr metric in Boyer-Lindquist coordinates. Each GPU thread traces one
// photon (one pixel) backward from the camera through the gravitational
// field, checking for accretion disk intersection or horizon absorption.
//
// Physics Reference:
//   - Kerr metric: ds² in Boyer-Lindquist coordinates
//   - Geodesic integration via Hamiltonian formulation with RK4
//   - ZAMO (zero angular momentum observer) tetrad for camera
//   - Oliver James et al., "Gravitational Lensing by Spinning Black Holes
//     in Astrophysics and in the Movie Interstellar" (2015)
//
// Memory Layout:
//   pixels: float4 array [r, g, b, a] per pixel — output image
//
// Units: Geometrized (G = c = 1), distances in units of M (black hole mass)

// ============================================================================
// Constants
// ============================================================================

#define M       1.0f        // Black hole mass
#define SPIN    0.998f      // Kerr spin parameter a = J/(Mc), range [0, 1)

#define DISK_INNER  2.0f    // Inner edge of accretion disk (units of M)
#define DISK_OUTER  20.0f   // Outer edge of accretion disk

#define MAX_STEPS   2000    // Max geodesic integration steps
#define ESCAPE_R    50.0f   // Ray considered escaped at this radius
#define HORIZON_EPS 0.02f   // Stop tracing this close to horizon

#define PI 3.14159265359f

// ============================================================================
// Kerr Metric Helpers
// ============================================================================

// Compute Sigma = r^2 + a^2 cos^2(theta)
float kerr_sigma(float r, float theta) {
    float a = SPIN;
    float ct = cos(theta);
    return r * r + a * a * ct * ct;
}

// Compute Delta = r^2 - 2Mr + a^2
float kerr_delta(float r) {
    float a = SPIN;
    return r * r - 2.0f * M * r + a * a;
}

// Compute A = (r^2 + a^2)^2 - a^2 * Delta * sin^2(theta)
float kerr_A(float r, float theta) {
    float a = SPIN;
    float rr_aa = r * r + a * a;
    float delta = kerr_delta(r);
    float st = sin(theta);
    return rr_aa * rr_aa - a * a * delta * st * st;
}

// Event horizon radius (outer): r+ = M + sqrt(M^2 - a^2)
float horizon_radius() {
    float a = SPIN;
    return M + sqrt(M * M - a * a);
}

// ============================================================================
// Hamiltonian for Null Geodesics in Kerr Spacetime
// ============================================================================

// H = (1/2) g^{mu nu} p_mu p_nu = 0 for photons
//
// With p_t = -E and p_phi = L as constants of motion:
// 2H = g^rr p_r^2 + g^tt p_t^2 + g^{theta theta} p_theta^2
//      + 2 g^{t phi} p_t p_phi + g^{phi phi} p_phi^2
//
// Contravariant metric components:
//   g^rr       = Delta / Sigma
//   g^{tt}     = -A / (Sigma * Delta)
//   g^{theta}  = 1 / Sigma
//   g^{t phi}  = -2Mar / (Sigma * Delta)   [note: a = SPIN]
//   g^{phi}    = (Delta - a^2 sin^2 theta) / (Sigma * Delta * sin^2 theta)

float hamiltonian(float r, float theta, float p_r, float p_theta,
                  float E, float L) {
    float a = SPIN;
    float sigma = kerr_sigma(r, theta);
    float delta = kerr_delta(r);
    float bigA = kerr_A(r, theta);
    float st = sin(theta);
    float st2 = st * st;

    // Contravariant metric components
    float g_rr = delta / sigma;
    float g_thth = 1.0f / sigma;
    float g_tt = -bigA / (sigma * delta);
    float g_tph = -2.0f * M * a * r / (sigma * delta);
    float g_phph = (st2 > 1e-10f)
        ? (delta - a * a * st2) / (sigma * delta * st2)
        : 0.0f;

    // H = (1/2) [g^rr p_r^2 + g^thth p_th^2 + g^tt E^2
    //           - 2 g^tph E L + g^phph L^2]
    // Note: p_t = -E, so g^tt p_t^2 = g^tt E^2, g^tph p_t p_phi = -g^tph E L
    return 0.5f * (g_rr * p_r * p_r
                 + g_thth * p_theta * p_theta
                 + g_tt * E * E
                 - 2.0f * g_tph * E * L
                 + g_phph * L * L);
}

// ============================================================================
// Geodesic Equations of Motion (Hamilton's Equations)
// ============================================================================

// dr/dlambda = dH/dp_r = g^rr * p_r
// dtheta/dlambda = dH/dp_theta = g^{theta theta} * p_theta
// dp_r/dlambda = -dH/dr   (computed via numerical derivative)
// dp_theta/dlambda = -dH/dtheta   (computed via numerical derivative)

// Also need dphi/dlambda for disk coloring:
// dphi/dlambda = g^{phi phi} L + g^{t phi} (-E)
//             = g^{phi phi} L - g^{t phi} E

typedef struct {
    float r;
    float theta;
    float p_r;
    float p_theta;
    float phi;
} GeoState;

typedef struct {
    float dr;
    float dtheta;
    float dp_r;
    float dp_theta;
    float dphi;
} GeoDerivs;

GeoDerivs geodesic_derivs(GeoState s, float E, float L) {
    float a = SPIN;
    float sigma = kerr_sigma(s.r, s.theta);
    float delta = kerr_delta(s.r);
    float st = sin(s.theta);
    float st2 = st * st;

    GeoDerivs d;

    // dr/dlambda = (Delta / Sigma) * p_r
    d.dr = (delta / sigma) * s.p_r;

    // dtheta/dlambda = (1 / Sigma) * p_theta
    d.dtheta = (1.0f / sigma) * s.p_theta;

    // dp_r/dlambda = -dH/dr via central finite difference
    float eps = 1e-4f;
    float Hp = hamiltonian(s.r + eps, s.theta, s.p_r, s.p_theta, E, L);
    float Hm = hamiltonian(s.r - eps, s.theta, s.p_r, s.p_theta, E, L);
    d.dp_r = -(Hp - Hm) / (2.0f * eps);

    // dp_theta/dlambda = -dH/dtheta via central finite difference
    Hp = hamiltonian(s.r, s.theta + eps, s.p_r, s.p_theta, E, L);
    Hm = hamiltonian(s.r, s.theta - eps, s.p_r, s.p_theta, E, L);
    d.dp_theta = -(Hp - Hm) / (2.0f * eps);

    // dphi/dlambda = g^{phi phi} * L - g^{t phi} * E
    float bigA = kerr_A(s.r, s.theta);
    float g_tph = -2.0f * M * a * s.r / (sigma * delta);
    float g_phph = (st2 > 1e-10f)
        ? (delta - a * a * st2) / (sigma * delta * st2)
        : 0.0f;
    d.dphi = g_phph * L - g_tph * E;

    return d;
}

// ============================================================================
// 4th-Order Runge-Kutta Integrator
// ============================================================================

GeoState rk4_step(GeoState s, float E, float L, float h) {
    // k1
    GeoDerivs k1 = geodesic_derivs(s, E, L);

    // k2
    GeoState s2;
    s2.r       = s.r       + 0.5f * h * k1.dr;
    s2.theta   = s.theta   + 0.5f * h * k1.dtheta;
    s2.p_r     = s.p_r     + 0.5f * h * k1.dp_r;
    s2.p_theta = s.p_theta + 0.5f * h * k1.dp_theta;
    s2.phi     = s.phi     + 0.5f * h * k1.dphi;
    GeoDerivs k2 = geodesic_derivs(s2, E, L);

    // k3
    GeoState s3;
    s3.r       = s.r       + 0.5f * h * k2.dr;
    s3.theta   = s.theta   + 0.5f * h * k2.dtheta;
    s3.p_r     = s.p_r     + 0.5f * h * k2.dp_r;
    s3.p_theta = s.p_theta + 0.5f * h * k2.dp_theta;
    s3.phi     = s.phi     + 0.5f * h * k2.dphi;
    GeoDerivs k3 = geodesic_derivs(s3, E, L);

    // k4
    GeoState s4;
    s4.r       = s.r       + h * k3.dr;
    s4.theta   = s.theta   + h * k3.dtheta;
    s4.p_r     = s.p_r     + h * k3.dp_r;
    s4.p_theta = s.p_theta + h * k3.dp_theta;
    s4.phi     = s.phi     + h * k3.dphi;
    GeoDerivs k4 = geodesic_derivs(s4, E, L);

    // Combine: y_{n+1} = y_n + (h/6)(k1 + 2*k2 + 2*k3 + k4)
    GeoState result;
    result.r       = s.r       + (h / 6.0f) * (k1.dr       + 2.0f * k2.dr       + 2.0f * k3.dr       + k4.dr);
    result.theta   = s.theta   + (h / 6.0f) * (k1.dtheta   + 2.0f * k2.dtheta   + 2.0f * k3.dtheta   + k4.dtheta);
    result.p_r     = s.p_r     + (h / 6.0f) * (k1.dp_r     + 2.0f * k2.dp_r     + 2.0f * k3.dp_r     + k4.dp_r);
    result.p_theta = s.p_theta + (h / 6.0f) * (k1.dp_theta + 2.0f * k2.dp_theta + 2.0f * k3.dp_theta + k4.dp_theta);
    result.phi     = s.phi     + (h / 6.0f) * (k1.dphi     + 2.0f * k2.dphi     + 2.0f * k3.dphi     + k4.dphi);

    return result;
}

// ============================================================================
// ZAMO Tetrad — Local Observer Frame at Camera Position
// ============================================================================

// The Zero Angular Momentum Observer is the natural local frame in Kerr
// spacetime. The camera "hovers" at fixed (r, theta) with zero angular
// momentum, co-rotating with the frame-dragging velocity.
//
// Lapse:  alpha = sqrt(Sigma * Delta / A)
// Shift:  omega = 2Mar / A  (frame-dragging angular velocity)
//
// Tetrad basis vectors (contravariant):
//   e_t^mu   = (1/alpha, 0, 0, omega/alpha)
//   e_r^mu   = (0, sqrt(Delta/Sigma), 0, 0)
//   e_th^mu  = (0, 0, 1/sqrt(Sigma), 0)
//   e_phi^mu = (0, 0, 0, 1/(sin(theta) * sqrt(A/Sigma)))

// Given a local ray direction (n_r, n_th, n_phi) in the ZAMO frame,
// compute the conserved quantities E and L and the initial momenta p_r, p_theta
void camera_ray_to_momenta(float cam_r, float cam_theta,
                           float n_r, float n_th, float n_phi,
                           float* E_out, float* L_out,
                           float* p_r_out, float* p_theta_out) {
    float a = SPIN;
    float sigma = kerr_sigma(cam_r, cam_theta);
    float delta = kerr_delta(cam_r);
    float bigA = kerr_A(cam_r, cam_theta);
    float st = sin(cam_theta);

    // ZAMO quantities
    float alpha = sqrt(sigma * delta / bigA);       // Lapse function
    float omega = 2.0f * M * a * cam_r / bigA;      // Frame-dragging velocity

    // Photon 4-momentum components in coordinate basis (setting E_local = 1):
    // p^t     = 1/alpha
    // p^r     = n_r * sqrt(Delta / Sigma)
    // p^theta = n_th / sqrt(Sigma)
    // p^phi   = omega/alpha + n_phi / (sin(theta) * sqrt(A/Sigma))
    float pt_up = 1.0f / alpha;
    float pr_up = n_r * sqrt(delta / sigma);
    float pth_up = n_th / sqrt(sigma);
    float pphi_up = omega / alpha + n_phi / (st * sqrt(bigA / sigma));

    // Lower indices using the metric:
    // p_t = g_tt p^t + g_tphi p^phi
    // p_phi = g_tphi p^t + g_phiphi p^phi
    // p_r = g_rr p^r
    // p_theta = g_thetatheta p^theta
    float g_tt = -(delta - a * a * st * st) / sigma;
    float g_tph = -2.0f * M * a * cam_r * st * st / sigma;
    float g_rr = sigma / delta;
    float g_thth = sigma;
    float g_phph = st * st * bigA / sigma;

    float p_t = g_tt * pt_up + g_tph * pphi_up;
    float p_phi = g_tph * pt_up + g_phph * pphi_up;

    *E_out = -p_t;  // Energy (conserved, positive for future-directed photons)
    *L_out = p_phi;  // Angular momentum (conserved)
    *p_r_out = g_rr * pr_up;
    *p_theta_out = g_thth * pth_up;
}

// ============================================================================
// Accretion Disk Model
// ============================================================================

// Thin disk in equatorial plane (theta = pi/2)
// Temperature profile: Novikov-Thorne T ~ r^(-3/4) with ISCO correction
// Doppler beaming from orbital motion included

// Blackbody-approximate color from temperature (simplified)
float3 temperature_to_color(float T) {
    // Normalized temperature (0 = cold outer edge, 1 = hot inner edge)
    // Map to a warm color ramp: deep red -> orange -> yellow -> white
    float3 color;
    if (T < 0.25f) {
        // Deep red to red
        color = (float3)(0.4f + T * 2.4f, T * 0.4f, T * 0.1f);
    } else if (T < 0.5f) {
        // Red to orange
        float t = (T - 0.25f) * 4.0f;
        color = (float3)(1.0f, 0.1f + t * 0.5f, t * 0.05f);
    } else if (T < 0.75f) {
        // Orange to yellow
        float t = (T - 0.5f) * 4.0f;
        color = (float3)(1.0f, 0.6f + t * 0.3f, 0.05f + t * 0.2f);
    } else {
        // Yellow to white
        float t = (T - 0.75f) * 4.0f;
        color = (float3)(1.0f, 0.9f + t * 0.1f, 0.25f + t * 0.75f);
    }
    return color;
}

// Compute disk color at a given radius and phi
float4 disk_color(float r, float phi, float E, float L) {
    if (r < DISK_INNER || r > DISK_OUTER) {
        return (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Temperature profile: T ~ (DISK_INNER / r)^(3/4)
    float T_norm = pow(DISK_INNER / r, 0.75f);

    // Keplerian orbital velocity at this radius
    float a = SPIN;
    float v_orb = 1.0f / (sqrt(r) + a);  // Prograde Kerr circular velocity (approx)

    // Doppler factor: g = 1 / (gamma * (1 + v * sin(phi)))
    // This creates the asymmetric brightness — approaching side brighter
    float gamma = 1.0f / sqrt(1.0f - v_orb * v_orb);
    float doppler = 1.0f / (gamma * (1.0f + v_orb * sin(phi)));

    // Boosted temperature and intensity
    float T_boosted = T_norm * doppler;
    float intensity = doppler * doppler * doppler * doppler;  // Relativistic beaming I ~ g^4

    float3 base_color = temperature_to_color(clamp(T_boosted, 0.0f, 1.0f));
    float brightness = clamp(intensity * 1.5f, 0.0f, 5.0f);

    return (float4)(base_color.x * brightness,
                    base_color.y * brightness,
                    base_color.z * brightness,
                    1.0f);
}

// ============================================================================
// Background Star Field
// ============================================================================

// Simple procedural star field using hash function
// Note: fract(x) = x - floor(x), defined manually for portability
float hash(float2 p) {
    float h = dot(p, (float2)(127.1f, 311.7f));
    float v = sin(h) * 43758.5453123f;
    return v - floor(v);
}

float4 starfield_color(float theta, float phi) {
    // Map (theta, phi) to a 2D texture coordinate
    float u = phi / (2.0f * PI);
    float v = theta / PI;

    // Grid for star placement
    float2 grid = (float2)(u * 200.0f, v * 100.0f);
    float2 cell = floor(grid);
    float2 frac_part = grid - cell;

    // Check if this cell has a star
    float star_chance = hash(cell);
    float4 color = (float4)(0.005f, 0.005f, 0.015f, 1.0f);  // Dark background

    if (star_chance > 0.97f) {
        // Star present — compute brightness and position within cell
        float2 star_pos = (float2)(hash(cell + (float2)(1.0f, 0.0f)),
                                   hash(cell + (float2)(0.0f, 1.0f)));
        float dist = length(frac_part - star_pos);
        float brightness = exp(-dist * dist * 50.0f) * (0.5f + 0.5f * hash(cell + (float2)(3.0f, 7.0f)));

        // Star color (slight variation: blue-white to yellow-white)
        float temp = hash(cell + (float2)(5.0f, 3.0f));
        float3 star_col = (float3)(0.8f + 0.2f * temp,
                                   0.85f + 0.15f * temp,
                                   1.0f - 0.3f * temp);
        color.xyz += star_col * brightness;
    }

    return color;
}

// ============================================================================
// Main Ray Tracing Kernel
// ============================================================================

// Each work item traces one photon from camera through curved spacetime.
// The photon follows a null geodesic of the Kerr metric, integrated via RK4.
// We check for:
//   1. Crossing the equatorial plane (disk intersection)
//   2. Falling below the event horizon (absorbed)
//   3. Escaping to large radius (background star field)

__kernel void raytrace(
    __global float4* pixels,     // Output RGBA per pixel
    const int width,
    const int height,
    const float cam_r,           // Camera radial distance
    const float cam_theta,       // Camera polar angle
    const float cam_phi,         // Camera azimuthal angle
    const float cam_fov          // Field of view in radians
) {
    int idx = get_global_id(0);
    if (idx >= width * height) return;

    int px = idx % width;
    int py = idx / width;

    // ====================================================================
    // Step 1: Compute ray direction in camera local frame
    // ====================================================================

    // Pixel coordinates to normalized device coordinates [-1, 1]
    float aspect = (float)width / (float)height;
    float ndc_x = (2.0f * (px + 0.5f) / width - 1.0f) * aspect;
    float ndc_y = (2.0f * (py + 0.5f) / height - 1.0f);

    // Convert to angles from camera forward direction
    float half_fov = cam_fov * 0.5f;
    float screen_x = ndc_x * tan(half_fov);
    float screen_y = -ndc_y * tan(half_fov);  // Flip Y for screen coords

    // Local ray direction in ZAMO frame:
    //   Camera looks toward BH (-r direction). Photon arriving from BH
    //   propagates outward (+r, toward camera). We set up the photon's
    //   propagation direction, then trace backward to find where it came from.
    float ray_len = sqrt(screen_x * screen_x + screen_y * screen_y + 1.0f);
    float n_r   =  1.0f / ray_len;           // Outward (photon heading toward camera)
    float n_phi =  screen_x / ray_len;       // Horizontal
    float n_th  =  screen_y / ray_len;       // Vertical (theta direction)

    // ====================================================================
    // Step 2: Set up initial conditions
    // ====================================================================

    float E, L, p_r_init, p_theta_init;
    camera_ray_to_momenta(cam_r, cam_theta,
                          n_r, n_th, n_phi,
                          &E, &L, &p_r_init, &p_theta_init);

    GeoState state;
    state.r       = cam_r;
    state.theta   = cam_theta;
    state.p_r     = p_r_init;
    state.p_theta = p_theta_init;
    state.phi     = cam_phi;

    float r_horizon = horizon_radius();

    // ====================================================================
    // Step 3: Integrate geodesic with RK4
    // ====================================================================

    float4 final_color = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
    bool hit_disk = false;
    float disk_alpha_accum = 0.0f;
    float4 disk_color_accum = (float4)(0.0f);

    for (int step = 0; step < MAX_STEPS; step++) {
        // Adaptive step size: smaller steps near the black hole
        float h = 0.04f;
        if (state.r < 10.0f) {
            h = 0.02f;
        }
        if (state.r < 5.0f) {
            h = 0.01f;
        }
        if (state.r < 3.0f) {
            h = 0.005f;
        }

        // Store previous theta for equatorial crossing detection
        float prev_theta = state.theta;
        float prev_r = state.r;
        float prev_phi = state.phi;

        // RK4 step (backward in time — negative affine parameter)
        state = rk4_step(state, E, L, -h);

        // Clamp theta to valid range [epsilon, pi-epsilon]
        if (state.theta < 0.01f) {
            state.theta = 0.01f;
            state.p_theta = fabs(state.p_theta);
        }
        if (state.theta > PI - 0.01f) {
            state.theta = PI - 0.01f;
            state.p_theta = -fabs(state.p_theta);
        }

        // ================================================================
        // Check: Equatorial plane crossing (accretion disk intersection)
        // ================================================================

        float half_pi = PI * 0.5f;
        bool crossed_equator = (prev_theta - half_pi) * (state.theta - half_pi) < 0.0f;

        if (crossed_equator) {
            // Linear interpolation to find crossing point
            float t_cross = (half_pi - prev_theta) / (state.theta - prev_theta);
            float cross_r = prev_r + t_cross * (state.r - prev_r);
            float cross_phi = prev_phi + t_cross * (state.phi - prev_phi);

            if (cross_r >= DISK_INNER && cross_r <= DISK_OUTER) {
                float4 dcolor = disk_color(cross_r, cross_phi, E, L);
                float opacity = dcolor.w * 0.85f;

                // Front-to-back compositing (disk is semi-transparent)
                disk_color_accum.xyz += (1.0f - disk_alpha_accum) * opacity * dcolor.xyz;
                disk_alpha_accum += (1.0f - disk_alpha_accum) * opacity;
                hit_disk = true;
            }
        }

        // ================================================================
        // Check: Event horizon absorption
        // ================================================================

        if (state.r <= r_horizon + HORIZON_EPS) {
            // Photon absorbed by black hole — black pixel with disk overlay
            final_color = (float4)(0.0f, 0.0f, 0.0f, 1.0f);

            if (hit_disk) {
                final_color.xyz = disk_color_accum.xyz;
            }
            pixels[idx] = final_color;
            return;
        }

        // ================================================================
        // Check: Escaped to background
        // ================================================================

        if (state.r > ESCAPE_R && state.p_r < 0.0f) {
            // Ray escaped — sample star field at exit angles
            float4 bg = starfield_color(state.theta, fmod(state.phi + 100.0f * PI, 2.0f * PI));

            // Composite disk hits over background
            final_color.xyz = disk_color_accum.xyz + (1.0f - disk_alpha_accum) * bg.xyz;
            pixels[idx] = final_color;
            return;
        }
    }

    // Max steps exceeded — use accumulated disk color or dim background
    final_color.xyz = disk_color_accum.xyz;
    if (!hit_disk) {
        final_color.xyz = (float3)(0.02f, 0.01f, 0.03f);
    }
    pixels[idx] = final_color;
}
