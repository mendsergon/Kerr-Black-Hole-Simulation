// blackhole.cl — Kerr Black Hole Gravitational Lensing
//
// v1.2 — Configurable parameters via OpenCL build-time defines.
//   All #defines use #ifndef guards so CLI values override defaults.

#ifndef M
#define M       1.0f
#endif
#ifndef SPIN
#define SPIN    0.998f
#endif
#ifndef DISK_INNER
#define DISK_INNER  2.0f
#endif
#ifndef DISK_OUTER
#define DISK_OUTER  20.0f
#endif
#ifndef MAX_STEPS
#define MAX_STEPS   2000
#endif
#ifndef ESCAPE_R
#define ESCAPE_R    50.0f
#endif
#ifndef HORIZON_EPS
#define HORIZON_EPS 0.02f
#endif
#define PI 3.14159265359f

// Max disk crossings stored per ray (primary + secondary + tertiary)
#define MAX_CROSSINGS 3

// ============================================================================
// Kerr Metric — precise math for geodesic integration
// ============================================================================

float kerr_sigma(float r, float theta) {
    float a = SPIN;
    float ct = cos(theta);
    return r * r + a * a * ct * ct;
}

float kerr_delta(float r) {
    float a = SPIN;
    return r * r - 2.0f * M * r + a * a;
}

float kerr_A(float r, float theta) {
    float a = SPIN;
    float rr_aa = r * r + a * a;
    float st = sin(theta);
    return rr_aa * rr_aa - a * a * kerr_delta(r) * st * st;
}

float horizon_radius() {
    float a = SPIN;
    return M + sqrt(M * M - a * a);
}

float hamiltonian(float r, float theta, float p_r, float p_theta,
                  float E, float L) {
    float a = SPIN;
    float sigma = kerr_sigma(r, theta);
    float delta = kerr_delta(r);
    float bigA = kerr_A(r, theta);
    float st = sin(theta);
    float st2 = st * st;

    float g_rr = delta / sigma;
    float g_thth = 1.0f / sigma;
    float g_tt = -bigA / (sigma * delta);
    float g_tph = -2.0f * M * a * r / (sigma * delta);
    float g_phph = (st2 > 1e-10f)
        ? (delta - a * a * st2) / (sigma * delta * st2)
        : 0.0f;

    return 0.5f * (g_rr * p_r * p_r + g_thth * p_theta * p_theta
                 + g_tt * E * E - 2.0f * g_tph * E * L + g_phph * L * L);
}

typedef struct { float r, theta, p_r, p_theta, phi; } GeoState;
typedef struct { float dr, dtheta, dp_r, dp_theta, dphi; } GeoDerivs;

GeoDerivs geodesic_derivs(GeoState s, float E, float L) {
    float a = SPIN;
    float sigma = kerr_sigma(s.r, s.theta);
    float delta = kerr_delta(s.r);
    float st = sin(s.theta);
    float st2 = st * st;

    GeoDerivs d;
    d.dr = (delta / sigma) * s.p_r;
    d.dtheta = (1.0f / sigma) * s.p_theta;

    float eps = 1e-4f;
    float Hp = hamiltonian(s.r + eps, s.theta, s.p_r, s.p_theta, E, L);
    float Hm = hamiltonian(s.r - eps, s.theta, s.p_r, s.p_theta, E, L);
    d.dp_r = -(Hp - Hm) * 5000.0f;

    Hp = hamiltonian(s.r, s.theta + eps, s.p_r, s.p_theta, E, L);
    Hm = hamiltonian(s.r, s.theta - eps, s.p_r, s.p_theta, E, L);
    d.dp_theta = -(Hp - Hm) * 5000.0f;

    float bigA = kerr_A(s.r, s.theta);
    float g_tph = -2.0f * M * a * s.r / (sigma * delta);
    float g_phph = (st2 > 1e-10f)
        ? (delta - a * a * st2) / (sigma * delta * st2)
        : 0.0f;
    d.dphi = g_phph * L - g_tph * E;

    return d;
}

GeoState rk4_step(GeoState s, float E, float L, float h) {
    GeoDerivs k1 = geodesic_derivs(s, E, L);
    float hh = 0.5f * h;

    GeoState s2 = { s.r + hh*k1.dr, s.theta + hh*k1.dtheta,
                    s.p_r + hh*k1.dp_r, s.p_theta + hh*k1.dp_theta,
                    s.phi + hh*k1.dphi };
    GeoDerivs k2 = geodesic_derivs(s2, E, L);

    GeoState s3 = { s.r + hh*k2.dr, s.theta + hh*k2.dtheta,
                    s.p_r + hh*k2.dp_r, s.p_theta + hh*k2.dp_theta,
                    s.phi + hh*k2.dphi };
    GeoDerivs k3 = geodesic_derivs(s3, E, L);

    GeoState s4 = { s.r + h*k3.dr, s.theta + h*k3.dtheta,
                    s.p_r + h*k3.dp_r, s.p_theta + h*k3.dp_theta,
                    s.phi + h*k3.dphi };
    GeoDerivs k4 = geodesic_derivs(s4, E, L);

    float h6 = h * 0.166666667f;
    GeoState result;
    result.r       = s.r       + h6 * (k1.dr       + 2.0f*k2.dr       + 2.0f*k3.dr       + k4.dr);
    result.theta   = s.theta   + h6 * (k1.dtheta   + 2.0f*k2.dtheta   + 2.0f*k3.dtheta   + k4.dtheta);
    result.p_r     = s.p_r     + h6 * (k1.dp_r     + 2.0f*k2.dp_r     + 2.0f*k3.dp_r     + k4.dp_r);
    result.p_theta = s.p_theta + h6 * (k1.dp_theta + 2.0f*k2.dp_theta + 2.0f*k3.dp_theta + k4.dp_theta);
    result.phi     = s.phi     + h6 * (k1.dphi     + 2.0f*k2.dphi     + 2.0f*k3.dphi     + k4.dphi);
    return result;
}

void camera_ray_to_momenta(float cam_r, float cam_theta,
                           float n_r, float n_th, float n_phi,
                           float* E_out, float* L_out,
                           float* p_r_out, float* p_theta_out) {
    float a = SPIN;
    float sigma = kerr_sigma(cam_r, cam_theta);
    float delta = kerr_delta(cam_r);
    float bigA = kerr_A(cam_r, cam_theta);
    float st = sin(cam_theta);

    float alpha = sqrt(sigma * delta / bigA);
    float omega = 2.0f * M * a * cam_r / bigA;

    float pt_up = 1.0f / alpha;
    float pr_up = n_r * sqrt(delta / sigma);
    float pth_up = n_th / sqrt(sigma);
    float pphi_up = omega / alpha + n_phi / (st * sqrt(bigA / sigma));

    float g_tt = -(delta - a * a * st * st) / sigma;
    float g_tph = -2.0f * M * a * cam_r * st * st / sigma;
    float g_rr = sigma / delta;
    float g_thth = sigma;
    float g_phph = st * st * bigA / sigma;

    float p_t = g_tt * pt_up + g_tph * pphi_up;
    float p_phi = g_tph * pt_up + g_phph * pphi_up;

    *E_out = -p_t;
    *L_out = p_phi;
    *p_r_out = g_rr * pr_up;
    *p_theta_out = g_thth * pth_up;
}

// ============================================================================
// KERNEL 1: raytrace — Trace geodesics, write geometry buffer
// ============================================================================
//
// G-buffer layout per pixel (4 float4s = 64 bytes):
//   gbuf[idx*4 + 0] = (hit_type, num_hits, bg_theta, bg_phi)
//     hit_type: 0=horizon, 1=escaped, -1=NaN/maxsteps
//   gbuf[idx*4 + 1] = (cross_r_0, cross_phi_0, cross_r_1, cross_phi_1)
//   gbuf[idx*4 + 2] = (cross_r_2, cross_phi_2, 0, 0)
//   gbuf[idx*4 + 3] = (opacity_0, opacity_1, opacity_2, total_alpha)

__kernel void raytrace(
    __global float4* gbuf,
    const int width, const int height,
    const float cam_r, const float cam_theta, const float cam_phi,
    const float cam_fov
) {
    int idx = get_global_id(0);
    if (idx >= width * height) return;

    int px = idx % width;
    int py = idx / width;

    float aspect = (float)width / (float)height;
    float ndc_x = (2.0f * (px + 0.5f) / width - 1.0f) * aspect;
    float ndc_y = (2.0f * (py + 0.5f) / height - 1.0f);
    float half_fov = cam_fov * 0.5f;
    float screen_x = ndc_x * tan(half_fov);
    float screen_y = -ndc_y * tan(half_fov);

    float ray_len = sqrt(screen_x * screen_x + screen_y * screen_y + 1.0f);
    float n_r   =  1.0f / ray_len;
    float n_phi =  screen_x / ray_len;
    float n_th  =  screen_y / ray_len;

    float E, L, p_r_init, p_theta_init;
    camera_ray_to_momenta(cam_r, cam_theta, n_r, n_th, n_phi,
                          &E, &L, &p_r_init, &p_theta_init);

    GeoState state;
    state.r       = cam_r;
    state.theta   = cam_theta;
    state.p_r     = p_r_init;
    state.p_theta = p_theta_init;
    state.phi     = cam_phi;

    float r_horizon = horizon_radius();

    // Disk crossing accumulator
    int num_hits = 0;
    float cross_r[MAX_CROSSINGS];
    float cross_phi[MAX_CROSSINGS];
    float cross_opacity[MAX_CROSSINGS];
    float total_alpha = 0.0f;

    float hit_type = -1.0f;  // Default: maxsteps/NaN
    float bg_theta = 0.0f;
    float bg_phi = 0.0f;

    for (int step = 0; step < MAX_STEPS; step++) {
        float h = 0.04f;
        if (state.r < 10.0f) h = 0.02f;
        if (state.r < 5.0f)  h = 0.01f;
        if (state.r < 3.0f)  h = 0.005f;

        float prev_theta = state.theta;
        float prev_r = state.r;
        float prev_phi = state.phi;

        state = rk4_step(state, E, L, -h);

        // NaN guard
        if (isnan(state.r) || isnan(state.theta) || isinf(state.r)) {
            hit_type = -1.0f;
            break;
        }

        // Theta clamp
        if (state.theta < 0.01f) { state.theta = 0.01f; state.p_theta = fabs(state.p_theta); }
        if (state.theta > PI - 0.01f) { state.theta = PI - 0.01f; state.p_theta = -fabs(state.p_theta); }

        // Equatorial crossing — record disk hit geometry
        float half_pi = PI * 0.5f;
        if ((prev_theta - half_pi) * (state.theta - half_pi) < 0.0f) {
            float t_cross = (half_pi - prev_theta) / (state.theta - prev_theta);
            float cr = prev_r + t_cross * (state.r - prev_r);
            float cp = prev_phi + t_cross * (state.phi - prev_phi);

            if (cr >= DISK_INNER && cr <= DISK_OUTER && num_hits < MAX_CROSSINGS) {
                cross_r[num_hits] = cr;
                cross_phi[num_hits] = cp;
                cross_opacity[num_hits] = 0.85f;
                total_alpha += (1.0f - total_alpha) * 0.85f;
                num_hits++;

                // If disk is nearly opaque, stop tracing
                if (total_alpha > 0.95f) {
                    hit_type = 0.0f;  // Treat as absorbed
                    break;
                }
            }
        }

        // Horizon
        if (state.r <= r_horizon + HORIZON_EPS) {
            hit_type = 0.0f;
            break;
        }

        // Escaped
        if (state.r > ESCAPE_R && state.p_r < 0.0f) {
            hit_type = 1.0f;
            bg_theta = state.theta;
            bg_phi = fmod(state.phi + 100.0f * PI, 2.0f * PI);
            break;
        }
    }

    // Max steps exceeded — ray is likely near photon sphere orbiting.
    // Treat as escaped at current angle rather than black (eliminates dark halo).
    if (hit_type < -0.5f && !isnan(state.theta)) {
        hit_type = 1.0f;
        bg_theta = state.theta;
        bg_phi = fmod(state.phi + 100.0f * PI, 2.0f * PI);
    }

    // Write g-buffer
    int base = idx * 4;
    gbuf[base + 0] = (float4)(hit_type, (float)num_hits, bg_theta, bg_phi);
    gbuf[base + 1] = (float4)(
        num_hits > 0 ? cross_r[0] : 0.0f,
        num_hits > 0 ? cross_phi[0] : 0.0f,
        num_hits > 1 ? cross_r[1] : 0.0f,
        num_hits > 1 ? cross_phi[1] : 0.0f
    );
    gbuf[base + 2] = (float4)(
        num_hits > 2 ? cross_r[2] : 0.0f,
        num_hits > 2 ? cross_phi[2] : 0.0f,
        0.0f, 0.0f
    );
    gbuf[base + 3] = (float4)(
        num_hits > 0 ? cross_opacity[0] : 0.0f,
        num_hits > 1 ? cross_opacity[1] : 0.0f,
        num_hits > 2 ? cross_opacity[2] : 0.0f,
        total_alpha
    );
}

// ============================================================================
// KERNEL 2: shade — Read g-buffer, compute colors (runs every frame, fast)
// ============================================================================

// Display helper functions — native math OK here (no precision needed)

float3 temperature_to_color(float T) {
    float3 color;
    if (T < 0.25f) {
        color = (float3)(0.4f + T * 2.4f, T * 0.4f, T * 0.1f);
    } else if (T < 0.5f) {
        float t = (T - 0.25f) * 4.0f;
        color = (float3)(1.0f, 0.1f + t * 0.5f, t * 0.05f);
    } else if (T < 0.75f) {
        float t = (T - 0.5f) * 4.0f;
        color = (float3)(1.0f, 0.6f + t * 0.3f, 0.05f + t * 0.2f);
    } else {
        float t = (T - 0.75f) * 4.0f;
        color = (float3)(1.0f, 0.9f + t * 0.1f, 0.25f + t * 0.75f);
    }
    return color;
}

float3 compute_disk_emission(float r, float phi, float sim_time) {
    float a = SPIN;
    float sqrt_r = native_sqrt(r);
    float omega_k = 1.0f / (sqrt_r + a);
    float disk_phi = phi - omega_k * sim_time;

    float T_norm = native_powr(DISK_INNER / r, 0.75f);
    float v_orb = omega_k;
    float gamma = 1.0f / native_sqrt(1.0f - v_orb * v_orb);
    float doppler = 1.0f / (gamma * (1.0f + v_orb * native_sin(disk_phi)));

    float T_boosted = clamp(T_norm * doppler, 0.0f, 1.0f);
    float intensity = doppler * doppler * doppler * doppler;
    float brightness = clamp(intensity * 1.5f, 0.0f, 5.0f);

    return temperature_to_color(T_boosted) * brightness;
}

float hash(float2 p) {
    float h = dot(p, (float2)(127.1f, 311.7f));
    float v = native_sin(h) * 43758.5453123f;
    return v - floor(v);
}

float3 compute_starfield(float theta, float phi) {
    float u = phi / (2.0f * PI);
    float v = theta / PI;

    float3 color = (float3)(0.003f, 0.003f, 0.005f);

    // Layer 1: dense dim stars (fine grid)
    // Check current cell + 4 neighbors to avoid cutoff at cell boundaries
    float2 grid1 = (float2)(u * 400.0f, v * 200.0f);
    float2 cell1 = floor(grid1);
    float2 fp1 = grid1 - cell1;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            float2 nc = cell1 + (float2)((float)dx, (float)dy);
            float sc = hash(nc);
            if (sc > 0.94f) {
                float2 sp = (float2)(hash(nc + (float2)(1.0f, 0.0f)),
                                     hash(nc + (float2)(0.0f, 1.0f)));
                float2 delta = fp1 - sp - (float2)((float)dx, (float)dy);
                float dist = length(delta);
                float br = native_exp(-dist * dist * 80.0f) * 0.3f
                         * (0.3f + 0.7f * hash(nc + (float2)(3.0f, 7.0f)));
                float t = hash(nc + (float2)(5.0f, 3.0f));
                color += (float3)(0.7f + 0.3f*t, 0.75f + 0.25f*t, 0.9f - 0.2f*t) * br;
            }
        }
    }

    // Layer 2: sparse bright stars (coarse grid)
    float2 grid2 = (float2)(u * 120.0f, v * 60.0f);
    float2 cell2 = floor(grid2);
    float2 fp2 = grid2 - cell2;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            float2 nc = cell2 + (float2)((float)dx, (float)dy);
            float sc = hash(nc + (float2)(77.0f, 33.0f));
            if (sc > 0.96f) {
                float2 sp = (float2)(hash(nc + (float2)(11.0f, 0.0f)),
                                     hash(nc + (float2)(0.0f, 11.0f)));
                float2 delta = fp2 - sp - (float2)((float)dx, (float)dy);
                float dist = length(delta);
                float br = native_exp(-dist * dist * 40.0f)
                         * (0.5f + 0.5f * hash(nc + (float2)(13.0f, 17.0f)));
                float t = hash(nc + (float2)(15.0f, 13.0f));
                color += (float3)(0.9f + 0.1f*t, 0.85f + 0.15f*t, 1.0f - 0.3f*t) * br;
            }
        }
    }

    return color;
}

__kernel void shade(
    __global const float4* gbuf,
    __global float4* pixels,
    const int width, const int height,
    const float sim_time,
    const int apply_tonemap  // 1 = output sRGB (fast path), 0 = output HDR (bloom path)
) {
    int idx = get_global_id(0);
    if (idx >= width * height) return;

    int base = idx * 4;
    float4 info = gbuf[base + 0];
    float hit_type = info.x;
    int num_hits = (int)info.y;
    float bg_theta = info.z;
    float bg_phi = info.w;

    float4 cross_01 = gbuf[base + 1];
    float4 cross_2  = gbuf[base + 2];
    float4 opacities = gbuf[base + 3];

    // Composite disk crossings with animated rotation
    float3 color = (float3)(0.0f);
    float alpha = 0.0f;

    // Crossing 0
    if (num_hits > 0) {
        float3 dcol = compute_disk_emission(cross_01.x, cross_01.y, sim_time);
        float op = opacities.x;
        color += (1.0f - alpha) * op * dcol;
        alpha += (1.0f - alpha) * op;
    }

    // Crossing 1
    if (num_hits > 1) {
        float3 dcol = compute_disk_emission(cross_01.z, cross_01.w, sim_time);
        float op = opacities.y;
        color += (1.0f - alpha) * op * dcol;
        alpha += (1.0f - alpha) * op;
    }

    // Crossing 2
    if (num_hits > 2) {
        float3 dcol = compute_disk_emission(cross_2.x, cross_2.y, sim_time);
        float op = opacities.z;
        color += (1.0f - alpha) * op * dcol;
        alpha += (1.0f - alpha) * op;
    }

    // Background: star field for escaped rays, black for horizon/NaN
    if (hit_type > 0.5f && alpha < 1.0f) {
        float3 bg = compute_starfield(bg_theta, bg_phi);
        color += (1.0f - alpha) * bg;
    }

    // Optional: apply Reinhard tone mapping + sRGB gamma on GPU
    // Eliminates CPU per-pixel post-processing on the fast path
    if (apply_tonemap) {
        // Reinhard
        color.x = color.x / (1.0f + color.x);
        color.y = color.y / (1.0f + color.y);
        color.z = color.z / (1.0f + color.z);
        // sRGB gamma (native_powr is fine for display math)
        color.x = (color.x < 0.0031308f) ? (12.92f * color.x) : (1.055f * native_powr(color.x, 0.41666667f) - 0.055f);
        color.y = (color.y < 0.0031308f) ? (12.92f * color.y) : (1.055f * native_powr(color.y, 0.41666667f) - 0.055f);
        color.z = (color.z < 0.0031308f) ? (12.92f * color.z) : (1.055f * native_powr(color.z, 0.41666667f) - 0.055f);
    }

    pixels[idx] = (float4)(color, 1.0f);
}
