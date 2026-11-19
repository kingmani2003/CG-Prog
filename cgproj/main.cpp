// =====================================================================
//  INTERACTIVE 3D CAR DESIGN STUDIO  -  modern OpenGL 3.3 edition
//
//  Libraries : GLFW (window + input), GLEW (GL loader), C++17 STL
//              (own tiny vector / matrix maths, no other dependency)
//
//  Models    : 1 Luxury Sedan 
//              2 Luxury SUV   
//              3 Sports Car   
//
//  Rendering : GLSL shaders, per-pixel lighting, procedural studio
//              environment reflections (clear-coat paint, glass, chrome),
//              fake ambient occlusion, soft contact shadow, planar floor
//              reflection, ACES tone mapping, 4x MSAA, wireframe mode.
//
//  Controls  : see printHelp() or press H
// =====================================================================
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

static const float PI_F = 3.14159265358979f;

// =====================================================================
//  MATHS
// =====================================================================
struct V3 {
    float x, y, z;
    V3() : x(0), y(0), z(0) {}
    V3(float a, float b, float c) : x(a), y(b), z(c) {}
};
static inline V3 operator+(V3 a, V3 b) { return V3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline V3 operator-(V3 a, V3 b) { return V3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline V3 operator*(V3 a, float s) { return V3(a.x * s, a.y * s, a.z * s); }
static inline float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 cross(V3 a, V3 b) {
    return V3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline float vlen(V3 a) { return std::sqrt(dot(a, a)); }
static inline V3 norm(V3 a) {
    float l = vlen(a);
    return l > 1e-12f ? a * (1.0f / l) : V3(0, 1, 0);
}
static inline float mixf(float a, float b, float t) { return a + (b - a) * t; }
static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline float sgnpow(float v, float e) {
    float a = std::pow(std::fabs(v), e);
    return v < 0 ? -a : a;
}

// column-major 4x4 matrix (OpenGL layout)
struct M4 { float m[16]; };

static M4 m4identity() {
    M4 r;
    for (int i = 0; i < 16; i++) r.m[i] = 0;
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1;
    return r;
}
static M4 m4mul(const M4& a, const M4& b) {
    M4 r;
    for (int c = 0; c < 4; c++)
        for (int row = 0; row < 4; row++) {
            float s = 0;
            for (int k = 0; k < 4; k++) s += a.m[k * 4 + row] * b.m[c * 4 + k];
            r.m[c * 4 + row] = s;
        }
    return r;
}
static M4 m4translate(V3 t) {
    M4 r = m4identity();
    r.m[12] = t.x; r.m[13] = t.y; r.m[14] = t.z;
    return r;
}
static M4 m4rotY(float a) {
    M4 r = m4identity();
    float c = std::cos(a), s = std::sin(a);
    r.m[0] = c; r.m[2] = -s; r.m[8] = s; r.m[10] = c;
    return r;
}
static M4 m4rotZ(float a) {
    M4 r = m4identity();
    float c = std::cos(a), s = std::sin(a);
    r.m[0] = c; r.m[1] = s; r.m[4] = -s; r.m[5] = c;
    return r;
}
static M4 m4perspective(float fovy, float aspect, float zn, float zf) {
    M4 r;
    for (int i = 0; i < 16; i++) r.m[i] = 0;
    float f = 1.0f / std::tan(fovy * 0.5f);
    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = (zf + zn) / (zn - zf);
    r.m[11] = -1.0f;
    r.m[14] = 2.0f * zf * zn / (zn - zf);
    return r;
}
static M4 m4lookAt(V3 eye, V3 center, V3 up) {
    V3 f = norm(center - eye);
    V3 s = norm(cross(f, up));
    V3 u = cross(s, f);
    M4 r = m4identity();
    r.m[0] = s.x; r.m[4] = s.y; r.m[8] = s.z;
    r.m[1] = u.x; r.m[5] = u.y; r.m[9] = u.z;
    r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
    r.m[12] = -dot(s, eye);
    r.m[13] = -dot(u, eye);
    r.m[14] = dot(f, eye);
    return r;
}

// =====================================================================
//  MESH BUILDING
// =====================================================================
enum MatId {
    MAT_PAINT = 0, MAT_GLASS, MAT_CHROME, MAT_TRIM, MAT_HEAD,
    MAT_TAIL, MAT_RIM, MAT_TYRE, MAT_ACCENT
};

struct Vertex { V3 p, n, c; float mat, ao; };

struct Mesh {
    std::vector<Vertex> v;
    std::vector<unsigned int> idx;
};

struct Surf {
    V3 col; int mat; float ao;
    Surf() : col(1, 1, 1), mat(0), ao(1) {}
    Surf(V3 c, int m, float a = 1.0f) : col(c), mat(m), ao(a) {}
};
typedef std::function<Surf(int, int)> SurfFn;

static Surf sChrome() { return Surf(V3(1, 1, 1), MAT_CHROME); }
static Surf sTrim(float g = 0.02f, float ao = 1.0f) { return Surf(V3(g, g, g), MAT_TRIM, ao); }
static Surf sPaint() { return Surf(V3(1, 1, 1), MAT_PAINT); }
static Surf sAccent(V3 c) { return Surf(c, MAT_ACCENT); }

// A grid of points  P[i*nr + j]  (i = station, j = ring) becomes a smooth
// shaded surface.  Winding / normals are made to point away from `ref`
// (or from the centroid) automatically.
static void addGrid(Mesh& m, const std::vector<V3>& P, int ns, int nr, bool periodic,
                    const SurfFn& fn, const std::vector<float>* vao = nullptr,
                    bool useRef = false, V3 ref = V3(), bool invert = false)
{
    int nq = periodic ? nr : nr - 1;
    V3 c = ref;
    if (!useRef) {
        c = V3();
        for (size_t k = 0; k < P.size(); k++) c = c + P[k];
        c = c * (1.0f / (float)P.size());
    }
    auto G = [&](int i, int j) { return i * nr + j; };

    double acc = 0.0;
    for (int i = 0; i < ns - 1; i++)
        for (int j = 0; j < nq; j++) {
            int j2 = (j + 1) % nr;
            V3 a = P[G(i, j)], b = P[G(i + 1, j)], cc = P[G(i + 1, j2)], d = P[G(i, j2)];
            V3 n1 = cross(b - a, cc - a), n2 = cross(cc - a, d - a);
            acc += dot(n1, (a + b + cc) * (1.0f / 3.0f) - c);
            acc += dot(n2, (a + cc + d) * (1.0f / 3.0f) - c);
        }
    bool flip = (acc < 0.0);
    if (invert) flip = !flip;

    std::vector<V3> N((size_t)ns * nr);
    for (int i = 0; i < ns - 1; i++)
        for (int j = 0; j < nq; j++) {
            int j2 = (j + 1) % nr;
            V3 a = P[G(i, j)], b = P[G(i + 1, j)], cc = P[G(i + 1, j2)], d = P[G(i, j2)];
            V3 n1 = cross(b - a, cc - a), n2 = cross(cc - a, d - a);
            if (flip) { n1 = n1 * -1.0f; n2 = n2 * -1.0f; }
            N[G(i, j)] = N[G(i, j)] + n1 + n2;
            N[G(i + 1, j)] = N[G(i + 1, j)] + n1;
            N[G(i + 1, j2)] = N[G(i + 1, j2)] + n1 + n2;
            N[G(i, j2)] = N[G(i, j2)] + n2;
        }
    for (int k = 0; k < ns * nr; k++) {
        if (vlen(N[k]) < 1e-9f) {
            V3 fb = norm(P[k] - c);
            if (invert) fb = fb * -1.0f;
            N[k] = fb;
        } else N[k] = norm(N[k]);
    }

    for (int i = 0; i < ns - 1; i++)
        for (int j = 0; j < nq; j++) {
            int j2 = (j + 1) % nr;
            int id[4] = { G(i, j), G(i + 1, j), G(i + 1, j2), G(i, j2) };
            Surf s = fn(i, j);
            unsigned int base = (unsigned int)m.v.size();
            for (int q = 0; q < 4; q++) {
                Vertex vt;
                vt.p = P[id[q]]; vt.n = N[id[q]]; vt.c = s.col;
                vt.mat = (float)s.mat;
                vt.ao = s.ao * (vao ? (*vao)[id[q]] : 1.0f);
                m.v.push_back(vt);
            }
            if (!flip) {
                m.idx.push_back(base); m.idx.push_back(base + 1); m.idx.push_back(base + 2);
                m.idx.push_back(base); m.idx.push_back(base + 2); m.idx.push_back(base + 3);
            } else {
                m.idx.push_back(base); m.idx.push_back(base + 2); m.idx.push_back(base + 1);
                m.idx.push_back(base); m.idx.push_back(base + 3); m.idx.push_back(base + 2);
            }
        }
}

static void addQuad(Mesh& m, V3 a, V3 b, V3 c, V3 d, V3 n, Surf s) {
    unsigned int base = (unsigned int)m.v.size();
    V3 pts[4] = { a, b, c, d };
    for (int q = 0; q < 4; q++) {
        Vertex vt; vt.p = pts[q]; vt.n = n; vt.c = s.col; vt.mat = (float)s.mat; vt.ao = s.ao;
        m.v.push_back(vt);
    }
    V3 g = cross(b - a, c - a);
    if (dot(g, n) >= 0) {
        m.idx.push_back(base); m.idx.push_back(base + 1); m.idx.push_back(base + 2);
        m.idx.push_back(base); m.idx.push_back(base + 2); m.idx.push_back(base + 3);
    } else {
        m.idx.push_back(base); m.idx.push_back(base + 2); m.idx.push_back(base + 1);
        m.idx.push_back(base); m.idx.push_back(base + 3); m.idx.push_back(base + 2);
    }
}

// box given by centre and three half-extent vectors
static void addBox(Mesh& m, V3 c, V3 ax, V3 ay, V3 az, Surf s) {
    V3 e[3] = { ax, ay, az };
    for (int k = 0; k < 3; k++)
        for (int sg = -1; sg <= 1; sg += 2) {
            V3 n = norm(e[k]) * (float)sg;
            V3 fc = c + e[k] * (float)sg;
            V3 u = e[(k + 1) % 3], v = e[(k + 2) % 3];
            addQuad(m, fc - u - v, fc + u - v, fc + u + v, fc - u + v, n, s);
        }
}
static void addBoxAA(Mesh& m, V3 c, V3 half, Surf s) {
    addBox(m, c, V3(half.x, 0, 0), V3(0, half.y, 0), V3(0, 0, half.z), s);
}

static void addEllipsoid(Mesh& m, V3 c, V3 r, Surf s, int nseg = 18, int nrng = 10) {
    int ns = nrng + 1, nr = nseg;
    std::vector<V3> P((size_t)ns * nr);
    for (int i = 0; i < ns; i++) {
        float phi = PI_F * i / nrng;
        float rad = std::sin(phi);
        for (int j = 0; j < nr; j++) {
            float a = 2.0f * PI_F * j / nr;
            P[i * nr + j] = V3(c.x + r.x * std::cos(phi),
                               c.y + r.y * rad * std::cos(a),
                               c.z + r.z * rad * std::sin(a));
        }
    }
    addGrid(m, P, ns, nr, true, [=](int, int) { return s; });
}

static void addTube(Mesh& m, const std::vector<V3>& path, float r, Surf s, int seg = 8) {
    int ns = (int)path.size();
    if (ns < 2) return;
    std::vector<V3> P((size_t)ns * seg);
    for (int i = 0; i < ns; i++) {
        V3 T = norm(path[std::min(i + 1, ns - 1)] - path[std::max(i - 1, 0)]);
        V3 rf = std::fabs(T.y) < 0.9f ? V3(0, 1, 0) : V3(0, 0, 1);
        V3 N1 = norm(cross(T, rf));
        V3 N2 = cross(T, N1);
        for (int j = 0; j < seg; j++) {
            float a = 2.0f * PI_F * j / seg;
            P[i * seg + j] = path[i] + (N1 * std::cos(a) + N2 * std::sin(a)) * r;
        }
    }
    addGrid(m, P, ns, seg, true, [=](int, int) { return s; });
}

// surface of revolution about the local z axis.  profile = (radius, z)
static void addLathe(Mesh& m, const std::vector<std::pair<float, float> >& prof, int nseg,
                     float a0, float a1, V3 off, Surf s,
                     bool useRef = false, V3 ref = V3(), bool invert = false)
{
    int ns = (int)prof.size();
    bool full = std::fabs((a1 - a0) - 2.0f * PI_F) < 1e-4f;
    int nr = full ? nseg : nseg + 1;
    std::vector<V3> P((size_t)ns * nr);
    for (int i = 0; i < ns; i++)
        for (int j = 0; j < nr; j++) {
            float a = a0 + (a1 - a0) * j / nseg;
            P[i * nr + j] = off + V3(prof[i].first * std::cos(a), prof[i].first * std::sin(a), prof[i].second);
        }
    addGrid(m, P, ns, nr, full, [=](int, int) { return s; }, nullptr, useRef, ref, invert);
}

// =====================================================================
//  CAR DEFINITION DATA
// =====================================================================
struct StationRow { float x, yb, yt, hw; };          // hull section
enum GlassType { GT_FRONT = 0, GT_SIDE, GT_PILLAR, GT_REAR, GT_END };
struct CabinRow { float x, yTop, w; int type; };      // greenhouse section

struct ModelDef {
    const char* name;
    std::vector<StationRow> hull; float hullN, endRound;
    std::vector<CabinRow> cabin; float cabinN, lean, thPillar, thSide;
    float wheelR, wheelW, rimR, archGap, axleF, axleR;
    int spokes; bool twin; V3 rimColor, caliperColor;
    std::vector<float> sideCuts, topCuts; float cutY0, cutY1;
    int defaultPaint;
    float halfLen, halfWid;
};

struct PaintDef { const char* name; V3 col; };
static const PaintDef PAINTS[] = {
    { "Pearl White",      V3(0.78f, 0.78f, 0.75f) },
    { "Obsidian Black",   V3(0.012f, 0.012f, 0.015f) },
    { "Midnight Blue",    V3(0.010f, 0.030f, 0.170f) },
    { "Racing Green",     V3(0.004f, 0.075f, 0.030f) },
    { "Candy Red",        V3(0.600f, 0.008f, 0.020f) },
    { "Titanium Silver",  V3(0.380f, 0.400f, 0.430f) },
    { "Sunset Orange",    V3(0.880f, 0.200f, 0.010f) },
    { "Royal Purple",     V3(0.120f, 0.015f, 0.260f) },
    { "Lemon Yellow",     V3(0.850f, 0.620f, 0.000f) },
};
static const int NUM_PAINTS = (int)(sizeof(PAINTS) / sizeof(PAINTS[0]));

struct Theme { const char* name; V3 horizon, top, key, rim, floorLight, floorDark; };
static const Theme THEMES[] = {
    { "Studio",   V3(0.13f, 0.14f, 0.17f), V3(0.015f, 0.020f, 0.035f), V3(1.0f, 0.97f, 0.92f), V3(0.75f, 0.88f, 1.0f),
                  V3(0.20f, 0.21f, 0.24f), V3(0.030f, 0.032f, 0.040f) },
    { "Sunset",   V3(0.75f, 0.28f, 0.09f), V3(0.040f, 0.050f, 0.160f), V3(1.0f, 0.70f, 0.45f), V3(1.0f, 0.45f, 0.15f),
                  V3(0.35f, 0.17f, 0.12f), V3(0.050f, 0.030f, 0.040f) },
    { "Showroom", V3(0.62f, 0.64f, 0.68f), V3(0.300f, 0.340f, 0.420f), V3(1.0f, 1.0f, 1.0f),  V3(1.0f, 1.0f, 1.0f),
                  V3(0.75f, 0.76f, 0.78f), V3(0.300f, 0.310f, 0.340f) },
    { "Neon Night", V3(0.15f, 0.02f, 0.22f), V3(0.000f, 0.005f, 0.030f), V3(0.7f, 0.6f, 1.0f), V3(0.1f, 0.9f, 1.0f),
                  V3(0.12f, 0.04f, 0.20f), V3(0.010f, 0.005f, 0.020f) },
};
static const int NUM_THEMES = (int)(sizeof(THEMES) / sizeof(THEMES[0]));

// ---------------------------------------------------------------------
static ModelDef defSedan() {
    ModelDef d;
    d.name = "Luxury Sedan";
    StationRow h[] = {
        { -2.65f, .46f, .96f, .78f }, { -2.45f, .30f, 1.02f, .93f }, { -2.00f, .24f, 1.04f, .99f },
        { -1.30f, .22f, 1.02f, 1.00f }, { 0.00f, .22f, 1.00f, 1.00f }, { 1.40f, .22f, 1.02f, .99f },
        { 2.00f, .26f, 1.04f, .97f }, { 2.45f, .32f, .96f, .90f }, { 2.65f, .42f, .86f, .78f } };
    d.hull.assign(h, h + sizeof(h) / sizeof(h[0]));
    d.hullN = 3.8f; d.endRound = 0.12f;
    CabinRow c[] = {
        { -2.05f, 1.06f, .80f, GT_REAR }, { -1.30f, 1.42f, .72f, GT_PILLAR }, { -1.18f, 1.45f, .73f, GT_SIDE },
        { -0.50f, 1.48f, .76f, GT_PILLAR }, { -0.40f, 1.485f, .77f, GT_SIDE }, { 0.34f, 1.485f, .77f, GT_PILLAR },
        { 0.44f, 1.47f, .77f, GT_FRONT }, { 1.08f, 1.02f, .80f, GT_END } };
    d.cabin.assign(c, c + sizeof(c) / sizeof(c[0]));
    d.cabinN = 2.8f; d.lean = 0.13f; d.thPillar = 0.40f; d.thSide = 0.85f;
    d.wheelR = 0.37f; d.wheelW = 0.255f; d.rimR = 0.265f; d.archGap = 0.07f;
    d.axleF = 1.65f; d.axleR = -1.70f;
    d.spokes = 5; d.twin = true;
    d.rimColor = V3(0.85f, 0.86f, 0.90f); d.caliperColor = V3(0.55f, 0.56f, 0.6f);
    d.sideCuts = { 1.05f, -0.45f, -1.15f }; d.topCuts = { 1.10f, -1.95f };
    d.cutY0 = 0.30f; d.cutY1 = 1.00f;
    d.defaultPaint = 2;
    d.halfLen = 2.8f; d.halfWid = 1.05f;
    return d;
}

static ModelDef defSUV() {
    ModelDef d;
    d.name = "Luxury SUV";
    StationRow h[] = {
        { -2.38f, .42f, 1.22f, .86f }, { -2.25f, .30f, 1.26f, .97f }, { -1.50f, .28f, 1.24f, 1.00f },
        { 0.00f, .28f, 1.22f, 1.00f }, { 1.40f, .28f, 1.20f, 1.00f }, { 2.00f, .30f, 1.18f, .98f },
        { 2.32f, .34f, 1.08f, .93f }, { 2.42f, .44f, 1.00f, .84f } };
    d.hull.assign(h, h + sizeof(h) / sizeof(h[0]));
    d.hullN = 4.4f; d.endRound = 0.08f;
    CabinRow c[] = {
        { -2.28f, 1.24f, .88f, GT_REAR }, { -2.02f, 1.80f, .85f, GT_PILLAR }, { -1.92f, 1.82f, .85f, GT_SIDE },
        { -1.12f, 1.84f, .86f, GT_PILLAR }, { -1.02f, 1.845f, .86f, GT_SIDE }, { -0.15f, 1.85f, .86f, GT_PILLAR },
        { -0.05f, 1.85f, .86f, GT_SIDE }, { 0.82f, 1.85f, .86f, GT_PILLAR }, { 0.92f, 1.84f, .86f, GT_FRONT },
        { 1.58f, 1.30f, .90f, GT_END } };
    d.cabin.assign(c, c + sizeof(c) / sizeof(c[0]));
    d.cabinN = 3.2f; d.lean = 0.08f; d.thPillar = 0.35f; d.thSide = 0.90f;
    d.wheelR = 0.43f; d.wheelW = 0.29f; d.rimR = 0.30f; d.archGap = 0.07f;
    d.axleF = 1.60f; d.axleR = -1.60f;
    d.spokes = 5; d.twin = true;
    d.rimColor = V3(0.12f, 0.12f, 0.14f); d.caliperColor = V3(0.75f, 0.03f, 0.03f);
    d.sideCuts = { 1.05f, -0.10f, -1.02f }; d.topCuts = { 1.55f, -2.30f };
    d.cutY0 = 0.34f; d.cutY1 = 1.20f;
    d.defaultPaint = 3;
    d.halfLen = 2.6f; d.halfWid = 1.1f;
    return d;
}

static ModelDef defSports() {
    ModelDef d;
    d.name = "Sports Car";
    StationRow h[] = {
        { -2.25f, .28f, .98f, .84f }, { -2.10f, .16f, 1.00f, .96f }, { -1.30f, .13f, .98f, 1.00f },
        { -0.50f, .12f, .92f, 1.00f }, { 0.50f, .12f, .84f, 1.00f }, { 1.25f, .12f, .80f, 1.00f },
        { 1.70f, .13f, .72f, 1.00f }, { 2.15f, .16f, .54f, .93f }, { 2.32f, .20f, .44f, .78f } };
    d.hull.assign(h, h + sizeof(h) / sizeof(h[0]));
    d.hullN = 3.0f; d.endRound = 0.07f;
    CabinRow c[] = {
        { -1.60f, .90f, .78f, GT_REAR }, { -0.62f, 1.10f, .70f, GT_PILLAR }, { -0.50f, 1.12f, .70f, GT_SIDE },
        { 0.12f, 1.13f, .70f, GT_PILLAR }, { 0.24f, 1.13f, .70f, GT_FRONT }, { 1.28f, .84f, .86f, GT_END } };
    d.cabin.assign(c, c + sizeof(c) / sizeof(c[0]));
    d.cabinN = 2.6f; d.lean = 0.20f; d.thPillar = 0.30f; d.thSide = 0.75f;
    d.wheelR = 0.345f; d.wheelW = 0.30f; d.rimR = 0.25f; d.archGap = 0.04f;
    d.axleF = 1.50f; d.axleR = -1.30f;
    d.spokes = 10; d.twin = false;
    d.rimColor = V3(0.05f, 0.05f, 0.06f); d.caliperColor = V3(0.95f, 0.75f, 0.0f);
    d.sideCuts = { 0.55f, -0.45f }; d.topCuts = { 1.75f, -1.20f };
    d.cutY0 = 0.22f; d.cutY1 = 0.82f;
    d.defaultPaint = 6;
    d.halfLen = 2.6f; d.halfWid = 1.1f;
    return d;
}

// =====================================================================
//  HULL  (lofted body shell, super-ellipse cross sections)
// =====================================================================
struct HS { float x, yb, yt, hw; };

struct Hull {
    std::vector<HS> s;
    float n;

    HS at(float x) const {
        if (x <= s.front().x) return s.front();
        if (x >= s.back().x) return s.back();
        size_t lo = 0, hi = s.size() - 1;
        while (hi - lo > 1) { size_t mid = (lo + hi) / 2; if (s[mid].x <= x) lo = mid; else hi = mid; }
        float t = (x - s[lo].x) / std::max(1e-6f, s[hi].x - s[lo].x);
        HS r; r.x = x;
        r.yb = mixf(s[lo].yb, s[hi].yb, t); r.yt = mixf(s[lo].yt, s[hi].yt, t); r.hw = mixf(s[lo].hw, s[hi].hw, t);
        return r;
    }
    bool contains(float x, float y, float z) const {
        if (x < s.front().x || x > s.back().x) return false;
        HS h = at(x);
        float hh = (h.yt - h.yb) * 0.5f, yc = (h.yt + h.yb) * 0.5f;
        if (hh <= 1e-4f || h.hw <= 1e-4f) return false;
        return std::pow(std::fabs(z) / h.hw, n) + std::pow(std::fabs(y - yc) / hh, n) <= 1.0f;
    }
    float sideZ(float x, float y) const {          // half width of the body at height y
        HS h = at(x);
        float hh = (h.yt - h.yb) * 0.5f, yc = (h.yt + h.yb) * 0.5f;
        float dy = std::fabs(y - yc) / std::max(hh, 1e-4f);
        if (dy >= 1.0f) return 0.0f;
        return h.hw * std::pow(1.0f - std::pow(dy, n), 1.0f / n);
    }
    float topY(float x, float z) const {           // height of the upper surface at lateral z
        HS h = at(x);
        float hh = (h.yt - h.yb) * 0.5f, yc = (h.yt + h.yb) * 0.5f;
        float q = std::fabs(z) / std::max(h.hw, 1e-4f);
        if (q >= 1.0f) return yc;
        return yc + hh * std::pow(1.0f - std::pow(q, n), 1.0f / n);
    }
    float frontX(float y, float z) const {
        for (float x = s.back().x; x > s.front().x; x -= 0.004f) if (contains(x, y, z)) return x;
        return 0.0f;
    }
    float rearX(float y, float z) const {
        for (float x = s.front().x; x < s.back().x; x += 0.004f) if (contains(x, y, z)) return x;
        return 0.0f;
    }
};

static Hull makeHull(const ModelDef& d) {
    Hull H; H.n = d.hullN;
    const std::vector<StationRow>& R = d.hull;
    float x0 = R.front().x, x1 = R.back().x;
    int N = (int)std::ceil((x1 - x0) / 0.05f);
    std::vector<HS> b(N + 1);
    for (int i = 0; i <= N; i++) {
        float x = x0 + (x1 - x0) * i / N;
        size_t k = 0;
        while (k + 2 < R.size() && R[k + 1].x < x) k++;
        float t = clampf((x - R[k].x) / (R[k + 1].x - R[k].x), 0.0f, 1.0f);
        b[i].x = x;
        b[i].yb = mixf(R[k].yb, R[k + 1].yb, t);
        b[i].yt = mixf(R[k].yt, R[k + 1].yt, t);
        b[i].hw = mixf(R[k].hw, R[k + 1].hw, t);
    }
    for (int pass = 0; pass < 2; pass++) {
        std::vector<HS> c = b;
        for (int i = 1; i < N; i++) {
            b[i].yb = (c[i - 1].yb + 2 * c[i].yb + c[i + 1].yb) * 0.25f;
            b[i].yt = (c[i - 1].yt + 2 * c[i].yt + c[i + 1].yt) * 0.25f;
            b[i].hw = (c[i - 1].hw + 2 * c[i].hw + c[i + 1].hw) * 0.25f;
        }
    }
    const int K = 8;
    std::vector<HS> rear, front;
    for (int e = 0; e < 2; e++) {
        HS end = (e == 0) ? b.front() : b.back();
        float yc = (end.yb + end.yt) * 0.5f, hh = (end.yt - end.yb) * 0.5f;
        for (int k = 1; k <= K; k++) {
            float ang = (PI_F * 0.5f) * k / K;
            float sc = (k == K) ? 0.0f : std::cos(ang);
            HS q;
            q.x = end.x + (e == 0 ? -1.0f : 1.0f) * d.endRound * std::sin(ang);
            q.yb = yc - hh * sc; q.yt = yc + hh * sc; q.hw = end.hw * sc;
            (e == 0 ? rear : front).push_back(q);
        }
    }
    for (int k = K - 1; k >= 0; k--) H.s.push_back(rear[k]);
    for (size_t i = 0; i < b.size(); i++) H.s.push_back(b[i]);
    for (int k = 0; k < K; k++) H.s.push_back(front[k]);
    return H;
}

struct ArchDef { float cx, cy, R, zin; };

static void buildHullMesh(Mesh& m, const Hull& H, const std::vector<ArchDef>& arches) {
    const int nr = 64;
    int ns = (int)H.s.size();
    std::vector<V3> P((size_t)ns * nr);
    std::vector<float> ao((size_t)ns * nr, 1.0f);
    float e = 2.0f / H.n;
    for (int i = 0; i < ns; i++) {
        const HS& st = H.s[i];
        float yc = (st.yb + st.yt) * 0.5f, hh = (st.yt - st.yb) * 0.5f;
        for (int j = 0; j < nr; j++) {
            float th = 2.0f * PI_F * j / nr;
            float c = std::cos(th), sn = std::sin(th);
            P[i * nr + j] = V3(st.x, yc + hh * sgnpow(sn, e), st.hw * sgnpow(c, e));
            if (sn < -0.55f) ao[i * nr + j] = 0.6f;
        }
    }
    for (size_t a = 0; a < arches.size(); a++) {
        const ArchDef& A = arches[a];
        for (int k = 0; k < ns * nr; k++) {
            V3& p = P[k];
            if (std::fabs(p.z) <= A.zin) continue;
            float dx = p.x - A.cx;
            if (std::fabs(dx) >= A.R) continue;
            float yArc = A.cy + std::sqrt(A.R * A.R - dx * dx);
            if (p.y < yArc) { p.y = yArc; ao[k] = std::min(ao[k], 0.32f); }
            else if (p.y < yArc + 0.06f) ao[k] = std::min(ao[k], 0.32f + 0.68f * (p.y - yArc) / 0.06f);
        }
    }
    SurfFn fn = [nr](int, int j) -> Surf {
        float th = 2.0f * PI_F * (j + 0.5f) / nr;
        if (std::sin(th) < -0.8f) return Surf(V3(0.015f, 0.015f, 0.017f), MAT_TRIM);
        return sPaint();
    };
    addGrid(m, P, ns, nr, true, fn, &ao);
}

// =====================================================================
//  CABIN  (windscreen, windows, roof, pillars)
// =====================================================================
struct CabinSample { float x, base, h, w; int type; };

static std::vector<CabinSample> makeCabin(const ModelDef& d, const Hull& H) {
    const std::vector<CabinRow>& R = d.cabin;
    float x0 = R.front().x, x1 = R.back().x;
    int N = (int)std::ceil((x1 - x0) / 0.03f);
    std::vector<float> yt(N + 1), w(N + 1), xs(N + 1);
    std::vector<int> ty(N + 1);
    for (int i = 0; i <= N; i++) {
        float x = x0 + (x1 - x0) * i / N;
        size_t k = 0;
        while (k + 2 < R.size() && R[k + 1].x <= x) k++;
        float t = clampf((x - R[k].x) / (R[k + 1].x - R[k].x), 0.0f, 1.0f);
        xs[i] = x; yt[i] = mixf(R[k].yTop, R[k + 1].yTop, t); w[i] = mixf(R[k].w, R[k + 1].w, t);
        ty[i] = R[k].type;
    }
    for (int pass = 0; pass < 3; pass++) {
        std::vector<float> a = yt, b = w;
        for (int i = 1; i < N; i++) {
            yt[i] = (a[i - 1] + 2 * a[i] + a[i + 1]) * 0.25f;
            w[i] = (b[i - 1] + 2 * b[i] + b[i + 1]) * 0.25f;
        }
    }
    std::vector<CabinSample> out(N + 1);
    for (int i = 0; i <= N; i++) {
        CabinSample& c = out[i];
        c.x = xs[i]; c.w = w[i]; c.type = ty[i];
        c.base = H.topY(c.x, c.w) - 0.02f;
        c.h = std::max(yt[i] - c.base, -0.03f);
    }
    return out;
}

static V3 cabinPoint(const ModelDef& d, const CabinSample& c, float th) {
    float e = 2.0f / d.cabinN;
    float s = std::sin(th), co = std::cos(th);
    float yf = std::pow(std::max(s, 0.0f), e);
    float z = c.w * sgnpow(co, e) * (1.0f - d.lean * yf);
    return V3(c.x, c.base + c.h * yf, z);
}

static void buildCabinMesh(Mesh& m, const ModelDef& d, const std::vector<CabinSample>& S) {
    const int nr = 25;
    int ns = (int)S.size();
    std::vector<V3> P((size_t)ns * nr);
    float avgBase = 0;
    for (int i = 0; i < ns; i++) {
        avgBase += S[i].base / ns;
        for (int j = 0; j < nr; j++) P[i * nr + j] = cabinPoint(d, S[i], PI_F * j / (nr - 1));
    }
    float xm = (S.front().x + S.back().x) * 0.5f;
    float thP = d.thPillar, thS = d.thSide;
    SurfFn fn = [&, nr, thP, thS](int i, int j) -> Surf {
        float th = PI_F * (j + 0.5f) / (nr - 1);
        int t = S[i].type;
        Surf glass(V3(0.02f, 0.03f, 0.04f), MAT_GLASS);
        if (t == GT_PILLAR) return sPaint();
        if (t == GT_FRONT || t == GT_REAR) return (th < thP || th > PI_F - thP) ? sPaint() : glass;
        return (th < thS || th > PI_F - thS) ? glass : sPaint();
    };
    addGrid(m, P, ns, nr, false, fn, nullptr, true, V3(xm, avgBase - 0.2f, 0));
}

static float cabinTopAt(const ModelDef& d, const std::vector<CabinSample>& S, float x, float z) {
    size_t best = 0; float bd = 1e9f;
    for (size_t i = 0; i < S.size(); i++) { float dd = std::fabs(S[i].x - x); if (dd < bd) { bd = dd; best = i; } }
    float az = std::fabs(z), y = S[best].base;
    for (float th = PI_F * 0.5f; th >= 0.0f; th -= 0.01f) {
        V3 p = cabinPoint(d, S[best], th);
        y = p.y;
        if (p.z >= az) break;
    }
    return y;
}

// =====================================================================
//  WHEEL
// =====================================================================
typedef std::vector<std::pair<float, float> > Prof;

static Mesh buildWheel(const ModelDef& d) {
    Mesh m;
    float R = d.wheelR, tw = d.wheelW, rr = d.rimR;
    const int SEG = 56;
    float tyreCol = 0.018f;
    // tyre
    Prof t;
    t.push_back(std::make_pair(rr * 0.97f, -0.50f * tw)); t.push_back(std::make_pair(R * 0.80f, -0.50f * tw));
    t.push_back(std::make_pair(R * 0.93f, -0.46f * tw));  t.push_back(std::make_pair(R * 0.985f, -0.36f * tw));
    t.push_back(std::make_pair(R, -0.22f * tw));          t.push_back(std::make_pair(R, 0.22f * tw));
    t.push_back(std::make_pair(R * 0.985f, 0.36f * tw));  t.push_back(std::make_pair(R * 0.93f, 0.46f * tw));
    t.push_back(std::make_pair(R * 0.80f, 0.50f * tw));   t.push_back(std::make_pair(rr * 0.97f, 0.50f * tw));
    addLathe(m, t, SEG, 0, 2 * PI_F, V3(), Surf(V3(tyreCol, tyreCol, tyreCol), MAT_TYRE));
    // rim barrel + lip
    Prof b;
    b.push_back(std::make_pair(rr * 0.90f, -0.45f * tw)); b.push_back(std::make_pair(rr * 1.0f, -0.40f * tw));
    b.push_back(std::make_pair(rr * 1.0f, 0.38f * tw));   b.push_back(std::make_pair(rr * 1.035f, 0.43f * tw));
    b.push_back(std::make_pair(rr * 1.0f, 0.48f * tw));   b.push_back(std::make_pair(rr * 0.93f, 0.46f * tw));
    addLathe(m, b, SEG, 0, 2 * PI_F, V3(), Surf(d.rimColor, MAT_RIM), true, V3(0, 0, 0));
    // dark disc behind spokes, brake disc, caliper
    Prof disc; disc.push_back(std::make_pair(0.0f, 0.20f * tw)); disc.push_back(std::make_pair(rr * 0.95f, 0.20f * tw));
    addLathe(m, disc, SEG, 0, 2 * PI_F, V3(), sTrim(0.01f, 0.5f), true, V3(0, 0, -1));
    Prof br; br.push_back(std::make_pair(rr * 0.30f, 0.26f * tw)); br.push_back(std::make_pair(rr * 0.80f, 0.26f * tw));
    addLathe(m, br, SEG, 0, 2 * PI_F, V3(), Surf(V3(0.35f, 0.35f, 0.37f), MAT_RIM), true, V3(0, 0, -1));
    {
        float ang = 0.9f;
        V3 u(std::cos(ang), std::sin(ang), 0), v(-std::sin(ang), std::cos(ang), 0);
        addBox(m, u * (rr * 0.70f) + V3(0, 0, 0.30f * tw), u * (rr * 0.14f), v * (rr * 0.20f), V3(0, 0, 0.07f * tw), sAccent(d.caliperColor));
    }
    // hub cap
    Prof hub;
    hub.push_back(std::make_pair(0.0f, 0.47f * tw)); hub.push_back(std::make_pair(rr * 0.16f, 0.47f * tw));
    hub.push_back(std::make_pair(rr * 0.19f, 0.43f * tw)); hub.push_back(std::make_pair(rr * 0.19f, 0.30f * tw));
    addLathe(m, hub, 24, 0, 2 * PI_F, V3(), sChrome(), true, V3(0, 0, -0.5f));
    // spokes
    int n = d.spokes;
    for (int k = 0; k < n; k++) {
        float a = 2.0f * PI_F * k / n;
        int copies = d.twin ? 2 : 1;
        for (int c = 0; c < copies; c++) {
            float aa = a + (d.twin ? (c == 0 ? -0.10f : 0.10f) : 0.0f);
            V3 u(std::cos(aa), std::sin(aa), 0), v(-std::sin(aa), std::cos(aa), 0);
            float r0 = rr * 0.17f, r1 = rr * 0.96f;
            float hwid = rr * (d.twin ? 0.045f : (n > 6 ? 0.045f : 0.08f));
            addBox(m, u * ((r0 + r1) * 0.5f) + V3(0, 0, 0.40f * tw), u * ((r1 - r0) * 0.5f), v * hwid,
                   V3(0, 0, 0.045f * tw), Surf(d.rimColor, MAT_RIM));
        }
    }
    return m;
}

// =====================================================================
//  CAR ASSEMBLY  (body mesh = hull + cabin + all detail parts)
// =====================================================================
struct Ctx {
    Mesh* m; const Hull* H; const std::vector<CabinSample>* cab; const ModelDef* d;
};

static void addArchLiner(Mesh& m, float x, float cy, float Rr, float zin, float zout, int side) {
    float z0 = side * zin, z1 = side * zout;
    Prof p; p.push_back(std::make_pair(Rr, z0)); p.push_back(std::make_pair(Rr, z1));
    addLathe(m, p, 20, 0, PI_F, V3(x, cy, 0), sTrim(0.01f, 0.3f), true, V3(x, cy, (z0 + z1) * 0.5f), true);
    Prof w; w.push_back(std::make_pair(0.0f, z0)); w.push_back(std::make_pair(Rr, z0));
    addLathe(m, w, 20, 0, PI_F, V3(x, cy, 0), sTrim(0.01f, 0.3f), true, V3(x, cy, z0 - side * 1.0f));
}

// ---- generic detail helpers -----------------------------------------
static void addGrille(Ctx& c, float yc, float hh, float zhalf, int count, bool vertical, float off) {
    const Hull& H = *c.H;
    float pitch = 2.0f * (vertical ? zhalf : hh) / (count - 1);
    for (int k = 0; k < count; k++) {
        std::vector<V3> pb, pc;
        if (vertical) {
            float z = -zhalf + pitch * k;
            for (int i = 0; i <= 8; i++) {
                float y = yc - hh + 2 * hh * i / 8.0f;
                float x = H.frontX(y, z);
                pb.push_back(V3(x + off, y, z)); pc.push_back(V3(x + off + 0.012f, y, z));
            }
        } else {
            float y = yc - hh + pitch * k;
            for (int i = 0; i <= 12; i++) {
                float z = -zhalf + 2 * zhalf * i / 12.0f;
                float x = H.frontX(y, z);
                pb.push_back(V3(x + off, y, z)); pc.push_back(V3(x + off + 0.012f, y, z));
            }
        }
        addTube(*c.m, pb, pitch * 0.58f, sTrim(0.006f), 6);
        addTube(*c.m, pc, 0.0075f, sChrome(), 6);
    }
    // chrome frame
    std::vector<V3> f;
    float zh = zhalf + pitch * 0.6f, y0 = yc - hh - 0.012f, y1 = yc + hh + 0.012f;
    for (int i = 0; i <= 10; i++) { float z = -zh + 2 * zh * i / 10.0f; f.push_back(V3(H.frontX(y1, z) + off + 0.012f, y1, z)); }
    for (int i = 1; i <= 8; i++) { float y = y1 - (y1 - y0) * i / 8.0f; f.push_back(V3(H.frontX(y, zh) + off + 0.012f, y, zh)); }
    for (int i = 1; i <= 10; i++) { float z = zh - 2 * zh * i / 10.0f; f.push_back(V3(H.frontX(y0, z) + off + 0.012f, y0, z)); }
    for (int i = 1; i <= 8; i++) { float y = y0 + (y1 - y0) * i / 8.0f; f.push_back(V3(H.frontX(y, -zh) + off + 0.012f, y, -zh)); }
    addTube(*c.m, f, 0.011f, sChrome(), 8);
}

static void addLamp(Ctx& c, bool front, float y, float z, V3 r, int mat, float inset) {
    const Hull& H = *c.H;
    float x = front ? H.frontX(y, std::fabs(z)) : H.rearX(y, std::fabs(z));
    float sg = front ? 1.0f : -1.0f;
    addEllipsoid(*c.m, V3(x - sg * inset, y, z), r, Surf(V3(1, 1, 1), mat), 18, 10);
}

static void addPlate(Ctx& c, bool front, float y, float hh, float hw) {
    const Hull& H = *c.H;
    float x = front ? H.frontX(y, 0) : H.rearX(y, 0);
    float sg = front ? 1.0f : -1.0f;
    addBoxAA(*c.m, V3(x + sg * 0.004f, y, 0), V3(0.008f, hh, hw), sAccent(V3(0.85f, 0.85f, 0.8f)));
}

static void addMirror(Ctx& c, float x, float y, float z, float sc) {
    int side = z > 0 ? 1 : -1;
    addEllipsoid(*c.m, V3(x, y, z), V3(0.11f * sc, 0.065f * sc, 0.075f * sc), sPaint(), 14, 8);
    std::vector<V3> st;
    st.push_back(V3(x + 0.03f, y - 0.04f, z - side * 0.14f)); st.push_back(V3(x + 0.02f, y - 0.02f, z - side * 0.05f));
    addTube(*c.m, st, 0.012f, sTrim(0.01f), 6);
}

static void addHandle(Ctx& c, float x, float y, float len, int side) {
    std::vector<V3> p;
    for (int i = 0; i <= 3; i++) {
        float xx = x - len * i / 3.0f;
        p.push_back(V3(xx, y, side * (c.H->sideZ(xx, y) + 0.008f)));
    }
    addTube(*c.m, p, 0.009f, sChrome(), 6);
}

static void addSideLine(Ctx& c, float x0, float x1, float y, float r, Surf s, int side) {
    std::vector<V3> p;
    for (float x = x0; x <= x1 + 1e-4f; x += 0.08f) p.push_back(V3(x, y, side * (c.H->sideZ(x, y) + 0.003f)));
    addTube(*c.m, p, r, s, 6);
}

static void addExhaust(Ctx& c, float y, float z, float r) {
    float x = c.H->rearX(y, std::fabs(z));
    Prof p; p.push_back(std::make_pair(r, 0.0f)); p.push_back(std::make_pair(r, 0.10f));
    // axis along x: rotate by building points manually
    std::vector<V3> path; path.push_back(V3(x + 0.02f, y, z)); path.push_back(V3(x - 0.07f, y, z));
    addTube(*c.m, path, r, sChrome(), 12);
    addTube(*c.m, std::vector<V3>{ V3(x - 0.071f, y, z), V3(x - 0.072f, y, z) }, r * 0.72f, sTrim(0.0f), 12);
}

// ---- Sedan ----------------------------------------------------------
static void detailsSedan(Ctx& c) {
    const Hull& H = *c.H;
    for (int s = -1; s <= 1; s += 2) {
        addSideLine(c, -2.20f, 1.95f, 0.86f, 0.010f, sChrome(), s);
        addSideLine(c, -1.15f, 1.20f, 0.33f, 0.014f, sChrome(), s);
        addHandle(c, 0.20f, 0.93f, 0.20f, s);
        addHandle(c, -0.75f, 0.93f, 0.20f, s);
        addMirror(c, 0.45f, 1.06f, s * 0.97f, 1.0f);
        addLamp(c, true, 0.80f, s * 0.58f, V3(0.07f, 0.045f, 0.17f), MAT_HEAD, 0.035f);
        addLamp(c, false, 0.85f, s * 0.64f, V3(0.06f, 0.05f, 0.18f), MAT_TAIL, 0.035f);
        addLamp(c, true, 0.40f, s * 0.60f, V3(0.05f, 0.03f, 0.06f), MAT_HEAD, 0.02f);   // fog lamps
    }
    addGrille(c, 0.64f, 0.17f, 0.34f, 15, true, 0.0f);
    // bonnet ornament
    float ox = 2.42f, oy = H.topY(ox, 0);
    addEllipsoid(*c.m, V3(ox, oy + 0.04f, 0), V3(0.045f, 0.055f, 0.014f), sChrome(), 12, 8);
    addTube(*c.m, std::vector<V3>{ V3(ox, oy - 0.01f, 0), V3(ox, oy + 0.03f, 0) }, 0.008f, sChrome(), 6);
    addPlate(c, true, 0.43f, 0.065f, 0.26f);
    addPlate(c, false, 0.58f, 0.075f, 0.26f);
    // rear chrome bar
    std::vector<V3> rb;
    for (int i = 0; i <= 10; i++) { float z = -0.55f + 1.1f * i / 10.0f; rb.push_back(V3(H.rearX(0.83f, z) - 0.008f, 0.83f, z)); }
    addTube(*c.m, rb, 0.008f, sChrome(), 6);
}

// ---- SUV ------------------------------------------------------------
static void detailsSUV(Ctx& c) {
    const Hull& H = *c.H;
    for (int s = -1; s <= 1; s += 2) {
        addSideLine(c, -2.05f, 2.05f, 0.33f, 0.028f, sTrim(0.03f), s);
        addSideLine(c, -2.0f, 1.9f, 1.05f, 0.008f, sChrome(), s);
        addHandle(c, 0.40f, 1.02f, 0.20f, s);
        addHandle(c, -0.55f, 1.02f, 0.20f, s);
        addMirror(c, 0.95f, 1.30f, s * 1.05f, 1.2f);
        addLamp(c, true, 0.96f, s * 0.68f, V3(0.07f, 0.04f, 0.20f), MAT_HEAD, 0.04f);
        addLamp(c, false, 0.96f, s * 0.80f, V3(0.05f, 0.17f, 0.06f), MAT_TAIL, 0.03f);
        // roof rails
        std::vector<V3> rail;
        for (float x = -1.95f; x <= 0.85f; x += 0.1f) rail.push_back(V3(x, cabinTopAt(*c.d, *c.cab, x, s * 0.58f) + 0.045f, s * 0.58f));
        addTube(*c.m, rail, 0.016f, sChrome(), 8);
        // skid / fog
        addLamp(c, true, 0.40f, s * 0.62f, V3(0.05f, 0.035f, 0.07f), MAT_HEAD, 0.025f);
    }
    addGrille(c, 0.74f, 0.17f, 0.58f, 7, false, 0.0f);
    std::vector<V3> sk;
    for (int i = 0; i <= 12; i++) { float z = -0.70f + 1.4f * i / 12.0f; sk.push_back(V3(H.frontX(0.40f, z) + 0.005f, 0.40f, z)); }
    addTube(*c.m, sk, 0.03f, sTrim(0.04f), 8);
    addPlate(c, true, 0.50f, 0.065f, 0.26f);
    addPlate(c, false, 0.62f, 0.075f, 0.26f);
    // roof spoiler
    addBoxAA(*c.m, V3(-2.05f, cabinTopAt(*c.d, *c.cab, -2.05f, 0) - 0.02f, 0), V3(0.10f, 0.015f, 0.70f), sPaint());
}

// ---- Sports car -----------------------------------------------------
static void detailsSports(Ctx& c) {
    const Hull& H = *c.H;
    for (int s = -1; s <= 1; s += 2) {
        // side intake
        float yi = 0.50f, xi = -0.85f;
        addEllipsoid(*c.m, V3(xi, yi, s * (H.sideZ(xi, yi) - 0.012f)), V3(0.36f, 0.10f, 0.05f), sTrim(0.005f), 16, 8);
        addSideLine(c, -0.9f, 1.0f, 0.30f, 0.012f, sAccent(V3(0.9f, 0.7f, 0.0f)), s);
        addHandle(c, 0.05f, 0.62f, 0.12f, s);
        addMirror(c, 0.80f, 0.90f, s * 0.97f, 0.9f);
        // Y headlights
        addLamp(c, true, 0.40f, s * 0.58f, V3(0.06f, 0.020f, 0.24f), MAT_HEAD, 0.04f);
        addLamp(c, true, 0.38f, s * 0.80f, V3(0.05f, 0.018f, 0.12f), MAT_HEAD, 0.04f);
        // tail lights
        addLamp(c, false, 0.80f, s * 0.60f, V3(0.05f, 0.035f, 0.24f), MAT_TAIL, 0.03f);
        // front corner intakes
        addLamp(c, true, 0.31f, s * 0.62f, V3(0.05f, 0.045f, 0.18f), MAT_TRIM, 0.03f);
        // exhausts
        addExhaust(c, 0.52f, s * 0.20f, 0.055f);
        addExhaust(c, 0.40f, s * 0.46f, 0.055f);
        // wing end plates + uprights
        addBoxAA(*c.m, V3(-2.12f, 1.16f, s * 0.97f), V3(0.20f, 0.07f, 0.012f), sPaint());
        addBoxAA(*c.m, V3(-2.08f, 1.04f, s * 0.50f), V3(0.025f, 0.09f, 0.02f), sTrim(0.01f));
    }
    // front intake + splitter
    addLamp(c, true, 0.31f, 0.0f, V3(0.06f, 0.05f, 0.30f), MAT_TRIM, 0.03f);
    float fx = H.frontX(0.30f, 0);
    addBoxAA(*c.m, V3(fx - 0.07f, 0.175f, 0), V3(0.08f, 0.010f, 0.72f), sTrim(0.01f));
    addPlate(c, true, 0.36f, 0.05f, 0.22f);
    // rear wing blade
    addBoxAA(*c.m, V3(-2.12f, 1.17f, 0), V3(0.17f, 0.012f, 0.97f), sTrim(0.012f));
    // engine louvres
    for (int i = 0; i < 6; i++) {
        float x = -1.30f - 0.13f * i;
        addBoxAA(*c.m, V3(x, H.topY(x, 0) + 0.004f, 0), V3(0.035f, 0.004f, 0.36f), sTrim(0.004f));
    }
    // high mount tail light bar + plate
    std::vector<V3> tb;
    for (int i = 0; i <= 8; i++) { float z = -0.35f + 0.7f * i / 8.0f; tb.push_back(V3(H.rearX(0.62f, z) - 0.01f, 0.62f, z)); }
    addTube(*c.m, tb, 0.012f, Surf(V3(1, 1, 1), MAT_TAIL), 6);
    addPlate(c, false, 0.40f, 0.07f, 0.26f);
}

// ---------------------------------------------------------------------
struct GpuMesh {
    GLuint vao, vbo, ebo; GLsizei count;
    GpuMesh() : vao(0), vbo(0), ebo(0), count(0) {}
    void upload(const Mesh& m) {
        count = (GLsizei)m.idx.size();
        glGenVertexArrays(1, &vao); glGenBuffers(1, &vbo); glGenBuffers(1, &ebo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, m.v.size() * sizeof(Vertex), m.v.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, m.idx.size() * sizeof(unsigned int), m.idx.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)12);
        glEnableVertexAttribArray(2); glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)24);
        glEnableVertexAttribArray(3); glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)36);
        glBindVertexArray(0);
    }
    void draw() const {
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, (void*)0);
        glBindVertexArray(0);
    }
};

struct Car {
    ModelDef def; Hull hull; Mesh bodyMesh, wheelMesh;
    GpuMesh body, wheel;
    V3 wheelPos[4]; int wheelSide[4];
};

static void buildCarMeshes(Car& car, int type) {
    car.def = (type == 0) ? defSedan() : (type == 1 ? defSUV() : defSports());
    const ModelDef& d = car.def;
    car.hull = makeHull(d);
    std::vector<CabinSample> cab = makeCabin(d, car.hull);

    std::vector<ArchDef> arches;
    float axles[2] = { d.axleF, d.axleR };
    int w = 0;
    float zin[2], Rr = d.wheelR + d.archGap;
    for (int a = 0; a < 2; a++) {
        float cy = d.wheelR;
        float zOut = car.hull.sideZ(axles[a], cy);
        float zc = zOut - d.wheelW * 0.5f + 0.004f;
        zin[a] = zc - d.wheelW * 0.5f - 0.07f;
        ArchDef A; A.cx = axles[a]; A.cy = cy; A.R = Rr; A.zin = zin[a];
        arches.push_back(A);
        for (int s = -1; s <= 1; s += 2) {
            car.wheelPos[w] = V3(axles[a], cy, s * zc);
            car.wheelSide[w] = s; w++;
        }
    }
    Mesh& m = car.bodyMesh;
    buildHullMesh(m, car.hull, arches);
    buildCabinMesh(m, d, cab);
    for (int a = 0; a < 2; a++)
        for (int s = -1; s <= 1; s += 2)
            addArchLiner(m, axles[a], d.wheelR, Rr - 0.004f, zin[a], car.hull.sideZ(axles[a], d.wheelR) - 0.02f, s);
    Ctx c; c.m = &m; c.H = &car.hull; c.cab = &cab; c.d = &car.def;
    if (type == 0) detailsSedan(c); else if (type == 1) detailsSUV(c); else detailsSports(c);
    car.wheelMesh = buildWheel(d);
}

// =====================================================================
//  SHADERS
// =====================================================================
static const char* COMMON_GLSL = R"(
vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
vec3 finalColor(vec3 c) { return pow(aces(c), vec3(1.0 / 2.2)); }
)";

static const char* CAR_VS = R"(#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec2 aMat;
uniform mat4 uModel;
uniform mat4 uVP;
uniform float uFlip;
out vec3 vWorld;
out vec3 vNormal;
out vec3 vColor;
flat out float vMat;
out float vAO;
out vec3 vObj;
void main() {
    vec4 w = uModel * vec4(aPos, 1.0);
    vWorld = w.xyz;
    vNormal = mat3(uModel) * aNormal;
    vColor = aColor; vMat = aMat.x; vAO = aMat.y; vObj = aPos;
    vec4 wp = w;
    if (uFlip > 0.5) wp.y = -wp.y;
    gl_Position = uVP * wp;
}
)";

static const char* CAR_FS_BODY = R"(
in vec3 vWorld;
in vec3 vNormal;
in vec3 vColor;
flat in float vMat;
in float vAO;
in vec3 vObj;
out vec4 FragColor;

uniform vec3 uEye;
uniform vec3 uPaint;
uniform vec3 uHorizon;
uniform vec3 uTop;
uniform vec3 uKey;
uniform vec3 uRimCol;
uniform float uLights;
uniform float uBrake;
uniform float uReflect;
uniform float uWire;
uniform int uNSide;
uniform float uSide[4];
uniform int uNTop;
uniform float uTopC[4];
uniform vec2 uCutY;

float rectLight(vec3 d, vec3 c, vec3 t, float hw, float hh) {
    float k = dot(d, c);
    if (k <= 0.0) return 0.0;
    t = normalize(t - c * dot(t, c));
    vec3 b = cross(c, t);
    float u = dot(d, t) / k, v = dot(d, b) / k;
    return (1.0 - smoothstep(hw * 0.75, hw, abs(u))) * (1.0 - smoothstep(hh * 0.75, hh, abs(v)));
}

vec3 envMap(vec3 d) {
    float h = d.y;
    vec3 hz = uHorizon * 1.5;
    vec3 c;
    if (h >= 0.0) c = mix(hz, uTop, smoothstep(0.0, 0.9, h));
    else          c = mix(hz, hz * 0.2, smoothstep(0.0, -0.5, h));
    c += uHorizon * 0.8 * exp(-abs(h) * 9.0);
    c += uKey * 4.5 * rectLight(d, vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), 0.95, 0.40);
    c += uRimCol * 3.2 * rectLight(d, normalize(vec3( 1.0, 0.25,  1.0)), vec3(0.0, 1.0, 0.0), 0.95, 0.14);
    c += uRimCol * 3.2 * rectLight(d, normalize(vec3(-1.0, 0.25,  0.7)), vec3(0.0, 1.0, 0.0), 0.95, 0.14);
    c += uKey    * 2.0 * rectLight(d, normalize(vec3(-0.6, 0.30, -1.0)), vec3(0.0, 1.0, 0.0), 0.95, 0.22);
    c += uKey    * 2.0 * rectLight(d, normalize(vec3( 1.0, 0.30, -0.8)), vec3(0.0, 1.0, 0.0), 0.95, 0.22);
    return c;
}

float hash31(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

float panelLines(vec3 N) {
    float f = 0.0;
    if (abs(N.z) > 0.55 && vObj.y > uCutY.x && vObj.y < uCutY.y)
        for (int i = 0; i < uNSide; i++)
            f = max(f, 1.0 - smoothstep(0.003, 0.007, abs(vObj.x - uSide[i])));
    if (N.y > 0.6)
        for (int i = 0; i < uNTop; i++)
            f = max(f, 1.0 - smoothstep(0.003, 0.007, abs(vObj.x - uTopC[i])));
    return f;
}

void main() {
    int mat = int(vMat + 0.5);
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uEye - vWorld);
    float NdV = clamp(dot(N, V), 0.0, 1.0);
    vec3 R = reflect(-V, N);
    float fres = pow(1.0 - NdV, 5.0);
    float ao = vAO;
    float groundOcc = mix(0.45, 1.0, smoothstep(0.0, 0.30, vWorld.y));

    vec3 L1 = normalize(vec3(-0.35, 0.9, 0.45));
    vec3 L2 = normalize(vec3(0.8, 0.35, -0.6));
    vec3 hemi = mix(uHorizon * 0.5 + 0.02, uTop * 0.5 + uHorizon * 0.4 + 0.04, N.y * 0.5 + 0.5);
    float d1 = max(dot(N, L1), 0.0), d2 = max(dot(N, L2), 0.0);
    vec3 light = hemi * ao + uKey * (1.5 * d1 * (0.5 + 0.5 * ao)) + uRimCol * 0.5 * d2 * ao;
    vec3 Hh = normalize(L1 + V);
    float spec = pow(max(dot(N, Hh), 0.0), 180.0);
    vec3 env = envMap(R);
    vec3 col;

    if (mat == 0) {                                   // clear-coat metallic paint
        vec3 base = uPaint;
        float cut = panelLines(N);
        base *= 1.0 - 0.85 * cut;
        float flake = hash31(floor(vObj * 220.0));
        flake = step(0.985, flake);
        float F = 0.04 + 0.96 * fres;
        col = base * light * 0.9;
        col += env * ao * base * 0.45;
        col += env * ao * F * (1.0 - cut);
        col += uKey * spec * 1.4 * (1.0 - cut);
        col += base * flake * 0.9 * (0.3 + 0.7 * NdV);
    } else if (mat == 1) {                            // tinted glass
        col = vec3(0.012, 0.016, 0.024) * light * 3.0;
        col += env * (0.10 + 0.90 * fres) * 0.9;
        col += uKey * spec * 1.2;
    } else if (mat == 2) {                            // chrome
        col = env * vec3(0.92, 0.93, 0.97) + 0.12 * light;
    } else if (mat == 3) {                            // black trim / plastic
        col = vColor * light + env * (0.02 + 0.10 * fres) * ao;
    } else if (mat == 4) {                            // head light
        vec3 off = vColor * 0.55 * light + env * (0.15 + fres);
        vec3 on = vec3(3.0, 2.9, 2.6) * (0.6 + 0.4 * NdV);
        col = mix(off, on, uLights);
    } else if (mat == 5) {                            // tail light
        vec3 off = vec3(0.14, 0.01, 0.01) * light + env * (0.10 + 0.60 * fres);
        vec3 on = vec3(2.4, 0.10, 0.06) * (0.7 + 0.3 * NdV) * (1.0 + uBrake);
        col = mix(off, on, max(uLights, uBrake));
    } else if (mat == 6) {                            // alloy rim
        col = vColor * light * 0.5 + env * vColor * 0.9 * ao;
    } else if (mat == 7) {                            // tyre
        col = vColor * light * 0.6 + env * 0.015;
    } else {                                          // accent (calipers, plates)
        col = vColor * light + env * (0.12 + 0.5 * fres) * ao + uKey * spec * 0.5;
    }

    col *= groundOcc;
    if (uReflect < 0.999) col *= uReflect * mix(1.0, 0.35, clamp(vWorld.y / 1.6, 0.0, 1.0));

    if (uWire > 0.5) {
        vec3 w = mix(vec3(0.10, 0.80, 1.0), uPaint * 3.0, 0.35) * (0.35 + 0.65 * NdV);
        if (mat == 1) w = vec3(0.05, 0.30, 0.65) * (0.4 + 0.6 * NdV);
        if (mat == 4 || mat == 5) w = vec3(1.0, 0.8, 0.3);
        col = w;
    }
    FragColor = vec4(finalColor(col), 1.0);
}
)";

static const char* BG_VS = R"(#version 330 core
out vec2 vNdc;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vNdc = p * 2.0 - 1.0;
    gl_Position = vec4(vNdc, 0.999, 1.0);
}
)";

static const char* BG_FS_BODY = R"(
in vec2 vNdc;
out vec4 FragColor;
uniform vec3 uHorizon;
uniform vec3 uTop;
uniform vec3 uFwd;
uniform vec3 uRight;
uniform vec3 uUp;
uniform float uTan;
uniform float uAspect;
void main() {
    vec3 dir = normalize(uFwd + vNdc.x * uAspect * uTan * uRight + vNdc.y * uTan * uUp);
    vec3 hz = uHorizon * 1.5;
    vec3 c = mix(hz, uTop, smoothstep(0.0, 0.7, dir.y));
    c *= 1.0 - 0.35 * dot(vNdc, vNdc) * 0.5;
    FragColor = vec4(finalColor(c), 1.0);
}
)";

static const char* FLOOR_VS = R"(#version 330 core
layout(location = 0) in vec2 aPos;
uniform mat4 uVP;
out vec3 vW;
void main() {
    vW = vec3(aPos.x, 0.0, aPos.y);
    gl_Position = uVP * vec4(vW, 1.0);
}
)";

static const char* FLOOR_FS_BODY = R"(
in vec3 vW;
out vec4 FragColor;
uniform vec3 uHorizon;
uniform vec3 uFloorLight;
uniform vec3 uFloorDark;
uniform vec3 uRimCol;
uniform float uScroll;
uniform float uRoad;
uniform float uHalfLen;
uniform float uHalfWid;
uniform float uAlpha;
uniform float uGrid;

float gridLine(float v, float w) {
    float f = abs(fract(v - 0.5) - 0.5) / max(fwidth(v), 1e-5);
    return clamp(1.0 - f / w, 0.0, 1.0);
}

void main() {
    vec2 p = vW.xz;
    float r = length(p);
    float spot = exp(-r * r / (2.0 * 7.0 * 7.0));
    vec3 col = mix(uFloorDark, uFloorLight, spot);
    col = mix(col, vec3(0.025, 0.027, 0.030) + uFloorLight * 0.10 * spot, uRoad);

    vec2 g = vec2(p.x + uScroll, p.y);
    float minor = max(gridLine(g.x, 1.0), gridLine(g.y, 1.0));
    float major = max(gridLine(g.x * 0.2, 1.4), gridLine(g.y * 0.2, 1.4));
    float gi = uGrid * (1.0 - 0.75 * uRoad) * exp(-r / 16.0);
    col += uRimCol * (0.10 * minor + 0.30 * major) * gi;

    // platform ring (showroom)
    float ring = (1.0 - smoothstep(0.0, 0.08, abs(r - 5.2))) * (1.0 - uRoad);
    col += uRimCol * 0.35 * ring * uGrid;

    // road markings
    float dash = step(0.5, fract((p.x + uScroll) / 6.0));
    float lane = (1.0 - smoothstep(0.05, 0.09, abs(abs(p.y) - 2.6))) * dash;
    float edge = 1.0 - smoothstep(0.07, 0.12, abs(abs(p.y) - 5.4));
    col += vec3(0.75, 0.75, 0.70) * (lane + edge) * uRoad * exp(-r / 30.0);

    // soft contact shadow
    float d = length(vec2(p.x / (uHalfLen * 1.05), p.y / (uHalfWid * 1.12)));
    float sh = 0.70 * smoothstep(1.45, 0.50, d) + 0.30 * smoothstep(1.05, 0.55, d);
    col *= 1.0 - clamp(sh, 0.0, 0.92);

    float fade = smoothstep(14.0, 70.0, r);
    col = mix(col, uHorizon * 1.5, fade);
    float alpha = mix(uAlpha, 1.0, fade);
    FragColor = vec4(finalColor(col), alpha);
}
)";

#ifndef CAR_TEST
static GLuint compileShader(GLenum type, const std::string& src) {
    GLuint s = glCreateShader(type);
    const char* p = src.c_str();
    glShaderSource(s, 1, &p, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048]; glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        std::fprintf(stderr, "Shader compile error:\n%s\n", log);
        std::exit(1);
    }
    return s;
}
static GLuint makeProgram(const std::string& vs, const std::string& fs) {
    GLuint a = compileShader(GL_VERTEX_SHADER, vs), b = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, a); glAttachShader(p, b);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048]; glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        std::fprintf(stderr, "Program link error:\n%s\n", log);
        std::exit(1);
    }
    glDeleteShader(a); glDeleteShader(b);
    return p;
}

// =====================================================================
//  APPLICATION STATE
// =====================================================================
struct App {
    float yaw = 0.70f, pitch = 0.24f, dist = 8.6f;
    float tyaw = 0.70f, tpitch = 0.24f, tdist = 8.6f;
    bool dragging = false; double lx = 0, ly = 0;
    int model = 0, theme = 0; int paint[3] = { 2, 3, 6 };
    bool spin = true, drive = false, wire = false, grid = true, lights = true, reflect = true;
    float speed = 0, travel = 0, wheelAng = 0, brake = 0;
    bool shot = false;
};
static App A;
static Car CARS[3];

static void printHelp() {
    std::printf(
        "\n=========== 3D CAR DESIGN STUDIO ===========\n"
        " Mouse drag      orbit        Mouse wheel  zoom\n"
        " A / D, arrows   rotate       W / S        tilt\n"
        " + / -           zoom         1 2 3        Sedan / SUV / Sports\n"
        " C / X           next / previous paint colour\n"
        " V               environment (Studio, Sunset, Showroom, Neon)\n"
        " SPACE           turntable    M            drive mode\n"
        " L               lights       G            floor grid\n"
        " F               wireframe    B            floor reflection\n"
        " P               screenshot   R            reset view\n"
        " H               this help    ESC          exit\n"
        "============================================\n");
}

static void keyCb(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    switch (key) {
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(w, 1); break;
        case GLFW_KEY_1: A.model = 0; break;
        case GLFW_KEY_2: A.model = 1; break;
        case GLFW_KEY_3: A.model = 2; break;
        case GLFW_KEY_C: A.paint[A.model] = (A.paint[A.model] + 1) % NUM_PAINTS; break;
        case GLFW_KEY_X: A.paint[A.model] = (A.paint[A.model] + NUM_PAINTS - 1) % NUM_PAINTS; break;
        case GLFW_KEY_V: A.theme = (A.theme + 1) % NUM_THEMES; break;
        case GLFW_KEY_SPACE: A.spin = !A.spin; break;
        case GLFW_KEY_M: A.drive = !A.drive; break;
        case GLFW_KEY_L: A.lights = !A.lights; break;
        case GLFW_KEY_G: A.grid = !A.grid; break;
        case GLFW_KEY_F: A.wire = !A.wire; break;
        case GLFW_KEY_B: A.reflect = !A.reflect; break;
        case GLFW_KEY_P: A.shot = true; break;
        case GLFW_KEY_H: printHelp(); break;
        case GLFW_KEY_R: A.tyaw = 0.70f; A.tpitch = 0.24f; A.tdist = 8.6f; break;
        default: break;
    }
}
static void mouseCb(GLFWwindow* w, int button, int action, int) {
    if (button == GLFW_MOUSE_BUTTON_LEFT || button == GLFW_MOUSE_BUTTON_RIGHT) {
        A.dragging = (action == GLFW_PRESS);
        if (A.dragging) { glfwGetCursorPos(w, &A.lx, &A.ly); A.spin = false; }
    }
}
static void cursorCb(GLFWwindow*, double x, double y) {
    if (!A.dragging) return;
    A.tyaw += (float)(x - A.lx) * 0.006f;
    A.tpitch += (float)(y - A.ly) * 0.005f;
    A.lx = x; A.ly = y;
}
static void scrollCb(GLFWwindow*, double, double dy) { A.tdist *= std::pow(0.92f, (float)dy); }

static void saveShot(int w, int h) {
    static int n = 0;
    std::vector<unsigned char> px((size_t)w * h * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, px.data());
    char name[64]; std::snprintf(name, sizeof(name), "carshot_%03d.ppm", n++);
    FILE* f = std::fopen(name, "wb");
    if (!f) return;
    std::fprintf(f, "P6\n%d %d\n255\n", w, h);
    for (int y = h - 1; y >= 0; y--) std::fwrite(&px[(size_t)y * w * 3], 1, (size_t)w * 3, f);
    std::fclose(f);
    std::printf("Saved %s\n", name);
}

static inline GLint U(GLuint p, const char* n) { return glGetUniformLocation(p, n); }
static void setV3(GLuint p, const char* n, V3 v) { glUniform3f(U(p, n), v.x, v.y, v.z); }

// =====================================================================
//  MAIN
// =====================================================================
int main() {
    if (!glfwInit()) { std::fprintf(stderr, "GLFW init failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    glfwWindowHint(GLFW_SAMPLES, 4);
    GLFWwindow* win = glfwCreateWindow(1280, 720, "3D Car Design Studio", nullptr, nullptr);
    if (!win) { std::fprintf(stderr, "Window creation failed (OpenGL 3.3 needed)\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { std::fprintf(stderr, "GLEW init failed\n"); return 1; }
    glGetError();

    glfwSetKeyCallback(win, keyCb);
    glfwSetMouseButtonCallback(win, mouseCb);
    glfwSetCursorPosCallback(win, cursorCb);
    glfwSetScrollCallback(win, scrollCb);

    GLuint carProg = makeProgram(CAR_VS, std::string("#version 330 core\n") + COMMON_GLSL + CAR_FS_BODY);
    GLuint bgProg = makeProgram(BG_VS, std::string("#version 330 core\n") + COMMON_GLSL + BG_FS_BODY);
    GLuint flProg = makeProgram(FLOOR_VS, std::string("#version 330 core\n") + COMMON_GLSL + FLOOR_FS_BODY);

    for (int i = 0; i < 3; i++) {
        buildCarMeshes(CARS[i], i);
        CARS[i].body.upload(CARS[i].bodyMesh);
        CARS[i].wheel.upload(CARS[i].wheelMesh);
        A.paint[i] = CARS[i].def.defaultPaint;
        std::printf("Built %-13s : %zu body triangles, %zu wheel triangles\n", CARS[i].def.name,
                    CARS[i].bodyMesh.idx.size() / 3, CARS[i].wheelMesh.idx.size() / 3);
    }

    GLuint bgVao; glGenVertexArrays(1, &bgVao);
    GLuint flVao, flVbo;
    {
        float q[] = { -150, -150, 150, -150, 150, 150, -150, -150, 150, 150, -150, 150 };
        glGenVertexArrays(1, &flVao); glGenBuffers(1, &flVbo);
        glBindVertexArray(flVao); glBindBuffer(GL_ARRAY_BUFFER, flVbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(q), q, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, (void*)0);
        glBindVertexArray(0);
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glEnable(GL_MULTISAMPLE);
    glPolygonOffset(2.0f, 2.0f);
    printHelp();

    double last = glfwGetTime(), titleT = 0; int frames = 0; float fps = 0;
    float simTime = 0;

    while (!glfwWindowShouldClose(win)) {
        double now = glfwGetTime();
        float dt = (float)(now - last); last = now;
        if (dt > 0.1f) dt = 0.1f;
        simTime += dt;
        glfwPollEvents();

        // ---- continuous input
        float rs = 1.8f * dt;
        if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_LEFT) == GLFW_PRESS)  { A.tyaw -= rs; A.spin = false; }
        if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_RIGHT) == GLFW_PRESS) { A.tyaw += rs; A.spin = false; }
        if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_UP) == GLFW_PRESS)    A.tpitch += rs * 0.7f;
        if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_DOWN) == GLFW_PRESS)  A.tpitch -= rs * 0.7f;
        if (glfwGetKey(win, GLFW_KEY_EQUAL) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_KP_ADD) == GLFW_PRESS)       A.tdist *= std::pow(0.4f, dt);
        if (glfwGetKey(win, GLFW_KEY_MINUS) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS)  A.tdist /= std::pow(0.4f, dt);
        A.tpitch = clampf(A.tpitch, -0.12f, 1.45f);
        A.tdist = clampf(A.tdist, 4.5f, 22.0f);
        if (A.spin) A.tyaw += 0.45f * dt;
        float k = std::min(1.0f, dt * 9.0f);
        A.yaw += (A.tyaw - A.yaw) * k; A.pitch += (A.tpitch - A.pitch) * k; A.dist += (A.tdist - A.dist) * k;

        // ---- drive simulation
        float target = A.drive ? 14.0f : 0.0f;
        float prev = A.speed;
        A.speed += (target - A.speed) * std::min(1.0f, dt * 1.6f);
        if (A.speed < 0.02f) A.speed = 0;
        A.brake = (A.speed < prev - 1e-4f && A.speed > 0.3f) ? 1.0f : 0.0f;
        A.travel += A.speed * dt;
        const Car& car = CARS[A.model];
        A.wheelAng += A.speed * dt / car.def.wheelR;

        int fbw, fbh;
        glfwGetFramebufferSize(win, &fbw, &fbh);
        if (fbw < 1 || fbh < 1) continue;
        glViewport(0, 0, fbw, fbh);

        const Theme& T = THEMES[A.theme];
        float aspect = (float)fbw / (float)fbh;
        float fov = 38.0f * PI_F / 180.0f;
        V3 target3(0, 0.62f, 0);
        V3 eye = target3 + V3(std::cos(A.pitch) * std::cos(A.yaw), std::sin(A.pitch), std::cos(A.pitch) * std::sin(A.yaw)) * A.dist;
        M4 VP = m4mul(m4perspective(fov, aspect, 0.1f, 400.0f), m4lookAt(eye, target3, V3(0, 1, 0)));
        V3 fwd = norm(target3 - eye), right = norm(cross(fwd, V3(0, 1, 0))), up = cross(right, fwd);

        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // ---- 1. background
        glDisable(GL_DEPTH_TEST); glDisable(GL_CULL_FACE);
        glUseProgram(bgProg);
        setV3(bgProg, "uHorizon", T.horizon); setV3(bgProg, "uTop", T.top);
        setV3(bgProg, "uFwd", fwd); setV3(bgProg, "uRight", right); setV3(bgProg, "uUp", up);
        glUniform1f(U(bgProg, "uTan"), std::tan(fov * 0.5f)); glUniform1f(U(bgProg, "uAspect"), aspect);
        glBindVertexArray(bgVao); glDrawArrays(GL_TRIANGLES, 0, 3);
        glEnable(GL_DEPTH_TEST); glEnable(GL_CULL_FACE);

        // ---- car drawing lambda
        float bob = A.speed > 0.1f ? 0.004f * std::sin(simTime * 37.0f) * std::min(1.0f, A.speed / 8.0f) : 0.0f;
        M4 bodyM = m4translate(V3(0, bob, 0));
        auto drawCar = [&](bool flip) {
            glUseProgram(carProg);
            glUniformMatrix4fv(U(carProg, "uVP"), 1, GL_FALSE, VP.m);
            V3 e = flip ? V3(eye.x, -eye.y, eye.z) : eye;
            setV3(carProg, "uEye", e);
            setV3(carProg, "uPaint", PAINTS[A.paint[A.model]].col);
            setV3(carProg, "uHorizon", T.horizon); setV3(carProg, "uTop", T.top);
            setV3(carProg, "uKey", T.key); setV3(carProg, "uRimCol", T.rim);
            glUniform1f(U(carProg, "uLights"), A.lights ? 1.0f : 0.0f);
            glUniform1f(U(carProg, "uBrake"), A.brake);
            glUniform1f(U(carProg, "uReflect"), flip ? 0.55f : 1.0f);
            glUniform1f(U(carProg, "uWire"), A.wire ? 1.0f : 0.0f);
            glUniform1f(U(carProg, "uFlip"), flip ? 1.0f : 0.0f);
            float sc[4] = { 0, 0, 0, 0 }, tc[4] = { 0, 0, 0, 0 };
            int ns = (int)std::min<size_t>(4, car.def.sideCuts.size()), nt = (int)std::min<size_t>(4, car.def.topCuts.size());
            for (int i = 0; i < ns; i++) sc[i] = car.def.sideCuts[i];
            for (int i = 0; i < nt; i++) tc[i] = car.def.topCuts[i];
            glUniform1i(U(carProg, "uNSide"), ns); glUniform1fv(U(carProg, "uSide"), 4, sc);
            glUniform1i(U(carProg, "uNTop"), nt); glUniform1fv(U(carProg, "uTopC"), 4, tc);
            glUniform2f(U(carProg, "uCutY"), car.def.cutY0, car.def.cutY1);

            glUniformMatrix4fv(U(carProg, "uModel"), 1, GL_FALSE, bodyM.m);
            car.body.draw();
            for (int i = 0; i < 4; i++) {
                int sd = car.wheelSide[i];
                M4 flipM = (sd > 0) ? m4identity() : m4rotY(PI_F);
                M4 spinM = m4rotZ(sd > 0 ? -A.wheelAng : A.wheelAng);
                M4 wm = m4mul(bodyM, m4mul(m4translate(car.wheelPos[i]), m4mul(flipM, spinM)));
                glUniformMatrix4fv(U(carProg, "uModel"), 1, GL_FALSE, wm.m);
                car.wheel.draw();
            }
        };

        // ---- 2. mirrored car (floor reflection)
        if (A.reflect && !A.wire) {
            glFrontFace(GL_CW);
            drawCar(true);
            glFrontFace(GL_CCW);
        }

        // ---- 3. floor
        glUseProgram(flProg);
        glUniformMatrix4fv(U(flProg, "uVP"), 1, GL_FALSE, VP.m);
        setV3(flProg, "uHorizon", T.horizon); setV3(flProg, "uFloorLight", T.floorLight);
        setV3(flProg, "uFloorDark", T.floorDark); setV3(flProg, "uRimCol", T.rim);
        glUniform1f(U(flProg, "uScroll"), A.travel);
        glUniform1f(U(flProg, "uRoad"), A.drive || A.speed > 0.5f ? std::min(1.0f, A.speed / 4.0f) : 0.0f);
        glUniform1f(U(flProg, "uHalfLen"), car.def.halfLen); glUniform1f(U(flProg, "uHalfWid"), car.def.halfWid);
        glUniform1f(U(flProg, "uAlpha"), (A.reflect && !A.wire) ? 0.74f : 1.0f);
        glUniform1f(U(flProg, "uGrid"), A.grid ? 1.0f : 0.0f);
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDisable(GL_CULL_FACE);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glBindVertexArray(flVao); glDrawArrays(GL_TRIANGLES, 0, 6);
        glDisable(GL_POLYGON_OFFSET_FILL);
        glDisable(GL_BLEND);
        glEnable(GL_CULL_FACE);

        // ---- 4. the car
        if (A.wire) { glDisable(GL_CULL_FACE); glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); }
        drawCar(false);
        if (A.wire) { glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); glEnable(GL_CULL_FACE); }

        if (A.shot) { saveShot(fbw, fbh); A.shot = false; }
        glfwSwapBuffers(win);

        // ---- window title as HUD
        frames++; titleT += dt;
        if (titleT >= 0.4) {
            fps = (float)(frames / titleT); frames = 0; titleT = 0;
            char buf[256];
            std::snprintf(buf, sizeof(buf), "3D Car Studio | %s | %s | %s | %s%s%s | %.0f km/h | %.0f FPS | H = help",
                          car.def.name, PAINTS[A.paint[A.model]].name, T.name,
                          A.spin ? "[Spin] " : "", A.drive ? "[Drive] " : "", A.wire ? "[Wire]" : "",
                          A.speed * 3.6f, fps);
            glfwSetWindowTitle(win, buf);
        }
    }
    glfwTerminate();
    return 0;
}
#endif  // CAR_TEST
