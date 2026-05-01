(function(){
'use strict';

console.log('WebGL2 supported:', !!document.createElement('canvas').getContext('webgl2'));

/* ---------- Physical constants ---------- */
const RHO_BLOOD = 1060.0;
const L_PHYS = 0.01;
const U_LAT_MIN = 0.04;
const U_LAT_MAX = 0.10;

const CELL_FLUID = 0, CELL_WALL = 1, CELL_INLET = 2, CELL_OUTLET = 3;
const CY_MU0 = 56e-3, CY_MUI = 3.45e-3, CY_LAM = 3.313, CY_N = 0.3568, CY_A = 2.0;

/* ---------- DOM ---------- */
const $ = id => document.getElementById(id);
const canvas = $('cfdCanvas');
const cbCv = $('cbCanvas');
const cbCtx = cbCv.getContext('2d');

/* ---------- WebGL2 context ---------- */
const gl = canvas.getContext('webgl2', { alpha: false, depth: false, stencil: false, antialias: false, preserveDrawingBuffer: false });
if (!gl) { showErr('WebGL2 is not supported in this browser.'); return; }
const extCBF = gl.getExtension('EXT_color_buffer_float');
const extLin = gl.getExtension('OES_texture_float_linear');
if (!extCBF) { showErr('Your GPU/driver lacks EXT_color_buffer_float, required for float simulation.'); return; }

function showErr(msg) {
  const ov = $('errOv');
  $('errMsg').textContent = msg;
  ov.classList.add('show');
  console.error(msg);
}

let NX = 360, NY = 144;
let DX = L_PHYS / NY;
let geomData = new Float32Array(NX * NY * 4);

const state = {
  uin: 0.30,
  muN: 3.5e-3,
  viscModel: 'newtonian',
  inletType: 'velocity',
  inletProfile: 'plug',
  outletType: 'zerograd',
  pIn_mmHg: 100.0,
  pOut_mmHg: 80.0,
  gravity: true,
  field: 'vel',
  showStream: true,
  showParticles: false,
  scaleMode: 'true',
  xform: 'linear',
  manualMin: 0.0,
  manualMax: 1.0,
  detectSteady: false,
  stepsPerFrame: 3,
  preset: 'sphere',
  cy_mu0: CY_MU0, cy_muI: CY_MUI, cy_lam: CY_LAM, cy_n: CY_N, cy_a: CY_A,
  inletYMin: 0, inletYMax: 1,
  dt: 1e-5, tau: 0.6, uScale: 1, pScale: 1, Re: 0, uLat: 0.05,
};
let pOutClamped = false, pInClamped = false;

/* ---------- GL Utilities ---------- */
function compile(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    console.error('Shader error:\n' + gl.getShaderInfoLog(s));
    throw new Error('Shader compile failed');
  }
  return s;
}
function link(vsSrc, fsSrc) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS))
    throw new Error('Link: ' + gl.getProgramInfoLog(p));
  return p;
}
function cacheUniforms(prog, names) {
  const out = {};
  for (const n of names) out[n] = gl.getUniformLocation(prog, n);
  return out;
}
const quadBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);
function bindQuad(prog) {
  gl.useProgram(prog);
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
  const loc = gl.getAttribLocation(prog, 'aP');
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
}

/* ---------- Shaders (same as before, verified) ---------- */
const VS = `#version 300 es
in vec2 aP;
out vec2 vUV;
void main(){ vUV = aP * 0.5 + 0.5; gl_Position = vec4(aP, 0.0, 1.0); }`;

const FS_INIT = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uGeom;
uniform float uUlat;
layout(location=0) out vec4 outA;
layout(location=1) out vec4 outB;
layout(location=2) out vec4 outC;
const float W0=4.0/9.0, W14=1.0/9.0, W58=1.0/36.0;
void main(){
  float ct = texture(uGeom, vUV).r;
  float u = 0.0, v = 0.0, rho = 1.0;
  if (ct > 1.5 && ct < 2.5) {
    u = uUlat;
  } else if (ct < 0.5) {
    u = uUlat;
    v = 0.025 * uUlat * sin(vUV.x * 18.8496) * (vUV.y - 0.5);
  }
  float usq = 1.5*(u*u + v*v);
  float cu;
  float f[9];
  cu=0.0;       f[0]=W0 *rho*(1.0 - usq);
  cu=u;         f[1]=W14*rho*(1.0+3.0*cu+4.5*cu*cu - usq);
  cu=v;         f[2]=W14*rho*(1.0+3.0*cu+4.5*cu*cu - usq);
  cu=-u;        f[3]=W14*rho*(1.0+3.0*cu+4.5*cu*cu - usq);
  cu=-v;        f[4]=W14*rho*(1.0+3.0*cu+4.5*cu*cu - usq);
  cu=u+v;       f[5]=W58*rho*(1.0+3.0*cu+4.5*cu*cu - usq);
  cu=-u+v;      f[6]=W58*rho*(1.0+3.0*cu+4.5*cu*cu - usq);
  cu=-u-v;      f[7]=W58*rho*(1.0+3.0*cu+4.5*cu*cu - usq);
  cu=u-v;       f[8]=W58*rho*(1.0+3.0*cu+4.5*cu*cu - usq);
  outA = vec4(f[0], f[1], f[2], f[3]);
  outB = vec4(f[4], f[5], f[6], f[7]);
  outC = vec4(f[8], 0.0, 0.0, 0.0);
}`;

const FS_STEP = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uFA, uFB, uFC, uGeom;
uniform vec2 uRes;
uniform float uTau, uGx, uGy, uUlat, uRhoIn, uRhoOut;
uniform int uNonNewt, uInletType, uOutletType, uInletProfile;
uniform vec2 uInletYRange;
uniform float uMu0, uMuI, uLam, uN, uA;
layout(location=0) out vec4 outA;
layout(location=1) out vec4 outB;
layout(location=2) out vec4 outC;
const float W0=4.0/9.0, W14=1.0/9.0, W58=1.0/36.0;

ivec2 clm(ivec2 c){ return clamp(c, ivec2(0), ivec2(uRes) - ivec2(1)); }

void main(){
  ivec2 coord = ivec2(gl_FragCoord.xy);

  float f0 = texelFetch(uFA, clm(coord                ), 0).r;
  float f1 = texelFetch(uFA, clm(coord - ivec2( 1, 0)), 0).g;
  float f2 = texelFetch(uFA, clm(coord - ivec2( 0, 1)), 0).b;
  float f3 = texelFetch(uFA, clm(coord - ivec2(-1, 0)), 0).a;
  float f4 = texelFetch(uFB, clm(coord - ivec2( 0,-1)), 0).r;
  float f5 = texelFetch(uFB, clm(coord - ivec2( 1, 1)), 0).g;
  float f6 = texelFetch(uFB, clm(coord - ivec2(-1, 1)), 0).b;
  float f7 = texelFetch(uFB, clm(coord - ivec2(-1,-1)), 0).a;
  float f8 = texelFetch(uFC, clm(coord - ivec2( 1,-1)), 0).r;

  float ct = texelFetch(uGeom, coord, 0).r;

  #define SANE(v, w) (((v) == (v)) && (v) < 1e30 && (v) > -1e30 ? max(v, 0.0) : (w))
  f0 = SANE(f0, 4.0/9.0);
  f1 = SANE(f1, 1.0/9.0);  f2 = SANE(f2, 1.0/9.0);
  f3 = SANE(f3, 1.0/9.0);  f4 = SANE(f4, 1.0/9.0);
  f5 = SANE(f5, 1.0/36.0); f6 = SANE(f6, 1.0/36.0);
  f7 = SANE(f7, 1.0/36.0); f8 = SANE(f8, 1.0/36.0);

  if (ct > 0.5 && ct < 1.5) {
    float t;
    t = f1; f1 = f3; f3 = t;
    t = f2; f2 = f4; f4 = t;
    t = f5; f5 = f7; f7 = t;
    t = f6; f6 = f8; f8 = t;
    outA = vec4(f0, f1, f2, f3);
    outB = vec4(f4, f5, f6, f7);
    outC = vec4(f8, 0.0, 0.0, 0.0);
    return;
  }

  float rho = f0+f1+f2+f3+f4+f5+f6+f7+f8;
  rho = clamp(rho, 0.1, 5.0);
  float ux = (f1 - f3 + f5 - f6 - f7 + f8) / rho;
  float uy = (f2 - f4 + f5 + f6 - f7 - f8) / rho;
  ux = clamp(ux, -0.20, 0.20);
  uy = clamp(uy, -0.20, 0.20);

  if (ct > 1.5 && ct < 2.5) {
    if (uInletType == 0) {
      float u_local = uUlat;
      if (uInletProfile == 1) {
        float yy = float(coord.y);
        float yc = 0.5 * (uInletYRange.x + uInletYRange.y);
        float yh = 0.5 * (uInletYRange.y - uInletYRange.x);
        if (yh > 0.5) {
          float t = (yy - yc) / yh;
          u_local = 1.2 * uUlat * max(0.0, 1.0 - t * t);
        }
      }
      ux = clamp(u_local, -0.20, 0.20);
      uy = 0.0;
      rho = 1.0;
    } else {
      rho = clamp(uRhoIn, 0.5, 2.0);
      ivec2 dn = clm(coord + ivec2(1, 0));
      vec4 nA = texelFetch(uFA, dn, 0), nB = texelFetch(uFB, dn, 0), nC = texelFetch(uFC, dn, 0);
      float n0=SANE(nA.r,4.0/9.0), n1=SANE(nA.g,1.0/9.0), n2=SANE(nA.b,1.0/9.0),
            n3=SANE(nA.a,1.0/9.0), n4=SANE(nB.r,1.0/9.0), n5=SANE(nB.g,1.0/36.0),
            n6=SANE(nB.b,1.0/36.0), n7=SANE(nB.a,1.0/36.0), n8=SANE(nC.r,1.0/36.0);
      float nrho = clamp(n0+n1+n2+n3+n4+n5+n6+n7+n8, 0.1, 5.0);
      ux = clamp((n1 - n3 + n5 - n6 - n7 + n8) / nrho, -0.20, 0.20);
      uy = 0.0;
    }
  } else if (ct > 2.5 && ct < 3.5) {
    if (uOutletType == 0) {
      ivec2 up = clm(coord - ivec2(1, 0));
      vec4 nA = texelFetch(uFA, up, 0);
      vec4 nB = texelFetch(uFB, up, 0);
      f3 = SANE(nA.a, 1.0/9.0);
      f6 = SANE(nB.b, 1.0/36.0);
      f7 = SANE(nB.a, 1.0/36.0);
      rho = clamp(f0+f1+f2+f3+f4+f5+f6+f7+f8, 0.1, 5.0);
      ux = clamp((f1 - f3 + f5 - f6 - f7 + f8) / rho, -0.20, 0.20);
      uy = clamp((f2 - f4 + f5 + f6 - f7 - f8) / rho, -0.20, 0.20);
    } else {
      rho = clamp(uRhoOut, 0.5, 2.0);
      ivec2 up = clm(coord - ivec2(1, 0));
      vec4 nA = texelFetch(uFA, up, 0), nB = texelFetch(uFB, up, 0), nC = texelFetch(uFC, up, 0);
      float n0=SANE(nA.r,4.0/9.0), n1=SANE(nA.g,1.0/9.0), n2=SANE(nA.b,1.0/9.0),
            n3=SANE(nA.a,1.0/9.0), n4=SANE(nB.r,1.0/9.0), n5=SANE(nB.g,1.0/36.0),
            n6=SANE(nB.b,1.0/36.0), n7=SANE(nB.a,1.0/36.0), n8=SANE(nC.r,1.0/36.0);
      float nrho = clamp(n0+n1+n2+n3+n4+n5+n6+n7+n8, 0.1, 5.0);
      ux = clamp((n1 - n3 + n5 - n6 - n7 + n8) / nrho, -0.20, 0.20);
      uy = clamp((n2 - n4 + n5 + n6 - n7 - n8) / nrho, -0.20, 0.20);
    }
  }

  float uxE = clamp(ux + uGx, -0.20, 0.20);
  float uyE = clamp(uy + uGy, -0.20, 0.20);

  float tauL = uTau;
  if (uNonNewt == 1) {
    float usq = 1.5*(ux*ux + uy*uy);
    float cu, feq;
    cu=0.0;    feq = W0 *rho*(1.0 - usq);                                 float n0 = f0 - feq;
    cu=ux;     feq = W14*rho*(1.0+3.0*cu+4.5*cu*cu - usq);                float n1 = f1 - feq;
    cu=uy;     feq = W14*rho*(1.0+3.0*cu+4.5*cu*cu - usq);                float n2 = f2 - feq;
    cu=-ux;    feq = W14*rho*(1.0+3.0*cu+4.5*cu*cu - usq);                float n3 = f3 - feq;
    cu=-uy;    feq = W14*rho*(1.0+3.0*cu+4.5*cu*cu - usq);                float n4 = f4 - feq;
    cu=ux+uy;  feq = W58*rho*(1.0+3.0*cu+4.5*cu*cu - usq);                float n5 = f5 - feq;
    cu=-ux+uy; feq = W58*rho*(1.0+3.0*cu+4.5*cu*cu - usq);                float n6 = f6 - feq;
    cu=-ux-uy; feq = W58*rho*(1.0+3.0*cu+4.5*cu*cu - usq);                float n7 = f7 - feq;
    cu=ux-uy;  feq = W58*rho*(1.0+3.0*cu+4.5*cu*cu - usq);                float n8 = f8 - feq;
    float Pxx = n1 + n3 + n5 + n6 + n7 + n8;
    float Pyy = n2 + n4 + n5 + n6 + n7 + n8;
    float Pxy = n5 - n6 + n7 - n8;
    float piSq = max(Pxx*Pxx + Pyy*Pyy + 2.0*Pxy*Pxy, 0.0);
    float tauRef = 0.6;
    float shear = 3.0 / (rho * tauRef) * sqrt(piSq * 0.5);
    shear = min(shear, 1e4);
    float lamShear = max(uLam * shear, 1e-8);
    float mu = uMuI + (uMu0 - uMuI) * pow(1.0 + pow(lamShear, uA), (uN - 1.0) / uA);
    if (!(mu > 0.0)) mu = uMuI;
    tauL = clamp(0.5 + 3.0 * mu / max(rho, 0.1), 0.55, 2.5);
  }

  float usq = 1.5*(uxE*uxE + uyE*uyE);
  float cu; float feq[9];
  cu=0.0;       feq[0] = W0 *rho*(1.0 - usq);
  cu=uxE;       feq[1] = W14*rho*(1.0+3.0*cu+4.5*cu*cu - usq);
  cu=uyE;       feq[2] = W14*rho*(1.0+3.0*cu+4.5*cu*cu - usq);
  cu=-uxE;      feq[3] = W14*rho*(1.0+3.0*cu+4.5*cu*cu - usq);
  cu=-uyE;      feq[4] = W14*rho*(1.0+3.0*cu+4.5*cu*cu - usq);
  cu=uxE+uyE;   feq[5] = W58*rho*(1.0+3.0*cu+4.5*cu*cu - usq);
  cu=-uxE+uyE;  feq[6] = W58*rho*(1.0+3.0*cu+4.5*cu*cu - usq);
  cu=-uxE-uyE;  feq[7] = W58*rho*(1.0+3.0*cu+4.5*cu*cu - usq);
  cu=uxE-uyE;   feq[8] = W58*rho*(1.0+3.0*cu+4.5*cu*cu - usq);
  float w = 1.0 / tauL;
  f0 += w*(feq[0]-f0); f1 += w*(feq[1]-f1); f2 += w*(feq[2]-f2);
  f3 += w*(feq[3]-f3); f4 += w*(feq[4]-f4); f5 += w*(feq[5]-f5);
  f6 += w*(feq[6]-f6); f7 += w*(feq[7]-f7); f8 += w*(feq[8]-f8);

  bool pinToEq = (ct > 1.5 && ct < 2.5) ||
                 (ct > 2.5 && ct < 3.5 && uOutletType == 1);
  if (pinToEq) {
    f0=feq[0]; f1=feq[1]; f2=feq[2]; f3=feq[3];
    f4=feq[4]; f5=feq[5]; f6=feq[6]; f7=feq[7]; f8=feq[8];
  }

  f0 = clamp(f0, 1e-7, 10.0); f1 = clamp(f1, 1e-7, 10.0);
  f2 = clamp(f2, 1e-7, 10.0); f3 = clamp(f3, 1e-7, 10.0);
  f4 = clamp(f4, 1e-7, 10.0); f5 = clamp(f5, 1e-7, 10.0);
  f6 = clamp(f6, 1e-7, 10.0); f7 = clamp(f7, 1e-7, 10.0);
  f8 = clamp(f8, 1e-7, 10.0);

  outA = vec4(f0, f1, f2, f3);
  outB = vec4(f4, f5, f6, f7);
  outC = vec4(f8, 0.0, 0.0, 0.0);
}`;

const FS_MACRO = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uFA, uFB, uFC, uGeom;
uniform vec2 uRes;
layout(location=0) out vec4 outMacro;
void main(){
  ivec2 c = ivec2(gl_FragCoord.xy);
  vec4 A = texelFetch(uFA, c, 0);
  vec4 B = texelFetch(uFB, c, 0);
  vec4 C = texelFetch(uFC, c, 0);
  vec4 G = texelFetch(uGeom, c, 0);
  float rho = A.r+A.g+A.b+A.a + B.r+B.g+B.b+B.a + C.r;
  float ux = (A.g - A.a + B.g - B.b - B.a + C.r) / max(rho, 1e-6);
  float uy = (A.b - B.r + B.g + B.b - B.a - C.r) / max(rho, 1e-6);
  outMacro = vec4(ux, uy, rho, G.r);
}`;

const FS_RENDER = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uMacro, uGeom, uStreamTex;
uniform vec2 uRes;
uniform float uField;
uniform float uVmin, uVmax;
uniform int uXform;
uniform float uLinthresh;
uniform float uShowStream;
uniform float uUScale, uPScale;
uniform float uTime;

out vec4 outCol;

vec3 jet(float t) {
  t = clamp(t, 0.0, 1.0);
  float r = clamp(1.5 - abs(4.0 * t - 3.0), 0.0, 1.0);
  float g = clamp(1.5 - abs(4.0 * t - 2.0), 0.0, 1.0);
  float b = clamp(1.5 - abs(4.0 * t - 1.0), 0.0, 1.0);
  return vec3(r, g, b);
}
vec3 rwb(float t) {
  t = clamp(t, 0.0, 1.0);
  if (t < 0.5) {
    float s = t * 2.0;
    return mix(vec3(0.02, 0.20, 0.72), vec3(0.95, 0.95, 0.95), s);
  }
  float s = (t - 0.5) * 2.0;
  return mix(vec3(0.95, 0.95, 0.95), vec3(0.76, 0.02, 0.02), s);
}

float normalize01(float raw) {
  float EPS = 1e-12;
  if (uXform == 1) {
    float lr = log(max(raw, EPS));
    float lmn = log(max(uVmin, EPS));
    float lmx = log(max(uVmax, EPS));
    return (lr - lmn) / max(lmx - lmn, 1e-9);
  }
  if (uXform == 2) {
    float lt = max(uLinthresh, EPS);
    float sl = sign(raw) * log(1.0 + abs(raw) / lt);
    float mn = sign(uVmin) * log(1.0 + abs(uVmin) / lt);
    float mx = sign(uVmax) * log(1.0 + abs(uVmax) / lt);
    return (sl - mn) / max(mx - mn, 1e-9);
  }
  return (raw - uVmin) / max(uVmax - uVmin, 1e-9);
}

void main(){
  vec4 m = texture(uMacro, vUV);
  float ux = m.r, uy = m.g, rho = m.b, ct = m.a;

  vec3 col;
  float raw;
  if (uField < 0.5) {
    raw = length(vec2(ux, uy)) * uUScale;
    col = jet(normalize01(raw));
  } else if (uField < 1.5) {
    raw = (rho - 1.0) * uPScale;
    col = jet(normalize01(raw));
  } else {
    vec2 ts = 1.0 / uRes;
    vec4 mR = texture(uMacro, vUV + vec2( ts.x, 0.0));
    vec4 mL = texture(uMacro, vUV + vec2(-ts.x, 0.0));
    vec4 mT = texture(uMacro, vUV + vec2(0.0,  ts.y));
    vec4 mB = texture(uMacro, vUV + vec2(0.0, -ts.y));
    float vort_lat = (mR.g - mL.g) * 0.5 - (mT.r - mB.r) * 0.5;
    float vort = vort_lat * uUScale * uRes.x;
    col = rwb(normalize01(vort));
  }

  if (ct > 0.5 && ct < 1.5) {
    col = vec3(0.45, 0.52, 0.62);
  } else if (ct > 1.5 && ct < 2.5) {
    col = mix(col, vec3(0.31, 0.80, 0.77), 0.28);
  } else if (ct > 2.5 && ct < 3.5) {
    col = mix(col, vec3(0.96, 0.65, 0.14), 0.28);
  }

  if (uShowStream > 0.5 && ct < 0.5) {
    float sn = texture(uStreamTex, vUV).r;
    sn = smoothstep(0.12, 0.88, sn);
    col = mix(col, vec3(1.0), sn * 0.75);
  }

  outCol = vec4(col, 1.0);
}`;

const FS_STREAM_ADVECT = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uNoise;
uniform sampler2D uMacro;
uniform vec2 uRes;
uniform float uDt;
uniform float uTime;
uniform float uInjectProb;
uniform float uDecay;
out vec4 outC;

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

void main() {
  vec4 m = texture(uMacro, vUV);
  vec2 v = vec2(m.r, m.g);
  float ct = m.a;

  if (ct > 0.5 && ct < 1.5) {
    outC = vec4(0.0);
    return;
  }

  vec2 prev = vUV - v * uDt;
  vec4 adv = texture(uNoise, prev) * uDecay;

  float r = hash12(vUV * uRes + vec2(uTime * 123.4, uTime * 456.7));
  if (r < uInjectProb) adv = vec4(1.0);

  outC = adv;
}`;

const FS_PART = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uParticles;
uniform sampler2D uMacro;
uniform sampler2D uGeom;
uniform vec2 uRes;
uniform vec2 uInletYRange;
uniform float uDt;
uniform float uTime;
out vec4 outP;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

vec2 findInletSpawn(float seed) {
  float yr = hash(vec2(seed * 1.1, uTime * 0.173));
  float xr = hash(vec2(seed * 1.7 + 0.3, uTime * 0.089));
  float yMin = uInletYRange.x / uRes.y;
  float yMax = uInletYRange.y / uRes.y;
  float margin = 0.02;
  float y = mix(yMin + margin * (yMax - yMin),
                yMax - margin * (yMax - yMin), yr);
  float x = 0.005 + xr * 0.015;
  return vec2(x, y);
}

void main() {
  vec4 p = texture(uParticles, vUV);
  vec2 pos = p.xy;
  float age = p.z;
  float state_ = p.w;

  if (state_ < 0.5) {
    age -= 1.0 / 60.0;
    if (age <= 0.0) {
      float newSeed = hash(vec2(vUV.x * 71.3 + vUV.y * 29.7, uTime * 1.11));
      vec2 newPos = findInletSpawn(newSeed);
      outP = vec4(newPos, 0.0, 0.5 + newSeed * 0.5);
      return;
    }
    outP = vec4(pos, age, 0.0);
    return;
  }

  vec4 m = texture(uMacro, pos);
  vec2 v = vec2(m.r, m.g);
  float ct = m.a;

  bool destroy = false;
  if (pos.x < 0.001 || pos.x > 0.999 || pos.y < 0.001 || pos.y > 0.999) destroy = true;
  if (ct > 0.5 && ct < 1.5) destroy = true;
  if (ct > 2.5 && ct < 3.5) destroy = true;
  if (length(v) < 1e-5 && age > 1.2) destroy = true;

  if (destroy) {
    float delay = 0.1 + hash(vec2(state_, uTime)) * 0.7;
    outP = vec4(vec2(-10.0), delay, 0.0);
    return;
  }

  pos += v * uDt;
  age += 1.0 / 60.0;
  outP = vec4(pos, age, state_);
}`;

const VS_PART = `#version 300 es
uniform sampler2D uParticles;
uniform float uNside;
out float vAge;
out float vActive;
out float vSpeed;
uniform sampler2D uMacro;
void main() {
  int id = gl_VertexID;
  int nx = int(uNside);
  int x = id - (id / nx) * nx;
  int y = id / nx;
  vec2 uvPart = (vec2(float(x), float(y)) + 0.5) / uNside;
  vec4 p = texture(uParticles, uvPart);
  vec2 pos = p.xy;
  vAge = p.z;
  vActive = p.w;
  if (vActive < 0.5) {
    gl_Position = vec4(-10.0, -10.0, 0.0, 1.0);
    gl_PointSize = 0.0;
    vSpeed = 0.0;
    return;
  }
  vec4 m = texture(uMacro, pos);
  vSpeed = length(vec2(m.r, m.g));
  gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
  gl_PointSize = 2.6;
}`;
const FS_PARTREND = `#version 300 es
precision highp float;
in float vAge;
in float vActive;
in float vSpeed;
out vec4 outC;
void main() {
  if (vActive < 0.5) { discard; }
  vec2 d = gl_PointCoord - 0.5;
  float r2 = dot(d, d);
  if (r2 > 0.25) discard;
  float fade = smoothstep(0.0, 0.1, vAge);
  float soft = smoothstep(0.25, 0.0, r2);
  float spdFade = smoothstep(0.0, 5e-4, vSpeed) * 0.5 + 0.5;
  float bright = 0.95 * fade * soft * spdFade;
  outC = vec4(vec3(bright * 0.88, bright * 1.0, bright * 1.0), 1.0);
}`;

/* ---------- Programs ---------- */
const progInit   = link(VS, FS_INIT);
const progStep   = link(VS, FS_STEP);
const progMacro  = link(VS, FS_MACRO);
const progRender = link(VS, FS_RENDER);
const progPart   = link(VS, FS_PART);
const progPartRender = link(VS_PART, FS_PARTREND);
const progStream = link(VS, FS_STREAM_ADVECT);

const uInit = cacheUniforms(progInit, ['uGeom', 'uUlat']);
const uStep = cacheUniforms(progStep, [
  'uFA','uFB','uFC','uGeom','uRes','uTau','uGx','uGy','uUlat',
  'uRhoIn','uRhoOut','uNonNewt','uInletType','uOutletType','uInletProfile','uInletYRange',
  'uMu0','uMuI','uLam','uN','uA']);
const uMacro = cacheUniforms(progMacro, ['uFA','uFB','uFC','uGeom','uRes']);
const uRender = cacheUniforms(progRender, [
  'uMacro','uGeom','uStreamTex','uRes','uField','uVmin','uVmax',
  'uXform','uLinthresh','uShowStream','uUScale','uPScale','uTime']);
const uPart = cacheUniforms(progPart, ['uParticles','uMacro','uGeom','uRes','uDt','uTime','uInletYRange']);
const uPartR = cacheUniforms(progPartRender, ['uParticles','uMacro','uNside']);
const uStream = cacheUniforms(progStream, ['uNoise','uMacro','uRes','uDt','uTime','uInjectProb','uDecay']);

/* ---------- Textures & FBOs ---------- */
function mkTex(w, h, filter) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}
function mkFBO(texArr) {
  const fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  for (let i = 0; i < texArr.length; ++i) {
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, texArr[i], 0);
  }
  gl.drawBuffers(texArr.map((_, i) => gl.COLOR_ATTACHMENT0 + i));
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
    console.warn('FBO incomplete');
  return fb;
}

let fA, fB, fC, fboStep, texGeom, texMacro, fboMacro;
let texPart, fboPart;
let texStream, fboStream;
let macroReadBuf = null;
let pp = 0, pPart = 0, pStream = 0, stepN = 0, paused = false;
const linearFilt = extLin ? gl.LINEAR : gl.NEAREST;

const PSIDE = 60;
const NPART = PSIDE * PSIDE;
const STREAM_W = 900, STREAM_H = 360;
let streamTexAllocated = false;

function allocStreamTextures() {
  if (streamTexAllocated) return;
  texStream = [mkTex(STREAM_W, STREAM_H, linearFilt), mkTex(STREAM_W, STREAM_H, linearFilt)];
  fboStream = [mkFBO([texStream[0]]), mkFBO([texStream[1]])];
  for (const f of fboStream) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    gl.viewport(0, 0, STREAM_W, STREAM_H);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }
  streamTexAllocated = true;
}

function allocTextures() {
  fA = [mkTex(NX, NY, gl.NEAREST), mkTex(NX, NY, gl.NEAREST)];
  fB = [mkTex(NX, NY, gl.NEAREST), mkTex(NX, NY, gl.NEAREST)];
  fC = [mkTex(NX, NY, gl.NEAREST), mkTex(NX, NY, gl.NEAREST)];
  fboStep = [mkFBO([fA[0], fB[0], fC[0]]), mkFBO([fA[1], fB[1], fC[1]])];
  texGeom = mkTex(NX, NY, gl.NEAREST);
  texMacro = mkTex(NX, NY, linearFilt);
  fboMacro = mkFBO([texMacro]);
  texPart = [mkTex(PSIDE, PSIDE, gl.NEAREST), mkTex(PSIDE, PSIDE, gl.NEAREST)];
  fboPart = [mkFBO([texPart[0]]), mkFBO([texPart[1]])];
  seedParticles();
  macroReadBuf = new Float32Array(NX * NY * 4);
  allocStreamTextures();
}
function seedParticles() {
  const init = new Float32Array(PSIDE * PSIDE * 4);
  for (let i = 0; i < NPART; ++i) {
    init[i * 4    ] = -10.0;
    init[i * 4 + 1] = -10.0;
    init[i * 4 + 2] = Math.random() * 2.0;
    init[i * 4 + 3] = 0.0;
  }
  for (const t of texPart) {
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, PSIDE, PSIDE, 0, gl.RGBA, gl.FLOAT, init);
  }
  pPart = 0;
}
function clearStream() {
  if (!streamTexAllocated) return;
  for (const f of fboStream) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    gl.viewport(0, 0, STREAM_W, STREAM_H);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }
  pStream = 0;
}

/* ---------- Geometry ---------- */
function setCell(x, y, t) {
  if (x < 0 || x >= NX || y < 0 || y >= NY) return;
  geomData[(y * NX + x) * 4] = t;
}
function getCell(x, y) {
  if (x < 0 || x >= NX || y < 0 || y >= NY) return -1;
  return geomData[(y * NX + x) * 4];
}
function uploadGeomFull() {
  gl.bindTexture(gl.TEXTURE_2D, texGeom);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, NX, NY, 0, gl.RGBA, gl.FLOAT, geomData);
}
function uploadGeomPatch(x0, y0, x1, y1) {
  x0 = Math.max(0, x0); y0 = Math.max(0, y0);
  x1 = Math.min(NX - 1, x1); y1 = Math.min(NY - 1, y1);
  const pw = x1 - x0 + 1, ph = y1 - y0 + 1;
  if (pw <= 0 || ph <= 0) return;
  const buf = new Float32Array(pw * ph * 4);
  for (let y = 0; y < ph; ++y) {
    for (let x = 0; x < pw; ++x) {
      const srcI = ((y0 + y) * NX + (x0 + x)) * 4;
      const dstI = (y * pw + x) * 4;
      buf[dstI] = geomData[srcI];
    }
  }
  gl.bindTexture(gl.TEXTURE_2D, texGeom);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, x0, y0, pw, ph, gl.RGBA, gl.FLOAT, buf);
}
function clearGeom() { geomData.fill(0); }

function baseChannel() {
  clearGeom();
  for (let x = 0; x < NX; ++x) {
    setCell(x, 0, CELL_WALL); setCell(x, 1, CELL_WALL);
    setCell(x, NY - 1, CELL_WALL); setCell(x, NY - 2, CELL_WALL);
  }
}
function setInletsOutlets() {
  for (let y = 0; y < NY; ++y) {
    if (getCell(2, y) === CELL_FLUID) setCell(0, y, CELL_INLET);
    else                              setCell(0, y, CELL_WALL);
    if (getCell(NX - 3, y) === CELL_FLUID) setCell(NX - 1, y, CELL_OUTLET);
    else                                   setCell(NX - 1, y, CELL_WALL);
  }
}
function drawDisk(cx, cy, r, type) {
  const r2 = r * r;
  for (let y = Math.max(0, cy - r); y <= Math.min(NY - 1, cy + r); ++y)
    for (let x = Math.max(0, cx - r); x <= Math.min(NX - 1, cx + r); ++x) {
      const dx = x - cx, dy = y - cy;
      if (dx*dx + dy*dy <= r2) setCell(x, y, type);
    }
}
function drawBox(x0, y0, x1, y1, type) {
  for (let y = y0; y <= y1; ++y)
    for (let x = x0; x <= x1; ++x) setCell(x, y, type);
}
function fillAllWalls() {
  for (let x = 0; x < NX; ++x)
    for (let y = 0; y < NY; ++y) setCell(x, y, CELL_WALL);
}

function loadPreset(name) {
  switch (name) {
    case 'channel': baseChannel(); break;
    case 'sphere':
      baseChannel();
      drawDisk(Math.floor(NX * 0.32), Math.floor(NY * 0.5), Math.floor(NY * 0.18), CELL_WALL);
      break;
    case 'cube': {
      baseChannel();
      const cx = Math.floor(NX * 0.32), cy = Math.floor(NY * 0.5), s = Math.floor(NY * 0.30);
      drawBox(cx - Math.floor(s / 2), cy - Math.floor(s / 2), cx + Math.floor(s / 2), cy + Math.floor(s / 2), CELL_WALL);
      break;
    }
    case 'stenosis': {
      baseChannel();
      const xc = Math.floor(NX * 0.42), hw = Math.floor(NY * 0.5), occl = 0.50;
      for (let dx = -hw; dx <= hw; ++dx) {
        const x = xc + dx;
        if (x < 2 || x >= NX - 2) continue;
        const prof = 0.5 * (1.0 + Math.cos(Math.PI * dx / hw));
        const th = Math.floor(prof * NY * occl * 0.5);
        for (let k = 0; k < th; ++k) {
          setCell(x, 2 + k, CELL_WALL);
          setCell(x, NY - 3 - k, CELL_WALL);
        }
      }
      break;
    }
    case 'plaque': {
      baseChannel();
      const xc = Math.floor(NX * 0.40), hw = Math.floor(NY * 0.65), peak = Math.floor(NY * 0.40);
      for (let dx = -hw; dx <= hw; ++dx) {
        const x = xc + dx;
        if (x < 2 || x >= NX - 2) continue;
        let prof;
        if (dx < 0) prof = Math.pow(1.0 - Math.abs(dx) / hw, 1.2);
        else prof = Math.pow(Math.max(0, 1.0 - dx / hw), 2.2);
        const th = Math.floor(prof * peak);
        for (let k = 0; k < th; ++k) setCell(x, 2 + k, CELL_WALL);
      }
      break;
    }
    case 'aneurysm': {
      clearGeom();
      const topEdge = Math.floor(NY * 0.32), botEdge = Math.floor(NY * 0.82);
      for (let x = 0; x < NX; ++x)
        for (let y = 0; y < NY; ++y) {
          if (y < topEdge || y > botEdge) setCell(x, y, CELL_WALL);
          else setCell(x, y, CELL_FLUID);
        }
      const xc = Math.floor(NX * 0.50), rAn = Math.floor(NY * 0.22);
      for (let x = xc - rAn; x <= xc + rAn; ++x)
        for (let y = Math.max(0, topEdge - rAn); y <= topEdge; ++y) {
          const dx = x - xc, dy = y - topEdge;
          if (dx * dx + dy * dy <= rAn * rAn) setCell(x, y, CELL_FLUID);
        }
      break;
    }
    case 'bifurcation': {
      fillAllWalls();
      const splitX = Math.floor(NX * 0.38);
      const parentHalf = Math.floor(NY * 0.14), dauHalf = Math.floor(NY * 0.09);
      const parentY = Math.floor(NY * 0.50);
      const dauTopY = Math.floor(NY * 0.25), dauBotY = Math.floor(NY * 0.75);
      for (let x = 0; x < NX; ++x) {
        if (x < splitX) {
          for (let y = parentY - parentHalf; y <= parentY + parentHalf; ++y) setCell(x, y, CELL_FLUID);
        } else {
          const t = (x - splitX) / Math.max(1, NX - 1 - splitX);
          const topC = parentY + t * (dauTopY - parentY);
          const botC = parentY + t * (dauBotY - parentY);
          const hw = parentHalf + t * (dauHalf - parentHalf);
          for (let y = Math.floor(topC - hw); y <= Math.ceil(topC + hw); ++y) setCell(x, y, CELL_FLUID);
          for (let y = Math.floor(botC - hw); y <= Math.ceil(botC + hw); ++y) setCell(x, y, CELL_FLUID);
        }
      }
      break;
    }
    case 'trifurcation': {
      fillAllWalls();
      const splitX = Math.floor(NX * 0.32);
      const parentHalf = Math.floor(NY * 0.16), dauHalf = Math.floor(NY * 0.07);
      const parentY = Math.floor(NY * 0.50);
      const centres = [0.18, 0.50, 0.82].map(f => Math.floor(NY * f));
      for (let x = 0; x < NX; ++x) {
        if (x < splitX) {
          for (let y = parentY - parentHalf; y <= parentY + parentHalf; ++y) setCell(x, y, CELL_FLUID);
        } else {
          const t = (x - splitX) / Math.max(1, NX - 1 - splitX);
          const hw = parentHalf + t * (dauHalf - parentHalf);
          for (const c0 of centres) {
            const cy = parentY + t * (c0 - parentY);
            for (let y = Math.floor(cy - hw); y <= Math.ceil(cy + hw); ++y) setCell(x, y, CELL_FLUID);
          }
        }
      }
      break;
    }
  }
  setInletsOutlets();
  computeInletYRange();
  uploadGeomFull();
  resetSim();
  resetSteady();
  forceDiagnosticsUpdate();
}

function computeInletYRange() {
  let yMin = NY, yMax = -1;
  for (let y = 0; y < NY; ++y) {
    if (getCell(0, y) === CELL_INLET) {
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
    }
  }
  if (yMax < yMin) { yMin = 0; yMax = NY - 1; }
  state.inletYMin = yMin;
  state.inletYMax = yMax;
}

/* ---------- Parameter updates ---------- */
function updateLatticeParams() {
  DX = L_PHYS / NY;
  const ulatRaw = U_LAT_MAX * (0.3 / Math.max(state.uin, 0.05));
  state.uLat = Math.max(U_LAT_MIN, Math.min(U_LAT_MAX, ulatRaw));
  const dt = state.uLat * DX / Math.max(state.uin, 1e-4);
  const mu_eff = state.muN;
  const nu = mu_eff / RHO_BLOOD;
  const nu_lat = nu * dt / (DX * DX);
  state.tau = Math.max(0.55, Math.min(2.5, 0.5 + 3 * nu_lat));
  state.dt = dt;
  state.uScale = DX / dt;
  state.pScale = RHO_BLOOD * (DX / dt) * (DX / dt) / 3.0;
  state.Re = state.uin * L_PHYS / nu;
}

/* ---------- Simulation passes ---------- */
function resetSim() {
  stepN = 0;
  updateLatticeParams();
  gl.viewport(0, 0, NX, NY);
  bindQuad(progInit);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fboStep[0]);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texGeom);
  gl.uniform1i(uInit.uGeom, 0);
  gl.uniform1f(uInit.uUlat, state.uLat);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fboStep[1]);
  gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fboMacro);
  gl.clear(gl.COLOR_BUFFER_BIT);
  pp = 0;
}

function setStepUniforms() {
  gl.useProgram(progStep);
  gl.uniform2f(uStep.uRes, NX, NY);
  gl.uniform1f(uStep.uTau, state.tau);
  gl.uniform1f(uStep.uUlat, state.uLat);
  let gx = 0, gy = 0;
  if (state.gravity) {
    const g = 9.81;
    const gLat = g * state.dt * state.dt / DX;
    gy = -gLat;
  }
  gl.uniform1f(uStep.uGx, gx);
  gl.uniform1f(uStep.uGy, gy);

  const P_MMHG_TO_PA = 133.322;
  let rhoIn = 1.0, rhoOut = 1.0;
  pInClamped = false;
  pOutClamped = false;
  const bothP = state.inletType === 'pressure' && state.outletType === 'pressure';
  const inP   = state.inletType === 'pressure';
  const outP  = state.outletType === 'pressure';
  if (bothP) {
    const dP = state.pIn_mmHg - state.pOut_mmHg;
    const tgt = (dP * P_MMHG_TO_PA) / Math.max(state.pScale, 1e-9);
    const dR  = Math.max(-0.10, Math.min(0.10, tgt));
    if (Math.abs(dR - tgt) > 1e-4) { pInClamped = true; pOutClamped = true; }
    rhoIn = 1.0 + dR;
    rhoOut = 1.0;
  } else if (inP) {
    const tgt = (state.pIn_mmHg * P_MMHG_TO_PA) / Math.max(state.pScale, 1e-9);
    const dR  = Math.max(-0.10, Math.min(0.10, tgt));
    if (Math.abs(dR - tgt) > 1e-4) pInClamped = true;
    rhoIn = 1.0 + dR;
  } else if (outP) {
    rhoOut = 1.0;
  }
  gl.uniform1f(uStep.uRhoIn, rhoIn);
  gl.uniform1f(uStep.uRhoOut, rhoOut);

  gl.uniform1i(uStep.uNonNewt, state.viscModel === 'carreau' ? 1 : 0);
  gl.uniform1i(uStep.uInletType, state.inletType === 'velocity' ? 0 : 1);
  gl.uniform1i(uStep.uOutletType, state.outletType === 'zerograd' ? 0 : 1);
  gl.uniform1i(uStep.uInletProfile, state.inletProfile === 'parabolic' ? 1 : 0);
  gl.uniform2f(uStep.uInletYRange, state.inletYMin, state.inletYMax);

  const mulat = muP => muP / RHO_BLOOD * state.dt / (DX * DX);
  gl.uniform1f(uStep.uMu0, mulat(state.cy_mu0));
  gl.uniform1f(uStep.uMuI, mulat(state.cy_muI));
  gl.uniform1f(uStep.uLam, state.cy_lam / state.dt);
  gl.uniform1f(uStep.uN, state.cy_n);
  gl.uniform1f(uStep.uA, state.cy_a);
}
function stepOnce() {
  bindQuad(progStep);
  gl.viewport(0, 0, NX, NY);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fboStep[1 - pp]);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, fA[pp]);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, fB[pp]);
  gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, fC[pp]);
  gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, texGeom);
  gl.uniform1i(uStep.uFA, 0);
  gl.uniform1i(uStep.uFB, 1);
  gl.uniform1i(uStep.uFC, 2);
  gl.uniform1i(uStep.uGeom, 3);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  pp = 1 - pp;
  stepN++;
}
function computeMacro() {
  bindQuad(progMacro);
  gl.viewport(0, 0, NX, NY);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fboMacro);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, fA[pp]);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, fB[pp]);
  gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, fC[pp]);
  gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, texGeom);
  gl.uniform1i(uMacro.uFA, 0);
  gl.uniform1i(uMacro.uFB, 1);
  gl.uniform1i(uMacro.uFC, 2);
  gl.uniform1i(uMacro.uGeom, 3);
  gl.uniform2f(uMacro.uRes, NX, NY);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
}

let vmin = 0, vmax = 1;
let trueMin = 0, trueMax = 1;
let p1 = 0, p5 = 0, p95 = 1, p99 = 1;

const DIAG_EVERY_STEADY = 120;
const DIAG_EVERY_STARTUP = 12;
const STARTUP_DIAGS = 20;
let diagFrames = 0;
let diagsRun = 0;
let keHistory = [];
const KE_HISTORY_LEN = 6;
let steadyState = false;

function resetSteady() {
  steadyState = false;
  keHistory = [];
  diagsRun = 0;
  diagFrames = 0;
}
function diagInterval() {
  return diagsRun < STARTUP_DIAGS ? DIAG_EVERY_STARTUP : DIAG_EVERY_STEADY;
}

function forceDiagnosticsUpdate() {
  diagFrames = diagInterval();
  diagsRun = 0;
}

function readMacroField() {
  gl.bindFramebuffer(gl.FRAMEBUFFER, fboMacro);
  gl.readPixels(0, 0, NX, NY, gl.RGBA, gl.FLOAT, macroReadBuf);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

function runDiagnostics() {
  readMacroField();
  const fluidVals = [];
  let ke = 0;
  for (let j = 0; j < NY; ++j) {
    for (let i = 0; i < NX; ++i) {
      const k = (j * NX + i) * 4;
      const ux = macroReadBuf[k], uy = macroReadBuf[k + 1];
      const rho = macroReadBuf[k + 2], ct = macroReadBuf[k + 3];
      if (ct < 0.5) {
        const sp2 = ux * ux + uy * uy;
        ke += sp2;
        let val;
        if (state.field === 'vel') {
          val = Math.sqrt(sp2) * state.uScale;
        } else if (state.field === 'pres') {
          val = (rho - 1.0) * state.pScale;
        } else {
          const iR = Math.min(NX - 1, i + 1), iL = Math.max(0, i - 1);
          const jT = Math.min(NY - 1, j + 1), jB = Math.max(0, j - 1);
          const kR = (j * NX + iR) * 4, kL = (j * NX + iL) * 4;
          const kT = (jT * NX + i) * 4, kB = (jB * NX + i) * 4;
          const dvdx = (macroReadBuf[kR + 1] - macroReadBuf[kL + 1]) * 0.5;
          const dudy = (macroReadBuf[kT] - macroReadBuf[kB]) * 0.5;
          val = (dvdx - dudy) * state.uScale * NX;
        }
        fluidVals.push(val);
      }
    }
  }

  if (fluidVals.length < 10) return;

  fluidVals.sort((a, b) => a - b);
  const n = fluidVals.length;
  trueMin = fluidVals[0];
  trueMax = fluidVals[n - 1];
  p1  = fluidVals[Math.max(0, Math.floor(n * 0.01))];
  p5  = fluidVals[Math.max(0, Math.floor(n * 0.05))];
  p95 = fluidVals[Math.min(n - 1, Math.floor(n * 0.95))];
  p99 = fluidVals[Math.min(n - 1, Math.floor(n * 0.99))];
  applyScaleMode();

  keHistory.push(ke);
  if (keHistory.length > KE_HISTORY_LEN) keHistory.shift();
  if (state.detectSteady && keHistory.length === KE_HISTORY_LEN) {
    const mean = keHistory.reduce((s, v) => s + v, 0) / keHistory.length;
    let maxDev = 0;
    for (const v of keHistory) maxDev = Math.max(maxDev, Math.abs(v - mean));
    const rel = maxDev / Math.max(mean, 1e-9);
    if (rel < 5e-3 && stepN > 200) {
      steadyState = true;
      paused = true;
      $('btnPlay').textContent = '▶ Resume';
      const stat = $('simStat');
      stat.textContent = '● steady state (solver stopped)';
      stat.className = 'sim-run paused';
      stat.style.color = '#4ade80';
    }
  }
  diagsRun++;
}

function applyScaleMode() {
  if (state.scaleMode === 'manual') {
    vmin = state.manualMin;
    vmax = state.manualMax;
    return;
  }
  let lo, hi;
  if (state.scaleMode === 'p1-99')     { lo = p1;       hi = p99;     }
  else if (state.scaleMode === 'true') { lo = trueMin;  hi = trueMax; }
  else                                 { lo = p5;       hi = p95;     }

  if (state.field === 'vel') {
    const peak = Math.max(trueMax, hi, state.uin * 1.5, 1e-3);
    vmax = peak;
    if (state.xform === 'log') {
      vmin = Math.max(vmax * 1e-4, 1e-6);
    } else {
      vmin = 0;
    }
  } else if (state.field === 'pres') {
    let mn = lo, mx = hi;
    if (mn < 0 && mx > 0) {
      const m = Math.max(Math.abs(mn), Math.abs(mx));
      mn = -m; mx = m;
    }
    if (Math.abs(mx - mn) < 1e-6) mx = mn + 1;
    vmin = mn; vmax = mx;
  } else {
    const m = Math.max(Math.abs(lo), Math.abs(hi), 1e-3);
    vmin = -m; vmax = m;
  }
}

/* ---------- Particle / Stream ---------- */
function advanceParticles() {
  bindQuad(progPart);
  gl.viewport(0, 0, PSIDE, PSIDE);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fboPart[1 - pPart]);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texPart[pPart]);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, texMacro);
  gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, texGeom);
  gl.uniform1i(uPart.uParticles, 0);
  gl.uniform1i(uPart.uMacro, 1);
  gl.uniform1i(uPart.uGeom, 2);
  gl.uniform2f(uPart.uRes, NX, NY);
  gl.uniform2f(uPart.uInletYRange, state.inletYMin, state.inletYMax);
  const visSpeed = 4.0 * state.stepsPerFrame;
  gl.uniform1f(uPart.uDt, visSpeed / NX);
  gl.uniform1f(uPart.uTime, performance.now() * 0.001);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  pPart = 1 - pPart;
}
function advanceStream() {
  bindQuad(progStream);
  gl.viewport(0, 0, STREAM_W, STREAM_H);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fboStream[1 - pStream]);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texStream[pStream]);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, texMacro);
  gl.uniform1i(uStream.uNoise, 0);
  gl.uniform1i(uStream.uMacro, 1);
  gl.uniform2f(uStream.uRes, STREAM_W, STREAM_H);
  gl.uniform1f(uStream.uDt, 30.0 / NX);
  gl.uniform1f(uStream.uTime, performance.now() * 0.001);
  gl.uniform1f(uStream.uInjectProb, 0.007);
  gl.uniform1f(uStream.uDecay, 0.982);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  pStream = 1 - pStream;
}
function renderParticles() {
  gl.useProgram(progPartRender);
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
  const loc = gl.getAttribLocation(progPartRender, 'aP');
  if (loc >= 0) {
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  }
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texPart[pPart]);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, texMacro);
  gl.uniform1i(uPartR.uParticles, 0);
  gl.uniform1i(uPartR.uMacro, 1);
  gl.uniform1f(uPartR.uNside, PSIDE);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
  gl.drawArrays(gl.POINTS, 0, NPART);
  gl.disable(gl.BLEND);
}

function renderToCanvas(time) {
  bindQuad(progRender);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texMacro);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, texGeom);
  gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, texStream[pStream]);
  gl.uniform1i(uRender.uMacro, 0);
  gl.uniform1i(uRender.uGeom, 1);
  gl.uniform1i(uRender.uStreamTex, 2);
  gl.uniform2f(uRender.uRes, NX, NY);
  const fieldIdx = state.field === 'vel' ? 0.0 : state.field === 'pres' ? 1.0 : 2.0;
  gl.uniform1f(uRender.uField, fieldIdx);
  gl.uniform1f(uRender.uVmin, vmin);
  gl.uniform1f(uRender.uVmax, vmax);
  let xformCode = 0;
  if (state.xform === 'log') {
    xformCode = (state.field === 'vel') ? 1 : 2;
  }
  gl.uniform1i(uRender.uXform, xformCode);
  const linthresh = Math.max(1e-6, (Math.abs(vmax) + Math.abs(vmin)) * 0.01);
  gl.uniform1f(uRender.uLinthresh, linthresh);
  gl.uniform1f(uRender.uShowStream, state.showStream ? 1.0 : 0.0);
  gl.uniform1f(uRender.uUScale, state.uScale);
  gl.uniform1f(uRender.uPScale, state.pScale);
  gl.uniform1f(uRender.uTime, time * 0.001);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
}

/* ---------- Colorbar ---------- */
function drawColorbar() {
  const w = cbCv.width, h = cbCv.height;
  cbCtx.clearRect(0, 0, w, h);
  const grad = cbCtx.createLinearGradient(0, 0, w, 0);
  if (state.field === 'vort') {
    for (let s = 0; s <= 20; ++s) {
      const t = s / 20;
      let r, g, b;
      if (t < 0.5) {
        const u = t * 2;
        r = Math.round((0.02 + 0.93 * u) * 255);
        g = Math.round((0.20 + 0.75 * u) * 255);
        b = Math.round((0.72 + 0.23 * u) * 255);
      } else {
        const u = (t - 0.5) * 2;
        r = Math.round((0.95 - 0.19 * u) * 255);
        g = Math.round((0.95 - 0.93 * u) * 255);
        b = Math.round((0.95 - 0.93 * u) * 255);
      }
      grad.addColorStop(t, `rgb(${r},${g},${b})`);
    }
  } else {
    for (let s = 0; s <= 20; ++s) {
      const t = s / 20;
      const r = Math.round(Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 3))) * 255);
      const g = Math.round(Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 2))) * 255);
      const b = Math.round(Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 1))) * 255);
      grad.addColorStop(t, `rgb(${r},${g},${b})`);
    }
  }
  cbCtx.fillStyle = grad;
  cbCtx.fillRect(0, 0, w, h);
}

/* ---------- Status UI ---------- */
const fmt2 = v => {
  if (!isFinite(v)) return '—';
  return v.toFixed(2);
};

let fpsN = 0, fpsT = 0, fpsV = 0;
let stepAccum = 0;
function updateStats() {
  $('sRe').textContent = state.Re < 1 ? fmt2(state.Re) : Math.round(state.Re);
  $('sTau').textContent = fmt2(state.tau);
  $('sSt').textContent = stepN;
  $('sFPS').textContent = fpsV;

  if (keHistory.length >= 2) {
    const mean = keHistory.reduce((s, v) => s + v, 0) / keHistory.length;
    let maxDev = 0;
    for (const v of keHistory) maxDev = Math.max(maxDev, Math.abs(v - mean));
    const rel = maxDev / Math.max(mean, 1e-9);
    $('sRes').textContent = (rel * 100).toFixed(2) + '%';
  } else {
    $('sRes').textContent = '—';
  }

  const lbl = $('cbLo'), hbl = $('cbHi');
  const tinyTol = 1e-3 * Math.max(Math.abs(vmax - vmin), 1e-6);
  const showTrue = state.scaleMode !== 'true' && state.scaleMode !== 'manual' &&
                   (Math.abs(trueMin - vmin) > tinyTol || Math.abs(trueMax - vmax) > tinyTol);
  if (showTrue) {
    lbl.textContent = fmt2(vmin) + '  (min ' + fmt2(trueMin) + ')';
    hbl.textContent = fmt2(vmax) + '  (max ' + fmt2(trueMax) + ')';
  } else {
    lbl.textContent = fmt2(vmin);
    hbl.textContent = fmt2(vmax);
  }
  const unit = state.field === 'vel' ? 'm/s' : state.field === 'pres' ? 'Pa' : '1/s';
  $('cbU').textContent = unit + (state.xform === 'log' ? '  (log)' : '');
  $('vReHint').textContent = `Re ≈ ${Math.round(state.Re)}`;
  $('pClampHint').style.display = pOutClamped ? 'inline' : 'none';
  $('pInClampHint').style.display = pInClamped ? 'inline' : 'none';
}

/* ---------- Main Loop ---------- */
function frame(time) {
  if (!paused) {
    updateLatticeParams();
    setStepUniforms();
    for (let k = 0; k < state.stepsPerFrame; ++k) {
      stepOnce();
      stepAccum++;
    }
  }
  computeMacro();

  if (!paused) {
    diagFrames++;
    if (diagFrames >= diagInterval()) {
      diagFrames = 0;
      runDiagnostics();
    }
    if (state.showParticles) advanceParticles();
    if (state.showStream)    advanceStream();
  }

  renderToCanvas(time);
  if (state.showParticles) renderParticles();
  drawColorbar();

  fpsN++;
  if (time - fpsT > 500) {
    const dt = (time - fpsT) / 1000;
    fpsV = Math.round(fpsN / dt);
    const mcs = (NX * NY * stepAccum) / dt * 1e-6;
    $('sMcs').textContent = mcs < 1 ? mcs.toFixed(2) : mcs.toFixed(1);
    stepAccum = 0;
    fpsN = 0; fpsT = time;
    updateStats();
  }
  requestAnimationFrame(frame);
}

/* ---------- User Input ---------- */
let tool = 'none', drawing = false, brushR = 3, isErasing = false;
function setTool(t) {
  tool = t;
  ['tNone','tWall','tErase'].forEach(id => $(id).classList.remove('on'));
  if (t === 'none')  { $('tNone').classList.add('on'); $('dhint').classList.remove('on'); }
  if (t === 'wall')  { $('tWall').classList.add('on'); $('dhint').classList.add('on'); }
  if (t === 'erase') { $('tErase').classList.add('on'); $('dhint').classList.add('on'); }
}

function canvasToCell(e) {
  const r = canvas.getBoundingClientRect();
  const px = (e.clientX - r.left) / r.width;
  const py = (e.clientY - r.top) / r.height;
  return { i: Math.round(px * (NX - 1)), j: Math.round((1 - py) * (NY - 1)) };
}
function applyBrush(i, j, typ) {
  const r = brushR, r2 = r * r;
  let x0 = i - r, y0 = j - r, x1 = i + r, y1 = j + r;
  for (let dy = -r; dy <= r; ++dy)
    for (let dx = -r; dx <= r; ++dx) {
      if (dx*dx + dy*dy > r2 + 0.5) continue;
      const ci = i + dx, cj = j + dy;
      if (ci <= 1 || ci >= NX - 2 || cj <= 1 || cj >= NY - 2) continue;
      setCell(ci, cj, typ);
    }
  uploadGeomPatch(x0, y0, x1, y1);
  forceDiagnosticsUpdate();
}

canvas.addEventListener('mousedown', e => {
  if (tool === 'none') return;
  drawing = true;
  isErasing = (e.button === 2) || tool === 'erase';
  const {i, j} = canvasToCell(e);
  applyBrush(i, j, isErasing ? CELL_FLUID : CELL_WALL);
});
canvas.addEventListener('mousemove', e => {
  const r = canvas.getBoundingClientRect();
  const ind = $('bInd');
  if (tool !== 'none') {
    const pxPerCell = r.width / NX;
    const sz = brushR * 2 * pxPerCell;
    ind.style.display = 'block';
    ind.style.width = sz + 'px';
    ind.style.height = sz + 'px';
    ind.style.left = e.clientX + 'px';
    ind.style.top = e.clientY + 'px';
  } else {
    ind.style.display = 'none';
  }
  if (drawing && tool !== 'none') {
    const {i, j} = canvasToCell(e);
    applyBrush(i, j, isErasing ? CELL_FLUID : CELL_WALL);
  }
});
canvas.addEventListener('mouseup', () => { drawing = false; });
canvas.addEventListener('mouseleave', () => {
  drawing = false;
  $('bInd').style.display = 'none';
});
canvas.addEventListener('contextmenu', e => { e.preventDefault(); });
canvas.addEventListener('touchstart', e => {
  if (tool === 'none') return;
  e.preventDefault();
  drawing = true; isErasing = (tool === 'erase');
  const t = e.touches[0];
  const {i, j} = canvasToCell(t);
  applyBrush(i, j, isErasing ? CELL_FLUID : CELL_WALL);
}, { passive: false });
canvas.addEventListener('touchmove', e => {
  if (!drawing) return;
  e.preventDefault();
  const t = e.touches[0];
  const {i, j} = canvasToCell(t);
  applyBrush(i, j, isErasing ? CELL_FLUID : CELL_WALL);
}, { passive: false });
canvas.addEventListener('touchend', () => { drawing = false; });

/* ---------- UI Bindings ---------- */
$('slUin').addEventListener('input', e => {
  state.uin = parseFloat(e.target.value);
  $('vUin').textContent = state.uin.toFixed(2) + ' m/s';
  updateLatticeParams();
  resetSteady();
  updateStats();
  forceDiagnosticsUpdate();
});
$('slMu').addEventListener('input', e => {
  const mPas = parseFloat(e.target.value);
  state.muN = mPas * 1e-3;
  $('vMu').textContent = mPas.toFixed(2) + ' mPa·s';
  updateLatticeParams();
  resetSteady();
  updateStats();
  forceDiagnosticsUpdate();
});
$('selVisc').addEventListener('change', e => {
  state.viscModel = e.target.value;
  $('grpMu').style.opacity = state.viscModel === 'newtonian' ? '1' : '0.4';
  $('grpMu').style.pointerEvents = state.viscModel === 'newtonian' ? 'auto' : 'none';
  $('cyCard').style.display = state.viscModel === 'carreau' ? 'flex' : 'none';
  resetSteady();
  forceDiagnosticsUpdate();
});
$('selField').addEventListener('change', e => {
  state.field = e.target.value;
  keHistory = [];
  diagsRun = 0;
  diagFrames = diagInterval();
  forceDiagnosticsUpdate();
});
$('selRes').addEventListener('change', e => {
  const [nx, ny] = e.target.value.split('x').map(Number);
  NX = nx; NY = ny; DX = L_PHYS / NY;
  geomData = new Float32Array(NX * NY * 4);
  allocTextures();
  loadPreset(state.preset);
});
$('chkSL').addEventListener('change', e => {
  state.showStream = e.target.checked;
  if (!e.target.checked) clearStream();
});
$('chkPa').addEventListener('change', e => { state.showParticles = e.target.checked; });
$('selAs').addEventListener('change', e => {
  state.scaleMode = e.target.value;
  $('grpManRange').style.display = state.scaleMode === 'manual' ? 'flex' : 'none';
  if (state.scaleMode === 'manual') {
    $('txtMin').value = vmin.toFixed(2);
    $('txtMax').value = vmax.toFixed(2);
    state.manualMin = vmin;
    state.manualMax = vmax;
  }
  applyScaleMode();
  diagFrames = diagInterval();
  forceDiagnosticsUpdate();
});
$('selXf').addEventListener('change', e => {
  state.xform = e.target.value;
  applyScaleMode();
  forceDiagnosticsUpdate();
});
$('txtMin').addEventListener('input', e => {
  const v = parseFloat(e.target.value);
  if (isFinite(v)) { state.manualMin = v; applyScaleMode(); }
});
$('txtMax').addEventListener('input', e => {
  const v = parseFloat(e.target.value);
  if (isFinite(v)) { state.manualMax = v; applyScaleMode(); }
});
$('chkGr').addEventListener('change', e => { state.gravity = e.target.checked; resetSteady(); forceDiagnosticsUpdate(); });
$('chkSS').addEventListener('change', e => {
  state.detectSteady = e.target.checked;
  if (!e.target.checked) resetSteady();
});
$('slSpd').addEventListener('input', e => {
  state.stepsPerFrame = parseInt(e.target.value);
  $('vSpd').textContent = '×' + state.stepsPerFrame;
});
$('selInlet').addEventListener('change', e => {
  state.inletType = e.target.value;
  $('grpPin').style.display = state.inletType === 'pressure' ? 'flex' : 'none';
  $('grpInletProfile').style.display = state.inletType === 'velocity' ? 'flex' : 'none';
  resetSteady();
  forceDiagnosticsUpdate();
});
$('selInletProfile').addEventListener('change', e => {
  state.inletProfile = e.target.value;
  resetSteady();
  forceDiagnosticsUpdate();
});
$('selOutlet').addEventListener('change', e => {
  state.outletType = e.target.value;
  $('grpPout').style.display = state.outletType === 'pressure' ? 'flex' : 'none';
  resetSteady();
  forceDiagnosticsUpdate();
});
$('slPin').addEventListener('input', e => {
  state.pIn_mmHg = parseFloat(e.target.value);
  $('vPin').textContent = state.pIn_mmHg.toFixed(2);
  resetSteady();
  forceDiagnosticsUpdate();
});
$('slPout').addEventListener('input', e => {
  state.pOut_mmHg = parseFloat(e.target.value);
  $('vPout').textContent = state.pOut_mmHg.toFixed(2);
  resetSteady();
  forceDiagnosticsUpdate();
});

function wireCy(id, key, unit) {
  const s = $(id);
  const v = $('v' + id.substring(2));
  s.addEventListener('input', e => {
    const val = parseFloat(e.target.value);
    state[key] = unit ? val * unit : val;
    v.textContent = val.toFixed(2);
    resetSteady();
    forceDiagnosticsUpdate();
  });
}
wireCy('slCy0', 'cy_mu0', 1e-3);
wireCy('slCyI', 'cy_muI', 1e-3);
wireCy('slCyL', 'cy_lam', 1);
wireCy('slCyN', 'cy_n',   1);
wireCy('slCyA', 'cy_a',   1);
$('btnCyDefault').addEventListener('click', () => {
  state.cy_mu0 = CY_MU0; $('slCy0').value = 56; $('vCy0').textContent = '56.00';
  state.cy_muI = CY_MUI; $('slCyI').value = 3.45; $('vCyI').textContent = '3.45';
  state.cy_lam = CY_LAM; $('slCyL').value = 3.31; $('vCyL').textContent = '3.31';
  state.cy_n = CY_N;     $('slCyN').value = 0.36; $('vCyN').textContent = '0.36';
  state.cy_a = CY_A;     $('slCyA').value = 2.00; $('vCyA').textContent = '2.00';
  resetSteady();
  forceDiagnosticsUpdate();
});

$('selPre').addEventListener('change', e => {
  state.preset = e.target.value;
  loadPreset(state.preset);
});

$('btnPlay').addEventListener('click', () => {
  if (steadyState) { steadyState = false; keHistory = []; }
  paused = !paused;
  $('btnPlay').textContent = paused ? '▶ Resume' : '⏸ Pause';
  const stat = $('simStat');
  stat.textContent = paused ? '● paused' : '● running';
  stat.className = paused ? 'sim-run paused' : 'sim-run';
  stat.style.color = '';
});
$('btnReset').addEventListener('click', () => {
  resetSim();
  resetSteady();
  seedParticles();
  clearStream();
  forceDiagnosticsUpdate();
  if (paused) $('btnPlay').click();
});
$('btnClear').addEventListener('click', () => {
  clearGeom();
  uploadGeomFull();
  resetSim();
  resetSteady();
  clearStream();
  forceDiagnosticsUpdate();
});

$('tNone').addEventListener('click', () => setTool('none'));
$('tWall').addEventListener('click', () => setTool('wall'));
$('tErase').addEventListener('click', () => setTool('erase'));
$('slBr').addEventListener('input', e => {
  brushR = parseInt(e.target.value);
  $('vBr').textContent = brushR + ' px';
});

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
  switch (e.key) {
    case ' ': e.preventDefault(); $('btnPlay').click(); break;
    case 'r': case 'R': $('btnReset').click(); break;
    case 'c': case 'C': $('btnClear').click(); break;
    case 'w': case 'W': setTool('wall'); break;
    case 'e': case 'E': setTool('erase'); break;
    case 'Escape': setTool('none'); break;
    case 's': case 'S': {
      const c = $('chkSL'); c.checked = !c.checked; state.showStream = c.checked; break;
    }
    case '1': $('selField').value = 'vel';  state.field = 'vel';  forceDiagnosticsUpdate(); break;
    case '2': $('selField').value = 'pres'; state.field = 'pres'; forceDiagnosticsUpdate(); break;
    case '3': $('selField').value = 'vort'; state.field = 'vort'; forceDiagnosticsUpdate(); break;
  }
});

// Accessibility panel
const accBtn = $('accBtn'), accPanel = $('accPanel');
accBtn.addEventListener('click', () => accPanel.classList.toggle('open'));
document.addEventListener('click', e => {
  if (!accPanel.contains(e.target) && e.target !== accBtn) accPanel.classList.remove('open');
});
$('accDark').addEventListener('change', e => {
  document.body.classList.toggle('dark-mode', e.target.checked);
  try { localStorage.setItem('acc_dark', e.target.checked ? '1' : ''); } catch {}
});
document.querySelectorAll('[data-sz]').forEach(b => b.addEventListener('click', () => {
  document.body.classList.remove('size-sm', 'size-md', 'size-lg');
  if (b.dataset.sz) document.body.classList.add(b.dataset.sz);
  document.querySelectorAll('[data-sz]').forEach(x => x.classList.toggle('active', x === b));
  try { localStorage.setItem('acc_sz', b.dataset.sz); } catch {}
}));
document.querySelectorAll('[data-fi]').forEach(b => b.addEventListener('click', () => {
  document.body.classList.remove('font-serif', 'font-mono');
  if (b.dataset.fi) document.body.classList.add(b.dataset.fi);
  document.querySelectorAll('[data-fi]').forEach(x => x.classList.toggle('active', x === b));
  try { localStorage.setItem('acc_fi', b.dataset.fi); } catch {}
}));
try {
  if (localStorage.getItem('acc_dark') === '1') {
    document.body.classList.add('dark-mode');
    $('accDark').checked = true;
  }
  const sz = localStorage.getItem('acc_sz');
  if (sz) {
    document.body.classList.remove('size-sm', 'size-md', 'size-lg');
    document.body.classList.add(sz);
    document.querySelectorAll('[data-sz]').forEach(b => b.classList.toggle('active', b.dataset.sz === sz));
  }
  const fi = localStorage.getItem('acc_fi');
  if (fi) {
    document.body.classList.add(fi);
    document.querySelectorAll('[data-fi]').forEach(b => b.classList.toggle('active', b.dataset.fi === fi));
  }
} catch {}

/* ---------- Boot ---------- */
allocTextures();
$('grpPin').style.display = state.inletType === 'pressure' ? 'flex' : 'none';
$('grpPout').style.display = state.outletType === 'pressure' ? 'flex' : 'none';
$('cyCard').style.display = state.viscModel === 'carreau' ? 'flex' : 'none';
$('grpInletProfile').style.display = state.inletType === 'velocity' ? 'flex' : 'none';

function renderCyEquation() {
  if (typeof katex === 'undefined') {
    setTimeout(renderCyEquation, 50);
    return;
  }
  katex.render(
    String.raw`\mu(\dot\gamma) = \mu_\infty + (\mu_0 - \mu_\infty)\left[1 + (\lambda\dot\gamma)^a\right]^{(n-1)/a}`,
    $('cyEq'),
    { throwOnError: false, displayMode: false }
  );
}
renderCyEquation();

loadPreset('sphere');
setTool('none');
updateStats();
requestAnimationFrame(frame);

})();