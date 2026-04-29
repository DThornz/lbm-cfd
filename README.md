# lbm-cfd

**2D LBM Flow Solver — GPU Hemodynamics Simulator** — browser-based educational CFD tool.

Part of the [A. Mirza academic tools portfolio](https://dthornz.github.io/website-cv-tools/).

🌐 **Live:** [dthornz.github.io/lbm-cfd](https://dthornz.github.io/lbm-cfd/)

---

Browser-based GPU lattice Boltzmann (LBM) solver for 2D incompressible blood flow at the vessel scale. Built as an educational tool for hemodynamics and computational fluid mechanics.

**Physics & Numerics:**
- D2Q9 lattice with BGK single-relaxation-time collision operator
- Half-way bounce-back no-slip boundary conditions (Ladd 1994, 2nd-order accurate)
- Optional Carreau–Yasuda non-Newtonian blood rheology (μ₀ = 56 mPa·s, μ∞ = 3.45 mPa·s)
- Local shear rate computed from Chapman–Enskog momentum-flux tensor
- WebGL2 GPU compute via fragment shaders — one texel per cell, one draw call per timestep
- Blood properties: ρ = 1060 kg/m³ · Reynolds range Re ≤ 2000 · Max grid 1280×480

**Simulator features:**
- 8 vascular preset geometries: empty channel, sphere/disk, square block, symmetric stenosis (50%), asymmetric plaque, saccular aneurysm, Y-bifurcation, trifurcation
- Inlet BC: velocity (plug or fully-developed parabolic) or pressure
- Display fields: velocity |U| (m/s), pressure (Pa), vorticity (ω, 1/s)
- 6 grid resolutions: 240×96 (Coarse) → 1280×480 (Giga)
- Newtonian / Non-Newtonian (Carreau–Yasuda) rheology toggle
- Draw tools: wall pencil, erase, brush radius control

**Educational content:**
- D2Q9 lattice derivation and BGK collision operator
- Chapman–Enskog expansion recovering incompressible Navier–Stokes
- Bounce-back BC analysis, Carreau–Yasuda closure, unit mapping
- Reynolds regimes across the vasculature
- 7 cited references (Qian 1992, Succi 2001, Ladd 1994, etc.)

**Tech:** Vanilla HTML/CSS/JS · WebGL2 · KaTeX · No build step · GitHub Pages

---

© 2026 Asad Mirza, Ph.D. · Research Assistant Professor · FIU Biomedical Engineering
