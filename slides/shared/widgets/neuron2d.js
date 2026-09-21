// A neuron and a layer as surfaces over a 2D input plane (x1, x2).
//   mode "neuron": h = relu(w·x + b): a hinge; faint plane = the linear part w·x + b.
//   mode "layer":  y = Σ_k v_k relu(w_k·x + b_k): a piecewise-linear surface; creases = the lines w_k·x + b_k = 0.
//   toggle "ReLU off": the same layer without the nonlinearity collapses to a single plane.
// config: { mode: "neuron" | "layer", units: 6, width, height }
import { LIGHT } from '../theme.js';
import { C, h, fmt } from './util.js';
import { THREE, makeStage, line, label, arrow3 } from './three-util.js';

const T = (x1, x2, z) => [x1, z, -x2];                 // math (x1, x2 horizontal, z up) -> three.js
function rng(seed) { let s = seed; return () => ((s = (s * 16807) % 2147483647) / 2147483647); }

export function mount(el, cfg) {
  const W = cfg.width || 560, H = cfg.height || 400;
  const st = { mode: cfg.mode || 'neuron', angle: 30, mag: 1, b: 0, units: cfg.units ?? 6, relu: true, seed: 4 };
  const canvas = h('canvas', { width: W, height: H });
  const stage = makeStage(canvas, { position: [2.9, 3.0, 3.4], target: [0, 0.15, 0] });
  const { scene, render } = stage;
  const N = 60, R = 1.5;                                // grid resolution and half-size of the input square

  // floor grid and axes
  const grid = new THREE.GridHelper(2 * R, 6, 0x3a3f4b, 0x2d313a); scene.add(grid);
  scene.add(line([T(-R, 0, 0), T(R * 1.15, 0, 0)], 0x9aa1ae));
  scene.add(line([T(0, -R, 0), T(0, R * 1.15, 0)], 0x9aa1ae));
  const l1 = label('input x₁', { color: C.blue, size: 0.34, font: '600 44px Inter, Arial' }); l1.position.set(...T(R * 1.35, 0, 0)); scene.add(l1);
  const l2 = label('input x₂', { color: C.blue, size: 0.5, font: '600 44px Inter, Arial' }); l2.position.set(...T(-0.2, R * 1.3, 0.25)); scene.add(l2);
  // vertical axis = the output (height of the surface)
  scene.add(arrow3(T(-R, -R, 0), T(-R, -R, 1.25), 0xf2b134, 0.12));
  const l3 = label('output (height)', { color: C.accent, size: 0.34, font: '600 44px Inter, Arial' }); l3.position.set(...T(-R + 0.7, -R, 1.42)); scene.add(l3);

  const geo = new THREE.PlaneGeometry(2 * R, 2 * R, N, N);
  geo.rotateX(-Math.PI / 2);                           // lie in three's XZ plane
  const colors = new Float32Array(geo.attributes.position.count * 3);
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const surf = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, side: THREE.DoubleSide, roughness: 0.75, flatShading: true }));
  scene.add(surf);
  const wire = new THREE.LineSegments(new THREE.WireframeGeometry(geo), new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.06 }));
  scene.add(wire);
  const planeGeo = new THREE.PlaneGeometry(2 * R, 2 * R, 1, 1); planeGeo.rotateX(-Math.PI / 2);
  const ghost = new THREE.Mesh(planeGeo, new THREE.MeshBasicMaterial({ color: 0x9aa1ae, transparent: true, opacity: 0.18, side: THREE.DoubleSide, depthWrite: false }));
  scene.add(ghost);
  const dyn = new THREE.Group(); scene.add(dyn);
  const readout = h('div', { style: 'line-height:1.55; min-height:5.5em' });

  function layerParams() {
    const r = rng(st.seed * 31 + st.units);
    return Array.from({ length: st.units }, () => {
      const a = 2 * Math.PI * r(), m = 0.8 + 0.8 * r();
      const w = [m * Math.cos(a), m * Math.sin(a)], d = (r() - 0.5) * 1.6;
      return { w, b: -d * m, v: (r() < 0.5 ? -1 : 1) * (0.25 + 0.35 * r()) };
    });
  }

  let this_k = 1, this_z0 = 0;
  function draw() {
    const P = st.mode === 'neuron'
      ? [{ w: [st.mag * Math.cos(st.angle * Math.PI / 180), st.mag * Math.sin(st.angle * Math.PI / 180)], b: st.b, v: 1 }]
      : layerParams();
    const act = z => (st.relu ? Math.max(0, z) : z);
    const f = (x1, x2) => P.reduce((s, u) => s + u.v * act(u.w[0] * x1 + u.w[1] * x2 + u.b), 0);
    const pos = geo.attributes.position;
    let zmin = Infinity, zmax = -Infinity;
    for (let i = 0; i < pos.count; i++) {
      const z = f(pos.getX(i), -pos.getZ(i)); zmin = Math.min(zmin, z); zmax = Math.max(zmax, z);
    }
    // heights shown to scale for one neuron; a layer is rescaled so its surface always fits the view
    const k = st.mode === 'neuron' ? 0.35 : 1.1 / Math.max(1e-6, zmax - zmin);
    const z0 = st.mode === 'neuron' ? 0 : zmin;
    this_k = k; this_z0 = z0;
    for (let i = 0; i < pos.count; i++) pos.setY(i, (f(pos.getX(i), -pos.getZ(i)) - z0) * k);
    zmin = (zmin - z0) * k; zmax = (zmax - z0) * k;
    // colour by height: blue (low) → surface grey → amber (high)
    const lo = [0.22, 0.53, 0.9], mid = LIGHT ? [0.78, 0.8, 0.84] : [0.2, 0.22, 0.27], hi = [0.79, 0.52, 0.0];
    const zc = st.mode === 'neuron' ? 0 : (zmin + zmax) / 2, span = Math.max(1e-6, Math.max(Math.abs(zmin - zc), Math.abs(zmax - zc)));
    for (let i = 0; i < pos.count; i++) {
      const t = (pos.getY(i) - zc) / span, c = t >= 0 ? hi : lo, a = Math.min(1, Math.abs(t));
      colors[3 * i] = mid[0] + (c[0] - mid[0]) * a; colors[3 * i + 1] = mid[1] + (c[1] - mid[1]) * a; colors[3 * i + 2] = mid[2] + (c[2] - mid[2]) * a;
    }
    pos.needsUpdate = true; geo.attributes.color.needsUpdate = true; geo.computeVertexNormals();
    wire.geometry.dispose(); wire.geometry = new THREE.WireframeGeometry(geo);

    // the linear part (neuron mode) as a faint plane
    ghost.visible = st.mode === 'neuron';
    if (ghost.visible) {
      const u = P[0], pp = planeGeo.attributes.position;
      for (let i = 0; i < pp.count; i++) { const x1 = pp.getX(i), x2 = -pp.getZ(i); pp.setY(i, (u.w[0] * x1 + u.w[1] * x2 + u.b) * this_k); }
      pp.needsUpdate = true;
    }
    // creases: the lines w·x + b = 0 on the floor
    dyn.clear();
    if (st.relu) P.forEach(u => {
      const pts = [];
      for (const t of [-3, 3]) {
        // point on the line closest to origin + t · direction along the line
        const n2 = u.w[0] ** 2 + u.w[1] ** 2, p0 = [-u.b * u.w[0] / n2, -u.b * u.w[1] / n2], d = [-u.w[1] / Math.sqrt(n2), u.w[0] / Math.sqrt(n2)];
        pts.push([p0[0] + t * d[0], p0[1] + t * d[1]]);
      }
      // clip to the square by sampling
      const seg = [];
      for (let k = 0; k <= 200; k++) {
        const x1 = pts[0][0] + (pts[1][0] - pts[0][0]) * k / 200, x2 = pts[0][1] + (pts[1][1] - pts[0][1]) * k / 200;
        if (Math.abs(x1) <= R && Math.abs(x2) <= R) seg.push(T(x1, x2, (f(x1, x2) - this_z0) * this_k + 0.01));
      }
      if (seg.length > 1) dyn.add(line(seg, 0xb28dff));
    });
    render();

    const n = st.mode === 'neuron' ? 3 : 4 * st.units + 0;
    readout.innerHTML = st.mode === 'neuron'
      ? `<div><b>one neuron</b>: $h = \\mathrm{ReLU}(\\mathbf{w}^\\top \\mathbf{x} + b)$</div><div class="dim">${st.relu ? 'a hinge: zero on one side of the purple line, a tilted plane on the other' : 'without ReLU: just the plane'}</div><div>parameters: <b>3</b></div>`
      : `<div><b>a layer of ${st.units}</b>: $\\sum_k v_k\\,${st.relu ? '\\mathrm{ReLU}' : ''}(\\mathbf{w}_k^\\top \\mathbf{x} + b_k)$</div><div class="dim">${st.relu ? 'a sum of hinges: flat facets meeting at the purple creases' : 'ReLU off: the sum of planes is <b>one plane</b>, however many units'}</div><div>parameters: <b>${n}</b></div>`;
    window.renderMathInElement?.(readout, { delimiters: [{ left: '$', right: '$', display: false }], throwOnError: false });
  }

  const sl = (key, min, max, step, name) => {
    const s = h('input', { type: 'range', min, max, step, value: st[key] });
    s.addEventListener('input', () => { st[key] = +s.value; draw(); });
    return h('div', { class: 'neuron-only' }, h('label', { style: 'display:inline-block; width:5.5em' }, name), s);
  };
  const modeBtns = [['neuron', 'one neuron'], ['layer', 'a layer']].map(([m, txt]) => h('button', { onclick: () => { st.mode = m; sync(); draw(); } }, txt));
  const reluBtn = h('button', { onclick: () => { st.relu = !st.relu; sync(); draw(); } }, 'ReLU on');
  const unitsSl = h('input', { type: 'range', min: 1, max: 16, step: 1, value: st.units });
  unitsSl.addEventListener('input', () => { st.units = +unitsSl.value; draw(); });
  const layerCtl = h('div', {}, h('label', { style: 'display:inline-block; width:5.5em' }, 'units'), unitsSl, ' ', h('button', { onclick: () => { st.seed++; draw(); } }, 'shuffle'));
  const neuronCtl = h('div', {}, sl('angle', -180, 180, 1, 'direction'), sl('mag', 0.2, 2, 0.05, 'steepness'), sl('b', -1.5, 1.5, 0.05, 'bias b'));
  function sync() {
    modeBtns.forEach((b, i) => b.classList.toggle('active', ['neuron', 'layer'][i] === st.mode));
    reluBtn.textContent = st.relu ? 'ReLU: on' : 'ReLU: off'; reluBtn.classList.toggle('active', !st.relu);
    neuronCtl.style.display = st.mode === 'neuron' ? '' : 'none'; layerCtl.style.display = st.mode === 'layer' ? '' : 'none';
  }
  el.classList.add('widget');
  el.append(h('div', { style: 'display:flex; gap:0.8em; align-items:flex-start' }, canvas,
    h('div', { style: 'display:flex; flex-direction:column; gap:0.5em; width:16em; flex:none' }, readout,
      h('div', { class: 'wctl interactive-only', style: 'flex-direction:column; align-items:flex-start; gap:0.35em' },
        h('div', { style: 'display:flex; gap:0.3em' }, modeBtns), reluBtn, neuronCtl, layerCtl, h('div', { class: 'dim' }, 'drag to orbit')))));
  sync(); draw();
}
