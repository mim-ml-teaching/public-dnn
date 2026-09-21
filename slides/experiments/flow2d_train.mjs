// Pretrains the 2D flow-matching velocity fields shown in lecture 5 (so that the slides are instant and deterministic).
// Usage: node experiments/flow2d_train.mjs            (from dnn/slides; ~1 min)
// Writes assets/lecture05/data/flow_<source>_<target>.json, loaded by shared/widgets/flow2d.js.
import { writeFileSync } from 'node:fs';
import { rng, makeNet, cfmStep, exportNet, SOURCES, TARGETS, euler } from '../shared/widgets/flowlib.js';

const CASES = [['gauss', 'moons'], ['gauss8', 'moons'], ['gauss', 'spiral'], ['gauss', 'rings']];
const STEPS = +process.argv[2] || 6000;

for (const [src, tgt] of CASES) {
  const r = rng(7), net = makeNet([3, 64, 64, 64, 2], r), opt = { lr: 2e-3, step: 0 };
  const t0 = Date.now();
  let avg = 0;
  for (let s = 1; s <= STEPS; s++) {
    if (s === Math.floor(STEPS * 0.6)) opt.lr = 7e-4;
    if (s === Math.floor(STEPS * 0.85)) opt.lr = 2e-4;
    const loss = cfmStep(net, opt, r, SOURCES[src], TARGETS[tgt], 256);
    avg = s === 1 ? loss : 0.98 * avg + 0.02 * loss;
    if (s % 1000 === 0) console.log(`${src}→${tgt} step ${s}: loss ${avg.toFixed(3)} (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
  }
  // quick quality check: how far do generated samples land from the target set?
  const r2 = rng(99), data = Array.from({ length: 2000 }, () => TARGETS[tgt](r2));
  let dist = 0;
  for (let i = 0; i < 300; i++) {
    const x = euler(net, SOURCES[src](r2), 50).at(-1);
    dist += Math.sqrt(Math.min(...data.map(p => (p[0] - x[0]) ** 2 + (p[1] - x[1]) ** 2)));
  }
  console.log(`  mean distance of a generated sample to the nearest data point: ${(dist / 300).toFixed(3)}`);
  writeFileSync(`assets/lecture05/data/flow_${src}_${tgt}.json`, JSON.stringify(exportNet(net)));
}
