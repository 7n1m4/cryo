/**
 * Shared bending primitives.
 *
 * The five spells differ in what they do and agree on how they move. This is the
 * "how": easing that never snaps, a frame that never flips, and the two queries
 * against the world every one of them needs.
 *
 * Nothing here allocates.
 */

/** @param {number} v */
export function clamp01(v: any) {
 return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** @param {number} v @param {number} lo @param {number} hi */
export function clampRange(v: any, lo: any, hi: any) {
 return v < lo ? lo : v > hi ? hi : v;
}

/** Hermite smoothstep on an already-normalised parameter. @param {number} t */
export function smooth01(t: any) {
 const x = clamp01(t);
 return x * x * (3 - 2 * x);
}

/**
 * A bell that is zero at both ends and one in the middle, with zero slope
 * everywhere it matters.
 *
 * Used for every amplitude envelope in the spells. A linear ramp leaves a
 * visible corner at both ends of every arc, and the grammar here is "no instant
 * spawns, no instant despawns".
 * @param {number} t
 */
export function bell(t: any) {
 const x = clamp01(t);
 const s = Math.sin(Math.PI * x);
 return s * s;
}

/** Framerate-independent exponential approach. @param {number} cur @param {number} target @param {number} rate @param {number} dt */
export function expDamp(cur: any, target: any, rate: any, dt: any) {
 return target + (cur - target) * Math.exp(-rate * dt);
}

/**
 * Rotate a reference vector by the minimal rotation taking `t0` to `t1`.
 *
 * This is what keeps a swept section from spinning as its spine curves. The
 * obvious alternatives both fail on the shapes these spells draw:
 *
 *   Frenet frame       undefined wherever the spine is momentarily straight,
 *                      and it flips through 180 degrees at every inflection —
 *                      which a figure-eight has two of, by definition.
 *   up-referenced      degenerate wherever the spine passes through vertical,
 *                      which a ribbon thrown overhead does constantly.
 *
 * Parallel transport has no degeneracy at all: each frame is the previous one
 * rotated by the smallest rotation that lines the tangents up, so it only ever
 * turns as much as the curve does.
 *
 * Writes the transported right vector into `out[o..o+2]`.
 *
 * @param {Float32Array} out @param {number} o
 * @param {number} rx @param {number} ry @param {number} rz
 * @param {number} t0x @param {number} t0y @param {number} t0z
 * @param {number} t1x @param {number} t1y @param {number} t1z
 */
export function transport(out: any, o: any, rx: any, ry: any, rz: any, t0x: any, t0y: any, t0z: any, t1x: any, t1y: any, t1z: any) {
 // Axis and angle of the rotation taking t0 to t1.
 let ax = t0y * t1z - t0z * t1y;
 let ay = t0z * t1x - t0x * t1z;
 let az = t0x * t1y - t0y * t1x;
 const s = Math.hypot(ax, ay, az);

 if (s < 1e-7) {
  // Parallel or antiparallel. Antiparallel cannot happen between adjacent
  // samples of a spine sampled every few centimetres, so this is the
  // straight-line case and the frame simply carries over.
  out[o] = rx; out[o + 1] = ry; out[o + 2] = rz;
  return;
 }

 ax /= s; ay /= s; az /= s;
 const c = t0x * t1x + t0y * t1y + t0z * t1z;
 const ang = Math.atan2(s, c);
 const cs = Math.cos(ang);
 const sn = Math.sin(ang);

 // Rodrigues.
 const dot = ax * rx + ay * ry + az * rz;
 const cx = ay * rz - az * ry;
 const cy = az * rx - ax * rz;
 const cz = ax * ry - ay * rx;

 let ox = rx * cs + cx * sn + ax * dot * (1 - cs);
 let oy = ry * cs + cy * sn + ay * dot * (1 - cs);
 let oz = rz * cs + cz * sn + az * dot * (1 - cs);

 const l = Math.hypot(ox, oy, oz) || 1;
 out[o] = ox / l; out[o + 1] = oy / l; out[o + 2] = oz / l;
}

/**
 * Where a ray meets the snow.
 *
 * A coarse march followed by a bisection refine, against the CPU height mirror.
 * Eight refinement steps put the hit inside a centimetre, which is finer than
 * anything downstream of this can express.
 *
 * The march deliberately does not stop at the first sample below the surface if
 * the ray *starts* below it — a targeting ray fired from inside a drift should
 * come out of the drift and hit the far side, not report a hit at the origin.
 *
 * @param {{heightAt(x:number,z:number):number}} terrain
 * @param {number} ox @param {number} oy @param {number} oz
 * @param {number} dx @param {number} dy @param {number} dz
 * @param {number} maxDist
 * @returns {number} distance along the ray, or -1 for a miss
 */
export function groundRay(terrain: any, ox: any, oy: any, oz: any, dx: any, dy: any, dz: any, maxDist: any) {
 const STEP = 0.6;
 let prevT = 0;
 let prevAbove = oy - terrain.heightAt(ox, oz);
 if (prevAbove < 0) prevAbove = 0.001; // started buried; treat as above

 for (let t = STEP; t <= maxDist; t += STEP) {
  const x = ox + dx * t;
  const y = oy + dy * t;
  const z = oz + dz * t;
  const above = y - terrain.heightAt(x, z);

  if (above <= 0) {
   let lo = prevT;
   let hi = t;
   for (let k = 0; k < 8; k++) {
    const mid = (lo + hi) * 0.5;
    const my = oy + dy * mid;
    if (my - terrain.heightAt(ox + dx * mid, oz + dz * mid) > 0) lo = mid;
    else hi = mid;
   }
   return (lo + hi) * 0.5;
  }
  prevT = t;
  prevAbove = above;
 }
 return -1;
}

/**
 * Where the player is aiming, on the ground.
 *
 * Falls back to a fixed distance ahead when the ray misses — looking up at the
 * sky and pressing a key has to do *something*, and putting the effect at a sane
 * distance in front of the player is the only answer that is never surprising.
 *
 * Writes (x, y, z) into `out`.
 *
 * @param {Float32Array} out
 * @param {{heightAt(x:number,z:number):number}} terrain
 * @param {number} ox @param {number} oy @param {number} oz
 * @param {number} dx @param {number} dy @param {number} dz
 * @param {number} maxDist @param {number} fallback
 */
export function aimPoint(out: any, terrain: any, ox: any, oy: any, oz: any, dx: any, dy: any, dz: any, maxDist: any, fallback: any) {
 const t = groundRay(terrain, ox, oy, oz, dx, dy, dz, maxDist);
 if (t > 0) {
  out[0] = ox + dx * t;
  out[1] = oy + dy * t;
  out[2] = oz + dz * t;
  return t;
 }
 // Flatten the direction and step out, then drop onto the surface.
 const fl = Math.hypot(dx, dz) || 1;
 out[0] = ox + (dx / fl) * fallback;
 out[2] = oz + (dz / fl) * fallback;
 out[1] = terrain.heightAt(out[0], out[2]);
 return fallback;
}
