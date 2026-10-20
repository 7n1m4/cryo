/**
 * The audio engine — one graph, decoded buffers, and a mixer wired to `S`.
 *
 * Shape of it:
 *
 *   one-shots ─┐
 *   loops ─────┴─► bus gain (ambience | music | sfx | voice) ─► master ─► limiter ─► out
 *
 * Three rules this file exists to enforce:
 *
 *  1. **Nothing decodes at play time.** `load()` runs behind the boot screen and
 *     fetches every entry in the manifest into an `AudioBuffer`. A one-shot fired
 *     from the render loop is a `createBufferSource` and a `start()` — no fetch,
 *     no decode, no promise, and nothing that can hitch a frame.
 *
 *  2. **The context is created suspended and only ever resumed from a gesture.**
 *     Browsers refuse to start audio otherwise, and a demo that silently fails
 *     to play is worse than one that asks. `unlock()` is the only resume path
 *     and it is called from the boot gate or the sound button — both clicks.
 *
 *  3. **Gain is a ramp, never an assignment.** Writing `gain.value` from the
 *     render loop steps the parameter at frame boundaries, which is audible as a
 *     zipper on anything continuous. Everything here goes through
 *     `setTargetAtTime` or an explicit ramp.
 *
 * The mixer reads `S.audioMuted` / `S.masterVolume` / `S.ambienceVolume` /
 * `S.musicVolume` / `S.sfxVolume` and subscribes to them, so the overlay sliders
 * are live.
 */

import { S, onChange, type SettingKey } from "../core/settings.js";
import { assetCandidates } from "../core/assets.js";
import { AUDIO_MANIFEST } from "./manifest.js";

/**
 * Hard ceiling on simultaneous one-shots. Reached only by a bug — the manifest's
 * `minGap` floors keep the real count in single digits — but an unbounded voice
 * count is how a stuck trigger turns into a locked-up audio thread.
 */
const MAX_VOICES = 24;

/** How far ahead the loop scheduler queues overlap segments, seconds. */
const LOOKAHEAD = 0.6;

/**
 * Release applied to a voice cut short by a new one in its exclusive group. Long
 * enough not to click, short enough that the two do not audibly overlap.
 */
const SOLO_RELEASE = 0.06;

const BUS_NAMES = ["ambience", "music", "sfx", "voice"] as const;

/** A mixer bus name — one gain node per name, all wired into `master`. */
type BusName = (typeof BUS_NAMES)[number];

/** The manifest's own key set, not `string`. */
type AudioKey = keyof typeof AUDIO_MANIFEST;

/**
 * The per-field view `play`/`loop` need. Every field is optional on the entry
 * itself — `minGap`, `exclusive`, `rate`, `loop` each exist only on some rows —
 * so this is the union of them rather than one row's shape, and the entries are
 * read through it rather than through the union TS cannot express.
 */
type AudioFields = Partial<
 Record<"url" | "bus" | "gain" | "rate" | "loop" | "trim" | "crossfade" | "minGap" | "exclusive" | "needs", unknown>
>;

/** Call-site options for `play`/`loop`. */
type PlayOpts = {
 gain?: number;
 rate?: number;
 delay?: number;
 bus?: string;
 minGap?: number;
 exclusive?: boolean | string;
};

/** A manifest numeric field — `minGap`, `rate` — or `undefined` if absent. */
function num(v: unknown): number | undefined {
 return typeof v === "number" ? v : undefined;
}

/**
 * Resolve the `AudioContext` constructor.
 *
 * Old Safari exposes Web Audio only as `window.webkitAudioContext`, which
 * `lib.dom` does not declare. The prefixed name is read through an index
 * into `unknown` and only accepted when the value is function-shaped, so the
 * vendor-prefix fallback keeps working without a fabricated object type or a
 * `Window` augmentation. `undefined` means "no Web Audio at all".
 */
function readAudioContextCtor(): typeof AudioContext | undefined {
 if (window.AudioContext) return window.AudioContext;
 const w = window as unknown as Record<string, unknown>;
 const prefixed = w["webkitAudioContext"];
 return typeof prefixed === "function"
  ? (prefixed as typeof AudioContext)
  : undefined;
}

export class AudioEngine {
 [key: string]: any;
 constructor() {
  /** @type {AudioContext|null} */
  this.ctx = null;
  /** @type {GainNode|null} */
  this.master = null;
  /** @type {DynamicsCompressorNode|null} */
  this.limiter = null;
  /** @type {Record<BusName, GainNode>} */
  this.buses = Object.create(null);

  /** @type {Map<string, AudioBuffer>} */
  this.buffers = new Map();

  /**
   * Keys whose fetch/decode threw. The load loop is deliberately non-fatal:
   * one asset missing is not a reason to drop the other forty, so `load()`
   * still resolves and `buffers` simply keeps that key absent. Recording it
   * here is what makes that policy auditable — `failed.has(k)` now tells a
   * caller "this asset was attempted and threw" apart from "never attempted",
   * where both used to read as an absent key in `buffers`.
   * @type {Set<string>}
   */
  this.failed = new Set();

  /** True once every manifest entry has been fetched and decoded (or failed). */
  this.loaded = false;
  /** True once a user gesture has resumed the context. */
  this.unlocked = false;
  /**
   * The rejection from the last refused `ctx.resume()`, or null. The return
   * value of `unlock()` stays the boolean callers branch on; this keeps the
   * reason it came back false.
   * @type {unknown}
   */
  this.lastResumeError = null;

  /** @type {LoopVoice[]} */
  this._loops = [];
  /** Live one-shot count, for the voice cap. */
  this._voices = 0;
  /** Context time of the last start, per key, for `minGap`. */
  this._last = Object.create(null);
  /**
   * Live voice per exclusive group — see the `exclusive` manifest field.
   * @type {Record<string, {src: AudioBufferSourceNode, gain: GainNode}>}
   */
  this._solo = Object.create(null);

  this._boundSettings = false;
 }

 /** Whether anything actually decoded — false means offer no sound at all. */
 get hasAssets() {
  return this.buffers.size > 0;
 }

 // ------------------------------------------------------------------ graph

 /**
  * Build the context and the mixer. Called by `load()`, so the graph exists
  * long before the gate — the context simply sits in `suspended` until a
  * gesture resumes it, which is the one state browsers let us start in.
  */
 _ensureContext() {
  if (this.ctx) return this.ctx;

  // Safari ships Web Audio only behind the vendor prefix, which lib.dom does
  // not declare — see `readAudioContextCtor`.
  const Ctor = readAudioContextCtor();
  if (!Ctor) {
   console.warn("[audio] Web Audio is unavailable; running silent");
   return null;
  }

  // `interactive` asks for the smallest buffer the platform will give us.
  // A wind bed would be happy with `playback`, but the carve hits are
  // gameplay feedback and latency reads as sloppiness.
  const ctx = new Ctor({ latencyHint: "interactive" });
  this.ctx = ctx;

  // A gentle limiter across the whole mix. The wind bed, the board, a carve
  // hit, a cast and a voice line can all be up in the same frame, and that
  // sum measures close enough to full scale to clip on a loud setting. This
  // is set high and fast so it only ever catches those stacks — it is not a
  // tone control, and on anything short of a pile-up it is transparent.
  this.limiter = ctx.createDynamicsCompressor();
  this.limiter.threshold.value = -6;
  this.limiter.knee.value = 6;
  this.limiter.ratio.value = 8;
  this.limiter.attack.value = 0.004;
  this.limiter.release.value = 0.16;
  this.limiter.connect(ctx.destination);

  this.master = ctx.createGain();
  this.master.gain.value = 0;
  this.master.connect(this.limiter);

  for (const name of BUS_NAMES) {
   const g = ctx.createGain();
   g.gain.value = 1;
   g.connect(this.master);
   this.buses[name] = g;
  }

  if (!this._boundSettings) {
   this._boundSettings = true;
   onChange(
    ["audioMuted", "masterVolume", "ambienceVolume", "musicVolume", "sfxVolume"],
    () => this.applyMix()
   );
   // A suspended context burns no CPU and, more to the point, a wind bed
   // droning out of a background tab is the kind of thing people close
   // the tab over.
   document.addEventListener("visibilitychange", () => {
    if (!this.unlocked || !this.ctx) return;
    if (document.hidden) this.ctx.suspend();
    else this.ctx.resume();
   });
  }

  this.applyMix();
  return ctx;
 }

 /**
  * The gain node for a bus, falling back to the sfx bus when the name is not
  * one of the four — so a manifest row or a call site naming an unknown bus
  * still lands somewhere audible rather than silencing the sound.
  * @param {string} [name]
  * @returns {GainNode}
  */
 _bus(name?: string): GainNode {
  const isBus = (n: string): n is BusName =>
   (BUS_NAMES as readonly string[]).includes(n);
  return isBus(name ?? "") ? this.buses[name as BusName] : this.buses.sfx;
 }

 /**
  * Push `S`'s mix values into the graph. Ramped rather than assigned so a
  * slider drag or a mute is a fade, not a click.
  *
  * @param {number} [smooth] time constant, seconds. The default is tuned for
  *   slider drags and mutes; `unlock` passes something much shorter, because
  *   40 ms of time constant is 120 ms to full scale and that is long enough to
  *   read as the demo being slow to make a sound.
  */
 applyMix(smooth = 0.04) {
  const ctx = this.ctx;
  if (!ctx || !this.master) return;
  const now = ctx.currentTime;
  const master = this.unlocked && !S.audioMuted ? (S.masterVolume as number) : 0;
  this.master.gain.setTargetAtTime(master, now, smooth);
  this.buses.ambience.gain.setTargetAtTime((S.ambienceVolume as number), now, smooth);
  this.buses.music.gain.setTargetAtTime((S.musicVolume as number), now, smooth);
  this.buses.sfx.gain.setTargetAtTime((S.sfxVolume as number), now, smooth);
  this.buses.voice.gain.setTargetAtTime((S.sfxVolume as number), now, smooth);
 }

 // ----------------------------------------------------------------- loading

 /**
  * Fetch and decode the whole manifest.
  *
  * Called early in `boot()` and awaited late, so the download overlaps device
  * creation, the heightfield bake and pipeline compilation — on a warm cache
  * it has been resolved for several seconds by the time anything waits on it.
  *
  * A single asset failing is not fatal: it is logged, its key stays absent
  * from `buffers`, and every later `play`/`loop` for that key is a no-op. A
  * missing carve sample must not take the demo down with it.
  *
  * @param {(p: number) => void} [onProgress] 0..1, monotonic
  */
 async load(onProgress?: (p: number) => void) {
  const ctx = this._ensureContext();
  if (!ctx) {
   this.loaded = true;
   return;
  }

  // `AUDIO_MANIFEST` is a `Record<string, AudioAsset>`, so `Object.keys` widens
  // to `string[]`; the cast restores the manifest's own key union and keeps the
  // lookups below checked rather than implicit-any.
  const keys = Object.keys(AUDIO_MANIFEST) as AudioKey[];
  // Per-asset completion, so the reported number is the mean of real
  // download fractions rather than a count of finished files.
  const frac = new Float64Array(keys.length);
  const report = () => {
   if (!onProgress) return;
   let sum = 0;
   for (let i = 0; i < frac.length; i++) sum += frac[i];
   onProgress(sum / frac.length);
  };

  await Promise.all(
   keys.map(async (k, i) => {
    const def = (AUDIO_MANIFEST as Record<AudioKey, (typeof AUDIO_MANIFEST)[AudioKey & string]>)[k];
    // Belongs to a feature that is switched off: not fetched, not
    // decoded, not waited for. `play` already no-ops on a key with no
    // buffer, so nothing downstream needs to know it is absent.
    //
    // `needs` is optional and per-asset, so it is read off a value typed
    // `unknown` (the object union TS cannot express `needs` on). A name absent
    // from `S` reads as falsy, exactly as the old `!S[def.needs]` did — the
    // settings key is then only used once it names a real one.
    const raw: unknown = def && typeof def === "object" && "needs" in def ? def.needs : undefined;
    const needs: SettingKey | undefined =
     typeof raw === "string" ? (raw as SettingKey) : undefined;
    if (needs && !S[needs]) {
     frac[i] = 1;
     report();
     return;
    }
    try {
     // Decode is the last 15%: it is real work, but it is not
     // measurable in progress, so it gets a flat reservation.
     const buf = await this._fetchDecode(def.url, (p: any) => {
      frac[i] = p * 0.85;
      report();
     });
     this.buffers.set(k, buf);
    } catch (err: unknown) {
     // Deliberately non-fatal: one missing asset must not take the other
     // forty with it, so the loop continues and `buffers` simply keeps this
     // key absent. The non-fatality is recorded in `this.failed` so it is
     // visible to a caller rather than a swallow.
     console.warn("[audio] failed to load '" + k + "':", err);
     this.failed.add(k);
    }
    frac[i] = 1;
    report();
   })
  );

  this.loaded = true;
 }

 /**
  * @param {string} path store-relative, e.g. `audio/ambiance.mp3`
  * @param {(p: number) => void} onProgress
  * @returns {Promise<AudioBuffer>}
  */
 async _fetchDecode(path: any, onProgress: any) {
  // The store first, this origin second. A sound that has been committed
  // but not yet uploaded still plays. See `core/assets.js`.
  let res = null;
  let lastError = null;
  for (const url of assetCandidates(path)) {
   try {
    const r = await fetch(url);
    if (r.ok) { res = r; break; }
    lastError = new Error(url + " — " + r.status + " " + r.statusText);
   } catch (err) {
    lastError = err;
   }
  }
  if (!res) throw lastError || new Error("not found: " + path);

  const total = Number(res.headers.get("content-length")) || 0;
  /** @type {ArrayBuffer} */
  let bytes;

  if (total > 0 && res.body) {
   const reader = res.body.getReader();
   /** @type {Uint8Array[]} */
   const chunks = [];
   let read = 0;
   for (; ;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    read += value.length;
    onProgress(Math.min(1, read / total));
   }
   const flat = new Uint8Array(read);
   let off = 0;
   for (let i = 0; i < chunks.length; i++) {
    flat.set(chunks[i], off);
    off += chunks[i].length;
   }
   bytes = flat.buffer;
  } else {
   // No content-length (compressed transfer, some dev servers): fall
   // back to a plain read and report the file as one step.
   bytes = await res.arrayBuffer();
   onProgress(1);
  }

  return await (this.ctx as AudioContext).decodeAudioData(bytes);
 }

 // ------------------------------------------------------------------ unlock

 /**
  * Resume the context. Must be called synchronously from inside a user
  * gesture handler — the boot gate's button or the sound button.
  *
  * The boolean is the contract callers branch on (`false` = no sound, stay
  * mute), so a refused resume still returns `false`. What was lost was the
  * reason: it went to the console and nowhere else. `lastResumeError` holds
  * the rejection so a caller or a test can read why, not just that.
  * @returns {Promise<boolean>}
  */
 async unlock() {
  const ctx = this._ensureContext();
  if (!ctx) return false;
  try {
   if (ctx.state !== "running") await ctx.resume();
  } catch (err: unknown) {
   console.warn("[audio] resume refused:", err);
   this.lastResumeError = err;
   return false;
  }
  this.unlocked = ctx.state === "running";
  // Open the master fast — this is the gesture that asked for sound, so the
  // fade here exists only to keep the first sample from clicking.
  this.applyMix(0.008);
  return this.unlocked;
 }

 // ----------------------------------------------------------------- playback

 /**
  * Fire a one-shot. Silently returns null when the engine is locked, the key
  * is unknown, the asset failed to load, the voice cap is reached, or the
  * key's `minGap` has not elapsed — every one of those is a normal condition
  * for a call made from a render loop, not an error.
  *
  * An asset marked `exclusive` holds one voice at a time: starting it again
  * releases the voice already playing rather than layering onto it. Spells are
  * the case that needs it — nothing rate-limits how fast a player can press
  * `1`, the spell simply restarts, and a five-second effect stacked six deep is
  * a wall of noise where the sound should have restarted with the spell.
  *
  * @param {string} key manifest key
  * @param {{ gain?: number, rate?: number, delay?: number, bus?: string,
  *           minGap?: number, exclusive?: boolean|string }} [opts]
  * @returns {AudioBufferSourceNode|null}
  */
 play(key: string, opts?: PlayOpts) {
  if (!this.unlocked || !this.ctx) return null;
  // An unknown key is a no-op, not an error — every call site is a render
  // loop — so it is narrowed to the manifest's key union before the lookup,
  // and an absent entry falls through the `!def` guard below.
  const entry = key in AUDIO_MANIFEST ? AUDIO_MANIFEST[key as AudioKey] : undefined;
  const buf = this.buffers.get(key);
  if (!entry || !buf) return null;

  // Every field below is optional on the entry and absent on some rows, so it
  // is read through the field view rather than off one row's shape.
  const def = entry as AudioFields;

  const ctx = this.ctx;
  const now = ctx.currentTime;

  const gap = opts?.minGap ?? num(def.minGap) ?? 0;
  if (gap > 0 && now - (this._last[key] ?? -1e9) < gap) return null;
  if (this._voices >= MAX_VOICES) return null;
  this._last[key] = now;

  // One voice per exclusive group. The group defaults to the key itself, so
  // re-casting a spell cuts its own sound and leaves every other spell's
  // alone.
  const solo = opts?.exclusive ?? def.exclusive;
  const group: string = solo === true ? key : (typeof solo === "string" ? solo : "");
  if (group) this._releaseSolo(group, now);

  const src = ctx.createBufferSource();
  src.buffer = buf;
  const rate = opts?.rate ?? num(def.rate) ?? 1;
  src.playbackRate.value = rate;

  const g = ctx.createGain();
  const gain = (opts?.gain ?? 1) * (num(def.gain) ?? 1);
  g.gain.value = gain;

  src.connect(g).connect(this._bus(opts?.bus ?? (def.bus as string | undefined)));

  this._voices++;
  src.onended = () => {
   this._voices--;
   src.disconnect();
   g.disconnect();
   // Only clear the slot if it is still this voice: a newer one may have
   // taken it over, and this handler runs after that voice has started.
   if (group && this._solo[group]?.src === src) delete this._solo[group];
  };

  src.start(now + (opts?.delay ?? 0));
  if (group) this._solo[group] = { src, gain: g };
  return src;
 }

 /**
  * Fade out and stop whatever holds an exclusive group, so a new voice can
  * take it. Ramped rather than stopped outright — cutting a buffer mid-cycle
  * is a click.
  *
  * @param {string} group
  * @param {number} now context time
  */
 _releaseSolo(group: any, now: any) {
  const live = this._solo[group];
  if (!live) return;
  delete this._solo[group];
  const p = live.gain.gain;
  p.cancelScheduledValues(now);
  p.setValueAtTime(p.value, now);
  p.linearRampToValueAtTime(0, now + SOLO_RELEASE);
  // `onended` does the disconnecting and the voice-count bookkeeping.
  try {
   live.src.stop(now + SOLO_RELEASE + 0.005);
  } catch {
   // Already stopped — nothing to release.
  }
 }

 /**
  * Build a continuous voice for a looped asset. Returned stopped and silent;
  * the caller starts it and rides its gain.
  *
  * @param {string} key manifest key, must be `loop: true`
  * @param {{ gain?: number, rate?: number, bus?: string }} [opts]
  * @returns {LoopVoice|null}
  */
 loop(key: string, opts?: PlayOpts) {
  const entry = key in AUDIO_MANIFEST ? AUDIO_MANIFEST[key as AudioKey] : undefined;
  const buf = this.buffers.get(key);
  if (!entry || !buf || !this.ctx) return null;
  // `loop` is declared on the rows that declare it and read as optional.
  const def = entry as AudioFields;
  if (!def.loop) {
   console.warn("[audio] '" + key + "' is not declared as a loop");
   return null;
  }
  const voice = new LoopVoice(this, key, entry, buf, opts);
  this._loops.push(voice);
  return voice;
 }

 /** Stop every loop and let one-shots ring out. */
 stopAll(fade = 0.4) {
  for (let i = 0; i < this._loops.length; i++) this._loops[i].stop(fade);
  this._loops.length = 0;
 }
}

/**
 * One continuous sound.
 *
 * Two modes, chosen by whether the manifest entry declares a `crossfade`:
 *
 *  - **native** — a single source with `loop = true` and the loop points pulled
 *    inside the encoder padding. Cheap, and the only mode whose playback rate
 *    can be modulated, which the surf loop needs.
 *
 *  - **overlap** — segments scheduled back to back with an equal-power-ish
 *    crossfade over the join. Costs a timer and two live sources, and in exchange
 *    it does not care whether the file's head and tail match, which is the
 *    difference between a wind bed that breathes and one that ticks every
 *    seventeen seconds.
 */
export class LoopVoice {
 [key: string]: any;
 /**
  * @param {AudioEngine} engine
  * @param {string} key
  * @param {import("./manifest.js").AudioAsset} def
  * @param {AudioBuffer} buf
  * @param {{ gain?: number, rate?: number, bus?: string }} [opts]
  */
 constructor(engine: any, key: any, def: any, buf: any, opts: any) {
  this.engine = engine;
  this.key = key;
  this.buf = buf;
  this.unity = def.gain ?? 1;
  this.rate = (opts?.rate ?? 1) * (def.rate ?? 1);
  this.trim = def.trim ?? 0;
  this.crossfade = def.crossfade ?? 0;
  /** Playable seconds between the trims. */
  this.segment = Math.max(0.05, buf.duration - this.trim * 2);
  if (this.crossfade > this.segment * 0.45) this.crossfade = this.segment * 0.45;

  const ctx = /** @type {AudioContext} */ (engine.ctx);
  this.gainNode = ctx.createGain();
  this.gainNode.gain.value = (opts?.gain ?? 0) * this.unity;
  this.gainNode.connect(engine._bus(opts?.bus ?? def.bus));

  /** @type {AudioBufferSourceNode|null} native mode's single source */
  this._src = null;
  /** Context time the next overlap segment should start at. */
  this._nextAt = 0;
  /** True until the first overlap segment is queued; see `_spawn`. */
  this._first = false;
  /** @type {number} overlap-mode scheduler handle */
  this._timer = 0;
  this.running = false;
 }

 /** Begin playback. Idempotent. */
 start() {
  if (this.running || !this.engine.unlocked) return;
  const ctx = /** @type {AudioContext} */ (this.engine.ctx);
  this.running = true;

  if (this.crossfade > 0) {
   // As close to "now" as scheduling allows: a few milliseconds of head
   // room against the audio thread, not a musical offset.
   this._nextAt = ctx.currentTime + 0.005;
   // The first segment has nothing to crossfade against, so it starts at
   // full level. Fading it in over the overlap window would make every
   // entrance a slow swell no matter what the caller asked for — which
   // is the one thing a bed that should already be there must not do.
   this._first = true;
   this._pump();
   // 4 Hz against a 0.6 s look-ahead: comfortably ahead of the schedule
   // even if the main thread is busy, and free enough to ignore.
   this._timer = setInterval(() => this._pump(), 250);
   return;
  }

  const src = ctx.createBufferSource();
  src.buffer = this.buf;
  src.loop = true;
  src.loopStart = this.trim;
  src.loopEnd = this.trim + this.segment;
  src.playbackRate.value = this.rate;
  src.connect(this.gainNode);
  src.start(ctx.currentTime, this.trim);
  this._src = src;
 }

 /** Queue overlap segments up to the look-ahead horizon. */
 _pump() {
  if (!this.running) return;
  const ctx = /** @type {AudioContext} */ (this.engine.ctx);
  // A suspended context's clock is frozen, so this loop simply does not
  // fire — which is exactly the behaviour a hidden tab wants.
  let guard = 8;
  while (this._nextAt < ctx.currentTime + LOOKAHEAD && guard-- > 0) {
   this._spawn(this._nextAt);
  }
 }

 /** @param {number} at context time */
 _spawn(at: any) {
  const ctx = /** @type {AudioContext} */ (this.engine.ctx);
  const xf = this.crossfade;
  const seg = this.segment;

  const src = ctx.createBufferSource();
  src.buffer = this.buf;
  src.playbackRate.value = this.rate;

  const g = ctx.createGain();
  if (this._first) {
   // Open at level — see `start()`. A couple of milliseconds of ramp
   // anyway, because a buffer that begins on a non-zero sample and is
   // switched on at full gain is a click.
   this._first = false;
   g.gain.setValueAtTime(0, at);
   g.gain.linearRampToValueAtTime(1, at + 0.006);
  } else {
   g.gain.setValueAtTime(0, at);
   g.gain.linearRampToValueAtTime(1, at + xf);
  }
  g.gain.setValueAtTime(1, at + seg - xf);
  g.gain.linearRampToValueAtTime(0, at + seg);

  src.connect(g).connect(this.gainNode);
  src.onended = () => {
   src.disconnect();
   g.disconnect();
  };
  src.start(at, this.trim, seg + 0.02);
  src.stop(at + seg + 0.01);

  // The next segment starts where this one begins fading out, so the pair
  // sums to roughly constant energy across the join.
  this._nextAt = at + seg - xf;
 }

 /**
  * Follow a continuously varying level. Safe to call every frame — this is a
  * one-pole approach to the target, not a per-frame parameter step.
  * @param {number} to 0..1 of the asset's unity gain
  * @param {number} [smooth] time constant, seconds
  */
 set(to: any, smooth = 0.09) {
  if (!this.running) return;
  const ctx = /** @type {AudioContext} */ (this.engine.ctx);
  this.gainNode.gain.setTargetAtTime(to * this.unity, ctx.currentTime, smooth);
 }

 /**
  * An explicit fade with a known duration — for entrances and exits, where
  * "arrives at the target" matters and `set`'s asymptote does not.
  * @param {number} to 0..1 of the asset's unity gain
  * @param {number} seconds
  */
 fade(to: any, seconds: any) {
  if (!this.running) return;
  const ctx = /** @type {AudioContext} */ (this.engine.ctx);
  const now = ctx.currentTime;
  const p = this.gainNode.gain;
  p.cancelScheduledValues(now);
  p.setValueAtTime(p.value, now);
  p.linearRampToValueAtTime(to * this.unity, now + Math.max(0.01, seconds));
 }

 /**
  * Retune. Native mode only — an overlap schedule is laid out in real seconds,
  * so changing the rate under it would walk the joins out of alignment.
  * @param {number} r
  */
 setRate(r: any) {
  this.rate = r;
  if (!this._src) return;
  const ctx = /** @type {AudioContext} */ (this.engine.ctx);
  this._src.playbackRate.setTargetAtTime(r, ctx.currentTime, 0.08);
 }

 /** @param {number} [fadeSeconds] */
 stop(fadeSeconds = 0.25) {
  if (!this.running) return;
  // Fade before clearing the flag — `fade` no-ops on a stopped voice.
  this.fade(0, fadeSeconds);
  this.running = false;
  if (this._timer) {
   clearInterval(this._timer);
   this._timer = 0;
  }
  const ctx = /** @type {AudioContext} */ (this.engine.ctx);
  const src = this._src;
  if (src) {
   src.stop(ctx.currentTime + fadeSeconds + 0.02);
   this._src = null;
  }
 }
}

/**
 * The engine is a singleton because an `AudioContext` is: browsers cap how many
 * a page may create, and a second mixer would be a second set of faders for the
 * same output.
 */
export const audio = new AudioEngine();
