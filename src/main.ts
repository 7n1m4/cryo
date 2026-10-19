/**
 * SNOWFLOW — entry point and frame orchestration.
 *
 * WebGL2 only, by design (this build; the original was WebGPU-only). Three.js
 * is a rasterizer here — no feature-detect branches: if the context or the
 * float render targets aren't there we say so once and stop.
 */

import * as THREE from "three";

import { registerShaders } from "./shaders/registry.js";
import { S, onChange, applyPreset, type SettingValue } from "./core/settings.js";
import {
 sample, checkSpike, stats, mark, installDrawCounter, endFrameDraws,
} from "./core/perf.js";
import { initInput, pollInput, endFrame, input, isDown } from "./core/input.js";
import { CameraRig, expDamp } from "./core/camera.js";
import { createGfx } from "./core/gfx.js";
import { CharacterController } from "./character/controller.js";
import { Character } from "./character/character.js";
import { SnowContact } from "./character/snowContact.js";
import { SprayField } from "./vfx/particles.js";

import { MuzzleMarkers } from "./vfx/muzzleMarkers.js";
import { SmokeTrails } from "./vfx/smokeTrails.js";
import { SurfWake } from "./vfx/surfWake.js";
import { SpellSystem } from "./spells/spellSystem.js";
import { Speeder } from "./player/speeder.js";
import { CollisionPass, emitBurnTrail } from "./physics/collisionPass.js";
import { Overlay } from "./ui/overlay.js";
import { createFpsMeter } from "./ui/fpsMeter.js";
import { createPerfHud } from "./ui/perfHud.js";
import { Sky } from "./render/sky.js";
import { ShadowSystem } from "./render/shadows.js";
import { Terrain } from "./terrain/terrain.js";
import { DepthPass } from "./render/depthPass.js";
import { PostChain } from "./post/postChain.js";
import { whenReady } from "./core/gpuUtil.js";
import * as loading from "./core/loading.js";
import { audio } from "./audio/engine.js";
import { createSoundButton } from "./ui/soundButton.js";
import { createTouchControls } from "./ui/touchControls.js";
import { OPENING, applyOpening, installShotCapture } from "./core/openingShot.js";
import { loadModelAsset } from "./walkers/modelAsset.js";

// --------------------------------------------------------- recovered contracts
// The flight state `main.ts` carries for the code the walker/escort purge left
// behind. Both arrays are always empty at runtime — nothing populates them any
// more — but the collision pass and the presentation loop are still written
// against *lists* of craft, and their loops stay.

/**
 * The console debug handle `window.SNOWFLOW`.
 *
 * Nothing in this repo reads it — the only access is from the browser console,
 * so the global is genuinely unowned by any type here. It is declared rather
 * than reached through an `as unknown as` double cast so the one assignment
 * below needs no suppression and the shape it publishes is checked.
 */
declare global {
 interface Window {
  SNOWFLOW?: unknown;
 }
}

/** An escort. Its module was removed; the shape is expressed by its uses. */
type Wingman = {
 effects: {
  smoke: SmokeTrails,
  enemy: EnemyFire[],
  raid: { turn: number, size: number },
  onKillHit: (x: number, y: number, z: number) => void,
  onBounce: (x: number, y: number, z: number, facing: number) => void,
  onCrash: (x: number, y: number, z: number, facing: number) => void,
 },
 pilot: { bounced: boolean, _fx: unknown, position: THREE.Vector3, facing: number },
 bounced: boolean,
 _bounceT: number,
 flyover: (target: THREE.Vector3, y: number, distance: number) => void,
 warmUp: () => unknown,
 tick: (dt: number) => void,
 update: (dt: number) => void,
 sync: (cameraPos: THREE.Vector3) => void,
 triangles: number,
};

/** The infantry's reaction pipe, read (never written) through the effects bag. */
type EnemyFire = unknown;

/**
 * The controller stand-in a wreck's hull is hung off.
 *
 * A real `CharacterController` minus the flight machinery — the wreck is never
 * integrated, so `ctl` is only ever read for pose numbers, which is why the
 * wreck build constructs these fields by hand rather than taking a controller.
 */
type WreckController = {
 position: THREE.Vector3,
 velocity: THREE.Vector3,
 facing: number,
 lean: number,
 steerRate: number,
 speed01: number,
 speedRaw: number,
 climb: number,
 lift01: number,
 pathY: number,
 groundY: number,
 driveHeld: boolean,
 boostHeld: boolean,
 fireHeld: boolean,
};

/** One hull in the graveyard: the player's own craft, stilled after the crash. */
type Wreck = {
 hull: Speeder,
 ctl: WreckController,
 active: boolean,
 burnT: number,
 age: number,
 smokeT: number,
 pilotX: number,
 pilotZ: number,
};

/**
 * The fake burst `SNOWFLOW.squadImpact` publishes, and the pass is handed.
 *
 * Deleted with the trooper herd: nothing calls it now. The binding stays so
 * both publish sites keep resolving, and it is `undefined` rather than absent
 * so a compile-time error is raised the moment it is actually called instead
 * of the call silently vanishing — an inert purge remnant. `| undefined` also
 * keeps tsc from narrowing a singly-assigned `const` to the literal type,
 * which would make the declaration itself dead code. It is suppressed at the
 * two publish sites with a non-null assertion rather than `?.`, so calling
 * the console handle throws the TypeError instead of no-oping.
 */
const squadImpact: ((x: number, y: number, z: number) => void) | undefined = undefined;

// ------------------------------------------------------- module-scope scratch
const _vel = new THREE.Vector3();
/** Scratch: where the crash camera is pointed. See the death-cam block. */
const _camAt = new THREE.Vector3();

/** Beauty clear — linear (§2.6). */
const CLEAR_COLOR = [0.02, 0.03, 0.05, 1];

/**
 * Metres at which a craft coming apart stops being audible.
 *
 * Deliberately more than double the cannons' 420 m (`SHOT_AUDIBLE` in the
 * soundscape): a chin cannon is one gun among many and stops mattering across
 * the line, while a snowspeeder exploding is the loudest event in the battle
 * and the one the player most wants to hear happen. The falloff is the same
 * cubed curve either way, so a crash at 700 m is a faint distant thump — which
 * is the point — rather than nothing at all.
 */
const CRASH_AUDIBLE = 900;

/**
 * The crash camera.
 *
 * `RATE` is radians a second around the wreck — roughly forty degrees, so the
 * fall and the wait together carry the shot well past a full circle. It was
 * a third of this to begin with and the orbit was real but invisible: a
 * hundred degrees spread over five seconds, from a camera already sitting
 * behind the craft, reads as the camera simply drifting rather than as a shot
 * moving around something. `DIST` stands the frame off far enough to hold the
 * whole hull, the crater and the machines still walking past behind it, and
 * `PITCH` looks down on it rather than at it.
 */
const DEATH_CAM_RATE = 0.72;
const DEATH_CAM_DIST = 26;
/** …and the standoff once the subject is a man rather than a machine. */
const DEATH_CAM_DIST_PILOT = 13;
/** Metres short of the battle a respawning craft comes back in at — near
 *  enough to be in the fight within seconds, far enough to be a flight back
 *  rather than a teleport into the middle of it. */
const RESPAWN_OUT = 260;
const DEATH_CAM_PITCH = 0.42;
/**
 * …and the pitch once the subject is the pilot: nearly level with the snow.
 *
 * Looking down at 0.42 is the right angle on a wrecked machine — it shows the
 * crater and the skid line. It is the wrong angle on a man lying flat, which
 * from above is a shape on a white field. Down at the deck he is *in front of*
 * the camera, read against the sky and the burning hull behind him.
 */
const DEATH_CAM_PITCH_PILOT = 0.1;



async function boot() {
 const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById("view"));

 // A coarse pointer means a phone or a tablet, and a phone is a tile-based GPU
 // with a fraction of the bandwidth this was tuned against. Drop to the
 // balanced preset *here*, before anything is built: the deformation buffer's
 // resolution is read once at construction, so a preset applied after the
 // terrain exists would leave the single largest target at its desktop size.
 if (window.matchMedia?.("(pointer: coarse)").matches) {
  applyPreset("balanced");
  S.resolutionScale = 0.7;
 }

 // Kick the audio download off before the device exists and await it at the
 // far end of the boot sequence. It is the only thing here that touches the
 // network, and it has nothing to do with the GPU work that follows — so it
 // runs underneath all of it and is, on any warm cache, long finished by the
 // time anything asks. Not awaited here on purpose; see "loading audio" below.
 const audioReady = audio.load();
 // The intro transmission rides the same reasoning: a megabyte of video
 // fetched to a blob underneath the GPU work, so the boot gate's click
 // plays local bytes instantly. Stub-tolerant — a miss costs the film,
 // not the boot, and the audio-only line takes over.
 // The hosted cut is the ONE source, deliberately: a local copy in
 // `public/video/` used to be tried first and win, which meant a freshly
 // uploaded clip silently lost to a stale file nobody remembered shipping.
 // Only the speeder (player) needs a baked asset now — the walker
 // herd, AT-ST escort and trooper infantry were purged.
 // Only when it is actually going to fly. Off — which is the default — the
 // speeder costs nothing at all: no fetch, no decode, no pipeline.
 // Build-time gate: whether this build includes the ship asset. NOT runtime mode.
 // Runtime mode is the `flying` variable below, toggled live by toggleMode().
 const FLYING = S.speeder === true;
 let flying = FLYING;
 // `srgbAlbedo` is deliberately *off*.
 //
 // The hull reading dark looked like an sRGB-decoded map being read as linear,
 // and it is not: decoding it moves the mid-tones down and the craft got
 // darker still, when a T-47 is a light grey aircraft. So the maps arrive in
 // the space they are wanted in and something else is eating the value. The
 // option stays because the next model through here may genuinely need it.
 // Latched at boot like S.speeder — the toggle needs a reload to *create* the
 // craft, and only hides it live.
 const speederReady = FLYING ? loadModelAsset("models/speeder") : null;

 await loading.phase("creating device", 0.05);

 // WebGL2 + EXT_color_buffer_float gate lives inside createGfx; it fails
 // the boot screen itself with the precise reason, so a null is just "stop".
 const gfx = createGfx(canvas);
 if (!gfx) return;

 const applyScale = () => gfx.setRenderScale(S.resolutionScale);
 applyScale();
 onChange("resolutionScale", applyScale);
 window.addEventListener("resize", () => gfx.resize());

 installDrawCounter(gfx);
 registerShaders();

 await loading.phase("building scene", 0.12);

 const rig = new CameraRig(gfx, canvas);

 // ------------------------------------------------------------------ sky
 await loading.phase("integrating atmosphere", 0.2);
 const sky = new Sky(gfx);
 await sky.solve();

 // -------------------------------------------------------------- shadows
 const shadows = new ShadowSystem(gfx);

 // The camera-space depth prepass. Babylon scheduled it by registration
 // order; here the schedule is the explicit pass list in `renderFrame`.
 const depthPass = new DepthPass(gfx);

 // -------------------------------------------------------------- terrain
 await loading.phase("baking heightfield", 0.34);
 const terrain = new Terrain(gfx, sky, shadows);
 await terrain.build();
 onChange("showTerrain", (v) => {
  if (terrain.mesh) terrain.mesh.visible = v;
 });
 depthPass.registerCaster(terrain.mesh, terrain.makePrepassMaterial());

 await loading.phase("placing character", 0.62);

 const character = new CharacterController(terrain);
 character.position.set(0, 0, 0);
 character.position.y = terrain.heightAt(0, 0);
 // Flying, the arm is a chase camera and A/D drags it round with the nose —
 // so it has to *start* behind the nose. The figure's opening bearing is a
 // composition (it frames the walkers); a craft's is simply where it points.
 if (S.speeder === true) rig.yaw = character.facing;

 /** @type {CharacterController|null} */
 const shipController = S.speeder === true ? new CharacterController(terrain) : null;
 if (shipController) {
  shipController.position.copy(character.position);
  shipController.facing = character.facing;
 }

 // The figure: skeleton, garment simulation, shell fur. Hidden while the
 // player is flying — it is still simulated, because the speeder reads the
 // same controller and the cloth solve costs a fraction of a millisecond, but
 // nothing draws it.
 const figure = new Character(gfx, terrain, sky, shadows, character);
 figure.registerPrepass(depthPass);
 const showFigure = () => figure.setVisible(S.showCharacter !== false && !flying);
 onChange("showCharacter", showFigure);

 // Airborne snow: footfall kick now, the surf plume and spell spray later.
 const spray = new SprayField(gfx, terrain, sky, shadows);
 // Tuning rings on the gun heads, live under the overlay's muzzle sliders.
 const muzzleMarkers = new MuzzleMarkers(gfx);
 muzzleMarkers.bindCamera(rig.camera.viewProjection);
 // Damage smoke and fire off wounded craft — the wingmen's ladder.
 const smoke = new SmokeTrails(gfx);
 smoke.bindCamera(rig.camera.viewProjection);

 // ------------------------------------------------------------ the speeder
 /** @type {Speeder|null} */
 const speeder = FLYING
  ? new Speeder(gfx, terrain, sky, shadows, await speederReady, shipController, spray)
  : null;
 speeder?.registerPrepass(depthPass);

 /**
  * The escorts. The escort module was removed with the walker herd, so
  * nothing populates this and it is always empty at runtime — but it is
  * retained, because the collision pass and the presentation loop are
 *  written against a *list* of other craft (the wingman is one of n, not
 *  a singleton) and rewriting them to drop the loop is the very code loss
 *  this file has already suffered twice.
 * @type {Wingman[]}
 */
 const wingmen: Wingman[] = [];
 // Where a cannon bolt lands, the infantry reacts — the player's guns report
 // through this handler. Three rings around the burst:
 //
 //   < 1.7 m   a hit. One of the three deaths, held on its final frame; the
 //             body lies in the snow a while, then the herd recycles it
 //             back in at its squad's station like any other re-entry.
 //   < 6 m     a near miss. The trooper dives *away from* the burst —
 //             jumpLeft for an impact on his right, jumpRight for his
 //             left — then the get-up, then the walk again.
 //   < 9 m     close enough to flinch.
 //
 // The get-up starts five and a half seconds in: Mixamo's clip spends its
 // first act lying still and rolling over, and a soldier who dove clear of
 // a shellburst stirs and rises rather than napping through the battle.
 // Walkers purged: no AT-AT/AT-ST bolt chains remain.

 let wrecks: Wreck[] = [];
 let leaveWreck: ((x: number, z: number, facing: number) => null) | null = null;
 if (FLYING) {
  const speederAsset2 = await speederReady;
  wrecks = [];
  const ctl = {
   position: new THREE.Vector3(0, -500, 0),
   velocity: new THREE.Vector3(),
   facing: 0, lean: 0.32, steerRate: 0,
   speed01: 0, speedRaw: 0,
   climb: -2.35, lift01: 0, pathY: 0, groundY: 0,
   driveHeld: false, boostHeld: false, fireHeld: false,
  };
  const hull = new Speeder(gfx, terrain, sky, shadows, speederAsset2, ctl, null);
  hull.grounded = true;
  hull.registerPrepass(depthPass);
  hull.setVisible(false);
  wrecks.push({
   hull, ctl, active: false, burnT: 0, age: 0, smokeT: 0,
   pilotX: 0, pilotZ: 0,
  });
  leaveWreck = (x: number, z: number, facing: number) => {
   const wk = wrecks[0];
   if (!wk) return null;
   wk.active = true; wk.burnT = 32; wk.age = 0; wk.smokeT = 0;
   wk.ctl.position.set(x, 0, z); wk.ctl.facing = facing;
   wk.ctl.lean = (Math.random() < 0.5 ? -1 : 1) * (0.25 + Math.random() * 0.15);
   wk.ctl.groundY = terrain.heightAt(x, z); wk.ctl.pathY = wk.ctl.groundY;
   wk.hull.snowOverride = 0; wk.hull.setVisible(true);
   const side = facing + (Math.random() < 0.5 ? 1 : -1) * (0.9 + Math.random() * 0.7);
   const throwOut = 4 + Math.random() * 1.5;
   wk.pilotX = x + Math.sin(side) * throwOut;
   wk.pilotZ = z + Math.cos(side) * throwOut;
   return null;
  };
 }
 /**
  * Where the wingmen's guns report a burst: the infantry's reaction pipe,
  * handed to each escort's `effects` bag as `enemy`. Nothing writes it
 *  (the trooper herd is gone), but the bag's shape still carries the key
 *  and the pass still reads it.
 * @type {EnemyFire[]}
 */
 const enemyFire: EnemyFire[] = [];
 // The shared fate token: one ship wounded and falling at a time, ever.
 const raid = { turn: 0, size: Math.max(1, wingmen.length) };
 /** Milliseconds inside which a second announcement of the same skip is
  *  the same skip. The two ground contacts of one fall are ~1.5 s apart
  *  and the crash does not route through `onBounce` at all, so anything
  *  under that is safe; a full second is comfortably clear of the
  *  duplicate-in-the-same-frame case it exists for. */
 const BOUNCE_DEBOUNCE = 1000;
 for (const w of wingmen) {
  w.effects = {
   smoke,
   enemy: enemyFire,
   raid,
   // The killing hit: a burst on the airframe itself, in the air —
   // the fireball the dying machine then flies out of. The volume's
   // centre normally sits LIFT above a ground impact, so the y is
   // walked down to land the burst at hull height.
   //
   // This is also where the *whole fall* starts speaking. Since the
   // crash choreography was re-timed onto `speeder-crash-2.mp3`, that
   // sample is no longer a report — it is the soundtrack of the next
   // seven and a half seconds: the hit, the long windmilling descent,
   // the first ground impact at 4.0 s, the final crash at 5.5 s and
   // the debris settling after. So it is fired exactly ONCE, here at
   // t = 0, and nothing downstream is allowed to speak again — not
   // the skip, not `onCrash`. Playing it at t=0 is what puts its
   // internal 4.0 s and 5.5 s marks on the frames where the craft
   // actually touches the snow; delaying it or restarting it at the
   // ground would smear the whole sequence out of sync.
   onKillHit: (x: number, y: number, z: number) => {
    // The cannon distance model, to the letter: linear to nothing
    // at 420 m, CUBED so it does not read as loud from across the
    // battle line, and 343 m/s of honest travel time on top.
    const kd = Math.hypot(
     x - character.position.x, z - character.position.z
    );
    // …but over a longer reach than a chin cannon. 420 m is the
    // range at which one gun firing stops mattering; a snowspeeder
    // coming apart is the loudest thing on this battlefield, and
    // it is *falling for five and a half seconds* — a fall that
    // usually carries it a couple of hundred metres closer to the
    // player before it lands. Cut at 420 and the deaths that read
    // as most cinematic, the ones out over the line, happen in
    // total silence. The cube keeps a distant one honestly faint.
    const kn = Math.max(0, 1 - kd / CRASH_AUDIBLE);
    if (kn > 0) {
     audio.play("speederCrash2", {
      gain: kn * kn * kn,
      delay: Math.min(1.6, kd / 343),
     });
    }
   },
   /**
    * The skip — first ground contact at t ≈ 4.0 s, the craft striking
    * the snow and bouncing back off it rather than ending there.
    *
    * Everything the crash does, one size down and one register
    * quieter: the same plume-and-ring splash shape, a short shallow
    * gouge along the skid line, a small dirty puff. Deliberately no
    * visual burst — the airframe is still whole, it is only touching —
    * and deliberately NO sound: `speeder-crash-2.mp3` has this exact
    * impact baked into it at 4.0 s and has been playing since the
    * killing hit, so anything fired here would double it.
    *
    * See `pumpBounces` for how this is reached: it is called both
    * ways, as a callback and off a one-frame latch. The debounce
    * lives here rather than at either call site precisely because
    * there are two of them — a pilot that announces the skip both
    * ways still only skips once, and a fall skips once by nature
    * (the next contact is the crash, and a respawn separates falls).
    */
   onBounce: (x: number, y: number, z: number, facing: number) => {
    const nowMs = performance.now();
    if (nowMs - (w._bounceT ?? -1e9) < BOUNCE_DEBOUNCE) return;
    w._bounceT = nowMs;
    const gy = terrain.heightAt(x, z);
    const fx2 = Math.sin(facing ?? 0), fz2 = Math.cos(facing ?? 0);
    // The scuff: shallow, long, laid along the direction of travel.
    terrain.deform.brush(
     x, z, 2.2, 0.3, 0.22, 0.75, 0.2, facing ?? 0, 3.4, 0.9
    );
    // The splash, at a third of the crash's count and half its
    // throw — a skip lifts a sheet of powder, not a column.
    for (let k = 0; k < 20; k++) {
     const a = Math.random() * Math.PI * 2;
     const out = 3 + Math.random() * 6;
     spray.emit(
      x + Math.cos(a) * 0.4, gy + 0.2, z + Math.sin(a) * 0.4,
      Math.cos(a) * out + fx2 * 4,
      3 + Math.random() * 6,
      Math.sin(a) * out + fz2 * 4,
      0.05 + Math.random() * 0.07, 0.5 + Math.random() * 0.7,
      Math.random() < 0.4 ? 1 : 0, 1.2
     );
    }
    // The low ring racing out from the touch point.
    for (let k = 0; k < 12; k++) {
     const a = (k / 12) * Math.PI * 2 + Math.random() * 0.3;
     const out = 6 + Math.random() * 5;
     spray.emit(
      x + Math.cos(a) * 1.0, gy + 0.15, z + Math.sin(a) * 1.0,
      Math.cos(a) * out, 1.0 + Math.random() * 1.6,
      Math.sin(a) * out,
      0.04 + Math.random() * 0.05, 0.45 + Math.random() * 0.4,
      1, 1.5
     );
    }
    // A puff, not a ball: grey, small, gone in a few seconds.
    for (let k = 0; k < 5; k++) {
     const a = Math.random() * Math.PI * 2;
     const rr = Math.random() * 0.9;
     smoke.emit(
      x + Math.cos(a) * rr, gy + 0.8 + Math.random() * 1.0,
      z + Math.sin(a) * rr,
      (Math.random() - 0.5) * 1.6 + fx2 * 2,
      0.6 + Math.random() * 1.0,
      (Math.random() - 0.5) * 1.6 + fz2 * 2,
      0.9 + Math.random() * 0.7, -0.1,
      2.4 + Math.random() * 2.0,
      0.09, 0.09, 0.09, 0.5, true
     );
    }
   },
   onCrash: (x: number, y: number, z: number, facing: number) => {
    // No report here on purpose. `speederCrash2` started at the
    // killing hit and its own timeline reaches the final crash at
    // 5.5 s, which is this frame — a second sample fired now would
    // land on top of the one already playing and smear the very
    // moment it is meant to punctuate. The fireball, the crater
    // and the snow are the whole of what this handler contributes.
    //
    // The arrival scar: the crater it dug and the trench it
    // ploughed getting there.
    terrain.deform.brush(x, z, 3.4, 0.85, 0.6, 1.0, 0.6, 0, 1.3, 1.0);
    terrain.deform.brush(
     x - Math.sin(facing) * 4.5, z - Math.cos(facing) * 4.5,
     2.4, 0.5, 0.55, 1.0, 0.3, facing, 3.0, 1.0
    );
    // The splash: a machine-sized slap of snow — a tall central
    // plume and a low fast ring racing out from the hull, the way
    // a heavy impact into powder actually reads.
    for (let k = 0; k < 60; k++) {
     const a = Math.random() * Math.PI * 2;
     const out = 4 + Math.random() * 11;
     spray.emit(
      x + Math.cos(a) * 0.5, y + 0.2, z + Math.sin(a) * 0.5,
      Math.cos(a) * out, 6 + Math.random() * 12, Math.sin(a) * out,
      0.06 + Math.random() * 0.1, 0.9 + Math.random() * 1.3,
      Math.random() < 0.4 ? 1 : 0, 1.2
     );
    }
    for (let k = 0; k < 30; k++) {
     const a = (k / 30) * Math.PI * 2 + Math.random() * 0.2;
     const out = 12 + Math.random() * 8;
     spray.emit(
      x + Math.cos(a) * 1.4, y + 0.15, z + Math.sin(a) * 1.4,
      Math.cos(a) * out, 1.5 + Math.random() * 2.5, Math.sin(a) * out,
      0.05 + Math.random() * 0.06, 0.6 + Math.random() * 0.6,
      1, 1.6
     );
    }
    // The smoke ball: a fat black cloud boiling up off the
    // impact, then *shrinking* — negative growth — as it thins
    // and lets go over the next ten seconds.
    for (let k = 0; k < 14; k++) {
     const a = Math.random() * Math.PI * 2;
     const rr = Math.random() * 1.8;
     smoke.emit(
      x + Math.cos(a) * rr, y + 1.2 + Math.random() * 2.2,
      z + Math.sin(a) * rr,
      (Math.random() - 0.5) * 1.2,
      0.7 + Math.random() * 1.4,
      (Math.random() - 0.5) * 1.2,
      1.8 + Math.random() * 1.6, -0.22,
      6 + Math.random() * 5,
      0.05, 0.05, 0.05, 0.6, true
     );
    }
    leaveWreck?.(x, z, facing);
   },
  };
  // The pilot's own back-channel to the same handlers. SimPilot is a
  // bare character controller — it has never held an effects object,
  // the Wingman wrapper does — but the skip happens deep inside the
  // downed ballistic, where only the pilot knows the frame the snow was
  // touched. Handing it the same bag lets it call `this._fx?.onBounce?.()`
  // directly if it wants to; if it would rather raise a one-frame latch
  // instead, `pumpBounces` below picks that up. Both roads lead here,
  // and the debounce makes taking both harmless.
  w.pilot._fx = w.effects;
 }

 /**
  * The skip, delivered whichever way the flight code chose to announce it.
  *
  * Two contracts are honoured because the fall is authored on the other
  * side of the wall and either shape is reasonable there:
  *
  *   the callback   the pilot (or the wrapper) calls `onBounce(x,y,z,facing)`
  *                  at the contact frame, through `_fx` or `effects`.
  *   the latch      the pilot raises `bounced = true` for one frame — the
  *                  same shape as the existing `crashed` latch — and this
  *                  the wrapper forwards it rather than the pilot raising it.
  *
  * A pilot that does both fires once — see the debounce inside `onBounce`.
  */
 const pumpBounces = () => {
  for (const w of wingmen) {
   const P = w.pilot;
   if (!P) continue;
   if (!P.bounced && !w.bounced) continue;
   P.bounced = false;
   w.bounced = false;
   w.effects?.onBounce?.(P.position.x, P.position.y, P.position.z, P.facing);
  }
 };

 // ------------------------------------------------------------ collisions
 // The craft meet the machines: swept spheres against the walkers' capsule
 // sets, the wrecks and each other, with the consequences routed to their
 // owners — deflection and scramble to the craft, a stumble to the walker,
 // snow, smoke, fire and the report to the field. Gated live by
 // S.collisions; every collaborator is optional, so a boot without a
 // speeder or without wingmen runs the parts it has.
 let storyMode = false;
 const collision = new CollisionPass({
  activeCharacter: () => (flying && shipController) ? shipController : character,
  speeder, wingmen,
  wrecks, terrain, spray, smoke, rig, audio, troopers: null,
  parked: () => {
   if (flying || !speeder) return null;
   if (wrecks.length && wrecks[0]?.active) return null;
   const dx = character.position.x - speeder.position.x;
   const dz = character.position.z - speeder.position.z;
   if (dx * dx + dz * dz < 3.5 * 3.5) return null;
   return speeder.position;
  },
  // Inert purge remnant: never assigned, so a call throws exactly as it
  // would have without the `| undefined` on the binding above. `?.` here
  // would turn that TypeError into a silent no-op.
  squadImpact: (x: number, z: number) => squadImpact!(x, 0, z),
  // Where the player's own crash ends up, and what happens after it.
  // Back in the air where the fight is, and flying immediately.
  //
  // Dying mid-battle is not the same event as arriving at one: the player's
  // own craft is staged back down the approach, not teleported into an echelon.
  // already making way, with control.
  onRestart: () => {
   if (S.speeder !== true) return;
   if (!storyMode) {
    // Free flight: back in the air where the fight is.
    const bx = character.position.x;
    const bz = character.position.z;
    const h = 0;
    const sc = shipController || character;
    sc.position.x = bx - Math.sin(h) * RESPAWN_OUT;
    sc.position.z = bz - Math.cos(h) * RESPAWN_OUT;
    sc.facing = h;
    sc.velocity.set(Math.sin(h) * 24, 0, Math.cos(h) * 24);
    character.position.copy(sc.position);
    character.facing = sc.facing;
    character.velocity.copy(sc.velocity);
    input.riseLevel = 3;
    flying = true;
    speeder?.setVisible?.(true);
    figure.setVisible(false);
    applyModeExtras();
    return;
   }
   // Story mode: strand the ship into the wreck and continue on foot.
   leaveWreck?.(character.position.x, character.position.z, character.facing);
   flying = false;
   speeder?.setVisible?.(false);
   character.position.y = terrain.heightAt(character.position.x, character.position.z);
   character.groundY = character.position.y;
   character.velocity.set(0, 0, 0);
   character.crashing = false; character.crashLanded = false; character.crashDown = false;
   applyModeExtras();
   showFigure();
  },
 });

 // The collision pass is told what is solid: the player's craft, the parked
 // hull, and the wreck graveyard. Every collaborator is optional, so a boot
 // without a speeder still runs the parts it has.

 /**
  * How near the line the player must get before Luke calls again, metres.
  * Wider than the flyby ring on purpose: the transmission is about having
  * *arrived at the battle*, not about passing under one machine, and the
  * square wants to be up before the shooting starts rather than during it.
  */
 const CROSSFIRE_AT = 165;
 let crossFired = false;
 speeder?.setVisible(true);
 showFigure();

 // Walkers purged: no herd spray/footfall wiring.

 // Feet and the surf groove write into the terrain state buffer through here.
 const contact = new SnowContact(character, terrain.deform, figure.figure, spray);

 // The breaking wave, its bow crest and the plume it sheds.
 const wake = new SurfWake(gfx, sky, shadows, character, spray, terrain);
 onChange("showWake", (v) => wake.setEnabled(v));
 wake.registerPrepass(depthPass);

 // The five spells, the water body they bend and the ice they leave.
 // Flying hides the whole snowboarding half of the demo: the spells are cast
 // by a figure that is not there, and the surf wake is thrown by a board.
 /** The figure's half of the demo is inert while the player is in the ship. */
 const applyModeExtras = () => {
  S.showSpells = !flying;
  S.showWake = !flying;
 };
 applyModeExtras();

 const spells = new SpellSystem(
  gfx, sky, shadows, terrain, character, figure.figure, rig, spray
 );
 // Every surface a spell can light.
 spells.addConsumers(
  terrain.material, figure.bodyMat, figure.clothMat,
  wake.material, spray.material
 );
 // Walkers purged: no herd materials.
 spells.registerPrepass(depthPass);

 // The rig needs ground heights to keep the spring arm above the snow.
 rig.groundAt = (x: number, z: number) => terrain.heightAt(x, z);

 // The opening shot, if one has been pinned. Before the warm-up, so the
 // frames rendered behind the loading screen are already the shot rather than
 // wherever the default rule put things.
 applyOpening(OPENING, rig, character, terrain);

 // The flight's entrance: the three escorts restaged behind the player at
 // altitude and speed, aimed at the centre of the walker line — the game
 // opens with the formation sweeping overhead toward the battle. Called on
 // the boot gate's click rather than here: the simulation runs behind the
 // boot screen, and a flyover staged now would be long gone before the
 // player ever saw the field.
 const startFlyover = () => {
  // The opening beat: hold the player's stick until the escorts have
  // passed overhead and Luke's line has landed. Sized off the actual
  // clip and the actual formation timing rather than guessed — the
  // last ship starts 160 m back at 32 m/s. The run loop counts this
  // down, freezing movement input (the look stays free) and seeding
  // the craft's forward way the moment the hold releases.
  // The voice is the film's when the film made it down, the mp3's when
  // it didn't — whichever is actually going to play. The film's voice
  // starts late by design: an open hold for the fly-in, then the static
  // tune-in, then Luke — `leadIn` is that whole runway.
  const lead = 0.5;
  const voice = audio.buffers.get("lukeIntro")?.duration ?? 3.5;
  introHold = Math.max(5.4, lead + voice + 0.4);
  if (!wingmen.length) return;
  for (const w of wingmen) w.flyover(character.position, 0, 300);
  // The player's own entrance: their craft staged 150 m back along the
  // formation's bearing, making way toward the spot the boot pinned.
  // Flown by feeding the controller's velocity each frame (the intro
  // block in the run loop), so the presentation, the camera and the
  // snow all read a genuinely moving craft rather than a teleport.
  if (S.speeder === true) {
   const h = Math.atan2(0 - character.position.x, 300 - character.position.z);
   introFly = {
    toX: 0, toZ: 300,
    heading: Math.atan2(0 - character.position.x, 300 - character.position.z),
   };
   character.position.x -= Math.sin(h) * 150;
   character.position.z -= Math.cos(h) * 150;
   character.facing = h;
  }
 };

 const post = new PostChain(gfx, rig, depthPass, sky);

 // ------------------------------------------------------------ the frame
 /**
  * The explicit render schedule (§4.3). The deform sim pass already ran
  * inside `terrain.update`; from here: cascades → prepass → beauty →
  * post chain. Beauty renders three layers into one target with a single
  * clear — depth persists from the opaques into the blended group.
  */
 const renderFrame = () => {
  gfx.syncTokenCamera(rig.camera);
  shadows.render(gfx);
  depthPass.render(gfx);
  gfx.runPass({
   target: post.sceneColor,
   clearColor: CLEAR_COLOR,
   clearDepth: true,
   layer: gfx.LAYER.SKY,
  });
  gfx.runPass({ target: post.sceneColor, layer: gfx.LAYER.OPAQUE });
  gfx.runPass({ target: post.sceneColor, layer: gfx.LAYER.BLEND });
  post.render(gfx);
 };

 // F2 snapshots the camera, the player and the herd as a paste-able block —
 // wired before the overlay so its "Start location · copy" button can share
 // the exact same capture. See `core/openingShot.js` for what to do with it.
 const captureShot = installShotCapture(rig, character);

 const overlay = new Overlay({ rig, character, actions: { captureShot } });
 const toggleOverlay = () => overlay.toggle();
 // The same two numbers the overlay carries, on their own, for when the panel
 // would cover the thing being measured.
 const fpsMeter = createFpsMeter();
 const perfHud = createPerfHud();
 initInput(canvas, {
  onToggleOverlay: toggleOverlay,
  onToggleFps: () => fpsMeter.toggle(),
 });
 // Builds hidden and reveals itself on the first real touch, so a desktop with
 // a touchscreen never sees it. Everything it produces goes into the same
 // `input` struct the keyboard writes.
 const touchControls = createTouchControls({ onToggleOverlay: toggleOverlay });

 // The soundscape reads game state; nothing in the game knows it exists. It
 // stays silent until `start()`, which only happens on the gesture that
 // unlocks the engine — the boot gate below, or the corner button later.
 const soundButton = createSoundButton(audio);
 // Mobile belt: some engines refuse an AudioContext resume that did not
 // ride a touch gesture, and a phone that entered through the gate without
 // sound would stay silent for ever. Any later touch retries the unlock;
 // once it lands, the listener has nothing left to do.
 window.addEventListener("pointerdown", () => {
  if (audio.hasAssets && !audio.unlocked) {
   audio.unlock().then((ok) => {
    if (ok) {
     soundButton.sync();
    }
   });
  }
 }, { passive: true });

 // ------------------------------------------------------------- warm-up
 // Everything that can compile, compiles here — behind the loading screen.
 await loading.phase("compiling pipelines", 0.78);
 shadows.update(rig.camera, sky.sunDir);
 sky.render(rig, 0);
 await terrain.warmUp();
 terrain.update(rig.camera.position, character.position, 0);
 figure.update(0);
 figure.sync(rig.camera.position);
 await figure.warmUp();
 // walkers/herds purged: no sync/warmUp.
 if (speeder) {
  speeder.update(0);
  speeder?.sync(rig.camera.position);
  await speeder.warmUp();
 }
 for (const w of wingmen) await w.warmUp();
 spray.update(0, rig.camera.position);
 await spray.warmUp();
 await wake.warmUp();
 await spells.warmUp(
  character.position.x + 3, character.position.y, character.position.z + 3
 );
 await whenReady(gfx, sky.material, "sky material");
 await depthPass.warmUp();
 post.update(0, 0, rig.distance);
 const passes = post.passes;
 for (let i = 0; i < passes.length; i++) {
  await whenReady(gfx, passes[i], "post:" + passes[i].name);
 }

 // The one-draw effect systems compile here too — an eye band or a smoke
 // puff whose shader compiles on its first visible frame is a hitch in the
 // opening seconds, which reads as the speeder stuttering.
 await whenReady(gfx, smoke.material, "smoke material");

 await loading.phase("warming render targets", 0.92);
 // A few real frames so every render target is allocated and every pipeline
 // has actually been bound at least once.
 for (let i = 0; i < 3; i++) {
  renderFrame();
  await loading.nextFrame();
 }
 // Only now: the spell meshes had to be standing *through* those frames for
 // their render pipelines to exist. See `WaterBody.warmUp`.
 spells.finishWarmUp();

 // ------------------------------------------------------------- run loop
 let prev = performance.now();
 let time = 0;
 /** Seconds of opening hold left — set by `startFlyover`, counted here. */
 let introHold = 0;
 /** Seconds into the player's crash, driving the camera's circle of the
  *  wreck. Zero whenever the craft is flying. */
 let deathCamT = 0;
 /** The framing the player had before the crash, handed back after it. */
 let deathCamDist = 0;
 /** The body the crash camera is circling — the player's own pilot, once
  *  they have been thrown clear. Null during the fall and after. */
 let deathCamPilot: { position: THREE.Vector3 } | null = null;
 /**
  * The cockpit's line to the pilot. Silent unless the player pushes the
  * stick during the opening hold — then "stand by" says the lock is
  * deliberate — and one "you have control" at the handover. `flightMsgT`
  * is the seconds left before the line fades.
  */
 const flightMsg = document.getElementById("flightmsg");
 let flightMsgT = 0;
 const showFlightMsg = (text: string, seconds: number, good?: boolean) => {
  if (!flightMsg) return;
  flightMsg.textContent = text;
  flightMsg.classList.toggle("good", !!good);
  flightMsg.classList.add("show");
  flightMsgT = seconds;
 };
 /** Cadence clock for the overlay's burn preview trail. */
 let burnPreviewT = 0;
 /**
  * The AT-AT line's advance bearing, latched from the opening line's own
  * mean heading on the first frame and never recomputed — the formation
  * marches *that* course for the whole battle, wherever the player goes.
  */
 let advanceBearing: number | null = null;
 /**
  * The player's fly-in, or null once they have arrived — set by
  * `startFlyover` alongside the escorts' own staging.
  */
 let introFly: { toX: number, toZ: number, heading: number } | null = null;

 let prevToggleKey = false;
 const toggleMode = () => {
  if (!speeder || !shipController) return;
  if (flying) {
   character.position.copy(shipController.position);
   character.facing = shipController.facing;
   terrain.heightfield.clampToPlayArea(character.position);
   terrain.heightfield.clampToPlayArea(shipController.position);
   flying = false;
   speeder.setVisible(true);
   speeder.grounded = false;
   showFigure();
   // Ground the figure at the surface instead of hover height.
   character.position.y = terrain.heightAt(character.position.x, character.position.z);
   character.groundY = character.position.y;
   character.velocity.y = 0;
   // Restore walk gait state; ship velocity is too high for stepping.
   character.surf = 0;
   character.surfActive = false;
   character.stepping = true;
   character.gaitPhase = 0;
   character.velocity.set(0, 0, 0);
   character.speed = 0; character.speed01 = 0; character.speedRaw = 0;
   character.lean = 0; character.streak01 = 0; character.carve = 0;
   character.crashing = false; character.crashLanded = false; character.crashDown = false;
   character.terrainSlam = null;
   character.cast = 0; character.castAimX = 0; character.castAimY = 0; character.castAimZ = 0;
   character.pathY = 0; character.lift01 = 0; character.impactSpin = 0;
   character.acceleration.set(0, 0, 0);
   character.footfall = false; character.footImpact = 0;
  } else {
   shipController.position.copy(character.position);
   shipController.velocity.copy(character.velocity);
   shipController.facing = character.facing;
   shipController.surf = 0;
   shipController.surfActive = false;
   shipController.stepping = true;
   shipController.gaitPhase = 0;
   shipController.cast = 0;
   shipController.impactSpin = 0;
   shipController._prevSpeed = 0;
   shipController._drive = 0;
   shipController._boost = 1;
   shipController._climb = S.speeder !== false ? Math.max(0, S.speederClimbMax || 0) : 0;
   shipController._climbWant = shipController._climb;
   shipController._vertEase = 0;
   shipController.climbRate = 0;
   shipController._cruiseGround = NaN;
   shipController.steer = 0;
   shipController.steerRate = 0;
   shipController.controlScramble = 0;
   shipController._scrambleMax = 0;
   shipController._scrambleT = 0;
   shipController.fireHeld = undefined;
   shipController.driveHeld = undefined;
   shipController.boostHeld = undefined;
   shipController._prevTargetY = NaN;
   shipController._slam = { closing: 0, nx: 0, nz: 0 };
   shipController.speed = 0;
   shipController.carve = 0;
   shipController.terrainSlam = null;
   terrain.heightfield.clampToPlayArea(shipController.position);
   terrain.heightfield.clampToPlayArea(character.position);
   flying = true;
   speeder.setVisible(true);
   figure.setVisible(false);
  }
  applyModeExtras();
 };
 // ------------------------------------------------------- main menu gate
 await new Promise<void>((resolve) => {
  const overlay = document.createElement("div");
  overlay.style.cssText = "position:fixed;inset:0;z-index:9999;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;background:radial-gradient(ellipse at 30% 20%,rgba(40,70,140,0.35),transparent 60%),radial-gradient(ellipse at 70% 80%,rgba(80,40,120,0.25),transparent 60%),rgba(4,6,14,0.92);color:#dfe6f5;font-family:ui-monospace,monospace;";
  const title = document.createElement("div");
  title.textContent = "SNOWFLOW";
  title.style.cssText = "font-size:1.6rem;letter-spacing:0.35em;text-transform:uppercase;font-weight:300;margin-bottom:4px;";
  const sub = document.createElement("div");
  sub.textContent = "S — story mode (ship stranded, repair to fly)    ·    Enter — free flight";
  sub.style.cssText = "font-size:0.72rem;letter-spacing:0.18em;text-transform:uppercase;opacity:0.6;margin-bottom:18px;";
  const btnStyle = "min-width:220px;padding:10px 20px;font-size:0.8rem;letter-spacing:0.2em;text-transform:uppercase;cursor:pointer;background:rgba(255,255,255,0.04);color:#dfe6f5;border:1px solid rgba(255,255,255,0.18);border-radius:2px;transition:background .15s,border-color .15s;";
  const storyBtn = document.createElement("button");
  storyBtn.textContent = "Story Mode  (S)";
  storyBtn.style.cssText = btnStyle;
  const flyBtn = document.createElement("button");
  flyBtn.textContent = "Free Flight  (Enter)";
  flyBtn.style.cssText = btnStyle;
  for (const b of [storyBtn, flyBtn]) {
   b.addEventListener("mouseenter", () => { b.style.background = "rgba(255,255,255,0.1)"; b.style.borderColor = "rgba(255,255,255,0.4)"; });
   b.addEventListener("mouseleave", () => { b.style.background = "rgba(255,255,255,0.04)"; b.style.borderColor = "rgba(255,255,255,0.18)"; });
  }
  overlay.append(title, sub, storyBtn, flyBtn);
  document.body.appendChild(overlay);
  // (overlay already appended above)

  const choose = (story: boolean) => {
   storyMode = story;
   overlay.remove();
   window.removeEventListener("keydown", onKey);
   resolve();
  };
  const onKey = (e: KeyboardEvent) => {
   if (e.code === "KeyS") choose(true);
   else if (e.code === "Enter") choose(false);
  };
  window.addEventListener("keydown", onKey);
  storyBtn.addEventListener("click", () => choose(true));
  flyBtn.addEventListener("click", () => choose(false));
 });

 if (storyMode) {
  const x = character.position.x, z = character.position.z, f = character.facing;
  leaveWreck?.(x, z, f);
  flying = false;
  speeder?.setVisible?.(false);
  applyModeExtras();
  showFigure();
 }

 gfx.renderer.setAnimationLoop(() => {
  const now = performance.now();
  let dtMs = now - prev;
  prev = now;
  if (dtMs > 100) dtMs = 100;
  const dt = S.freezeTime ? 0 : dtMs / 1000;
  time += dt;

  input.flying = flying;
  pollInput();

  // Mode toggle: exit/enter the ship.
  const toggleDown = isDown("KeyR");
  if (toggleDown && !prevToggleKey) {
   const wk = wrecks[0];
   if (!flying && wk && wk.active) {
    const dx = character.position.x - wk.ctl.position.x;
    const dz = character.position.z - wk.ctl.position.z;
    if (dx * dx + dz * dz <= 8 * 8) {
     // repair: bring the flyable ship back online at the wreck.
     // `wrecks` only fills when the build has a ship, which is exactly when
     // shipController and speeder were constructed — narrowing into
     // locals keeps that explicit rather than relying on it.
     const sc = shipController;
     const sp = speeder;
     if (sc && sp) {
      sc.position.copy(wk.ctl.position);
      sc.facing = wk.ctl.facing;
      sc.velocity.set(0, 0, 0);
      sp.position.copy(wk.ctl.position);
      character.position.copy(sc.position);
      character.facing = sc.facing;
      character.position.y = terrain.heightAt(character.position.x, character.position.z);
      character.groundY = character.position.y;
      wk.active = false;
      wk.hull.setVisible(false);
      toggleMode();
     }
    }
   } else {
    toggleMode();
   }
  }
  prevToggleKey = toggleDown;

  // The opening hold: escorts overhead, Luke on the wire, stick frozen.
  // Movement only — the mouse still looks around, which is the point of
  // an entrance. Releases with the craft already making way, so control
  // arrives mid-flight rather than from a standstill.
  if (introHold > 0) {
   introHold -= dt;
   // A push on the locked stick gets an answer, not silence: the
   // "stand by" line holds while they keep asking and fades ~1.5 s
   // after they stop.
   if (input.moving || input.fire || input.sprint
    || input.thrust || input.boost || input.vert !== 0) {
    showFlightMsg("⟨ stand by — escort inbound ⟩", 1.5);
   }
   input.moveX = 0;
   input.moveZ = 0;
   input.moving = false;
   input.sprint = false;
   input.thrust = false;
   input.boost = false;
   input.vert = 0;
   input.fire = false;
   if (introHold <= 0) {
    showFlightMsg("you have control", 2.6, true);
    if (shipController) {
     const sc = shipController;
     const f = sc.facing;
     sc.velocity.set(
      Math.sin(f) * 16, 0, Math.cos(f) * 16
     );
     character.position.copy(sc.position);
     character.facing = sc.facing;
     character.velocity.copy(sc.velocity);
    }
   }
   // The player's entrance: velocity fed straight into the
   // controller, which integrates, terrain-snaps and banks exactly
   // as it does under a stick. The approach slows as it closes, so
   // arrival hands over a craft settling in rather than slamming to
   // a stop — and the hold's own release seeds the forward way.
   if (introFly) {
    const fdx = introFly.toX - character.position.x;
    const fdz = introFly.toZ - character.position.z;
    const remaining = Math.hypot(fdx, fdz);
    if (remaining < 4 || introHold <= 0) {
     introFly = null;
    } else {
     const v = Math.min(34, Math.max(12, remaining * 0.45));
     const h = introFly.heading;
     const sc = shipController ?? character;
     sc.velocity.set(
      Math.sin(h) * v, 0, Math.cos(h) * v
     );
     character.position.copy(sc.position);
     character.facing = sc.facing;
     character.velocity.copy(sc.velocity);
    }
   }
  }

  // The flight message's clock, outside the hold: "you have control"
  // fades after the lock is already gone.
  if (flightMsgT > 0) {
   flightMsgT -= dt;
   if (flightMsgT <= 0 && flightMsg) flightMsg.classList.remove("show");
  }

  // Per-system CPU timing. There is no dependable GPU timer in WebGL2, so
  // the overlay's GPU row shows its dash and these rows stay `cpu`.
  const tFrame = performance.now();

  if (flying && shipController) {
   shipController.update(dt, rig);
   // Mirror into `character` so downstream camera/VFX/sound code keeps working.
   character.position.copy(shipController.position);
   character.velocity.copy(shipController.velocity);
   character.prevVelocity.copy(shipController.prevVelocity);
   character.acceleration.copy(shipController.acceleration);
   character.facing = shipController.facing;
   character.speed = shipController.speed;
   character.speed01 = shipController.speed01;
   character.speedRaw = shipController.speedRaw;
   character.lean = shipController.lean;
   character.carve = shipController.carve;
   character.streak01 = shipController.streak01;
   character.groundY = shipController.groundY;
   character.groundNormal.copy(shipController.groundNormal);
   character.surf = shipController.surf;
   character.surfActive = shipController.surfActive;
   character.crashing = shipController.crashing;
   character.crashLanded = shipController.crashLanded;
   character.crashDown = shipController.crashDown;
   character._crashT = shipController._crashT;
   character.terrainSlam = shipController.terrainSlam;
   character.pathY = shipController.pathY;
   character.lift01 = shipController.lift01;
   character.impactSpin = shipController.impactSpin;
   character.cast = shipController.cast;
   character.castAimX = shipController.castAimX;
   character.castAimY = shipController.castAimY;
   character.castAimZ = shipController.castAimZ;
   character.stepping = shipController.stepping;
   character.gaitPhase = shipController.gaitPhase;
   character.footfall = shipController.footfall;
   character.footIndex = shipController.footIndex;
   character.footPos.copy(shipController.footPos);
   // Presentation mirror: `character` is a view of the authoritative
   // shipController while flying. Keep this list in sync with the
   // CharacterController constructor field set.
   character._prevSpeed = shipController._prevSpeed;
   character._drive = shipController._drive;
   character._boost = shipController._boost;
   character._climb = shipController._climb;
   character._climbWant = shipController._climbWant;
   character._vertEase = shipController._vertEase;
   character.climb = shipController.climb;
   character.climbRate = shipController.climbRate;
   character._cruiseGround = shipController._cruiseGround;
   character.steer = shipController.steer;
   character.steerRate = shipController.steerRate;
   character.controlScramble = shipController.controlScramble;
   character._scrambleMax = shipController._scrambleMax;
   character._scrambleT = shipController._scrambleT;
   character.fireHeld = shipController.fireHeld;
   character.driveHeld = shipController.driveHeld;
   character.boostHeld = shipController.boostHeld;
   character._prevTargetY = shipController._prevTargetY;
   character._slam = shipController._slam;
   character.footImpact = shipController.footImpact;
  } else {
   character.footfall = false;
   character.update(dt, rig);
  }
  terrain.heightfield.clampToPlayArea(character.position);
  // Pose and simulate before the contact pass: the footprints are stamped
  // at the boot's actual planted position, which only exists once the
  // figure has been solved.
  figure.update(dt);
  contact.update(dt);
  // The walkers only read the player's position, so they can sit anywhere
  // after the controller has moved. Here, so their cost lands in the same
  // row of the overlay as the rest of the scene's inhabitants.
  // walkers/herds purged: no update.
  // The crash sites burn down and freeze over: smoke and fire for the
  // first half minute, thinning as the heat dies, and then the slow
  // frost — the hull whitening into the field over a couple of minutes.
  for (const wk of wrecks) {
   if (!wk.active) continue;
   wk.age += dt;
   if (wk.burnT > 0) {
    wk.burnT -= dt;
    wk.smokeT -= dt;
    if (wk.smokeT <= 0) {
     const p = wk.ctl.position;
     const gy = wk.ctl.groundY;
     const heat = Math.min(1, wk.burnT / 32);
     wk.smokeT = 0.1 + (1 - heat) * 0.4;
     smoke.emit(
      p.x + (Math.random() - 0.5) * 1.4, gy + 1.1,
      p.z + (Math.random() - 0.5) * 1.4,
      (Math.random() - 0.5) * 0.5, 1.2 + Math.random() * 0.8,
      (Math.random() - 0.5) * 0.5,
      0.55, 1.6, 2.0 + Math.random(),
      0.06, 0.06, 0.06, 0.5 * (0.4 + 0.6 * heat), true
     );
     if (wk.burnT > 3 && Math.random() < 0.75) {
      smoke.emit(
       p.x, gy + 0.7, p.z,
       (Math.random() - 0.5) * 0.8, 0.9,
       (Math.random() - 0.5) * 0.8,
       0.4, 0.8, 0.3 + Math.random() * 0.15,
       3.0, 1.0, 0.24, 0.7 * heat, false
      );
     }
    }
   }
   wk.hull.snowOverride = Math.min(1, wk.age / 150);
   wk.hull.update(dt);
  }
  if (flying) {
   speeder?.tick(dt);
   speeder?.update(dt);
  }
  for (const w of wingmen) {
   w.tick(dt);
   w.update(dt);
  }
  // The skip, if one of the falling craft touched the snow this frame
  // and said so with a latch rather than a call. Cheap — a flag test
  // per wingman — and it has to run after the pilots have moved, so
  // the contact point it reads is this frame's, not last frame's.
  pumpBounces();
  // After every craft has moved, before the rig frames the result: the
  // Collision pass works on whichever controller is actually driving this frame.
  collision.character = (flying && shipController) ? shipController : character;
  // contacts this frame's motion earned, resolved and reacted to.
  collision.update(dt);
  const tChar = performance.now();

  _vel.copy(character.velocity);
  // The death cam. While the craft is going in and lying there, the
  // frame stops being a cockpit view and becomes a shot *of* the crash:
  // the heading handed to the rig walks steadily round, and because the
  // rig chases whatever heading it is given, the camera circles the
  // wreck. The craft's own facing is untouched, so the hull lies still
  // while the world turns around it — turning the wreck instead would
  // spin it like a bottle. The bank is zeroed too: the roll belongs to
  // the falling craft, not to the camera watching it.
  let camFacing = flying ? character.facing : null;
  let camLean = character.lean;
  let camTarget = character.position;
  if (character.crashing) {
   if (deathCamT === 0) {
    // Remember the framing to give back afterwards — the player
    // may have zoomed it exactly where they like it.
    deathCamDist = rig.distanceTarget;
   }
   deathCamT += dt;
   camFacing = character.facing + deathCamT * DEATH_CAM_RATE;
   camLean = 0;
   // What the shot is ABOUT is the pilot.
   //
   // The wreck is scenery — it lands, it burns, it sits there. The
   // body thrown clear of it is the only thing in the frame still
   // doing something: three and a half seconds of dying, played out
   // in the snow. So once there is a body the camera takes it as its
   // subject and circles that, with the burning hull swinging
   // through the background as it goes. Until then (the fall itself,
   // before anything has hit) it stays on the craft.
   const body = deathCamPilot;
   if (body) {
    _camAt.set(body.position.x, body.position.y + 0.6, body.position.z);
    camTarget = _camAt;
   }
   // Stand off and look down — closer for a man than for a machine,
   // or the animation the shot exists for is four pixels of orange.
   rig.distanceTarget = body ? DEATH_CAM_DIST_PILOT : DEATH_CAM_DIST;
   rig.pitch = expDamp(
    rig.pitch,
    body ? DEATH_CAM_PITCH_PILOT : DEATH_CAM_PITCH, 1.4, dt
   );
  } else if (deathCamT > 0) {
   deathCamT = 0;
   deathCamPilot = null;
   rig.distanceTarget = deathCamDist || rig.distanceTarget;
  }
  // Flying, the rig is handed the craft's heading and chases it itself —
  // see CameraRig.update — rather than the controller stepping rig.yaw.
  rig.update(
   dt, camTarget, _vel, camLean, character.speed01, camFacing, character.facing
  );

  // Jitters the projection and republishes everything the screen-space
  // passes derive from the camera. Must be after the rig has moved and
  // before anything reads the view-projection — which the depth prepass
  // and the beauty pass both do.
  // The craft's streak when there is a craft: see `Speeder.streak01` for
  // why the character's is the wrong signal in the air.
  post.update(dt, (speeder && flying) ? speeder.streak01 : character.streak01, rig.distance);
  sky.update();
  sky.render(rig, time);
  shadows.update(rig.camera, sky.sunDir);
  // After the shadow refit, so the water and the ice carry this frame's
  // cascade matrices; before the terrain, so the brushes every spell
  // writes are in the staging array when the simulation pass runs.
  spells.update(dt, rig.camera.position);
  const tSpells = performance.now();
  terrain.update(rig.camera.position, character.position, dt);
  const tTerrain = performance.now();
  // After the shadow refit, so the figure's uniforms carry this frame's
  // cascade matrices rather than last frame's.
  figure.sync(rig.camera.position);
  // Also picks each walker's level of detail, which needs this frame's
  // camera and this frame's field of view.
  // walkers/herds purged: no sync.
  for (const wk of wrecks) {
   if (wk.active) wk.hull.sync(rig.camera.position);
  }
  speeder?.sync(rig.camera.position);
  for (const w of wingmen) w.sync(rig.camera.position);
  // The tuning rings, after every transform they pin to has settled.
  muzzleMarkers.begin();
  if (S.showMuzzles) {
   // walkers/herds purged: no muzzle collection.
   speeder?.collectMuzzles?.(muzzleMarkers);
  }
  muzzleMarkers.commit(rig.camera.position);
  // Before the spray: the wake decides where its own lip is, and the
  // grains it sheds have to be in the pool before the pool is uploaded.
  wake.update(dt, rig.camera.position);
  spray.update(dt, rig.camera.position);
  // The overlay's burn preview: the wingmen's damage trail, lit on the
  // player craft so the burn sliders can be dialled against a hull the
  // player can actually orbit.
  if (S.burnPreview && speeder && flying) {
   burnPreviewT -= dt;
   if (burnPreviewT <= 0) {
    burnPreviewT = 0.05;
    // Off the *drawn* hull, not the controller — the controller
    // rides hover-height below the craft the player is looking at.
    emitBurnTrail(
     smoke, speeder.position, speeder.yaw,
     character.velocity, true
    );
   }
  }
  smoke.update(dt, rig.camera.position);
  const tVfx = performance.now();

  // Last, and a pure reader: every signal it mixes on — surf blend, carve
  // load, speed, cast count — has settled by here.
  const tAudio = performance.now();

  renderFrame();
  post.endFrame();
  const tRender = performance.now();

  mark("cpu character", tChar - tFrame);
  mark("cpu spells", tSpells - tChar);
  mark("cpu terrain", tTerrain - tSpells);
  mark("cpu wake+spray", tVfx - tTerrain);
  mark("cpu audio", tAudio - tVfx);
  mark("cpu submit", tRender - tAudio);
  mark("cpu total", tRender - tFrame);

  endFrameDraws();
  stats.triangles =
   (terrain.mesh && terrain.mesh.metadata ? terrain.mesh.metadata.triangles : 0) +
   (S.showCharacter && !flying ? figure.triangles : 0) +
   (speeder ? speeder.triangles : 0) +
   wingmen.reduce((t, w) => t + w.triangles, 0) +
   // walkers/herds purged: no triangle counts.
   (wake.mesh && wake.mesh.visible ? wake.mesh.metadata.triangles : 0) +
   spells.triangles +
   spray.liveCount * 2;

  sample(dtMs);
  checkSpike(dtMs);
  overlay.update(dtMs, gfx);
  fpsMeter.update(dtMs);
  perfHud.update(dtMs, gfx);

  endFrame();
 });

 // ------------------------------------------------------- audio + the gate
 // Normally already resolved: the download has been running since before the
 // device existed. This phase only shows up on a cold cache or a slow line.
 await loading.phase("loading audio", 0.96);
 await audioReady;
 // The second transmission is fetched behind the boot with everything
 // else, so the moment it is called for it plays local bytes — a clip that
 // buffered mid-battle would arrive after the moment that earned it.

 // One click to enter, and that click is also the gesture that makes sound
 // legal. `unlock()` is issued from inside the handler — hence the callback —
 // and awaited here, so the context is resumed before the boot screen starts
 // to fade. Skipped entirely if nothing decoded: a button whose only job is
 // to start audio has none when there is no audio.
 //
 // Native TS annotation rather than JSDoc: `unlock()` is assigned inside the
 // gate callback closure, so control-flow analysis never widens the binding
 // and a `/** @type */` line above `let unlocking = null` is left dangling —
 // tsc then infers literal `null`, which makes the guard below dead code.
 let unlocking: Promise<boolean> | null = null;
 if (audio.hasAssets) {
  await loading.gate(() => {
   unlocking = audio.unlock();
  });
 }
 if (unlocking) {
  await unlocking;
 }
 soundButton.sync();

 // The instant the player is coming in — boot screen starting its fade —
 // stage the formation behind them, so the escorts thunder overhead in the
 // first seconds of the game rather than at some point during the loading.
 if (!storyMode) startFlyover();

 // After the loading screen has gone: it sits at a higher z-index than the
 // overlay, so opening the panel before this would hide it behind a full
 // screen of boot gradient.
 await loading.done();
 if (S.overlayOpen && !overlay.visible) overlay.toggle();
 soundButton.reveal();
 setTimeout(() => overlay.resetSpikes(), 800);

 const snowflowGlobal = window;
 snowflowGlobal.SNOWFLOW = {
  gfx, scene: gfx.scene, rig, character, figure, wrecks, speeder, collision, contact, spray, wake, spells,
  overlay, touchControls, terrain, sky, shadows, post, depthPass,
  audio,
  S, input, perfStats: stats,
  captureShot,
  /** Fake a cannon burst at (x, z) — for testing the squad's reactions. */
  // Inert purge remnant: never assigned, so a call throws exactly as it
  // would have without the `| undefined` on the binding above. `?.` here
  // would turn that TypeError into a silent no-op.
  squadImpact: (x: number, z: number) => squadImpact!(x, 0, z),
  /**
   * Print the speeder's look settings as a paste-able block.
   *
   * These four are dialled by eye against a moving craft over bright snow,
   * which is not a thing that can be solved on paper — so the loop is drag,
   * look, and read the numbers back out. Logged as source rather than as an
   * object so the answer can go straight into `settings.js`.
   */
  speederTuning() {
   const keys = ["speederFill", "speederTint", "speederRough", "speederAmbient",
    "speederDesat", "jetSpan", "jetDropY", "jetBackZ",
    "jetWidth", "jetLength", "jetFlare",
    "jetGlow", "jetHaloBack",
    "boltR", "boltG", "boltB", "boltWidth", "boltLength", "boltSpeed",
    "boltRange", "muzzleSpan", "muzzleDropY", "muzzleFwdZ",
    "speederClimbMax", "speederClimbRate",
    "surfWidth", "speederStreak",
    // The feel layer, so a session spent dragging the "Speeder
    // feel" sliders reads back out the same way the look does.
    "cinematicCam", "camYawLag", "speederBankLead",
    "speederTurbulence", "camBoostBreath"];
   const w = Math.max(...keys.map((k) => k.length));
   // The keys are literal names of settings this file does not otherwise
   // reference, so `S` cannot be indexed by a plain `string` — every one of
   // them exists on it, and the read is one-to-one with the list above.
   const readSetting = (k: string): SettingValue => S[k as keyof typeof S];
   const body = keys
    .map((k) => `    ${k}:${" ".repeat(w - k.length)} ${readSetting(k)},`)
    .join("\n");
   console.log("// speeder look — tuned\n" + body);
   return Object.fromEntries(keys.map((k) => [k, readSetting(k)]));
  },
 };
}

boot().catch((err) => {
 console.error(err);
 loading.fail("Startup failed — see console.");
});
