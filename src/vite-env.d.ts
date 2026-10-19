// Vite's ambient types. This project had no `vite-env.d.ts`, which meant
// `import.meta.env` and `import.meta.glob` were untyped and both call sites
// cast their way around it:
//
//   - `shaders/registry.ts` globs every `**/*.glsl` at build time.
//   - `core/assets.ts` reads the one env var the demo is built with.
//
// `vite/client` augments `ImportMeta` with both (`env: ImportMetaEnv`,
// `glob: ImportGlobFunction`), so the casts below can come out.

/// <reference types="vite/client" />

/**
 * The only env read in the codebase; the place it is read is
 * `core/assets.ts` (`VITE_ASSET_BASE` — a CDN base for the binaries, or unset
 * to serve them from this site's own origin).
 *
 * Optional because unset and empty both mean same-origin, which is the
 * default. Everything else on `ImportMetaEnv` (`BASE_URL`, `MODE`, `DEV`,
 * `PROD`, `SSR`) is Vite's and left alone.
 */
interface ImportMetaEnv {
 /** CDN base for the binaries, or unset to serve them from this origin. */
 readonly VITE_ASSET_BASE?: string;
}
