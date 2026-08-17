/** App version — injected by Vite at build time from git (`git describe --tags`),
 * so the software version and the repo tag stay in sync. Falls back to the
 * package.json version when there is no .git directory (e.g. the release zip). */
declare const __APP_VERSION__: string;
export const APP_VERSION = __APP_VERSION__;
