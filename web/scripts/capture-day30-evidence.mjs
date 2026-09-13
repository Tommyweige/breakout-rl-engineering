import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import process from 'node:process';

// Keep the historical command name, but route it to the only final-evidence
// path that reuses Day 29's 30 episodes and runs only seeds 323-342.
const incrementScript = fileURLToPath(new URL('./capture-day30-parallel-increment.mjs', import.meta.url));
const result = spawnSync(process.execPath, [incrementScript, ...process.argv.slice(2)], { stdio: 'inherit' });
if (result.error) throw result.error;
process.exitCode = result.status ?? 1;
