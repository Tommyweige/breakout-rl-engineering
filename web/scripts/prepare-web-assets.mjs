import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const webRoot = resolve(scriptDirectory, '..');
const repositoryRoot = resolve(webRoot, '..');
const alePackage = resolve(webRoot, 'node_modules/@farama/ale-wasm');
const alePublicDirectory = resolve(webRoot, 'public/ale');
const contractPublicPath = resolve(webRoot, 'public/configs/eval/breakout_contract_v2.json');
const alePackageMetadata = JSON.parse(await readFile(resolve(alePackage, 'package.json'), 'utf8'));

await mkdir(alePublicDirectory, { recursive: true });
await mkdir(dirname(contractPublicPath), { recursive: true });
await Promise.all([
  copyFile(resolve(alePackage, 'ale.js'), resolve(alePublicDirectory, 'ale.js')),
  copyFile(resolve(alePackage, 'ale.wasm'), resolve(alePublicDirectory, 'ale.wasm')),
  copyFile(resolve(alePackage, 'ale.data'), resolve(alePublicDirectory, 'ale.data')),
  copyFile(resolve(repositoryRoot, 'configs/eval/breakout_contract_v2.json'), contractPublicPath),
]);

await writeFile(
  resolve(alePublicDirectory, 'asset-manifest.json'),
  `${JSON.stringify({
    package: '@farama/ale-wasm',
    version: alePackageMetadata.version,
    upstreamCommit: alePackageMetadata.codexBuildProvenance?.upstreamCommit,
    paddleStrengthMethod: 'actWithPaddleStrength(action, strength)',
    romSource: 'preloaded /roms/breakout.bin',
  }, null, 2)}\n`,
  'utf8',
);
