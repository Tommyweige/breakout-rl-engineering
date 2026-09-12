import { copyFile, mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const webRoot = resolve(scriptDirectory, '..');
const repositoryRoot = resolve(webRoot, '..');
const alePackage = resolve(webRoot, 'node_modules/@farama/ale-wasm');
const alePublicDirectory = resolve(webRoot, 'public/ale');
const contractPublicPath = resolve(webRoot, 'public/configs/eval/breakout_contract_v2.json');
const nativeNoopManifestPath = resolve(webRoot, 'public/fixtures/day29-native-noop-reset.json');

await mkdir(alePublicDirectory, { recursive: true });
await mkdir(dirname(contractPublicPath), { recursive: true });
await mkdir(dirname(nativeNoopManifestPath), { recursive: true });
await Promise.all([
  copyFile(resolve(alePackage, 'ale.wasm'), resolve(alePublicDirectory, 'ale.wasm')),
  copyFile(resolve(alePackage, 'ale.data'), resolve(alePublicDirectory, 'ale.data')),
  copyFile(resolve(repositoryRoot, 'configs/eval/breakout_contract_v2.json'), contractPublicPath),
  copyFile(resolve(repositoryRoot, 'assets/day29/native-noop-reset-manifest.json'), nativeNoopManifestPath),
]);
await writeFile(
  resolve(alePublicDirectory, 'asset-manifest.json'),
  `${JSON.stringify({ package: '@farama/ale-wasm', version: '0.12.0', romSource: 'preloaded /roms/breakout.bin' }, null, 2)}\n`,
  'utf8',
);
