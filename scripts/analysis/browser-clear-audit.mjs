#!/usr/bin/env node
/**
 * Browser Unbeatable clear audit.
 *
 * Runs the checked-in browser "Unbeatable" agent (the predictive pixel vision
 * controller under web/src) against the pinned ALE WebAssembly runtime, one
 * controller decision per ALE-native frame, and records two independent clear
 * signals per episode:
 *
 *   (a) screen wall clear  - the rendered 18x6 brick grid is empty for at least
 *       one frame (what a viewer sees as "the whole wall is gone");
 *   (b) canonical clear    - ALE RAM[76:78] packed-BCD score == 864 (both brick
 *       walls, the documented maximum) and it equals the cumulative raw reward,
 *       exactly as breakout_rl/completion.py requires.
 *
 * The environment mirrors the browser agent path in web/src/app/App.ts and
 * web/src/environment/aleEnvironment.ts for the "unbeatable" difficulty:
 *   frame_skip = 1, repeat_action_probability = 0.25 (contract v3),
 *   max_num_frames_per_episode = 108000, one decision per ALE frame, and
 *   environment-owned FIRE serve after reset and after every life loss.
 *
 * Usage (from the repository root):
 *   node scripts/analysis/browser-clear-audit.mjs --seeds 501-600
 *   node scripts/analysis/browser-clear-audit.mjs --seeds 101,102 --frame-logs 1
 *   node scripts/analysis/browser-clear-audit.mjs --seeds 501-600 \
 *     --merge audits/browser-clear/summary-20.json --out /tmp/audit-100
 *
 * Requires web dependencies once: `(cd web && npm ci)` (esbuild).
 */
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createWriteStream, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createGzip } from 'node:zlib';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const VENDOR = join(REPO_ROOT, 'web', 'vendor', 'ale-wasm');
const CONTROLLER_SOURCE = join(REPO_ROOT, 'web', 'src', 'inference', 'PredictiveVisionController.ts');
const ESBUILD = join(REPO_ROOT, 'web', 'node_modules', '.bin', 'esbuild');

const WIDTH = 160;
const HEIGHT = 210;
const ALE_ACTION = { NOOP: 0, FIRE: 1, RIGHT: 3, LEFT: 4 };
const FULL_CLEAR_SCORE = 864;
const MAX_FRAMES = 108000;
const WALL_X0 = 8;
const WALL_Y0 = 57;
const CELL_W = 8;
const CELL_H = 6;
const COLS = 18;
const ROWS = 6;
const FULL_WALL_BRICKS = COLS * ROWS;
const ROM_SHA256 = '376323f051c3c373c887fd83abead39d87d844ff283d435f4addbfc1710c6fd5';

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const index = args.indexOf(`--${name}`);
  return index >= 0 ? args[index + 1] : fallback;
};
const options = (name) => {
  const values = [];
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === `--${name}` && args[index + 1] !== undefined) values.push(args[index + 1]);
  }
  return values;
};

function expandSeeds(spec) {
  const seeds = [];
  for (const part of spec.split(',')) {
    const trimmed = part.trim();
    if (!trimmed) continue;
    const range = /^(\d+)-(\d+)$/.exec(trimmed);
    if (range) {
      const start = Number(range[1]);
      const end = Number(range[2]);
      if (end < start) throw new Error(`invalid seed range ${trimmed}`);
      for (let seed = start; seed <= end; seed += 1) seeds.push(seed);
    } else {
      const seed = Number(trimmed);
      if (!Number.isInteger(seed)) throw new Error(`invalid seed ${trimmed}`);
      seeds.push(seed);
    }
  }
  if (new Set(seeds).size !== seeds.length) throw new Error('seed list contains duplicates');
  return seeds;
}

const seeds = expandSeeds(option('seeds', '501-600'));
const summarizeOnly = args.includes('--summarize-only');
const maxFrames = Number(option('max-frames', String(MAX_FRAMES)));
const outDir = resolve(option('out', join(tmpdir(), 'browser-clear-audit', new Date().toISOString().replace(/[:.]/g, '-'))));
const frameLogs = option('frame-logs', '0') === '1';
const mergePaths = options('merge').map((value) => resolve(value)).filter((value) => existsSync(value));
const dumpSteps = new Set((option('dump-steps', '') || '').split(',').filter(Boolean).map(Number));

mkdirSync(join(outDir, 'frames'), { recursive: true });

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function ensureControllerBundle() {
  const source = readFileSync(CONTROLLER_SOURCE);
  const sourceSha = sha256(source);
  const bundlePath = join(tmpdir(), 'browser-clear-audit', `controller-${sourceSha.slice(0, 12)}.mjs`);
  mkdirSync(dirname(bundlePath), { recursive: true });
  if (!existsSync(bundlePath)) {
    if (!existsSync(ESBUILD)) {
      throw new Error('web dependencies are missing; run `(cd web && npm ci)` so esbuild is available');
    }
    execFileSync(ESBUILD, [CONTROLLER_SOURCE, '--bundle', '--format=esm', `--outfile=${bundlePath}`], { stdio: 'pipe' });
  }
  return { bundlePath, sourceSha, bundleSha: sha256(readFileSync(bundlePath)) };
}

/** Count bricks on the rendered frame; each brick is an 8x6 cell in an 18x6 grid. */
function countBricks(rgb) {
  let bricks = 0;
  for (let row = 0; row < ROWS; row += 1) {
    for (let col = 0; col < COLS; col += 1) {
      const x0 = WALL_X0 + col * CELL_W;
      const y0 = WALL_Y0 + row * CELL_H;
      let lit = 0;
      for (let y = y0; y < y0 + CELL_H; y += 1) {
        for (let x = x0; x < x0 + CELL_W; x += 1) {
          const index = (y * WIDTH + x) * 3;
          if (Math.max(rgb[index], rgb[index + 1], rgb[index + 2]) > 40) lit += 1;
        }
      }
      if (lit >= 30) bricks += 1;
    }
  }
  return bricks;
}

/** breakout_rl/completion.py: packed BCD at RAM[76:78], reject nibbles greater than 9. */
function readRamScore(ram) {
  const digits = [];
  for (const address of [76, 77]) {
    const value = ram[address];
    const high = value >> 4;
    const low = value & 0x0f;
    if (high > 9 || low > 9) return null;
    digits.push(high, low);
  }
  return digits[0] * 1000 + digits[1] * 100 + digits[2] * 10 + digits[3];
}

function writePpm(path, rgb) {
  const header = Buffer.from(`P6\n${WIDTH} ${HEIGHT}\n255\n`, 'ascii');
  writeFileSync(path, Buffer.concat([header, Buffer.from(rgb)]));
}

function mean(values) {
  return values.length === 0 ? null : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
}

function distribution(values) {
  const buckets = [
    { range: '<200', test: (value) => value < 200 },
    { range: '200-399', test: (value) => value >= 200 && value < 400 },
    { range: '400-599', test: (value) => value >= 400 && value < 600 },
    { range: '600-799', test: (value) => value >= 600 && value < 800 },
    { range: '800-863', test: (value) => value >= 800 && value < 864 },
    { range: '864', test: (value) => value === 864 },
  ];
  return buckets.map((bucket) => ({ range: bucket.range, count: values.filter(bucket.test).length }));
}

function aggregate(episodes) {
  const scores = episodes.map((episode) => episode.score);
  const canonical = episodes.filter((episode) => episode.canonical_cleared).length;
  const wall = episodes.filter((episode) => episode.brick_clear_events > 0).length;
  return {
    episode_count: episodes.length,
    brick_clear: {
      episodes_with_at_least_one_wall_knocked_out: wall,
      share: episodes.length === 0 ? null : wall / episodes.length,
      total_wall_knockouts: episodes.reduce((sum, episode) => sum + episode.brick_clear_events, 0),
      episodes_with_level_advance: episodes.filter((episode) => episode.level_advance_events > 0).length,
      total_level_advances: episodes.reduce((sum, episode) => sum + episode.level_advance_events, 0),
      episodes_with_pre864_wall_knockout: episodes.filter((episode) => (episode.pre864_wall_clear_events ?? 0) > 0).length,
      share_pre864: episodes.length === 0 ? null : episodes.filter((episode) => (episode.pre864_wall_clear_events ?? 0) > 0).length / episodes.length,
      total_pre864_wall_knockouts: episodes.reduce((sum, episode) => sum + (episode.pre864_wall_clear_events ?? 0), 0),
      post864_wall_knockouts: episodes.reduce((sum, episode) => sum + (episode.post864_wall_clear_events ?? 0), 0),
    },
    canonical_clear: {
      count: canonical,
      share: episodes.length === 0 ? null : canonical / episodes.length,
    },
    score: {
      mean: mean(scores),
      median: median(scores),
      min: scores.length === 0 ? null : Math.min(...scores),
      max: scores.length === 0 ? null : Math.max(...scores),
      distribution: distribution(scores),
    },
    total_frames: episodes.reduce((sum, episode) => sum + episode.emulator_frames, 0),
  };
}

const { bundlePath, sourceSha, bundleSha } = ensureControllerBundle();
const { PredictiveVisionController } = await import(pathToFileURL(bundlePath).href);
const createALEModule = (await import(pathToFileURL(join(VENDOR, 'node.mjs')).href)).default;
const module = await createALEModule({
  locateFile: (name) => join(VENDOR, name),
  print: () => {},
  printErr: () => {},
});
const ALE_VERSION = module.ALEInterface.getVersion();

function makeAle(seed) {
  const ale = new module.ALEInterface();
  ale.setBool('display_screen', false);
  ale.setInt('random_seed', seed);
  ale.setInt('frame_skip', 1);
  ale.setFloat('repeat_action_probability', 0.25);
  ale.setInt('max_num_frames_per_episode', MAX_FRAMES);
  ale.loadROM('/roms/breakout.bin');
  ale.resetGame();
  return ale;
}

const startedAt = performance.now();
const episodes = [];
console.log(`[audit] controller source ${sourceSha.slice(0, 12)} bundle ${bundleSha.slice(0, 12)} ALE ${ALE_VERSION}`);
if (summarizeOnly) {
  if (mergePaths.length === 0) throw new Error('--summarize-only requires at least one --merge summary');
  console.log(`[audit] summarize-only from ${mergePaths.length} summaries -> ${outDir}`);
} else {
  console.log(`[audit] ${seeds.length} seeds ${seeds[0]}..${seeds[seeds.length - 1]} -> ${outDir}`);
}

for (const [episodeIndex, seed] of seeds.entries()) {
  if (summarizeOnly) break;
  const ale = makeAle(seed);
  const controller = new PredictiveVisionController();
  let frameStream = null;
  let frameLog = null;
  if (frameLogs) {
    frameLog = createGzip();
    frameStream = createWriteStream(join(outDir, 'frames', `${seed}.csv.gz`));
    frameLog.pipe(frameStream);
    frameLog.write('step,ale_frame,score_ram,cum_reward,lives,bricks,needs_fire,auto_fire,requested,executed\n');
  }

  let needsFire = true;
  let fireAttempts = 0;
  let fireActivityStreak = 0;
  let lastLives = ale.lives();
  let cumReward = 0;
  let scoreMismatchFrames = 0;
  const events = [];
  const lifeLossSteps = [];
  let minBricks = FULL_WALL_BRICKS;
  let wallEmptyFrames = 0;
  let wallRebuilds = 0;
  let canonicalClear = null;
  let prevBricks = countBricks(ale.getScreenRGB());
  let step = 0;
  const episodeStartedAt = performance.now();

  while (step < maxFrames) {
    step += 1;
    const before = ale.getScreenRGB();
    const policy = controller.select(before);
    const requested = policy.action;
    const autoFire = needsFire;
    const executed = autoFire ? 'FIRE' : requested;
    const reward = ale.act(ALE_ACTION[executed]);
    cumReward += reward;
    const after = ale.getScreenRGB();
    const lives = ale.lives();
    const ramScore = readRamScore(ale.getRAM());
    if (ramScore === null) throw new Error(`seed ${seed} step ${step}: invalid RAM score BCD`);
    if (Math.abs(ramScore - cumReward) > 1e-6) {
      scoreMismatchFrames += 1;
      if (scoreMismatchFrames <= 3) events.push({ step, frame: ale.getEpisodeFrameNumber(), kind: 'ram_reward_mismatch', ramScore, cumReward });
    }

    const bricks = countBricks(after);
    if (bricks < minBricks) minBricks = bricks;
    if (dumpSteps.has(step)) writePpm(join(outDir, 'frames', `${seed}-step${step}.ppm`), after);

    if (bricks === 0 && prevBricks > 0) {
      events.push({ step, frame: ale.getEpisodeFrameNumber(), kind: 'wall_empty', score: ramScore, cumReward, lives, bricksBefore: prevBricks });
    }
    if (bricks === 0) wallEmptyFrames += 1;
    if (bricks > prevBricks) {
      wallRebuilds += 1;
      events.push({ step, frame: ale.getEpisodeFrameNumber(), kind: 'wall_rebuild', score: ramScore, cumReward, lives, bricksBefore: prevBricks, bricksAfter: bricks });
    }
    prevBricks = bricks;

    if (needsFire) {
      fireAttempts += 1;
      let changed = 0;
      for (let index = 0; index < before.length; index += 1) if (before[index] !== after[index]) changed += 1;
      const changeFraction = changed / before.length;
      if (changeFraction >= 0.0001) fireActivityStreak += 1;
      else fireActivityStreak = 0;
      if (reward !== 0 || fireActivityStreak >= 2) {
        needsFire = false;
        fireAttempts = 0;
        fireActivityStreak = 0;
      } else if (fireAttempts >= 8) {
        throw new Error(`seed ${seed} step ${step}: FIRE serve was not confirmed after 8 attempts`);
      }
    } else {
      fireActivityStreak = 0;
    }

    if (lives < lastLives) {
      lifeLossSteps.push(step);
      events.push({ step, frame: ale.getEpisodeFrameNumber(), kind: 'life_loss', score: ramScore, cumReward, lives, bricks });
      needsFire = true;
      fireAttempts = 0;
      fireActivityStreak = 0;
    }
    lastLives = lives;

    if (canonicalClear === null && ramScore === FULL_CLEAR_SCORE && Math.abs(cumReward - FULL_CLEAR_SCORE) <= 1e-6) {
      canonicalClear = { step, frame: ale.getEpisodeFrameNumber(), score: ramScore, cumReward, lives };
      events.push({ step, frame: ale.getEpisodeFrameNumber(), kind: 'canonical_clear', score: ramScore, cumReward, lives, bricks });
      if (dumpSteps.size > 0) writePpm(join(outDir, 'frames', `${seed}-canonical-clear.ppm`), after);
    }

    if (frameLog) {
      frameLog.write(`${step},${ale.getEpisodeFrameNumber()},${ramScore},${cumReward},${lives},${bricks},${autoFire ? 1 : 0},${autoFire ? 'FIRE' : ''},${requested},${executed}\n`);
    }
    if (step % 10000 === 0) {
      console.error(`[frames] seed ${seed} step=${step} score=${ramScore} lives=${lives} bricks=${bricks}`);
    }

    const over = ale.gameOver();
    const truncated = ale.gameTruncated();
    if (over || truncated || step >= maxFrames) {
      events.push({ step, frame: ale.getEpisodeFrameNumber(), kind: truncated ? 'time_limit' : over ? 'game_over' : 'harness_cutoff', score: ramScore, cumReward, lives, bricks });
      const wallEmpty = events.filter((event) => event.kind === 'wall_empty');
      const pre864Wall = wallEmpty.filter((event) => event.score < FULL_CLEAR_SCORE).length;
      const row = {
        seed,
        ale_version: ALE_VERSION,
        score: ramScore,
        cum_reward: cumReward,
        lives_remaining: lives,
        bricks_remaining: bricks,
        min_bricks: minBricks,
        steps: step,
        emulator_frames: ale.getEpisodeFrameNumber(),
        terminated: over,
        truncated,
        brick_clear_events: wallEmpty.length,
        pre864_wall_clear_events: pre864Wall,
        post864_wall_clear_events: wallEmpty.length - pre864Wall,
        level_advance_events: events.filter((event) => event.kind === 'wall_rebuild').length,
        wall_empty_frames: wallEmptyFrames,
        board_empty_at_end: bricks === 0,
        canonical_cleared: canonicalClear !== null,
        canonical_clear: canonicalClear,
        life_loss_steps: lifeLossSteps,
        ram_reward_mismatch_frames: scoreMismatchFrames,
        events,
        duration_ms: Math.round(performance.now() - episodeStartedAt),
      };
      episodes.push(row);
      if (frameLog) {
        frameLog.end();
        await new Promise((done) => frameStream.on('finish', done));
      }
      const done = episodes.length;
      const running = aggregate(episodes);
      console.log(
        `[episode ${done}/${seeds.length}] seed=${seed} score=${row.score} lives=${row.lives_remaining} bricks=${row.bricks_remaining} ` +
        `steps=${row.steps} wallKO(pre864)=${row.pre864_wall_clear_events} canonical=${row.canonical_cleared ? 'yes' : 'no'} ` +
        `(${(row.duration_ms / 1000).toFixed(1)}s) | mean=${running.score.mean.toFixed(1)} ` +
        `canonical=${running.canonical_clear.count}/${done} wall=${running.brick_clear.episodes_with_pre864_wall_knockout}/${done}`,
      );
      break;
    }
  }
}

for (const path of mergePaths) {
  const payload = JSON.parse(readFileSync(path, 'utf8'));
  if (!Array.isArray(payload.episodes)) throw new Error(`${path}: missing episodes array`);
  for (const episode of payload.episodes) {
    if (episodes.some((existing) => existing.seed === episode.seed)) {
      throw new Error(`${path}: duplicate seed ${episode.seed} between runs`);
    }
    if (episode.pre864_wall_clear_events === undefined) {
      const wallEmpty = (episode.events ?? []).filter((event) => event.kind === 'wall_empty');
      episode.pre864_wall_clear_events = wallEmpty.filter((event) => event.score < FULL_CLEAR_SCORE).length;
      episode.post864_wall_clear_events = wallEmpty.length - episode.pre864_wall_clear_events;
    }
    episodes.push(episode);
  }
}

const summary = {
  harness: 'scripts/analysis/browser-clear-audit.mjs',
  ale_version: ALE_VERSION,
  rom_sha256: ROM_SHA256,
  contract: { frame_skip: 1, sticky_action_probability: 0.25, max_num_frames_per_episode: MAX_FRAMES, outer_action_repeat: 1 },
  controller: {
    source: 'web/src/inference/PredictiveVisionController.ts',
    source_sha256: sourceSha,
    bundle_sha256: bundleSha,
  },
  definitions: {
    screen_wall_clear: 'rendered brick grid (18x6 cells, 8x6 px) contains zero bricks on at least one frame',
    canonical_clear: 'RAM[76:78] packed-BCD score == 864 and equals cumulative raw reward (breakout_rl/completion.py)',
  },
  sources: { seeds: option('seeds', '501-600'), merged: mergePaths.map((path) => path) },
  aggregate: aggregate(episodes),
  episodes,
};
writeFileSync(join(outDir, 'summary.json'), `${JSON.stringify(summary, null, 2)}\n`);

const csvHeader = 'seed,score,lives_remaining,bricks_remaining,min_bricks,steps,emulator_frames,brick_clear_events,pre864_wall_clear_events,post864_wall_clear_events,level_advance_events,board_empty_at_end,canonical_cleared,terminated,truncated,duration_ms';
const csvLine = (episode) => [
  episode.seed, episode.score, episode.lives_remaining, episode.bricks_remaining, episode.min_bricks,
  episode.steps, episode.emulator_frames, episode.brick_clear_events, episode.pre864_wall_clear_events ?? '',
  episode.post864_wall_clear_events ?? '', episode.level_advance_events,
  episode.board_empty_at_end, episode.canonical_cleared, episode.terminated, episode.truncated, episode.duration_ms,
].join(',');
writeFileSync(join(outDir, 'episodes.csv'), `${[csvHeader, ...episodes.map(csvLine)].join('\n')}\n`);

console.log(JSON.stringify({
  outDir,
  elapsed_s: Math.round((performance.now() - startedAt) / 1000),
  aggregate: summary.aggregate,
}));
