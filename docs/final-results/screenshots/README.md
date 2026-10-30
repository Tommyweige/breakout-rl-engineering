# Closeout screenshots

These are real captures from the built browser application, not mock frames. They were taken on 2026-10-28 with headless Chromium at 1440×1000 (desktop) and 390×844 (mobile).

| File | What it shows |
| --- | --- |
| [`before-game.png`](before-game.png) | The game page on `origin/main` at `a8445df` before this closeout, with no case-study entry point. |
| [`after-game.png`](after-game.png) | The game page after the closeout, with the visible **Research results** link and the existing Human vs AI layout intact. |
| [`case-study-desktop.png`](case-study-desktop.png) | The case-study first screen at 1440×1000: hook, real gameplay poster, four headline metrics, and evidence statuses. |
| [`case-study-mobile.png`](case-study-mobile.png) | The case-study first screen at 390×844: hook, video, and the first verified metric card. |

The recorded gameplay used by the case study is [`tracker.mp4`](../../../web/public/case-study/tracker.mp4), with poster [`tracker-poster.png`](../../../web/public/case-study/tracker-poster.png). It is a real capture of the web demo’s “Unbeatable” AI (the tracker, the TypeScript port of the pixel controller) played at 6× speed; it is not a scored evaluation episode. [`gameplay-poster.png`](../../../web/public/case-study/gameplay-poster.png) is an earlier full-app capture of the DQN “Hard” mode, kept as the share image.

The before/after game-page pair demonstrates the additive entry point only. It is not evidence of a performance change.
