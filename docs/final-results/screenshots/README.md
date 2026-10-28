# Closeout screenshots

These are real captures from the built browser application, not mock frames. They were taken on 2026-10-28 with headless Chromium at 1440×1000 (desktop) and 390×844 (mobile).

| File | What it shows |
| --- | --- |
| [`before-game.png`](before-game.png) | The game page on `origin/main` at `a8445df` before this closeout, with no case-study entry point. |
| [`after-game.png`](after-game.png) | The game page after the closeout, with the visible **Research results** link and the existing Human vs AI layout intact. |
| [`case-study-desktop.png`](case-study-desktop.png) | The case-study first screen at 1440×1000: hook, real gameplay poster, four headline metrics, and evidence statuses. |
| [`case-study-mobile.png`](case-study-mobile.png) | The case-study first screen at 390×844: hook, video, and the first verified metric card. |

The recorded gameplay used by the case study is [`gameplay.mp4`](../../../web/public/case-study/gameplay.mp4), with poster [`gameplay-poster.png`](../../../web/public/case-study/gameplay-poster.png). The recording is a real capture of the DQN “Hard” mode; it is not a synchronized DQN-vs-controller match.

The before/after game-page pair demonstrates the additive entry point only. It is not evidence of a performance change.
