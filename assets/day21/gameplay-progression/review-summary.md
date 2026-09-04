# Gameplay progression review summary

Every row uses the same Contract v2 environment, showcase seed, episode cap, and real-time playback settings. The 250K row records the verified 200K nearest checkpoint because no 250K checkpoint exists in the source run.

| Stage | Actual checkpoint | Training seed | Showcase return | MP4 |
|---|---:|---:|---:|---|
| Untrained / random baseline | — | — | 1.0 | `00-untrained.mp4` |
| 100K transitions | 100,000 | 2022 | 7.0 | `01-100k.mp4` |
| 250K target / 200K nearest available | 200,000 | 2022 | 8.0 | `02-250k-target-200k-actual.mp4` |
| 500K transitions | 500,000 | 2022 | 31.0 | `03-500k.mp4` |
| 1M transitions | 1,000,000 | 2022 | 69.0 | `04-1m.mp4` |
| 2.5M selected Final Model | 2,500,000 | 2022 | 73.0 | `05-2_5m-final.mp4` |
| 5M transitions | 5,000,000 | 2022 | 45.0 | `06-5m.mp4` |

- Showcase evaluation seed: `101`
- Video: `30` FPS, codec `h264`, pixel format `yuv420p`, playback speed `1.0`
- Montage: `breakout-learning-progression.mp4`
- YouTube uploaded: `False`
