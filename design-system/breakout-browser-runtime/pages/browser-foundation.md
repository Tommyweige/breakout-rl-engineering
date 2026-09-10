# Browser Foundation Page Overrides

> **PROJECT:** Breakout Browser Runtime
> **Generated:** 2026-09-07 15:35:04
> **Page Type:** Runtime Console

> ⚠️ **IMPORTANT:** Rules in this file **override** the Master file (`design-system/MASTER.md`).
> Only deviations from the Master are documented here. For all other rules, refer to the Master.

---

## Page-Specific Rules

### Layout Overrides

- **Max Width:** 1320px
- **Layout:** Editorial masthead → shared command deck → asymmetric two-panel runtime surface → full-width evidence band
- **Responsive:** Collapse the command deck and panel grid at 960px; keep the evidence readable at 375px without horizontal scrolling

### Spacing Overrides

- **Section rhythm:** Use 14–22px panel gaps and 30–48px hero padding; keep the primary validation action visible in the first viewport

### Typography Overrides

- **Display:** Condensed, high-contrast system fallback for the editorial headline
- **Body:** Neutral sans-serif with a 16px minimum for explanatory text
- **Runtime facts:** Monospace for backend, Q-values, hashes, and state labels

### Color Overrides

- **Direction:** Light paper console with deep ink, electric cobalt, lime success, and orange warning accents
- **Do not use:** Claymorphism, purple study-app palette, gradients, glassmorphism, or decorative neon glow

### Component Overrides

- **Controls:** Sharp 3px corners, 46px minimum button height, visible orange focus ring, one cobalt primary action
- **Panels:** Crisp ink borders; Human Panel uses the dark signal channel, Agent Panel uses paper for data legibility
- **Status:** Rectangular status labels rather than pill-heavy dashboard chips

---

## Page-Specific Components

- **Signal stage:** Static ALE seam with grid/crosshair treatment; it must not imply a connected game loop
- **Q-value comparison:** Real Browser/reference values with data-driven bars, not illustrative numbers
- **Evidence band:** Fixed-fixture count and WASM PASS/FAIL state from the actual runtime artifact

---

## Recommendations

- Refer to MASTER.md for all design rules
- Add specific overrides as needed for this page
