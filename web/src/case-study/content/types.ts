export type Locale = 'en' | 'zh-TW';

/**
 * Copy strings may contain hydration tokens:
 * - `{{metric:id}}` — a value from the final-results summary, filled in after it loads;
 * - `{{seeds:id}}` — the episode count behind that summary value;
 * - `{{fact:id}}` — a story number from `facts.ts`, shared by both locales.
 *
 * A numeric reference (`NumberRef`) is the same thing without the braces,
 * e.g. `metric:rgb_controller_a8445df_mean_score` or `fact:day15_dqn_mean`.
 */
export type NumberRef = `metric:${string}` | `fact:${string}`;

export type Tone = 'lime' | 'coral' | 'blue' | 'muted';

export type Block =
  | { kind: 'p'; text: string }
  | { kind: 'callout'; tone?: Tone; title: string; text: string }
  | { kind: 'flow'; caption: string; steps: Array<{ label: string; detail: string }> }
  | {
      kind: 'table';
      caption: string;
      headers: string[];
      rows: string[][];
      /** Column indexes that hold numbers (right-aligned, monospace). */
      numeric?: number[];
    }
  | {
      kind: 'bars';
      caption: string;
      /** Bars are drawn as a share of this value. */
      max: NumberRef;
      maxLabel?: string;
      items: Array<{ label: string; value: NumberRef; tone?: Tone; note?: string }>;
    }
  | {
      kind: 'columns';
      caption: string;
      max: NumberRef;
      items: Array<{ label: string; value: NumberRef; tone?: Tone }>;
      note?: string;
    }
  | { kind: 'episodes'; caption: string; gameLabel: string; clearedLabel: string; firstWallLabel: string; bothWallsLabel: string }
  | { kind: 'stats'; items: Array<{ value: string; label: string; tone?: Tone }> }
  | { kind: 'code'; caption: string; code: string }
  | { kind: 'ledger'; items: Array<{ name: string; idea: string; result: string; detail: string }> }
  | { kind: 'links'; items: Array<{ label: string; detail: string; href: string }> };

export interface Chapter {
  id: string;
  /** Short label for the table of contents. */
  nav: string;
  kicker: string;
  title: string;
  blocks: Block[];
}

export interface CaseStudyCopy {
  header: {
    brandHomeLabel: string;
    brandMark: string;
    navLabel: string;
    tocLabel: string;
    playDemo: string;
    github: string;
    languageSwitch: {
      href: string;
      hreflang: Locale;
      lang: string;
      label: string;
      ariaLabel: string;
    };
  };
  hero: {
    kicker: string;
    title: string;
    lede: string;
    stats: Array<{ value: string; label: string; tone?: Tone }>;
    playCta: string;
    storyCta: string;
    summaryError: string;
    summaryErrorLink: string;
    videoAria: string;
    videoFallback: string;
    caption: string;
  };
  timeline: {
    nav: string;
    kicker: string;
    title: string;
    items: Array<{ days: string; title: string; text: string; tone?: Tone }>;
  };
  common: {
    notRun: string;
    yes: string;
    no: string;
    unknown: string;
    missingMetric: string;
  };
  chapters: Chapter[];
  footer: {
    updated: string;
  };
}
