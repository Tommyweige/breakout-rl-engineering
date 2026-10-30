export type Locale = 'en' | 'zh-TW';

export interface CaseStudyCopy {
  header: {
    brandHomeLabel: string;
    brandMark: string;
    brandSub: string;
    navLabel: string;
    playDemo: string;
    evidence: string;
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
    eyebrow: string;
    titleLead: string;
    titleTail: string;
    lede: string;
    hook: string;
    playCta: string;
    evidenceCta: string;
    summaryError: string;
    summaryErrorLink: string;
    videoAria: string;
    videoFallback: string;
    posterLink: string;
    videoFallbackAfter: string;
    figcaption: string;
  };
  common: {
    notRun: string;
    yes: string;
    no: string;
    unknown: string;
    missingMetric: string;
  };
  metricCards: Array<{
    metricId: string;
    label: string;
    /** Rendered after the value, e.g. "／15 局". */
    suffix?: string;
    note: string;
  }>;
  metricsAria: string;
  problem: {
    index: string;
    kicker: string;
    title: string;
    paragraphs: string[];
  };
  routes: {
    index: string;
    kicker: string;
    title: string;
    items: Array<{ lead: string; body: string }>;
  };
  results: {
    index: string;
    kicker: string;
    title: string;
    intro: string;
    headers: { method: string; mean: string; clears: string; trained: string };
    rows: Array<{ method: string; mean: string; clears: string; trained: string }>;
  };
  mistake: {
    index: string;
    kicker: string;
    title: string;
    paragraphs: string[];
  };
  laya: {
    index: string;
    kicker: string;
    title: string;
    paragraphs: string[];
  };
  negative: {
    index: string;
    kicker: string;
    title: string;
    paragraphs: string[];
  };
  footer: {
    brand: string;
    updated: string;
    limitsHeading: string;
    /** Two or three plain sentences. Each limit is stated once, in prose. */
    limits: string[];
    /** The single line that sends a reader to the repository. */
    evidenceLine: string;
  };
}
