export type Locale = 'en' | 'zh-TW';

/**
 * Copy templates may contain hydration tokens:
 * `{{metric:id}}` and `{{seeds:id}}` (episode count).
 */
export interface CaseStudyStat {
  value: string;
  unit?: string;
  label: string;
}

export interface CaseStudyCard {
  name: string;
  tagline?: string;
  body: string;
  /** Big one-line outcome, e.g. "No gain". */
  verdict?: string;
  /** Small monospace line with the numbers behind the verdict. */
  detail?: string;
}

export interface CaseStudyBar {
  name: string;
  metricId: string;
}

/** One block of the page. Each block says one thing. */
export interface CaseStudySection {
  id: string;
  kicker: string;
  title: string;
  /** One short sentence right under the title, before any numbers. */
  lead?: string;
  /** The block's headline number, shown large. */
  stat?: { value: string; unit?: string };
  /** Two or more numbers shown side by side. */
  stats?: CaseStudyStat[];
  /** Horizontal bars scaled against `barMaxMetricId`. */
  bars?: CaseStudyBar[];
  barMaxMetricId?: string;
  barMaxLabel?: string;
  cards?: CaseStudyCard[];
  body?: string[];
  /** Small print under the block. */
  note?: string;
  tone?: 'lime' | 'coral' | 'blue';
}

export interface CaseStudyCopy {
  header: {
    brandHomeLabel: string;
    brandMark: string;
    navLabel: string;
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
    answer: string;
    answerBody: string;
    scrollCta: string;
    summaryError: string;
    summaryErrorLink: string;
    videoAria: string;
    videoFallback: string;
    caption: string;
  };
  common: {
    notRun: string;
    yes: string;
    no: string;
    unknown: string;
    missingMetric: string;
  };
  sections: CaseStudySection[];
  evidence: {
    kicker: string;
    title: string;
    body: string;
    cta: string;
    playCta: string;
  };
  footer: {
    updated: string;
  };
}
