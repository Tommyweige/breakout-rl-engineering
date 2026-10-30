import type { VerificationStatus } from '../renderCaseStudy';

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
    summaryLoading: string;
    summaryLoaded: string;
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
    loading: string;
    source: string;
    missingMetric: string;
  };
  status: {
    labels: Record<VerificationStatus, string>;
    descriptions: Record<VerificationStatus, string>;
  };
  metricCards: Array<{
    metricId: string;
    label: string;
    note: string;
  }>;
  metricsAria: string;
  readingKey: {
    ariaLabel: string;
    title: string;
  };
  problem: {
    index: string;
    kicker: string;
    title: string;
    paragraphs: string[];
    callout: string;
    diagramAria: string;
    diagrams: Array<{
      label: string;
      steps: string[];
      note: string;
      accent?: boolean;
    }>;
  };
  method: {
    index: string;
    kicker: string;
    title: string;
    items: Array<{
      step: string;
      title: string;
      body: string;
      status: VerificationStatus;
    }>;
  };
  results: {
    index: string;
    kicker: string;
    title: string;
    intro: string;
    headers: {
      metric: string;
      value: string;
      scope: string;
      evidence: string;
      source: string;
    };
    metricLabels?: Record<string, string>;
  };
  fairness: {
    index: string;
    kicker: string;
    title: string;
    intro: string;
    headers: {
      property: string;
      v2: string;
      v3: string;
    };
    rows: string[];
    valuePhrases: {
      decisionPerFrames: string;
      decisionPerFrame: string;
      seedCount: string;
      yes: string;
      no: string;
    };
    comparisonNote: string;
    comparisonFallback: string;
    notes: string[];
  };
  laya: {
    index: string;
    kicker: string;
    title: string;
    paragraphs: Array<{ lead: string; body: string }>;
    callout: string;
    aside: {
      ariaLabel: string;
      heading: string;
      facts: Array<{ label: string; value: string; unit?: string }>;
      note: string;
      researchLinkLabel: string;
      prLinkLabel: string;
    };
  };
  negative: {
    index: string;
    kicker: string;
    title: string;
    cards: Array<{
      kicker: string;
      value: string;
      unit: string;
      body: string;
      status: VerificationStatus;
    }>;
  };
  ledger: {
    index: string;
    kicker: string;
    title: string;
    intro: string;
    downloads: Array<{ href: string; label: string }>;
  };
  cta: {
    kicker: string;
    title: string;
    body: string;
    playCta: string;
    githubCta: string;
  };
  footer: {
    brand: string;
    updated: string;
    limitsHeading: string;
    /** Two or three plain sentences. Each limit is stated once, in prose. */
    limits: string[];
    playDemo: string;
    evidence: string;
    github: string;
  };
}
