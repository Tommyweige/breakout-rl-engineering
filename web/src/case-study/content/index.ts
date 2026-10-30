import { enCopy } from './en';
import { zhTwCopy, zhTwResultLabelOverrides, type ResultLabelOverrides } from './zh-TW';
import type { CaseStudyCopy, Locale } from './types';

export type { CaseStudyCopy, Locale, ResultLabelOverrides };

export const CASE_STUDY_COPY: Record<Locale, CaseStudyCopy> = {
  en: enCopy,
  'zh-TW': zhTwCopy,
};

// Locales without an entry show summary.json's original unit and seed_role text.
export const CASE_STUDY_RESULT_LABELS: Partial<Record<Locale, ResultLabelOverrides>> = {
  'zh-TW': zhTwResultLabelOverrides,
};

export function resolveLocale(lang: string | undefined | null): Locale {
  return lang?.toLowerCase().startsWith('zh') ? 'zh-TW' : 'en';
}
