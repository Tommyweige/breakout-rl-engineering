import { enCopy } from './en';
import { zhTwCopy } from './zh-TW';
import type { CaseStudyCopy, Locale } from './types';

export type { CaseStudyCopy, Locale };

export const CASE_STUDY_COPY: Record<Locale, CaseStudyCopy> = {
  en: enCopy,
  'zh-TW': zhTwCopy,
};

export function resolveLocale(lang: string | undefined | null): Locale {
  return lang?.toLowerCase().startsWith('zh') ? 'zh-TW' : 'en';
}
