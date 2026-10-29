// @vitest-environment jsdom

import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';

import { describe, expect, it } from 'vitest';


const pages = [
  {
    file: 'case-study.html',
    lang: 'en',
    canonical: 'https://breakout.tommypan.com/case-study',
    ogLocale: 'en_US',
    alternateLocale: 'zh_TW',
    ogImage: 'https://breakout.tommypan.com/case-study/gameplay-poster.png',
  },
  {
    file: 'case-study.zh-TW.html',
    lang: 'zh-Hant-TW',
    canonical: 'https://breakout.tommypan.com/case-study.zh-TW',
    ogLocale: 'zh_TW',
    alternateLocale: 'en_US',
    ogImage: 'https://breakout.tommypan.com/case-study/gameplay-poster.zh-TW.png',
  },
] as const;

function load(file: string): Document {
  const html = readFileSync(resolve(process.cwd(), file), 'utf8');
  return new DOMParser().parseFromString(html, 'text/html');
}

function meta(document: Document, attribute: 'property' | 'name', value: string): string {
  return document.querySelector(`meta[${attribute}="${value}"]`)?.getAttribute('content') ?? '';
}

describe('Case study HTML entries', () => {
  it.each(pages)('declares language, canonical, and hreflang for $file', (page) => {
    const document = load(page.file);

    expect(document.documentElement.lang).toBe(page.lang);
    expect(document.querySelector('link[rel="canonical"]')?.getAttribute('href')).toBe(page.canonical);

    const alternates = new Map(
      [...document.querySelectorAll('link[rel="alternate"][hreflang]')].map((link) => [
        link.getAttribute('hreflang'),
        link.getAttribute('href'),
      ]),
    );
    expect(alternates.get('en')).toBe('https://breakout.tommypan.com/case-study');
    expect(alternates.get('zh-TW')).toBe(
      'https://breakout.tommypan.com/case-study.zh-TW',
    );
    expect(alternates.get('x-default')).toBe('https://breakout.tommypan.com/case-study');
    expect(alternates.size).toBe(3);
  });

  it.each(pages)('declares locale-specific Open Graph metadata for $file', (page) => {
    const document = load(page.file);

    expect(meta(document, 'property', 'og:title')).not.toBe('');
    expect(meta(document, 'property', 'og:description')).not.toBe('');
    expect(meta(document, 'property', 'og:image')).toBe(page.ogImage);
    expect(meta(document, 'property', 'og:url')).toBe(page.canonical);
    expect(meta(document, 'property', 'og:locale')).toBe(page.ogLocale);
    expect(meta(document, 'property', 'og:locale:alternate')).toBe(page.alternateLocale);
    expect(document.querySelector('script[type="module"]')?.getAttribute('src')).toBe(
      '/src/case-study/main.ts',
    );
  });

  it('gives the Chinese page its own title, description, and image instead of reusing English', () => {
    const en = load('case-study.html');
    const zh = load('case-study.zh-TW.html');

    expect(meta(zh, 'property', 'og:title')).not.toBe(meta(en, 'property', 'og:title'));
    expect(meta(zh, 'property', 'og:description')).not.toBe(
      meta(en, 'property', 'og:description'),
    );
    expect(meta(zh, 'property', 'og:image')).not.toBe(meta(en, 'property', 'og:image'));
    expect(zh.title).not.toBe(en.title);
  });

  it('uses Cloudflare Pages clean paths instead of .html routes', () => {
    for (const page of pages) {
      const document = load(page.file);
      expect(page.canonical).not.toContain('.html');
      expect(meta(document, 'property', 'og:url')).not.toContain('.html');
      for (const link of document.querySelectorAll('link[rel="alternate"][hreflang]')) {
        expect(link.getAttribute('href')).not.toContain('.html');
      }
    }
  });

  it('ships both Open Graph images as real PNG files', () => {
    for (const page of pages) {
      const asset = page.ogImage.replace('https://breakout.tommypan.com/', '');
      const path = resolve(process.cwd(), 'public', asset);
      expect(existsSync(path), `${asset} must exist`).toBe(true);
      const header = readFileSync(path).subarray(0, 8);
      expect([...header]).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
    }
  });
});
