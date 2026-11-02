// @vitest-environment jsdom

import { beforeEach, describe, expect, it, vi } from 'vitest';

import { initThemeToggle, resolvedTheme } from './theme';

function stubSystemTheme(dark: boolean): void {
  vi.stubGlobal(
    'matchMedia',
    vi.fn((query: string) => ({
      matches: query === '(prefers-color-scheme: dark)' && dark,
      media: query,
      addEventListener: vi.fn(),
    })),
  );
}

function mountToggle(): { button: HTMLButtonElement; meta: HTMLMetaElement } {
  const meta = document.createElement('meta');
  meta.name = 'theme-color';
  meta.content = '#ffffff';
  meta.dataset.light = '#ffffff';
  meta.dataset.dark = '#15171c';
  document.head.append(meta);

  const button = document.createElement('button');
  button.type = 'button';
  button.dataset.role = 'theme-toggle';
  button.setAttribute('aria-label', 'Dark theme');
  document.body.append(button);

  return { button, meta };
}

describe('theme switching', () => {
  beforeEach(() => {
    document.head.innerHTML = '';
    document.body.innerHTML = '';
    delete document.documentElement.dataset.theme;
    window.localStorage.clear();
    vi.unstubAllGlobals();
  });

  it('starts on the system preference and reports it on the switch', () => {
    stubSystemTheme(true);
    const { button, meta } = mountToggle();
    // The inline script in each HTML file paints this attribute before the module loads.
    document.documentElement.dataset.theme = 'dark';

    initThemeToggle();

    expect(resolvedTheme()).toBe('dark');
    expect(button.getAttribute('aria-pressed')).toBe('true');
    expect(meta.content).toBe('#15171c');
    expect(window.localStorage.getItem('breakout-theme')).toBeNull();
  });

  it('remembers the reader choice instead of the system preference', () => {
    stubSystemTheme(true);
    const { button, meta } = mountToggle();
    document.documentElement.dataset.theme = 'dark';
    initThemeToggle();

    button.click();

    expect(resolvedTheme()).toBe('light');
    expect(button.getAttribute('aria-pressed')).toBe('false');
    expect(meta.content).toBe('#ffffff');
    expect(window.localStorage.getItem('breakout-theme')).toBe('light');

    // A fresh page without the inline script resolves from the stored choice, not the OS.
    delete document.documentElement.dataset.theme;
    expect(resolvedTheme()).toBe('light');
  });
});
