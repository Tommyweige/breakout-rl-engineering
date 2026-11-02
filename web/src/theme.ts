/**
 * Dark/light switching shared by the game page and both case-study pages.
 *
 * The inline script in each HTML file applies the saved theme before first paint;
 * this module only keeps the switch, the stored choice, and the OS listener in sync
 * after the page has a toggle to show. Keep the storage key in step with that script.
 */

import './theme.css';

export type Theme = 'light' | 'dark';

const STORAGE_KEY = 'breakout-theme';
const DARK_QUERY = '(prefers-color-scheme: dark)';
const TOGGLE_SELECTOR = '[data-role="theme-toggle"]';

/**
 * Both states of the switch, so it can show the theme in use without a re-render.
 * theme.css hides the one that does not apply.
 */
export const THEME_TOGGLE_ICONS = `
  <svg class="theme-icon theme-icon-sun" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><circle cx="12" cy="12" r="4.1" /><path d="M12 2.6v2.2M12 19.2v2.2M4.7 4.7l1.6 1.6M17.7 17.7l1.6 1.6M2.6 12h2.2M19.2 12h2.2M4.7 19.3l1.6-1.6M17.7 6.3l1.6-1.6" /></svg>
  <svg class="theme-icon theme-icon-moon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M20.3 14.8A8.6 8.6 0 0 1 9.2 3.7a7.2 7.2 0 1 0 11.1 11.1Z" /></svg>`;

function readStored(): Theme | null {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    return stored === 'light' || stored === 'dark' ? stored : null;
  } catch {
    // Storage is blocked in some privacy modes; the switch still works for this visit.
    return null;
  }
}

function writeStored(theme: Theme): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // Same as above: an unpersisted choice beats a broken button.
  }
}

export function systemTheme(): Theme {
  return window.matchMedia?.(DARK_QUERY).matches ? 'dark' : 'light';
}

/** The theme on screen right now: the reader's choice, otherwise the system's. */
export function resolvedTheme(): Theme {
  const applied = document.documentElement.dataset.theme;
  if (applied === 'light' || applied === 'dark') return applied;
  return readStored() ?? systemTheme();
}

export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
  // The browser bar colour for each theme lives on the page's theme-color meta.
  const meta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');
  const colour = theme === 'dark' ? meta?.dataset.dark : meta?.dataset.light;
  if (meta && colour) meta.content = colour;
}

/**
 * Wires every toggle in the document. The button shows the current theme in its icon
 * and reports "dark theme on" through aria-pressed.
 */
export function initThemeToggle(root: ParentNode = document): void {
  const buttons = [...root.querySelectorAll<HTMLButtonElement>(TOGGLE_SELECTOR)];
  if (buttons.length === 0) return;

  const sync = (theme: Theme): void => {
    for (const button of buttons) button.setAttribute('aria-pressed', String(theme === 'dark'));
  };

  const set = (theme: Theme, persist: boolean): void => {
    applyTheme(theme);
    if (persist) writeStored(theme);
    sync(theme);
  };

  for (const button of buttons) {
    button.addEventListener('click', () => {
      set(resolvedTheme() === 'dark' ? 'light' : 'dark', true);
    });
  }

  // Keep following the OS until the reader picks a side themselves.
  window.matchMedia?.(DARK_QUERY).addEventListener?.('change', (event) => {
    if (readStored() === null) set(event.matches ? 'dark' : 'light', false);
  });

  // The inline script already painted this theme; re-applying keeps the browser bar and
  // the switch in step even when a page forgets that script.
  set(resolvedTheme(), false);
}
