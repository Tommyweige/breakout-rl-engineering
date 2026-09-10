import type { BrowserInfo } from './types';

interface UserAgentDataLike {
  getHighEntropyValues?: (hints: string[]) => Promise<{ brands?: Array<{ brand: string; version: string }>; fullVersionList?: Array<{ brand: string; version: string }> }>;
}

export async function detectBrowser(userAgent: string = navigator.userAgent): Promise<BrowserInfo> {
  const candidates: Array<[string, RegExp]> = [
    ['Edge', /Edg\/([\d.]+)/],
    ['Chrome', /Chrome\/([\d.]+)/],
    ['Firefox', /Firefox\/([\d.]+)/],
    ['Safari', /Version\/([\d.]+).*Safari\//],
  ];
  let detected: BrowserInfo = { name: 'Unknown', version: 'Unknown', userAgent };
  for (const [name, pattern] of candidates) {
    const match = userAgent.match(pattern);
    if (match?.[1]) {
      detected = { name, version: match[1], userAgent };
      break;
    }
  }
  const userAgentData = (navigator as Navigator & { userAgentData?: UserAgentDataLike }).userAgentData;
  if (userAgentData?.getHighEntropyValues) {
    try {
      const highEntropy = await userAgentData.getHighEntropyValues(['fullVersionList']);
      const brands = highEntropy.fullVersionList ?? highEntropy.brands ?? [];
      const browserBrand = brands.find(({ brand }) => /Google Chrome|Microsoft Edge|Chromium/i.test(brand));
      if (browserBrand) {
        detected = {
          name: /Edge/i.test(browserBrand.brand) ? 'Edge' : 'Chrome',
          version: browserBrand.version,
          userAgent,
        };
      }
    } catch {
      // The reduced UA version remains useful when high-entropy UA data is unavailable.
    }
  }
  return detected;
}

export function currentPageUrl(): string {
  return typeof window === 'undefined' ? 'unknown' : window.location.href;
}

export function currentPreviewUrl(): string | null {
  if (typeof window === 'undefined') return null;
  return window.location.hostname.endsWith('.pages.dev') ? window.location.origin : null;
}
