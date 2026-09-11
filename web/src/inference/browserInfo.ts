import type { BrowserInfo } from './types';

interface UserAgentDataLike {
  platform?: string;
  getHighEntropyValues?: (hints: string[]) => Promise<{
    brands?: Array<{ brand: string; version: string }>;
    fullVersionList?: Array<{ brand: string; version: string }>;
    platform?: string;
  }>;
}

export async function detectBrowser(userAgent: string = navigator.userAgent): Promise<BrowserInfo> {
  const candidates: Array<[string, RegExp]> = [
    ['Edge', /Edg\/([\d.]+)/],
    ['Chrome', /Chrome\/([\d.]+)/],
    ['Firefox', /Firefox\/([\d.]+)/],
    ['Safari', /Version\/([\d.]+).*Safari\//],
  ];
  const userAgentData = (navigator as Navigator & { userAgentData?: UserAgentDataLike }).userAgentData;
  const platform = userAgentData?.platform ?? navigator.platform ?? 'Unknown';
  let detected: BrowserInfo = { name: 'Unknown', version: 'Unknown', userAgent, platform };
  for (const [name, pattern] of candidates) {
    const match = userAgent.match(pattern);
    if (match?.[1]) {
      detected = { name, version: match[1], userAgent, platform };
      break;
    }
  }
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
          platform: highEntropy.platform ?? platform,
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
