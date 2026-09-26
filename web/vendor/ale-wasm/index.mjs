let browserFactoryPromise;

export default async function createALEModule(options = {}) {
  const factory = await loadBrowserFactory();
  return factory(options);
}

function loadBrowserFactory() {
  const existingFactory = globalThis.createALEModule;
  if (typeof existingFactory === 'function') return Promise.resolve(existingFactory);
  if (browserFactoryPromise) return browserFactoryPromise;

  browserFactoryPromise = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = new URL('/ale/ale.js', window.location.href).toString();
    script.async = true;
    script.onload = () => {
      const factory = globalThis.createALEModule;
      if (typeof factory === 'function') resolve(factory);
      else reject(new Error('ALE WebAssembly loader script did not expose createALEModule.'));
    };
    script.onerror = () => reject(new Error(`Failed to load ALE WebAssembly loader from ${script.src}.`));
    document.head.append(script);
  });
  return browserFactoryPromise;
}
