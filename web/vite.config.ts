import { defineConfig } from 'vite';

export default defineConfig({
  base: '/',
  publicDir: 'public',
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      input: {
        main: 'index.html',
        'case-study': 'case-study.html',
        'case-study.zh-TW': 'case-study.zh-TW.html',
      },
    },
  },
  server: {
    port: 5173,
    strictPort: true,
  },
});
