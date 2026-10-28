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
      },
    },
  },
  server: {
    port: 5173,
    strictPort: true,
  },
});
