import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig({
  plugins: [svelte()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8050',
      '/oauth': 'http://localhost:8050',
    },
  },
  build: {
    // Output goes into server/static so FastAPI serves it
    outDir: '../server/static',
    emptyOutDir: true,
  },
})
