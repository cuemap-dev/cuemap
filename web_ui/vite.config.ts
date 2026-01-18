import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '/ui/',
  server: {
    port: 3000,
    proxy: {
      // Proxy all API requests to the Rust engine
      '/graph': 'http://localhost:8080',
      '/recall': 'http://localhost:8080',
      '/stats': 'http://localhost:8080',
      '/projects': 'http://localhost:8080',
      '/ingest': 'http://localhost:8080',
      '/lexicon': 'http://localhost:8080',
      '/memory': 'http://localhost:8080',
      '/jobs': 'http://localhost:8080',
      '/sandbox': 'http://localhost:8080',
    }
  }
})
