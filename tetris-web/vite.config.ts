import { defineConfig } from 'vite';

// Content Security Policy applied both in dev (via Vite's dev server headers)
// and referenced by the static _headers file for production deploys.
//
// Key restrictions for the AI Manager upload feature:
//   worker-src blob:        — allows blob: workers (custom uploaded AIs)
//   connect-src <supabase>  — allows Supabase API calls from the main thread only;
//                             uploaded AI workers are still blocked at the JS level
//                             (fetch/XHR/WebSocket overridden in setupAiForGame)
const SUPABASE_URL = 'https://imecxemhqauzmxvdsgfh.supabase.co';
const CSP = [
  "default-src 'self'",
  "script-src 'self' 'unsafe-inline'",  // unsafe-inline needed for Vite dev HMR
  "style-src 'self' 'unsafe-inline'",
  "worker-src blob: 'self'",
  `connect-src 'self' ${SUPABASE_URL}`,
  "img-src 'self' data:",
  "font-src 'self'",
  "frame-ancestors 'none'",
].join('; ');

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: 'index.html',
        'ai-guide': 'ai-guide.html',
      },
    },
  },
  server: {
    headers: {
      'Content-Security-Policy': CSP,
    },
  },
  preview: {
    headers: {
      'Content-Security-Policy': CSP,
    },
  },
});
