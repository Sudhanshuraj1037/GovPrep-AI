const CACHE_NAME = 'govprep-ai-v2';
const STATIC_ASSETS = ['/', '/index.html', '/manifest.json'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE_NAME).then(c => c.addAll(STATIC_ASSETS).catch(() => {})));
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))))
  );
  self.clients.claim();
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  // Do not cache API calls or webpack hot reload/dev bundles
  if (url.pathname.startsWith('/api/') || url.hostname === 'localhost' || url.hostname === '127.0.0.1' || url.pathname.includes('hot-update')) {
    e.respondWith(fetch(e.request));
    return;
  }
  if (e.request.mode === 'navigate') {
    e.respondWith(fetch(e.request).then(r => { caches.open(CACHE_NAME).then(c => c.put(e.request, r.clone())); return r; }).catch(() => caches.match('/index.html')));
    return;
  }
  e.respondWith(caches.match(e.request).then(cached => cached || fetch(e.request).then(r => { caches.open(CACHE_NAME).then(c => c.put(e.request, r.clone())); return r; })));
});
