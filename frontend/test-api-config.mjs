import assert from 'node:assert/strict';

import { PROD_API_URL, resolveApiBaseUrl } from './src/services/apiConfig.js';

assert.equal(
  resolveApiBaseUrl({ VITE_API_URL: 'https://example.com///', DEV: true }),
  'https://example.com'
);

assert.equal(resolveApiBaseUrl({ DEV: true }), '/api');
assert.equal(resolveApiBaseUrl({ DEV: false }), PROD_API_URL);

console.log('apiConfig checks passed');
