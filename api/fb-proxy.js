// Vercel Serverless Function — Firebase API 代理
// 解决中国大陆无法直连 Google API 的问题
const BACKENDS = {
  identitytoolkit: 'https://identitytoolkit.googleapis.com',
  securetoken: 'https://securetoken.googleapis.com',
  firestore: 'https://firestore.googleapis.com',
};

// Disable Vercel's auto body parser so we can forward raw body as-is
// NOTE: config must be set AFTER function export (see bottom of file)

// Read raw body from request stream
function getRawBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', c => chunks.push(c));
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

module.exports = async function handler(req, res) {
  // CORS headers (always set)
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }

  // Path: /api/fb-proxy?p=identitytoolkit/v1/accounts:lookup&key=xxx
  const p = req.query.p || '';
  const slashIdx = p.indexOf('/');
  if (slashIdx < 0) {
    return res.status(400).json({ error: 'Missing service path' });
  }
  const service = p.substring(0, slashIdx);
  const rest = p.substring(slashIdx + 1);
  const backend = BACKENDS[service];
  if (!backend) {
    return res.status(404).json({ error: 'Unknown service' });
  }

  // Rebuild query string excluding 'p'
  const qs = Object.entries(req.query)
    .filter(([k]) => k !== 'p')
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
    .join('&');
  const targetUrl = `${backend}/${rest}${qs ? '?' + qs : ''}`;

  // Forward headers (keep content-type, authorization)
  const fwdHeaders = {};
  if (req.headers['content-type']) fwdHeaders['Content-Type'] = req.headers['content-type'];
  if (req.headers['authorization']) fwdHeaders['Authorization'] = req.headers['authorization'];

  try {
    // Read raw body and forward as-is (preserves form-encoded, JSON, etc.)
    const rawBody = req.method !== 'GET' && req.method !== 'HEAD'
      ? await getRawBody(req)
      : undefined;

    const upstream = await fetch(targetUrl, {
      method: req.method,
      headers: fwdHeaders,
      body: rawBody && rawBody.length > 0 ? rawBody : undefined,
    });

    const data = await upstream.text();
    const ct = upstream.headers.get('content-type');
    if (ct) res.setHeader('Content-Type', ct);
    res.status(upstream.status).send(data);
  } catch (err) {
    res.status(502).json({ error: 'Proxy upstream error', message: err.message });
  }
}

// Set config AFTER module.exports assignment so it doesn't get overwritten
module.exports.config = { api: { bodyParser: false } };
