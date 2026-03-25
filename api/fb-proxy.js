// Vercel Serverless Function — Firebase API 代理 v2
// 解决中国大陆无法直连 Google API 的问题
const BACKENDS = {
  identitytoolkit: 'https://identitytoolkit.googleapis.com',
  securetoken: 'https://securetoken.googleapis.com',
  firestore: 'https://firestore.googleapis.com',
};

module.exports = async function handler(req, res) {
  // CORS headers (always set)
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }

  // Debug mode: show body diagnostics
  if (req.query.debug === '1') {
    return res.status(200).json({
      method: req.method,
      ct: req.headers['content-type'],
      bodyType: typeof req.body,
      isBuffer: Buffer.isBuffer(req.body),
      bodyStr: typeof req.body === 'string' ? req.body.substring(0, 200) :
               Buffer.isBuffer(req.body) ? req.body.toString('utf8').substring(0, 200) :
               JSON.stringify(req.body).substring(0, 200),
    });
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

  // Forward headers
  const fwdHeaders = {};
  if (req.headers['content-type']) fwdHeaders['Content-Type'] = req.headers['content-type'];
  if (req.headers['authorization']) fwdHeaders['Authorization'] = req.headers['authorization'];

  // Reconstruct body from Vercel's parsed req.body
  let body = undefined;
  if (req.method !== 'GET' && req.method !== 'HEAD' && req.body != null) {
    if (typeof req.body === 'string') {
      body = req.body;
    } else if (Buffer.isBuffer(req.body)) {
      body = req.body;
    } else if (typeof req.body === 'object') {
      const ct = (req.headers['content-type'] || '').toLowerCase();
      if (ct.includes('x-www-form-urlencoded')) {
        // Re-encode as form data
        body = Object.entries(req.body)
          .map(([k, v]) => encodeURIComponent(k) + '=' + encodeURIComponent(v))
          .join('&');
      } else {
        body = JSON.stringify(req.body);
      }
    }
  }

  try {
    const upstream = await fetch(targetUrl, {
      method: req.method,
      headers: fwdHeaders,
      body: body || undefined,
    });

    const data = await upstream.text();
    const ct = upstream.headers.get('content-type');
    if (ct) res.setHeader('Content-Type', ct);
    res.status(upstream.status).send(data);
  } catch (err) {
    res.status(502).json({ error: 'Proxy upstream error', message: err.message });
  }
};
}
