const http = require('http');
const url = require('url');
function isPrime(n) {
  if (n < 2) return false;
  for (let i = 2; i <= Math.sqrt(n); i++) {
    if (n % i === 0) return false;
  }
  return true;
}
function getPrimes(limit) {
  const primes = [];
  for (let i = 2; i <= limit; i++) {
    if (isPrime(i)) primes.push(i);
  }
  return primes;
}
const server = http.createServer((req, res) => {
  const parsedUrl = url.parse(req.url, true);
  const query = parsedUrl.query;
  const limit = parseInt(query.upto, 10);
  res.writeHead(200, { 'Content-Type': 'text/html' });
  if (!limit || isNaN(limit)) {
    res.end(`<h2>Please provide a number like this: <code>?upto=100</code></h2>`);
    return;
  }
  const primes = getPrimes(limit);
  res.end(`
    <h2>Prime numbers up to ${limit}:</h2>
    <p>${primes.join(', ')}</p>
  `);
});

const PORT = 3007;
server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}/`);
});
