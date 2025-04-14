const express = require('express');
const app = express();

// Custom Middleware
const loggerMiddleware = (req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
  next(); // Call next() to pass control to the next middleware or route handler
};

// Use the middleware
app.use(loggerMiddleware);

// Route
app.get('/', (req, res) => {
  res.send('Hello, Middleware!');
});

// Start the server
const PORT = 3012;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});