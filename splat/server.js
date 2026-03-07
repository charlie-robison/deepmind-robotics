const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 8082;
const DIR = __dirname;

const mimeTypes = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.css': 'text/css',
    '.ply': 'application/octet-stream',
    '.splat': 'application/octet-stream',
};

http.createServer((req, res) => {
    let filePath = path.join(DIR, req.url === '/' ? 'gsplat-viewer.html' : req.url);
    const ext = path.extname(filePath);

    // Required headers for SharedArrayBuffer
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin');

    fs.readFile(filePath, (err, data) => {
        if (err) {
            res.writeHead(404);
            res.end('Not found');
            return;
        }
        res.setHeader('Content-Type', mimeTypes[ext] || 'application/octet-stream');
        res.writeHead(200);
        res.end(data);
    });
}).listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}/`);
});
