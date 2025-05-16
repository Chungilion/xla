const express = require('express');
const fileUpload = require('express-fileupload');
const cors = require('cors');
const path = require('path');

const app = express();

// Middleware
app.use(cors());
app.use(fileUpload());
app.use(express.static(path.join(__dirname, 'public')));

// Routes
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
