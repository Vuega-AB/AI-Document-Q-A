// my-chat-app-backend/server.js
const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const cors = require('cors');

const app = express();
const PORT = 3001;
const JWT_SECRET = 'your-super-secret-key-for-jwt-lawyer-app-v4-allusers';

const HARDCODED_ADMIN_EMAIL = 'nancyhisham2003@gmail.com';
const HARDCODED_ADMIN_PASSWORD = '123';

app.use(cors());
app.use(express.json());

const db = new sqlite3.Database('./chat_app.db', (err) => {
    if (err) {
        console.error('Error opening database', err.message);
    } else {
        console.log('Connected to the SQLite database.');
        db.serialize(() => {
            db.run(`CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                is_approved INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )`, (err) => {
                if (err) console.error("Error creating users table:", err.message);
                else {
                    console.log("Users table schema checked/created.");
                    // Attempt to add columns - simplified error logging for brevity
                    db.run("ALTER TABLE users ADD COLUMN is_approved INTEGER NOT NULL DEFAULT 0", () => {});
                    db.run("ALTER TABLE users ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP", () => {});
                }
            });
        });
    }
});

// Signup - is_approved defaults to 0, created_at defaults to CURRENT_TIMESTAMP
app.post('/api/signup', async (req, res) => {
    const { name, email, password } = req.body;
    if (!name || !email || !password) return res.status(400).json({ message: 'All fields are required' });
    if (email.toLowerCase() === HARDCODED_ADMIN_EMAIL.toLowerCase()) return res.status(409).json({ message: 'This email is reserved.' });

    db.get('SELECT email FROM users WHERE email = ?', [email.toLowerCase()], async (err, row) => {
        if (err) return res.status(500).json({ message: 'Database error', error: err.message });
        if (row) return res.status(409).json({ message: 'Email already registered.' });

        try {
            const password_hash = await bcrypt.hash(password, 10);
            db.run('INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)',
                [name, email.toLowerCase(), password_hash, 'user'],
                function (err) {
                    if (err) return res.status(500).json({ message: 'Could not register user', error: err.message });
                    res.status(201).json({ message: 'Registration successful! Your account is pending admin approval.' });
                }
            );
        } catch (e) { res.status(500).json({ message: 'Error processing registration', error: e.message }); }
    });
});

// Login
app.post('/api/login', (req, res) => {
    const { email, password } = req.body;
    if (!email || !password) return res.status(400).json({ message: 'Email and password are required' });

    const lowerEmail = email.toLowerCase();

    if (lowerEmail === HARDCODED_ADMIN_EMAIL.toLowerCase() && password === HARDCODED_ADMIN_PASSWORD) {
        const token = jwt.sign({ userId: -1, email: HARDCODED_ADMIN_EMAIL, role: 'admin', is_approved: 1 }, JWT_SECRET, { expiresIn: '1h' });
        return res.json({ message: 'Logged in as Hardcoded Admin', token, userId: -1, userName: "Nancy Hisham (Admin)", role: 'admin' });
    }

    db.get('SELECT * FROM users WHERE email = ?', [lowerEmail], async (err, user) => {
        if (err) return res.status(500).json({ message: 'Database error', error: err.message });
        if (!user) return res.status(401).json({ message: 'User not found or invalid credentials. If pending, wait for approval.' });
        if (user.is_approved !== 1) return res.status(403).json({ message: 'Account pending admin approval.' });

        const isMatch = await bcrypt.compare(password, user.password_hash);
        if (!isMatch) return res.status(401).json({ message: 'Invalid credentials.' });

        const token = jwt.sign({ userId: user.id, email: user.email, role: user.role, is_approved: user.is_approved }, JWT_SECRET, { expiresIn: '1h' });
        res.json({ message: 'Logged in successfully', token, userId: user.id, userName: user.name, role: user.role });
    });
});

// Middleware
function authenticateToken(req, res, next) {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];
    if (!token) return res.sendStatus(401);
    jwt.verify(token, JWT_SECRET, (err, decodedUser) => {
        if (err) return res.sendStatus(403);
        req.user = decodedUser;
        next();
    });
}

function isAdmin(req, res, next) {
    if (req.user && req.user.role === 'admin') return next();
    // Allow hardcoded admin via special userId
    if (req.user && req.user.userId === -1 && req.user.email === HARDCODED_ADMIN_EMAIL) return next();
    return res.status(403).json({ message: 'Forbidden: Admin access required' });
}

// Admin: Get All Users
// In server.js
app.get('/api/admin/all-users', authenticateToken, isAdmin, (req, res) => {
    db.all("SELECT id, name, email, role, is_approved, strftime('%Y-%m-%d %H:%M', created_at, 'localtime') as created_at FROM users ORDER BY created_at DESC", [], (err, rows) => {
        if (err) {
            console.error("Error fetching all users from DB:", err.message); // Add this log
            return res.status(500).json({ message: "Error fetching all users", error: err.message });
        }
        console.log("Sending all users:", rows); // Add this log to see if data is fetched
        res.json(rows);
    });
});

// Admin: Approve User (remains the same)
app.put('/api/admin/approve-user/:userId', authenticateToken, isAdmin, (req, res) => {
    const { userId } = req.params;
    db.run("UPDATE users SET is_approved = 1 WHERE id = ? AND role = 'user'", [userId], function(err) {
        if (err) return res.status(500).json({ message: "Error approving user", error: err.message });
        if (this.changes === 0) return res.status(404).json({ message: "User not found, already approved, or not a 'user' role." });
        res.json({ message: `User ID ${userId} approved successfully.` });
    });
});

app.put('/api/admin/user/:userId/set-approval', authenticateToken, isAdmin, (req, res) => {
    const { userId } = req.params;
    const { new_approval_status } = req.body; // Expecting { "new_approval_status": true } or { "new_approval_status": false }

    if (typeof new_approval_status !== 'boolean') {
        return res.status(400).json({ message: "Invalid approval status provided. Must be true or false." });
    }

    const approvalValue = new_approval_status ? 1 : 0;
    const actionText = new_approval_status ? "approved" : "set to pending";

    // Prevent changing approval status for the hardcoded admin if they have a DB entry
    // (though hardcoded admin login bypasses DB is_approved check anyway)
    db.get("SELECT email, role FROM users WHERE id = ?", [userId], (err, userToUpdate) => {
        if (err) return res.status(500).json({ message: "Database error checking user", error: err.message });
        if (!userToUpdate) return res.status(404).json({ message: "User not found." });

        if (userToUpdate.email.toLowerCase() === HARDCODED_ADMIN_EMAIL.toLowerCase()) {
            return res.status(403).json({ message: "Cannot change approval status of the primary hardcoded admin account via API."});
        } 
        // Prevent admin from pending another admin (can only pend 'user' roles)
        // Or prevent admin from approving a user to become admin directly (role change is separate)
        if (userToUpdate.role === 'admin' && !new_approval_status) {
            return res.status(403).json({ message: "Admin accounts cannot be set to pending. Change role to 'user' first if needed." });
        }


        db.run("UPDATE users SET is_approved = ? WHERE id = ? AND role = 'user'", [approvalValue, userId], function(err) {
            if (err) {
                return res.status(500).json({ message: `Error updating user approval status`, error: err.message });
            }
            if (this.changes === 0) {
                // Could be because user not found, or not a 'user' role, or status already set
                return res.status(404).json({ message: "User not found, not a 'user' role, or status already set as requested." });
            }
            res.json({ message: `User ID ${userId} has been ${actionText}.` });
        });
    });
});

// Admin: Update User Role
// app.put('/api/admin/update-user-role/:userId', authenticateToken, isAdmin, (req, res) => {
//     const { userId } = req.params;
//     const { newRole } = req.body; // Expecting { "newRole": "admin" } or { "newRole": "user" }

//     if (!newRole || (newRole !== 'admin' && newRole !== 'user')) {
//         return res.status(400).json({ message: "Invalid role specified. Must be 'admin' or 'user'." });
//     }

//     // Prevent hardcoded admin from having their role changed via API if they have a DB entry
//     // This check is more symbolic if the hardcoded admin doesn't have a DB entry
//     db.get("SELECT email FROM users WHERE id = ?", [userId], (err, userToUpdate) => {
//         if (err) return res.status(500).json({ message: "Database error", error: err.message });
//         if (userToUpdate && userToUpdate.email.toLowerCase() === HARDCODED_ADMIN_EMAIL.toLowerCase()) {
//             return res.status(403).json({ message: "Cannot change role of the primary hardcoded admin account via API."});
//         }

//         db.run("UPDATE users SET role = ? WHERE id = ?", [newRole, userId], function(err) {
//             if (err) return res.status(500).json({ message: "Error updating user role", error: err.message });
//             if (this.changes === 0) return res.status(404).json({ message: "User not found." });
//             res.json({ message: `User ID ${userId}'s role updated to ${newRole}.` });
//         });
//     });
// });


// Admin: Delete User (remains largely the same, but ensure not deleting hardcoded admin via this if they had a DB entry)
app.delete('/api/admin/user/:userId', authenticateToken, isAdmin, (req, res) => {
    const { userId } = req.params;

    db.get("SELECT email FROM users WHERE id = ?", [userId], (err, userToDelete) => {
        if (err) return res.status(500).json({ message: "Database error", error: err.message });
        if (userToDelete && userToDelete.email.toLowerCase() === HARDCODED_ADMIN_EMAIL.toLowerCase()) {
            return res.status(403).json({ message: "Primary hardcoded admin account cannot be deleted via API."});
        }

        db.run("DELETE FROM users WHERE id = ?", [userId], function(err) {
            if (err) return res.status(500).json({ message: "Error deleting user", error: err.message });
            if (this.changes === 0) return res.status(404).json({ message: "User not found." });
            res.json({ message: `User ID ${userId} deleted successfully.` });
        });
    });
});


app.listen(PORT, () => console.log(`Backend server running on http://localhost:${PORT}`));