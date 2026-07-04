const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");

const JWT_SECRET = process.env.JWT_SECRET || "govprep-dev-secret-change-in-production";

async function hashPassword(plain) { return bcrypt.hash(plain, 10); }
async function verifyPassword(plain, hash) { return bcrypt.compare(plain, hash); }
function signToken(user) { return jwt.sign({ id: user.id, name: user.name }, JWT_SECRET, { expiresIn: "30d" }); }

function requireAuth(req, res, next) {
  const header = req.headers.authorization || "";
  const token = header.startsWith("Bearer ") ? header.slice(7) : null;
  if (!token) return res.status(401).json({ error: "Not authenticated. Please log in." });
  try {
    const payload = jwt.verify(token, JWT_SECRET);
    req.userId = payload.id;
    req.userName = payload.name;
    next();
  } catch {
    return res.status(401).json({ error: "Session expired. Please log in again." });
  }
}

module.exports = { hashPassword, verifyPassword, signToken, requireAuth };
