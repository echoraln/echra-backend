/**
 * Echra Backend — server.js
 * Secure Express server. Handles all AI communication.
 * API key never leaves this file.
 *
 * Setup:
 *   npm install
 *   Create .env with ANTHROPIC_API_KEY=sk-ant-...
 *   node server.js
 *
 * Deploy to Railway / Render / Fly.io — set env var there.
 */

import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import dotenv from 'dotenv';
import Anthropic from '@anthropic-ai/sdk';

dotenv.config();

/* ─── Validate environment ─── */
if (!process.env.ANTHROPIC_API_KEY) {
  console.error('ERROR: ANTHROPIC_API_KEY is not set. Add it to your .env file.');
  process.exit(1);
}

const app  = express();
const PORT = process.env.PORT || 3001;

/* ─── Anthropic client (server-side only) ─── */
const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

/* ─── Middleware ─── */
app.use(helmet());
app.use(express.json({ limit: '64kb' }));

// CORS — restrict to your frontend origin in production
const allowedOrigins = process.env.ALLOWED_ORIGINS
  ? process.env.ALLOWED_ORIGINS.split(',').map(o => o.trim())
  : ['http://localhost:3000', 'http://localhost:5173', 'http://127.0.0.1:5500'];

app.use(cors({
  origin: (origin, cb) => {
    // Allow requests with no origin (file://, Postman, same-origin)
    if (!origin || allowedOrigins.includes(origin)) return cb(null, true);
    cb(new Error(`CORS: origin ${origin} not allowed`));
  },
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type', 'X-Session-ID'],
}));

/* ─── Rate limiting ─── */
const chatLimiter = rateLimit({
  windowMs: 60 * 1000,   // 1 minute
  max: 30,               // max 30 AI requests per minute per IP
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests — slow down a little.' },
});

/* ─── In-memory session store ─────────────────────────────────────────────
 * Maps session_id → { messages: [{role, content}], lastAccess: Date }
 *
 * For production: replace with Redis or a database (Supabase, Postgres, etc.)
 * Sessions expire after SESSION_TTL_MS of inactivity.
 * ─────────────────────────────────────────────────────────────────────── */
const SESSION_TTL_MS  = 2 * 60 * 60 * 1000; // 2 hours
const MAX_HISTORY_LEN = 40;                  // max messages kept per session
const sessionStore    = new Map();

function getOrCreateSession(session_id) {
  if (!sessionStore.has(session_id)) {
    sessionStore.set(session_id, { messages: [], lastAccess: Date.now() });
  }
  const sess = sessionStore.get(session_id);
  sess.lastAccess = Date.now();
  return sess;
}

// Prune stale sessions every 30 minutes
setInterval(() => {
  const now = Date.now();
  for (const [id, sess] of sessionStore.entries()) {
    if (now - sess.lastAccess > SESSION_TTL_MS) {
      sessionStore.delete(id);
    }
  }
}, 30 * 60 * 1000);

/* ─── System prompt ─── */
function buildSystemPrompt(mood) {
  return `You are Echra — a teen-focused AI chat companion that feels like a real friend texting back. You are NOT a therapist, counselor, or mental health tool.

CORE PURPOSE: Simulate a real, supportive friend in chat form. React naturally, stay present, help the user figure out what to do next in a simple human way. Like texting a close friend who replies fast, understands context, and keeps it real.

MEMORY: You have the full conversation history for this session. Use it naturally — reference what was said earlier if relevant. Don't repeat yourself, and don't ignore prior context.

STRICT RULES (never break these):
- Zero therapy/clinical/psychological language.
- Never say "it sounds like," "you've been feeling," or "this may indicate."
- No long explanations or structured advice blocks.
- Do NOT sound like a mental health app or chatbot assistant.

TONE:
- Natural Gen Z texting style (iMessage / Snapchat / DMs)
- 1–5 sentences MAX, lowercase always
- Human, casual, slightly emotional when appropriate
- Prioritize sounding real over sounding "correct"

RESPONSE STRUCTURE:
1. React directly to what happened — specific and human
2. Optional: one short relatable line
3. ONE of: practical next step, simple question, or light suggestion

VIRAL MOMENTS (when it naturally fits, not forced):
"yeah that would ruin my whole day ngl"
"that's actually so annoying I'd be mad too"
"you didn't deserve that fr"

LATE NIGHT: slightly more emotionally present if message feels late/reflective.
BLUNT MODE: direct if user asks for honesty, never rude.
FUNNY MODE: casual humor if vibe is light.

BAD EXAMPLES — NEVER DO THIS:
"You've been experiencing emotional distress"
"It sounds like this is part of a larger pattern"
Long advice paragraphs or bullet-pointed suggestions
${mood ? `\nUser's current mood tag: ${mood}.` : ''}`;
}

/* ─── Input validation ─── */
function validateChatInput(body) {
  const { message, session_id, mood } = body;

  if (!message || typeof message !== 'string') {
    return 'message is required and must be a string';
  }
  if (message.trim().length === 0) {
    return 'message cannot be empty';
  }
  if (message.length > 2000) {
    return 'message is too long (max 2000 characters)';
  }
  if (!session_id || typeof session_id !== 'string') {
    return 'session_id is required and must be a string';
  }
  if (!/^[a-z0-9_-]{4,64}$/i.test(session_id)) {
    return 'session_id contains invalid characters';
  }
  if (mood && !['happy','sad','stressed','anxious','bored','good'].includes(mood)) {
    return 'invalid mood value';
  }
  return null;
}

/* ─── POST /chat ─────────────────────────────────────────────────────────
 * Body: { message: string, session_id: string, mood?: string }
 * Returns: { reply: string, session_id: string }
 * ─────────────────────────────────────────────────────────────────────── */
app.get("/", (req, res) => {
  res.send("Server is working 🚀");
});app.get("/chat", (req, res) => {
  res.json({ message: "Chat route is working 🚀" });
});
app.post('/chat', chatLimiter, async (req, res) => {
  try {
    const validationError = validateChatInput(req.body);
    if (validationError) {
      return res.status(400).json({ error: validationError });
    }

    const { message, session_id, mood } = req.body;
    const trimmedMessage = message.trim();

    // Get or create session history
    const session = getOrCreateSession(session_id);

    // Append the new user message to session history
    session.messages.push({ role: 'user', content: trimmedMessage });

    // Trim to max history length (keep most recent messages)
    if (session.messages.length > MAX_HISTORY_LEN) {
      session.messages = session.messages.slice(-MAX_HISTORY_LEN);
    }

    // Call Anthropic with full session history
    const response = await anthropic.messages.create({
      model: 'claude-sonnet-4-5-20251001',
      max_tokens: 300,
      system: buildSystemPrompt(mood || null),
      messages: session.messages,
    });

    const reply = response.content
      .filter(b => b.type === 'text')
      .map(b => b.text)
      .join('')
      .trim();

    if (!reply) {
      throw new Error('Empty response from AI');
    }

    // Append AI reply to session history
    session.messages.push({ role: 'assistant', content: reply });

    return res.json({ reply, session_id });

  } catch (err) {
    console.error('[/chat error]', err.message);

    // Don't leak internal error details to client
    if (err.status === 429) {
      return res.status(429).json({ error: 'AI is busy right now — try again in a moment.' });
    }
    if (err.status === 401) {
      return res.status(500).json({ error: 'Server configuration error.' });
    }
    return res.status(500).json({ error: 'Something went wrong — try again.' });
  }
});

/* ─── POST /session/clear ─────────────────────────────────────────────────
 * Clears server-side history for a session_id.
 * Body: { session_id: string }
 * ─────────────────────────────────────────────────────────────────────── */
app.post('/session/clear', (req, res) => {
  const { session_id } = req.body;
  if (!session_id || typeof session_id !== 'string') {
    return res.status(400).json({ error: 'session_id required' });
  }
  sessionStore.delete(session_id);
  return res.json({ ok: true });
});

/* ─── GET /health ─── */
app.get('/health', (_, res) => {
  res.json({ status: 'ok', sessions: sessionStore.size, ts: new Date().toISOString() });
});

/* ─── 404 catch-all ─── */
app.use((_, res) => res.status(404).json({ error: 'Not found' }));

/* ─── Global error handler ─── */
app.use((err, _req, res, _next) => {
  console.error('[unhandled error]', err.message);
  res.status(500).json({ error: 'Internal server error' });
});
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Echra backend running on http://localhost:${PORT}`);
  console.log(
    `API key loaded: ${process.env.ANTHROPIC_API_KEY ? "YES" : "NO"}`
  );
});

