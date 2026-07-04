// database.js — Pure JS file-based database using lowdb v1
// No native compilation needed. Data saved to backend/govprep-db.json

const low = require('lowdb');
const FileSync = require('lowdb/adapters/FileSync');
const path = require('path');

const adapter = new FileSync(path.join(__dirname, 'govprep-db.json'));
const db = low(adapter);

// Set defaults (schema)
db.defaults({
  users: [],
  pdfs: [],
  tests: [],
  test_results: [],
  chat_messages: [],
}).write();

// Helper: sqlite-style prepare().get() / .all() / .run() interface
// so we don't have to rewrite server.js
const dbHelper = {
  prepare(sql) {
    return new Statement(db, sql);
  }
};

class Statement {
  constructor(db, sql) {
    this.db = db;
    this.sql = sql.trim();
  }

  // Parse table and operation from SQL
  _parse() {
    const s = this.sql;
    const upper = s.toUpperCase();

    if (upper.startsWith('INSERT INTO')) {
      const table = s.match(/INSERT INTO (\w+)/i)?.[1];
      return { op: 'insert', table };
    }
    if (upper.startsWith('SELECT COUNT(*)')) {
      const table = s.match(/FROM (\w+)/i)?.[1];
      return { op: 'count', table };
    }
    if (upper.startsWith('SELECT')) {
      const table = s.match(/FROM (\w+)/i)?.[1];
      const whereCol = s.match(/WHERE (\w+)\s*=\s*\?/i)?.[1];
      const andCol = s.match(/AND (\w+)\s*=\s*\?/i)?.[1];
      const orderMatch = s.match(/ORDER BY (\w+) (ASC|DESC)/i);
      const limitMatch = s.match(/LIMIT (\d+)/i);
      const cols = s.match(/SELECT (.+?) FROM/i)?.[1]?.trim();
      return { op: 'select', table, whereCol, andCol, orderMatch, limitMatch, cols };
    }
    if (upper.startsWith('DELETE FROM')) {
      const table = s.match(/DELETE FROM (\w+)/i)?.[1];
      const whereCol = s.match(/WHERE (\w+)\s*=\s*\?/i)?.[1];
      return { op: 'delete', table, whereCol };
    }
    return { op: 'unknown' };
  }

  _camel(col) {
    // Convert snake_case SQL column names to camelCase for lowdb object fields
    // but we store everything as snake_case keys anyway so just return as-is
    return col;
  }

  run(...args) {
    const { op, table } = this._parse();
    if (op === 'insert') {
      // Parse column names from SQL
      const colMatch = this.sql.match(/\(([^)]+)\)\s+VALUES/i)?.[1];
      const cols = colMatch ? colMatch.split(',').map(c => c.trim()) : [];
      const record = {};
      cols.forEach((col, i) => { record[col] = args[i] ?? null; });
      this.db.get(table).push(record).write();
    } else if (op === 'delete') {
      const { whereCol } = this._parse();
      const val = args[0];
      this.db.get(table).remove(r => r[whereCol] === val).write();
    }
    return this;
  }

  get(...args) {
    const { op, table, whereCol, andCol, cols } = this._parse();
    if (op === 'count') {
      const count = this.db.get(table).value().length;
      return { c: count };
    }
    if (op === 'select') {
      let chain = this.db.get(table);
      if (whereCol && args[0] !== undefined) {
        chain = chain.find(r => r[whereCol] === args[0] && (!andCol || r[andCol] === args[1]));
      } else {
        chain = chain.find(() => true);
      }
      return chain.value() || null;
    }
    return null;
  }

  all(...args) {
    const { op, table, whereCol, andCol, orderMatch, limitMatch } = this._parse();
    if (op === 'select') {
      let items = this.db.get(table).value();
      if (whereCol && args[0] !== undefined) {
        items = items.filter(r => r[whereCol] === args[0]);
        if (andCol && args[1] !== undefined) {
          items = items.filter(r => r[andCol] === args[1]);
        }
      }
      if (orderMatch) {
        const col = orderMatch[1];
        const dir = orderMatch[2].toUpperCase();
        items = [...items].sort((a, b) => {
          if (a[col] < b[col]) return dir === 'ASC' ? -1 : 1;
          if (a[col] > b[col]) return dir === 'ASC' ? 1 : -1;
          return 0;
        });
      }
      if (limitMatch) {
        items = items.slice(0, parseInt(limitMatch[1]));
      }
      return items;
    }
    return [];
  }
}

module.exports = dbHelper;
