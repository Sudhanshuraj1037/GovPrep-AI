<div align="center">

# 🚀 GovPrep AI v3.0

### AI-Powered Government Exam Preparation Platform

Generate intelligent mock tests, upload PDFs, translate study material, and prepare smarter with AI.

<br>

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-Visit_Website-success?style=for-the-badge)](https://govprep-ai.netlify.app/)
![Version](https://img.shields.io/badge/Version-v3.0-blue?style=for-the-badge)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react)
![Node.js](https://img.shields.io/badge/Node.js-20-339933?style=for-the-badge&logo=node.js)
![Express.js](https://img.shields.io/badge/Express.js-Framework-000000?style=for-the-badge&logo=express)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)
![PWA](https://img.shields.io/badge/PWA-Supported-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production-success?style=for-the-badge)

...

### 🌐 Live Website

### https://govprep-ai.netlify.app/

---

**Built with ❤️ using React, Node.js, Express, Groq AI & SQL.js**

</div>

---

# 📖 About The Project

GovPrep AI is a modern AI-powered government exam preparation platform that helps students generate customized mock tests, practice quizzes, upload study PDFs, translate learning material, and interact with an AI study assistant.

The platform combines modern web technologies with Generative AI to provide an intelligent learning experience for aspirants preparing for competitive examinations.

Unlike traditional quiz generators, GovPrep AI allows users to create personalized tests based on subjects, difficulty levels, and question counts while maintaining a clean, responsive, and installable user experience.

---

# ✨ Key Features

| Feature | Status |
|----------|:------:|
| 🔐 Secure Login & Signup | ✅ |
| 🔑 JWT Authentication | ✅ |
| ⏳ 30-Day Login Session | ✅ |
| 🤖 AI Question Generation | ✅ |
| 💬 AI Study Assistant | ✅ |
| 📄 PDF Upload | ✅ |
| 🌐 English → Hindi PDF Translation | ✅ |
| 📝 Custom Practice Test Generator | ✅ |
| 📊 Dashboard | ✅ |
| 📁 My Tests History | ✅ |
| 📈 Result Analysis | ✅ |
| 🌙 Dark / Light Theme | ✅ |
| 📱 Fully Responsive UI | ✅ |
| 📲 Progressive Web App (PWA) | ✅ |
| ☁️ Cloud Deployment | ✅ |

---

# 🚀 Why GovPrep AI?

GovPrep AI is designed to simplify the preparation process for competitive government examinations by combining Artificial Intelligence with modern web technologies.

Users can:

- Generate unlimited AI-powered practice tests.
- Upload study PDFs for better learning.
- Translate English PDFs into Hindi.
- Practice topic-wise examinations.
- Review previous test history.
- Get instant AI assistance.
- Access the platform from desktop, tablet, or mobile.
- Install the application like a native app using PWA support.

---

# 📱 Progressive Web App (PWA)

GovPrep AI can be installed directly on your device without downloading it from any app store.

### Supported Platforms

- ✅ Android
- ✅ Windows
- ✅ macOS
- ✅ Linux
- ✅ ChromeOS

### PWA Features

- Install as App
- Offline Ready
- Fast Loading
- Responsive Layout
- App Icons
- Splash Screen
- Native App Experience

---

# 📸 Application Preview

# 📸 Screenshots

## 🔐 Authentication

<p align="center">
<img width="1920" height="1020" alt="Screenshot 2026-07-04 093005" src="https://github.com/user-attachments/assets/6334f442-b437-43da-a5f5-1db8517cca46" />
  
<img width="1920" height="1020" alt="Screenshot 2026-07-04 093023" src="https://github.com/user-attachments/assets/317298be-53b3-4f8d-8656-bcc58834b1e5" />
</p>

---

## 📊 Dashboard

<p align="center">
  <img width="1920" height="1020" alt="Screenshot 2026-07-04 093038" src="https://github.com/user-attachments/assets/f409e9d8-fafa-4f6d-b7f9-d6fe94d48916" />

</p>

<p align="center">
  <img width="1920" height="1020" alt="Screenshot 2026-07-04 093048" src="https://github.com/user-attachments/assets/2f81979a-0fb4-41d5-9ac0-4a308ca60e8e" />

</p>

---


# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │      Browser        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ React Frontend (UI) │
                    └──────────┬──────────┘
                               │ REST API
                               ▼
                  ┌─────────────────────────┐
                  │ Express.js Backend API  │
                  └──────────┬──────────────┘
                             │
          ┌──────────────────┴──────────────────┐
          ▼                                     ▼
 ┌───────────────────┐               ┌─────────────────┐
 │     Groq API      │               │ LowDB / SQL.js  │
 └───────────────────┘               └─────────────────┘
```

---

# ⚡ Tech Stack

## Frontend

- React
- JavaScript
- CSS Modules
- Responsive Design
- Progressive Web App (PWA)

---

## Backend

- Node.js
- Express.js
- REST API

---

## Database

- LowDB
- SQL.js

---

## Artificial Intelligence

- Groq API
- Large Language Models (LLM)

---

## Authentication

- JWT Authentication
- bcrypt Password Hashing

---

## Deployment

| Service | Platform |
|----------|----------|
| Frontend | Netlify |
| Backend | Render |

---

# 📂 Project Structure

```text
GovPrep-AI
│
├── backend
│   ├── server.js
│   ├── auth.js
│   ├── database.js
│   ├── package.json
│   └── ...
│
├── frontend
│   ├── public
│   ├── src
│   │   ├── components
│   │   ├── pages
│   │   └── ...
│   ├── package.json
│   └── ...
│
├── .gitignore
├── LICENSE
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/Sudhanshuraj1037/GovPrep-AI.git
```

Move into project directory

```bash
cd GovPrep-AI
```

---

## Backend Setup

```bash
cd backend

npm install

npm start
```

---

## Frontend Setup

```bash
cd frontend

npm install

npm start
```

---

# 🔑 Environment Variables

Create a `.env` file inside the backend folder.

```env
GROQ_API_KEY=YOUR_GROQ_API_KEY
JWT_SECRET=YOUR_SECRET_KEY
PORT=5000
```

---

# 🌍 Live Deployment

| Service | URL |
|----------|-----|
| 🌐 Frontend | https://govprep-ai.netlify.app/ |
| 🚀 Backend | https://govprep-backend-pkn9.onrender.com |

---

# 🔌 API Overview

The backend exposes RESTful APIs that power the frontend application.

| Method | Endpoint | Description |
|---------|----------|-------------|
| POST | `/api/signup` | Register a new user |
| POST | `/api/login` | User authentication |
| GET | `/api/profile` | Get user profile |
| POST | `/api/generate-test` | Generate AI-powered test |
| POST | `/api/upload-pdf` | Upload PDF |
| POST | `/api/translate-pdf` | Translate English PDF to Hindi |
| GET | `/api/tests` | Fetch previous tests |
| POST | `/api/assistant` | AI Study Assistant |

---

# 📈 Performance & Optimizations

GovPrep AI has been optimized to provide a smooth and fast user experience.

### 🚀 Frontend

- Responsive UI
- CSS Modules
- Optimized React Components
- Fast Routing
- Lazy Loading Ready
- Mobile Friendly

### ⚡ Backend

- Express REST API
- JWT Authentication
- Secure Password Hashing
- Optimized API Structure
- AI Integration using Groq

### 📲 Progressive Web App

- Installable on Desktop & Mobile
- Offline Support
- Splash Screen
- App Icons
- Native App Experience

---

# 🔒 Security Features

Security is a core part of GovPrep AI.

Implemented features:

- JWT Authentication
- Password Hashing using bcrypt
- Protected API Routes
- Session Management
- Environment Variables
- Secure Authentication Flow

---

# 📱 Responsive Design

GovPrep AI is fully optimized for multiple screen sizes.

| Device | Supported |
|---------|:---------:|
| 📱 Mobile | ✅ |
| 📲 Tablet | ✅ |
| 💻 Laptop | ✅ |
| 🖥 Desktop | ✅ |

---

# 📦 Dependencies

## Frontend

- React
- React DOM
- React Router
- CSS Modules

## Backend

- Express
- bcrypt
- jsonwebtoken
- multer
- cors
- dotenv
- Groq SDK
- LowDB
- SQL.js

---

# 🚀 Deployment

GovPrep AI is deployed using modern cloud services.

### Frontend

**Netlify**

https://govprep-ai.netlify.app/

### Backend

**Render**

https://govprep-backend-pkn9.onrender.com

---

# 🛣️ Roadmap

## ✅ Completed

- [x] Secure Authentication
- [x] JWT Login
- [x] AI Test Generation
- [x] AI Assistant
- [x] PDF Upload
- [x] PDF Translation
- [x] Dashboard
- [x] Test History
- [x] Result Page
- [x] Responsive UI
- [x] Progressive Web App
- [x] Dark / Light Theme
- [x] Live Deployment

---

## 🚧 Upcoming Features

- [ ] Email Verification
- [ ] Forgot Password
- [ ] User Profile
- [ ] AI Notes Generator
- [ ] AI Study Planner
- [ ] Leaderboard
- [ ] Performance Analytics
- [ ] Cloud Database
- [ ] Push Notifications
- [ ] Admin Dashboard
- [ ] Multi-language Support

---

# 💡 Future Scope

GovPrep AI is designed to evolve into a complete AI-powered learning platform.

Future goals include:

- Personalized Study Plans
- AI Performance Tracking
- Adaptive Question Generation
- Mock Test Rankings
- Smart Recommendations
- Voice-enabled AI Assistant
- Cloud Synchronization
- Mobile Application

---

# 🤝 Contributing

Contributions are welcome!

If you'd like to improve GovPrep AI:

1. Fork the repository
2. Create your feature branch

```bash
git checkout -b feature/amazing-feature
```

3. Commit your changes

```bash
git commit -m "feat: add amazing feature"
```

4. Push to GitHub

```bash
git push origin feature/amazing-feature
```

5. Open a Pull Request

---

# 👨‍💻 Developer

<div align="center">

## Sudhanshu Raj

**B.Tech Computer Science & Engineering**

Passionate about:

🤖 Artificial Intelligence

🧠 Machine Learning

💻 Full Stack Development

☁️ DevOps

🚀 Building Real-World AI Applications

### GitHub

https://github.com/Sudhanshuraj1037

</div>

---

# ⭐ Show Your Support

If you found this project helpful:

⭐ Star this repository

🍴 Fork the repository

🛠 Contribute to the project

📢 Share it with others

---

# 📄 License

This project is licensed under the **MIT License**.

See the LICENSE file for more details.

---

# 🙏 Acknowledgements

Special thanks to the amazing open-source community and the technologies that made this project possible.

- React
- Node.js
- Express.js
- Groq
- SQL.js
- LowDB
- Netlify
- Render

---

<div align="center">

# 🌟 Thank You for Visiting!

### If you like this project, don't forget to leave a ⭐

### 🚀 Happy Learning with GovPrep AI 🚀

Made with ❤️ by **Sudhanshu Raj**

</div>
