# 🏋️ Stay Hard Fitness - AI-Powered Gym Workout App

A modern, production-grade Django web application that combines state-of-the-art Generative AI with real-time Computer Vision to deliver personalized training protocols, high-fidelity biomechanical pose checking, and complete nutrition logging.

---

## 🌟 Key Architecture & Upgrades

### 1. 🤖 Multi-Engine Cloud & Local AI Coach (`OS Architect`)
The application features **OS Architect**, a strict, clinical Senior Fitness & Nutrition Coach. It leverages a modern, highly resilient multi-engine routing system:
- **Primary Engine**: Cloud-based `gemini-1.5-flash` for ultra-low latency (< 1s) and sub-second nutrient analysis.
- **Secondary Engine**: Local Ollama integration targeting `mistral:7b` or `llama3`.
- **Offline Engine**: Fallback to high-fidelity, procedurally generated workout/macros structures when offline or without API keys.
- **Dynamic Context Rendering**: Displays active backend status (e.g. `Cloud Engine Active (Gemini 1.5 Flash)` vs `Local Fallback Active`) in real-time.

### 2. ⚡ Real-Time HUD Statistics Synchronization
- **Thread-Safe Coordinator**: Implements a global telemetry state registry (`WORKOUT_STATS`) protected by thread locks (`threading.Lock`) in Django.
- **AJAX Telemetry Poller**: The frontend HUD queries `/api/workout-stats/` every 500ms using asynchronous AJAX fetch requests to update DOM rep counters and stage trackers without page reloads.
- **Headless Simulated Mode**: Automatically falls back to a simulated workout routine (pushing reps and form feedback tips to the HUD every 3 seconds) when running in headless environments or on machines without a physical webcam.

### 3. 📊 Persistent Posture History Database
- **SQLite Analytical Logging**: Integrates a custom `PostureAnalysis` database model to persist workout history.
- **Interactive Completion Workflow**: Users can click "Complete Session" from their live HUD toolbar, which saves metrics (reps, stage, biomechanics corrective tips, calculated posture alignment score) to the database and redirects to their historical workout analytics chart.

### 4. 🎨 Premium Glassmorphism UI/UX
- Sleek dark-mode color scheme with glowing HSL green/gold accents.
- Immersive frosted-glass panel overlays (`.glass-panel`) sitting directly over video stream feeds.
- Smooth CSS transition animations, micro-interaction buttons, and professional visual placeholders.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Django 4.2+
- OpenCV & MediaPipe
- Web camera (optional, fallback demo mode available)

### Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/45nivas/Stay_hard_fitness.git
   cd Stay_hard_fitness
   ```

2. **Establish Virtual Environment**
   ```bash
   python -m venv venv
   # Activate on Windows
   venv\Scripts\activate
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. **Install Core Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Apply Telemetry Database Migrations**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Run System Architecture Checks**
   ```bash
   python manage.py check
   ```

6. **Fire Up the Web Server**
   ```bash
   python manage.py runserver
   ```
   Open your browser and navigate to `http://127.0.0.1:8000`.

---

## 🔧 Environment Configuration

Create a `.env` file in the root workspace directory:
```env
# Core Django Configs
SECRET_KEY=your-django-secret-key-here
DEBUG=True

# Cloud-First Intelligence (Optional but Recommended)
GEMINI_API_KEY=your-gemini-api-key-here

# Local Ollama Configuration (Optional Fallback)
OLLAMA_URL=http://localhost:11434
```

---

## 📊 Folder Structure

```
Stay_hard_fitness/
├── gym_project/           # Django project configurations & settings
├── workouts/              # Main application logic
│   ├── models.py          # Database models (UserProfile, PostureAnalysis, MealLogs)
│   ├── views.py           # Core views, Thread-Safe Registry, and REST API controllers
│   ├── urls.py            # Route mappings (live streams, AJAX APIs)
│   ├── rep_counter.py     # MediaPipe biomechanics algorithms
│   └── fitness_chatbot.py # Cloud-first Gemini & Ollama local router
├── templates/             # Premium HTML templates (workout pages, chatbot interface)
├── static/                # Premium custom CSS stylesheets and assets
├── manage.py              # Django project manager
└── requirements.txt       # Python packages list
```

---

## 🙏 Technical Stack & Attributions

- **Google Gemini API**: Dynamic nutrient analysis and transformative planning.
- **Google MediaPipe**: Fast and precise pose landmark calculations.
- **OpenCV**: Resilient video capture and stream processing.
- **Django**: The ultra-secure, rapid-development backend.
