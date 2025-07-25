# 🏋️ Stay Hard Fitness - AI-Powered Gym Workout App

A modern Django web application that combines artificial intelligence with computer vision to provide personalized fitness training, pose correction, and comprehensive workout tracking.

## 🌟 Features

### 🤖 AI Fitness Trainer
- **Smart Workout Planning**: Get personalized workout routines based on your profile
- **Nutrition Guidance**: AI-powered meal recommendations and calorie tracking  
- **Real-time Chat**: Interactive fitness coaching with Ollama LLM integration
- **Progress Tracking**: Monitor your fitness journey with detailed analytics

### 📸 Pose Correction System
- **Real-time Analysis**: Live camera feed with MediaPipe pose detection
- **Exercise Form Checking**: Automatic posture correction for squats, push-ups, and more
- **Visual Feedback**: Instant overlay guidance on your workout form
- **Rep Counting**: Automated counting with pose-based detection

### 💪 One Rep Max Calculator
- **Strength Testing**: Calculate your maximum lift potential
- **Multiple Formulas**: Brzycki, Epley, and other proven calculation methods
- **Progress Tracking**: Monitor strength gains over time
- **Exercise Library**: Support for major compound movements

### 🎙️ Voice Calorie Tracker
- **Speech Recognition**: Log meals using voice commands
- **Smart Parsing**: Automatically extract food items and quantities
- **Nutrition Database**: Comprehensive calorie and macro information
- **Daily Summaries**: Track your nutritional intake with visual reports

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Django 4.2+
- OpenCV
- MediaPipe
- Web camera for pose detection

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/45nivas/Stay_hard_fitness.git
cd Stay_hard_fitness
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac  
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Database setup**
```bash
python manage.py makemigrations
python manage.py migrate
```

5. **Create superuser (optional)**
```bash
python manage.py createsuperuser
```

6. **Run the application**
```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000` to start your fitness journey!

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:
```env
SECRET_KEY=your-secret-key-here
DEBUG=True
OLLAMA_URL=http://localhost:11434
```

### Ollama LLM Setup (Optional)
For AI trainer features, install Ollama:
1. Download from [ollama.ai](https://ollama.ai)
2. Install a model: `ollama pull llama3.2`
3. Start the server: `ollama serve`

## 📱 How to Use

### Getting Started
1. **Sign Up**: Create your account with basic information
2. **Profile Setup**: Complete your fitness profile (height, weight, goals, etc.)
3. **Choose Your Tool**: Select from AI Trainer, Pose Correction, 1RM Calculator, or Calorie Tracker

### AI Trainer Workflow
1. Chat with the AI about your fitness goals
2. Receive personalized workout recommendations
3. Get nutrition advice based on your profile
4. Track conversations for future reference

### Pose Correction Usage
1. Allow camera access when prompted
2. Position yourself in frame for the selected exercise
3. Follow the real-time pose guidance
4. View your rep count and form feedback

### Calorie Tracking
1. Use voice commands: "I ate 2 slices of pizza"
2. Review parsed food items and calories
3. Check daily nutrition summaries
4. Monitor macro breakdowns

## 🛠️ Technology Stack

- **Backend**: Django 4.2, Python 3.8+
- **Computer Vision**: OpenCV, MediaPipe
- **AI Integration**: Ollama (Local LLM)
- **Database**: SQLite (development), PostgreSQL ready
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Styling**: Custom CSS with responsive design

## 📊 Project Structure

```
Stay_hard_fitness/
├── gym_project/           # Django project settings
├── workouts/             # Main application
│   ├── models.py        # Database models
│   ├── views.py         # Application logic  
│   ├── urls.py          # URL routing
│   ├── rep_counter.py   # Pose detection logic
│   └── fitness_chatbot.py # AI trainer integration
├── templates/           # HTML templates
├── static/             # CSS, JavaScript, images
├── manage.py           # Django management
└── requirements.txt    # Dependencies
```

## 🎯 Key Features Deep Dive

### Pose Detection System
- **MediaPipe Integration**: Leverages Google's MediaPipe for accurate pose landmarks
- **Real-time Processing**: 30+ FPS pose detection with minimal latency
- **Exercise Specific**: Tailored algorithms for different workout types
- **Form Analysis**: Angle calculations for proper movement assessment

### AI Trainer Intelligence
- **Context Awareness**: Remembers user profile and conversation history
- **Personalized Responses**: Tailored advice based on individual metrics
- **Fallback System**: Works offline with pre-built knowledge base
- **Continuous Learning**: Adapts recommendations based on user progress

### Nutrition Tracking
- **Voice Processing**: Natural language parsing for food logging
- **Database Integration**: Comprehensive nutrition information
- **Smart Suggestions**: AI-powered meal recommendations
- **Visual Reports**: Charts and graphs for progress visualization

## 🚧 Development Roadmap

- [ ] Mobile app development (React Native)
- [ ] Advanced exercise library expansion
- [ ] Social features and workout sharing
- [ ] Wearable device integration
- [ ] Nutrition barcode scanning
- [ ] Video workout tutorials

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe Team** for the pose detection framework
- **Ollama Community** for local LLM capabilities
- **OpenCV Contributors** for computer vision tools
- **Django Community** for the robust web framework

## 📞 Support

Having issues? We're here to help!

- 📧 Email: support@stayhardness.com
- 💬 Discord: [Join our community](https://discord.gg/stayhardness)
- 🐛 Issues: [GitHub Issues](https://github.com/45nivas/Stay_hard_fitness/issues)

---

**Stay Hard Fitness** - Where AI meets iron. Transform your workout experience with intelligent training technology.

*Built with ❤️ for the fitness community*
