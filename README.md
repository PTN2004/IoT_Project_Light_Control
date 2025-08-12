# 💡 Light Controlling Using Hand Gestures

A Python-based project that allows users to control lights using predefined hand gestures.  
The system combines Google's **MediaPipe Gesture Recognizer** with a custom **MLP (Multi-Layer Perceptron)** deep learning model to detect gestures and execute corresponding light control commands.  
It supports both **simulation mode** and **real hardware control** via Modbus RTU RS485 relay modules.

---

## 📜 Features
- 🎯 **Real-time Hand Gesture Recognition** using MediaPipe.
- 🧠 **MLP Model Training** for classifying custom-defined gestures.
- 💡 **Light Control** in simulation or on real hardware (3 lights supported).
- ⚙ **Customizable Gestures** via YAML configuration (`hand_gesture.yaml`).
- 📂 **Automated Data Collection** using `generate_landmark_data.py`.
- 🔌 **Hardware Support** with 4-relay Modbus RTU RS485 modules.

---

## 🛠 Project Structure
```
├── generate_landmark_data.py # Collects hand gesture data
├── hand_gesture_recognition.py # Runs real-time detection & simulation/hardware control
├── hand_gesture.yaml # Gesture-to-action configuration
├── requirements.txt # Python dependencies
└── data2/ # Collected CSV landmark data
```


---

## 🚀 Installation

### 1️⃣ Create Python Environment
```bash
conda create -n gesture_env python=3.10
conda activate gesture_env
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

