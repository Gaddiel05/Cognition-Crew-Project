# ASL Sign Language Recognition System - Technical Summary

## **Project Overview**
An AI-powered sign language recognition system that detects hand gestures via webcam, converts them to text, and speaks the text aloud using text-to-speech technology.

---

## **System Architecture**

### **1. Data Collection Pipeline**
- **Script**: `Collect Images.py`
- **Function**: Captures training images using webcam
- **Process**: 
  - 300 images per gesture class
  - 39 total classes (A-Z, 0-9, del, space, nothing)
  - Interactive capture (press 'r' to start each class)
  - Organized in folders by class label

### **2. Dataset Creation**
- **Script**: `Create the dataset.py`
- **Technology**: MediaPipe Hands
- **Process**:
  - Detects 21 hand landmarks per image
  - Extracts x,y coordinates for each landmark
  - Normalizes coordinates (position-independent)
  - Creates 42 features per sample (21 landmarks × 2 coordinates)
  - Saves to `data.pickle` file

### **3. Model Training**
- **Script**: `Train the model.py`
- **Algorithm**: Random Forest Classifier (or similar ML model)
- **Input**: 42 normalized hand landmark features
- **Output**: Trained model saved as `model.p`
- **Dataset**: ~11,700 samples (300 images × 39 classes)

### **4. Real-Time Detection & Prediction**
- **Scripts**: 
  - `Make predictions & detect hand gestures.py` (basic detection)
  - `Convert hand gestures to text and speech.py` (full application)

---

## **Core Technologies**

| Technology | Purpose |
|------------|---------|
| **OpenCV** | Video capture, image processing, display |
| **MediaPipe** | Hand landmark detection (21 points per hand) |
| **scikit-learn** | Machine learning model (Random Forest) |
| **NumPy** | Numerical operations, array handling |
| **Google Text-to-Speech (gTTS)** | Convert captured text to speech |
| **playsound** | Audio playback |

---

## **How It Works (User Flow)**

### **For End Users (Non-Technical)**
1. **Launch Application** - Double-click the program
2. **Show Hand Gesture** - Make a sign language gesture in front of webcam
3. **See Live Prediction** - Green text above hand shows detected gesture + countdown
4. **Hold for 2 Seconds** - System captures the letter/action
5. **Build Words** - Continue making gestures to spell words
6. **Special Gestures**:
   - **Space gesture**: Add space between words
   - **Del gesture**: Delete last character
7. **Speak Text** - Press 's' key to hear captured text
8. **Clear/Start Over** - Press 'c' key
9. **Exit** - Press 'x' key

---

## **Technical Flow Diagram**

```
Webcam Input → MediaPipe Detection → Extract 21 Landmarks → 
Normalize Coordinates (42 features) → ML Model Prediction → 
Display Gesture (with countdown) → Capture After 2s → 
Build Text → Text-to-Speech Output
```

---

## **System Requirements**

### **Hardware**
- Webcam (built-in or USB)
- 4GB RAM minimum
- Windows/Mac/Linux OS

### **Software Dependencies**
```
Python 3.9+
opencv-python==4.8.1.78
mediapipe
scikit-learn
numpy==1.26.4
gtts
playsound
speech_recognition
```

---

## **Key Features**

### **1. Real-Time Visual Feedback**
- Live gesture prediction above hand
- Countdown timer (2s, 1s, 0s)
- Hand skeleton overlay
- Captured text display

### **2. Intelligent Gesture Recognition**
- 39 distinct gestures (A-Z, 0-9, special commands)
- Position-independent detection
- Handles various hand sizes and orientations

### **3. Text-to-Speech**
- Nigerian English accent
- Clear audio output
- On-demand playback

### **4. Error Correction**
- Delete last character (del gesture)
- Preview before capture (2-second hold)
- Clear entire text ('c' key)

---

## **Dataset Specifications**

- **Total Classes**: 39
  - Letters: A-Z (26)
  - Numbers: 0-9 (10)
  - Commands: del, space, nothing (3)
- **Images per Class**: 300
- **Total Images**: ~11,700
- **Feature Vector**: 42 dimensions (normalized x,y for 21 landmarks)
- **Storage**: `./data/asl_data_train/` folder structure

---

## **Use Cases**

1. **Accessibility Tool** - Help hearing/speech-impaired communicate
2. **Education** - Learn sign language with instant feedback
3. **Communication Bridge** - Translate sign language to spoken words
4. **Hands-Free Input** - Alternative text input method

---

## **Advantages**

✅ **No Special Hardware** - Just a standard webcam  
✅ **Real-Time Processing** - Instant gesture recognition  
✅ **Position Independent** - Works from various angles  
✅ **Visual Feedback** - User sees prediction before capture  
✅ **Expandable** - Can add more gestures/languages  
✅ **Offline Capable** - No internet needed (except TTS)

---

## **Limitations & Future Improvements**

**Current Limitations**:
- Single hand detection
- Static gestures only (no motion-based signs)
- Requires 2-second hold per character
- Needs good lighting

**Potential Enhancements**:
- Two-hand gesture support
- Motion-based signs (dynamic gestures)
- Faster capture time
- Word prediction/autocomplete
- Multiple sign language systems (ASL, BSL, etc.)
- Mobile app version

---

## **Performance Metrics**

- **Detection Speed**: ~30 FPS (real-time)
- **Landmark Detection**: 21 points per hand
- **Capture Delay**: 2 seconds (user configurable)
- **Model Accuracy**: Depends on training (typically >90% with good data)

---

## **Project Structure**

```
Project/
├── data/
│   └── asl_data_train/        # Training images organized by class
├── Collect Images.py           # Data collection tool
├── Create the dataset.py       # Feature extraction
├── Train the model.py          # Model training
├── model.p                     # Trained model file
├── data.pickle                 # Processed dataset
├── Convert hand gestures...py  # Main application
└── Make predictions...py       # Basic detection demo
```

---

## **Installation & Setup**

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Africagen Hack FinalProduct"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   cd "Sign Language Detection & Conversion to Text & Speech"
   python "Convert hand gestures to text and speech.py"
   ```

---

## **Presentation Talking Points**

1. **Problem**: Communication barrier for hearing/speech-impaired individuals
2. **Solution**: AI-powered real-time sign language translator
3. **Innovation**: Visual feedback with countdown prevents errors
4. **Technology**: Combines computer vision (MediaPipe) + ML (Random Forest)
5. **Impact**: Makes communication accessible with just a webcam
6. **Demo**: Live demonstration of spelling words and text-to-speech

---

## **Contributing**
Contributions are welcome! Please feel free to submit a Pull Request.

## **License**
This project is licensed under the MIT License.

## **Contact**
For questions or feedback, please open an issue in the repository.
