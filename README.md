# Pest Detection and Recommendation System

This project is a complete system for identifying agricultural pests from images and providing recommendations. It features a hybrid deep learning model, a Python (Flask) backend, and an Android mobile application built with Capacitor.


## 📋 Features

* **Pest Detection:** Identifies 12 different types of agricultural pests using a powerful hybrid deep learning model.
* **Hybrid Model:** Combines the strengths of **Vision Transformer (ViT)**, **ResNet50**, and a **Custom CNN** for 94% accuracy.
* **Backend API:** A Flask server that exposes the model via a simple `/predict` endpoint.
* **Mobile App:** An Android application (built from web technologies) that allows users to upload an image from their phone and get instant predictions.
* **Recommendations:** Provides practical advice and control measures for detected pests (via `recommendations.json`).

## 💻 Tech Stack

* **Backend:** Python, Flask, PyTorch, Torchvision
* **Frontend:** HTML/CSS/JavaScript
* **Mobile Wrapper:** Capacitor
* **App Build:** Android Studio
* **Dataset:** [Agricultural Pests Image Dataset (Kaggle)](https://www.kaggle.com/datasets/vencerlanz09/agricultural-pests-image-dataset)

## 🚀 Setup and Installation

### Prerequisites

* Python 3.8+
* Node.js and npm
* Android Studio
* A physical Android device or emulator

---

### 1. Backend Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Pest-Detection-App.git](https://github.com/YourUsername/Pest-Detection-App.git)
    cd Pest-Detection-App/backend
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate
    
    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Pre-trained Model:**
    * Download the model file (`pest_detection_hybrid_model_optimized.pth`) from this link:
    * **[➡️ Add Your Google Drive/Dropbox Download Link Here ⬅️]**
    * Place the downloaded `.pth` file inside the `backend/` folder.

---

### 2. Frontend (Android App) Setup

1.  **Install Node.js dependencies:**
    ```bash
    cd ../frontend
    npm install
    ```

2.  **Update the API Endpoint:**
    * Find your computer's **Local IP Address** (e.g., `192.168.1.10`).
    * Open `frontend/index.html` (or your main JavaScript file).
    * Find the `API_ENDPOINT` variable and change it to point to your backend:
        ```javascript
        // Example:
        const API_ENDPOINT = '[http://192.168.1.10:5000/predict](http://192.168.1.10:5000/predict)'; 
        ```

3.  **Build and Run in Android Studio:**
    ```bash
    # Add the Android platform
    npx cap add android
    
    # Open the project in Android Studio
    npx cap open android
    ```
    * Wait for Android Studio to open and build the project.
    * Connect your Android phone (with USB Debugging enabled) or start an emulator.
    * Click the **Run 'app'** button (▶️) in Android Studio to install the app.

---

### 3. Usage

1.  **Start the Backend Server:**
    * Make sure you are in the `backend/` directory with your virtual environment activated.
    * Run the Flask app:
        ```bash
        # This makes the server visible on your network
        python app.py
        ```
    * Allow access if your firewall prompts you.

2.  **Use the App:**
    * Ensure your phone is connected to the **same Wi-Fi network** as your computer.
    * Open the "Pest Detector" app on your phone.
    * Upload an image and tap "Detect" to get a prediction from your local server.

## 🧠 Model Training

The model was trained using the `train_model.py` script.

* **Dataset:** [Agricultural Pests Image Dataset](https://www.kaggle.com/datasets/vencerlanz09/agricultural-pests-image-dataset) from Kaggle.
* **Split:** The data was split into `train` (80%), `valid` (10%), and `test` (10%).
* **Architecture:** A hybrid model combining features from a pre-trained `vit_b_16`, `resnet50`, and a custom shallow CNN. The combined features are passed through a final classifier.

To train the model yourself, download the dataset, organize it into `train/`, `valid/`, and `test/` folders, update the `DATA_DIR` path in `train_model.py`, and run the script.

## 🧠 Glimpse of APP:-
<img width="300" height="400" alt="image" src="https://github.com/user-attachments/assets/54153b8f-0c9b-403a-8a9d-1825177d1748" />
<img width="300" height="400" alt="image" src="https://github.com/user-attachments/assets/301dee8c-7389-4e8c-b269-b86fa2c5a149" />
<img width="300" height="400" alt="image" src="https://github.com/user-attachments/assets/1093710e-17fe-478b-97d9-526b01b45b96" />
<img width="300" height="400" alt="image" src="https://github.com/user-attachments/assets/1770110a-b60f-4eab-9695-38dde5fbedbb" />
<img width="300" height="400" alt="image" src="https://github.com/user-attachments/assets/6df7e332-e84e-43b5-9136-2e19faa85299" />
<img width="300" height="400" alt="image" src="https://github.com/user-attachments/assets/d5c545b7-288b-4761-bccc-f76b1ab00f9e" />

<p align="center">
  <i>Copyright © 2025 Nupur-Bopche20. All rights reserved.</i>
</p>





