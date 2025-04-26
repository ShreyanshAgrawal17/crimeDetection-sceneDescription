🕵️‍♂️ Crime Surveillance System using BiLSTM + Self-Attention and MediaPipe Holistic
=====================================================================================

This project showcases a smart Crime Surveillance system capable of detecting and describing **Shoplifting** and **Vandalism** activities from surveillance videos. We leverage:

*   **BiLSTM with Self-Attention** for crime classification.
    
*   **MediaPipe Holistic** (Pose + Hands only, no Face landmarks) for feature extraction.
    
*   **InternVL 2.5** model to generate detailed video descriptions.
    

🚀 Project Overview
-------------------

*   **Input**: A short surveillance video (mp4).
    
*   **Feature Extraction**: MediaPipe Holistic (225 features per frame: pose + hands only).
    
*   **Classification**: A BiLSTM model with self-attention trained to classify Shoplifting or Vandalism.
    
*   **Description Generation**: InternVL model creates a textual description based on frames.
    

📂 Folder Structure
-------------------
```bash
├── Crime_Surviellance.ipynb   # Main Colab notebook (contains all code and setup)
├── crime_detection_model_2class.h5  # Pre-trained BiLSTM model weights (upload separately)
└── README.md  # Project Documentation
```


⚙️ How To Run
-------------

### 1️⃣ Open "Crime_Surviellance.ipynb" in Google Colab 

### 2️⃣ Upload model weights

### 3️⃣ Set your HuggingFace token
```bash
os.environ["HF_TOKEN"] = "YOUR_HUGGINGFACE_TOKEN"
```
Instead of "YOUR_HUGGINGFACE_TOKEN", insert your Hugging Face API token.

### 4️⃣ Provide your own video path
```bash
analyze_video("/content/content/test_video_1.mp4")
```
### 5️⃣ Simply Run All Cells — everything including pip installs and imports is handled inside the notebook!



🧠 Model Architecture
---------------------

*   **BiLSTM + Self-Attention**:
    
    *   First Bidirectional LSTM layer: 128 units.
        
    *   Second LSTM layer: 64 units.
        
    *   Self-attention mechanism on LSTM outputs.
        
    *   Dense layers with **Swish** activation for final classification.
        
*   **Feature Extraction**:
    
    *   33 pose landmarks × (x,y,z) + 21 left-hand landmarks × (x,y,z) + 21 right-hand landmarks × (x,y,z).
        
    *   Total: **225 features** per frame.
        
    *   Face landmarks are **excluded** intentionally.
        

## 🏷️ Classes

| Class      | Description                     |
|------------|----------------------------------|
| Shoplifting | Stealing an item from a store    |
| Vandalism   | Damaging or defacing property    |


🔥 Key Highlights
-----------------

*   ✅ Clean feature extraction using only Pose and Hands (no face privacy concerns).
    
*   ✅ Lightweight BiLSTM model with Attention for high accuracy.
    
*   ✅ HuggingFace InternVL model integrated for smart scene description.
    
*   ✅ Full pipeline: video ➔ pose+hand ➔ classify ➔ describe.
    

⚡ Important Notes
-----------------

*   The .h5 model file must be **uploaded manually** to your Colab environment.
    
*   Make sure to **replace** YOUR\_HUGGINGFACE\_TOKEN with your real token.
    
*   Update the video path to analyze your own videos.
    

📸 Example
----------

🔹Predicted Crime: Shoplifting

📝 **Description:** In the video, a person is seen taking an item from the counter and putting it in their bag. This action is suspicious and could be considered theft.

<br>

![aunty_shoplifting](https://github.com/user-attachments/assets/3b6cf7ce-24ca-4028-93a8-c7d5980519c4)


🤝 Contributors
---------------

*   **[Shreyansh Agrawal](https://github.com/ShreyanshAgrawal17)**
    
*   **[Harshil Vijay](https://github.com/HarshilVj)**
    

🎯 Happy Crime Detection!
=========================
