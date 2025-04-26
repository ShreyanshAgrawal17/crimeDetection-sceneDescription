🕵️‍♂️ Crime Surveillance System using BiLSTM + Self-Attention and MediaPipe Holistic
=====================================================================================

* * * * *

This project presents a smart **Crime Surveillance System** capable of detecting and describing **Shoplifting** and **Vandalism** activities from surveillance videos.

We leverage:

-   **BiLSTM with Self-Attention** for crime classification.

-   **MediaPipe Holistic** (Pose + Hands only) for video feature extraction.

-   **InternVL 2.5** (via HuggingFace) for generating detailed video scene descriptions.

> 🛡️ **Dataset Source**: UCF-Crime Dataset [Download Here (Dropbox)](https://www.dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/AOjRokSTaiKxXmgUyqdcI6k?rlkey=5bg7mxxbq46t7aujfch46dlvz&e=3&dl=0)

* * * * *

🚀 Project Overview
-------------------

-   **Input**: Short surveillance videos (mp4).

-   **Feature Extraction**: MediaPipe Holistic (only Pose + Hands, excluding Face landmarks).

-   **Classification**: BiLSTM model with self-attention classifies between Shoplifting and Vandalism.

-   **Scene Description**: InternVL model generates a textual description of the activity.

* * * * *

📂 Folder Structure
-------------------

```

├── Crime_Surviellance.ipynb   # Main Colab Notebook (integrated pipeline: classification + description)
├── extract_features.py        # Script to extract MediaPipe features and save as NumPy arrays
├── preprocess_data.py         # Script to load features and labels for training
├── build_model.py             # Defines BiLSTM + Self-Attention model architecture
├── train_model.py             # Trains the model on extracted features
├── evaluate.py                # Evaluates model on test set
├── crime_detection_model_2class.h5  # Pre-trained model weights (upload separately)
├── preprocessed_features/     # Folder containing NumPy arrays for faster training
└── README.md                  # Project Documentation (you are here!)`

```
* * * * *

⚙️ How To Run
-------------

### 1️⃣ For Inference (Google Colab)

-   Open `Crime_Surviellance.ipynb` in Google Colab.

-   Upload the pre-trained `.h5` model file.

-   Set your Hugging Face token:

```
os.environ["HF_TOKEN"] = "YOUR_HUGGINGFACE_TOKEN"`
```

-   Provide your video path:

```
`analyze_video("/content/test_video_1.mp4")`
```

* * * * *

### 2️⃣ For Training (Local or Colab)

-   Run the scripts step-by-step:

```

python extract_features.py
python preprocess_data.py
python build_model.py
python train_model.py
python evaluate.py

```

-   Alternatively, use the available preprocessed `.npy` feature files for quick training without extraction.

* * * * *

🧠 Model Architecture
---------------------

-   **BiLSTM + Self-Attention**:

    -   First BiLSTM Layer: 128 units

    -   Second LSTM Layer: 64 units

    -   Self-attention on sequence outputs

    -   Dense layers with **Swish** activation

-   **Feature Extraction**:

    -   33 pose landmarks × (x, y, z)

    -   21 left hand landmarks × (x, y, z)

    -   21 right hand landmarks × (x, y, z)

    -   **Total**: 225 features per frame

-   **Face landmarks** are excluded for privacy.

* * * * *

🏷️ Classes
-----------

| Class | Description |
| --- | --- |
| Shoplifting | Stealing an item from a store |
| Vandalism | Damaging or defacing property |

* * * * *

🔥 Key Highlights
-----------------

-   ✅ **Privacy-respecting** feature extraction (no face landmarks).

-   ✅ **Attention-enhanced BiLSTM** for accurate classification.

-   ✅ **Video scene description** generated using HuggingFace **InternVL**.

-   ✅ **Pre-extracted feature files** provided to skip slow extraction.

-   ✅ **Colab notebook** fully integrates classification + description pipeline.

* * * * *

📸 Example
----------

🔹 **Predicted Crime**: Shoplifting

📝 **Description**:\
In the video, a person is seen taking an item from the counter and putting it into their bag. This action is suspicious and could be considered theft.

<br>

![aunty_shoplifting](https://github.com/user-attachments/assets/3b6cf7ce-24ca-4028-93a8-c7d5980519c4)

* * * * *

🤝 Contributors
---------------

-   **[Shreyansh Agrawal](https://github.com/ShreyanshAgrawal17)**

-   **[Harshil Vijay](https://github.com/HarshilVj)**

* * * * *

🎯 Happy Crime Detection!
-------------------------

* * * * *

📢 Notes:
=========

-   Make sure the `.h5` model file is uploaded before running inference.

-   HuggingFace token is mandatory for InternVL 2.5 video description.

-   Always check the input video format (should be `.mp4`).

* * * * *

* * * * *

🛠️ Future Work
===============

-   Expand the number of crime classes (e.g., Assault, Robbery).

-   Fine-tune scene description model on crime-specific datasets.

-   Deploy the complete system as a real-time web application.
