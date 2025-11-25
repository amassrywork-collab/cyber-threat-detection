# 🚀 Cyber Threat Detection using Deep Learning

This project implements a simple yet effective **PyTorch-based neural network** to detect potential **cybersecurity threats** from structured system log data.  
The model classifies each record as:

- **0 → Normal behavior**  
- **1 → Cyber threat**

The goal of the project is to practice deep learning fundamentals in a realistic cybersecurity setting.

---

## 📌 Features

- Built with **PyTorch**
- Fully connected neural network (MLP)
- Uses **CrossEntropyLoss** for classification
- Tracks validation accuracy each epoch
- Clean and modular structure (model + training scripts)
- Easy to run and extend

---

## 📂 Project Structure
cyber-threat-detection/
│
├── model.py # Neural network architecture
├── train.py # Training + validation loop
├── requirements.txt # Dependencies
├── data/ # Dataset (CSV files go here)
│ ├── labelled_train.csv
│ ├── labelled_validation.csv
│ └── labelled_test.csv
└── README.md # Project documentation
---

## 🧠 Model Architecture

A simple MLP classifier:

- Input layer → based on dataset columns  
- Hidden layer 1 → 32 neurons + ReLU  
- Hidden layer 2 → 16 neurons + ReLU  
- Output layer → 2 neurons (binary classification)

---

## 🏃 How to Run the Project

### 1. Install dependencies:
pip install -r requirements.txt



### 2. Run the training script:
python train.py


### 3. The script will:

- Train the model for **10 epochs**
- Compute **validation accuracy**
- Print accuracy every epoch

---

## 📊 Dataset

Place the dataset files inside the `data/` folder:

data/
├── labelled_train.csv
├── labelled_validation.csv
└── labelled_test.csv


These files contain preprocessed cybersecurity event logs.  
Each row represents a system process activity with the target field:

- `sus_label` = 0 or 1

---

## 🛡️ Use Case

This project demonstrates how deep learning can help detect:

- Anomalous system behavior  
- Suspicious processes  
- Potential cyber attacks  
- Malicious activity logs  

It’s ideal as a starting point for building more advanced **intrusion detection systems**.

---

## 🤝 Author

Ahmed Monir Almassri , a deep learning student practicing model training, cybersecurity data handling, and PyTorch fundamentals.

---

## ⭐ Future Improvements

- Add dropout for regularization  
- Add batch normalization  
- Add test set evaluation  
- Use a deeper model  
- Save metrics & plots  

---

⭐ If you like this project, feel free to ⭐ star the repository!
