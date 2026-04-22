# Satellite Infrastructure Damage Detection

## 🚀 Overview

This project implements an end-to-end machine learning pipeline including:

* Data preprocessing
* Model training
* Evaluation

The workflow is modular and organized for scalability and clarity.

---

## ⚙️ Features

* Clean data preparation pipeline
* Model training with configurable parameters
* Evaluation with performance metrics
* Reproducible workflow

---

🧠 Model Used

The model is trained on processed satellite imagery to classify and detect infrastructure damage.

Model type: **Siamese U-Net**
Input: Pre and post-disaster satellite images
Output: Damage classification / prediction

👉 Trained model can be downloaded from:
https://drive.google.com/file/d/1KtiqfH6rE5NCh097cJShbZ0hLaGFjYkB/view?usp=drive_link

---

## 🧠 Tech Stack

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib / Seaborn (if used)
* PyTorch
* tqdm

---

## 📥 Dataset

The dataset used in this project is too large to be hosted on GitHub.

👉 Download it from here:
**https://xview2.org/dataset**

After downloading, place it in the appropriate directory before running the scripts.

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

```bash
python src/prepare_data.py
```

### 3. Train model

```bash
python src/train.py
```

### 4. Evaluate model

```bash
python src/evaluate.py
```

---

🌐 Deployment (Streamlit)

Run the Streamlit app locally:

streamlit run app/projectapp.py

Then open in browser:

http://localhost:8501

🔹 Features of Web App
Upload satellite images
Run damage detection
View prediction results

---

## 📈 Results

* Model performance metrics are generated during evaluation.
* Modify parameters to experiment and improve results.
* Achieved Mean IoU Score of ~**0.43** 

---

## 🛠️ Customization

You can:

* Change model parameters in training script
* Use different datasets
* Extend evaluation metrics

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repository and submit a pull request.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 🙌 Acknowledgements

* Open-source libraries and tools used in this project
* Dataset providers


## Author Deatils

* Name: **Atharva Desai**
* E-Mail: **desaiatharva703@gmail.com**
