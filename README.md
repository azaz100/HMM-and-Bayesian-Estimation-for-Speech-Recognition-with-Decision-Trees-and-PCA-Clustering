# Dysarthric Speech Recognition Pipeline 🎙️🧠

## Project Overview
This project implements a complete Pattern Recognition pipeline designed to classify speech as either **Dysarthric** (pathological) or **Control** (typical). The system focuses on detecting acoustic biomarkers associated with motor speech disorders such as Cerebral Palsy (CP) and Amyotrophic Lateral Sclerosis (ALS).

The pipeline utilizes advanced signal processing for feature extraction, rigorous statistical hypothesis testing, and comparative machine learning modeling (Random Forest, GMM, HMM) to achieve robust classification.

## 🗂️ Datasets Used

This project analyzes two distinct datasets to validate model performance:

1.  **TORGO Database of Dysarthric Articulation**
    * **Source:** University of Toronto & Holland-Bloorview Kids Rehabilitation Hospital.
    * **Subjects:** Speakers with CP/ALS and matched controls.
    * **Nature:** High-quality recordings including non-words, short words, and restricted sentences.
    * **Goal:** Used for primary training and detailed acoustic analysis.

2.  **RAWDysPeech Dataset**
    * **Source:** Mendeley Data (Aggregated open-source dataset).
    * **Nature:** Preprocessed binary classification dataset (0: Control, 1: Dysarthric).
    * **Goal:** Used to test model generalization and robustness.

## ⚙️ Methodology & Pipeline

### 1. Feature Extraction (Acoustic Biomarkers)
We extract distinct features that map to specific physiological impairments in dysarthria:
* **MFCCs (13 Coefficients):** Capture the phonetic content and timbre of speech.
* **Pitch (F0 - Fundamental Frequency):** Detected using the YIN algorithm to identify monopitch or tremors.
* **Formants (F1 & F2):** Extracted via **Linear Predictive Coding (LPC)** to analyze tongue height and advancement (detecting "formant centralization" or slurring).

*Optimization:* Feature extraction is optimized using **Parallel Processing** (`joblib`) to handle large datasets efficiently.

### 2. Exploratory Data Analysis (EDA)
* **Spectrograms:** Visual comparison of spectral energy smearing in dysarthric speech vs. sharp transitions in control speech.
* **Boxplots:** Distribution analysis of pitch and formants across classes.

### 3. Statistical Hypothesis Testing
Before modeling, features are validated using **Welch’s T-Test** (independent samples with unequal variance).
* **Null Hypothesis ($H_0$):** No difference in feature means between groups.
* **Threshold:** $p < 0.05$ indicates statistical significance.

### 4. Modeling
Three classifiers were trained and compared:
* **Random Forest (RF):** Ensemble learning for handling non-linear feature relationships.
* **Bayesian Gaussian Mixture Model (B-GMM):** Probabilistic modeling of feature distributions.
* **Hidden Markov Model (HMM):** Gaussian HMM applied to feature vectors.

## 📊 Key Results

* **Feature Significance:** Statistical tests confirmed that **Pitch (F0)** and **Formant 1 (F1)** are highly significant discriminators ($p \approx 0.00$), confirming their clinical relevance.
* **Model Performance:**
    * **Random Forest** consistently outperformed GMM and HMM on both datasets.
    * The static nature of the aggregated feature vectors favored the decision-boundary approach of Random Forests over the temporal modeling of HMMs.

![Model Comparison Plot](path/to/your/comparison_plot_image.png)
*(Place your comparison bar chart image here)*

## 🛠️ Installation & Requirements

To run this pipeline, install the necessary dependencies:

```bash
pip install numpy pandas librosa matplotlib seaborn scikit-learn scipy joblib

## 🚀 Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/dysarthric-speech-pipeline.git](https://github.com/yourusername/dysarthric-speech-pipeline.git)
2. **Prepare Data:**
  Ensure the TORGO and RAWDysPeech datasets are extracted in the main directory.
3. **Run the Notebook:**
  Open Dysarthric_Speech_Recognition_Pipeline.ipynb in Jupyter Notebook or Google Colab.
4. **Execute Cells:**
  Run the cells sequentially to perform extraction, visualization, testing, and training.

## 📈 Visualizations

All Visualizations are shown in the code

## 📝 Conclusion

This project demonstrates that acoustic features derived from signal processing (LPC, MFCC) can effectively distinguish dysarthric speech. The Random Forest classifier serves as a robust tool for this binary classification task, offering potential applications in automated severity assessment and assistive technology.

## 📄 License

This project is licensed under the Apache 2.0 License - see the(LICENSE) file for details.

## 👥 Contributors

1. Abhinav Saluja
2. Amitesh Panda
3. Suhas Kesavan
