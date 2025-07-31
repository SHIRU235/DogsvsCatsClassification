# DogsvsCatsClassification

#  Dogs vs Cats – Fine-Tuning Deep Learning Lab

This project demonstrates the application of deep learning for binary image classification (Cats vs Dogs). It involves training a custom CNN and fine-tuning a pre-trained VGG16 model using transfer learning.

---
# Dataset
### Download Dataset
[Download dataset from Google Drive]https://drive.google.com/file/d/1LHGxX1djND8mh3Rw9wXK0jmIkLM328Xx/view?usp=sharing

##  Project Structure

- `Dogs_vs_Cats_Classification.ipynb`:  
  Main notebook with data exploration, model training, evaluation, and conclusions.

---

##  Objectives

- Perform exploratory data analysis (EDA) on Cats vs Dogs dataset
- Train a **Custom CNN model** from scratch
- Fine-tune **VGG16** pre-trained on ImageNet
- Compare models using accuracy, confusion matrix, precision, recall, F1-score, PR curve
- Visualize misclassified examples
- Draw performance-based conclusions

---

##  Tools & Technologies

- Python (TensorFlow, Keras, matplotlib, scikit-learn)
- Google Colab (for GPU acceleration)
- Visual Studio Code (initial development)

---

## Why Google Colab + VS Code?

This project was developed using **both Visual Studio Code (Jupyter Notebook)** and **Google Colab** to leverage the best of both platforms:

| Tool          | Purpose                                       | Advantage                                 |
|---------------|-----------------------------------------------|--------------------------------------------|
| **VS Code**   | Local prototyping, editing & EDA              | Easy debugging, Git integration            |
| **Google Colab** | Final training with GPU & larger models     | Faster training, shareable, cloud-based    |

>  Using both tools ensured clean, well-structured code, with GPU acceleration for better performance.

---

##  Key Results

| Model           | Accuracy | F1-Score | Misclassified |
|----------------|----------|----------|----------------|
| Custom CNN     | 75.40%   | ~0.74    | 246 images     |
| VGG16 (Tuned)  | 89.80%   | ~0.90    | Much fewer     |

---

##  Visualizations

- Accuracy/Loss plots
- Confusion Matrix
- Precision–Recall Curve
- Misclassified Image Display

---

## Conclusion

Fine-tuning a pre-trained model like VGG16 provides a significant performance advantage over training from scratch, especially on small datasets. Transfer learning is a valuable technique in modern deep learning workflows.

---