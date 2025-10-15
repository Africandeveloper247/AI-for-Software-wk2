# AI-for-Software-wk2
##  Project Overview  
Diabetes is one of the fastest-growing health challenges worldwide, affecting over **422 million people** (WHO, 2023). Early detection can significantly reduce complications such as kidney failure, blindness, and cardiovascular diseases.  

This project builds a **Logistic Regression model** that predicts whether a patient is likely to have diabetes based on health and demographic indicators. The project also aligns with the **United Nations Sustainable Development Goals (SDGs)**, particularly:  
- **SDG 3 – Good Health & Well-being**  
- **SDG 9 – Innovation in Healthcare**  
- **SDG 10 – Reduced Inequalities**  

---

##  Problem Statement  
Limited access to early screening and diagnosis in low-resource settings leads to late detection of diabetes, increasing mortality and healthcare costs. There is a need for **affordable, data-driven tools** that can identify high-risk individuals early.  

---

##  Solution  
Using the **Pima Indians Diabetes Dataset**, this project trains a Logistic Regression model with the following features:  
- Pregnancies  
- Glucose  
- Diastolic Blood Pressure  
- Triceps Skin Fold Thickness  
- Insulin Levels  
- BMI (Body Mass Index)  
- Diabetes Pedigree Function (DPF)  
- Age  

The target variable is whether the patient has diabetes (**1**) or not (**0**).  

---

##  Tech Stack  
- **Python 3.9+**  
- **Pandas & NumPy** – data preprocessing  
- **Scikit-learn** – model building & evaluation  
- **Matplotlib & Seaborn** – data visualization  
- **Jupyter Notebook** – analysis & experimentation  

## Model Performance

The Logistic Regression model was evaluated using accuracy, precision, recall, and F1-score.

Accuracy: ~65%

* Interpretability *: Identifies key risk factors (e.g., glucose, BMI, age) that drive diabetes predictions.

Note: Model performance can be further improved with hyperparameter tuning or advanced algorithms (e.g., Random Forest, XGBoost).

## Social Impact (SDG Alignment)

**SDG 3**: Improves early detection & reduces complications of diabetes.
**SDG 9**: Demonstrates how AI/ML can innovate in healthcare.
**SDG 10**: Provides low-cost, scalable solutions for underserved communities.

## Future Improvements

- Deploy the model as a web app using Streamlit or Flask.

- Add more advanced ML models for comparison.

- Collect real-world patient data for higher generalization.

- Integrate with mobile health apps for wider accessibility.

## Author

- Jebilla Ibrahim

- Junior Data Scientist | Machine Learning Enthusiast| AI learner

- Abuja, Nigeria
