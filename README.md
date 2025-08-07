# Insurance Cost Estimator

This is a web application that predicts medical insurance charges based on user input such as age, sex, BMI, number of children, smoking status, and region. It was built using Python, Dash, and a Ridge Regression model trained on a synthetic dataset.

This project was developed as part of a Predictive Analytics course to demonstrate end-to-end model deployment and application design.

---

## What the App Does

- Takes user inputs via a form  
- Preprocesses inputs to match model requirements  
- Predicts estimated insurance costs using a trained model  
- Displays the output interactively on the screen  

---

## Model Info

The Ridge Regression model was trained using scikit-learn on a synthetic dataset (`insurance.csv`). Key steps:  
- One-hot encoding for categorical features  
- Normalization of numerical variables  
- Ridge regularization to improve generalization  
