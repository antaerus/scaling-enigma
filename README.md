# Practical-Assignment-20.1-Initial-Report-and-EDA

# Job Scam Classification - Initial Report and EDA

[Click here for exploratory data analysis](/exploratory_data_analysis.ipynb)

The Jupyter notebook can be found in `exploratory_data_analysis.ipynb` 

## Research Question

What are the strongest indicators of a fraudulent job posting?

## Summary

For this module, I built an initial EDA and baseline binary classification model to classify a job posting as legitimate or fraudulent. My long-term goal is to implement an application to help job seekers determine the legitimacy of a job posting, as the current state of the job market is fraught with recruitment scams.

While platforms like Hugging Face and Kaggle contain datasets of unclassified job postings, the *Employment Scam Aegean Dataset* is the one dataset that focuses on this specific problem. As I work on this project, I may consider enrichment through other datasets.

## Dataset

- *Dataset used*: Employment Scam AEGEAN Dataset: https://github.com/FelixLuciano/Fake-JobPosting-Prediction
- This dataset contains job posting information, e.g. job title, description, company profile, telecommute, salary range, etc.

## Initial Results

![Confusion Matrix](/images/baseline_model_confusion_matrix.png)

A logistic regression model leveraging one hot encoding and TFIDF for the following features:

- title
- department
- company_profile
- description
- requirements
- benefits
- telecommuting
- has_company_logo
- has_questions
- required_experience
- required_education
- industry
- function

to classify a job posting as fraudulent or legitimate, resulted in the following scores

F1 Score: ~0.7125
ROC AUC Score: ~0.8591
Recall: ~0.7368

and a fit time of 2.95 seconds, which is a significant baseline for model deployment.

These are decent results for the baseline model, which can be further improved via additional datasets that identify "scam" key words, and other more advanced algorithms e.g. XGBoost, BERT, Transformers, to improve the above scores.

## Interpretation

![Feature Importances](images/feature_importances_and_coefficients_baseline.png)

The baseline result indicates IT is one of the departments most susceptible to fraudulency, followed closely by Oil and Energy, Product Development, and Computer Networking. Of the keywords from using TFIDF, "achieve" and "exciting" align with the expectations of suspicious agencies using emotionally charged buzz words to attract candidates. Words like "industry" could indicate that fraudulent postings employ vague language.

Stronger indicators of legitimate job postings are words like "level" and "team" which could relate the seniority, team focus that could align more with the specificity for a legitimate job posting. 

## Reflection

While the initial EDA employed SMOTE for handling this severely imbalanced dataset, the oversampling of the fraudulent class led to:

- the model catching almost 3/4 fraudulent postings
- 63 legitimate postings flagged as fraud. This could be a heavier cost for a desperate job searcher, whereas negligible for a casual job seeker. 
- 50 missed fraudulent job postings.

![ROC AUC CURVE](/images/baseline_model_roc_auc_curve.png)

The curve rises steeply to ~0.73 TPR before the elbow. At a very low false positive rate the model already catches nearly 3/4 of fraud.

## Next Steps

Per next steps, threshold tuning offers the most immediate gain - by lowering the classification threshold below the default 0.5, the model could become more aggressive in flagging fraud at a cost of modest increase in false positives. Optimizing this trade-off via a precision-recall curve would allow the threshold to be set deliberately based on business tolerance.

Feather enrichment through transfer learning on external corpora - such as the UCI SMS Spam Dataset - could expose the model to broader deceptive language patterns beyond just job postings. 

Ensemble methods like XGBoost could capture non-linear feature interactions and thus surface fraud signals that logistic regression's lienar decision boundaries would miss entirely.

For deeper semantic understanding, neural appraoches e.g. Keras-based deep networks or transformer models like BERT could also prove useful. BERT, having been pre-trained on vast text corpora, could contextualize buzz words and other galvanizing language often employed by fraudulent postings.

A combination of these strategies can improve this baseline towards a deployable job scam detection system.