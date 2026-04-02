# FUTURE_ML_02

This project focuses on classifying and prioritizing support tickets using NLP and basic machine learning techniques in a simple and practical way.

## Problem Statement

Support teams receive a lot of tickets every day, and reading each one manually takes time. This project helps by:

- classifying each support ticket into a business category
- predicting how urgent the ticket is

The system uses ticket subject and description text to predict:

- `Billing`
- `Technical Issue`
- `Account`
- `General Query`

and also:

- `High`
- `Medium`
- `Low`

## Business Use Case

This kind of system is useful for SaaS companies, e-commerce support teams, and customer care teams that want to:

- route tickets to the right team faster
- detect urgent issues earlier
- reduce manual triage work
- improve first response time

## Dataset

- Source: Kaggle `suraj520/customer-support-ticket-dataset`
- Local file used here: `data/customer_support_tickets.csv`
- Total records after cleaning: generated from the Kaggle customer support dataset

The original dataset has ticket types such as `Refund request`, `Billing inquiry`, `Technical issue`, `Cancellation request`, and `Product inquiry`.

To match the project objective, the labels were adapted like this:

- `Billing inquiry` and `Refund request` -> `Billing`
- `Technical issue` -> `Technical Issue`
- `Cancellation request` -> `Account`
- `Product inquiry` -> `General Query`

For priority:

- `Critical` was merged into `High`
- `High`, `Medium`, and `Low` stayed the same

If a priority value is missing, the code also includes a simple keyword-based fallback rule.

## Approach

1. Load the CSV data
2. Merge ticket subject and description into one text field
3. Clean the text using lowercase conversion, punctuation removal, tokenization, stopword removal, and optional lemmatization
4. Compare `TF-IDF` and `Bag of Words`
5. Train separate models for:
   - category classification
   - priority prediction
6. Evaluate with accuracy, precision, recall, F1-score, and confusion matrix
7. Save trained models, predictions, metrics, and visuals

## Models Used

- Logistic Regression
- Multinomial Naive Bayes
- Random Forest

## Project Structure

```text
FUTURE_ML_02/
|-- README.md
|-- notebooks/
|   |-- support_ticket_classification.ipynb
|-- src/
|   |-- data_preprocessing.py
|   |-- feature_engineering.py
|   |-- model_training.py
|   |-- evaluation.py
|   |-- predict.py
|-- data/
|-- models/
|-- outputs/
|   |-- metrics/
|   |-- predictions/
|   |-- visuals/
```

## Results Summary

After running the project, the main outputs are saved in the `outputs/` folder.

Some useful files to look at:

- `metrics/best_model_summary.csv` – shows the final selected model
- `metrics/model_comparison.csv` – compares performance of different models
- `metrics/category_classification_report.csv` – evaluation for ticket category prediction
- `metrics/priority_classification_report.csv` – evaluation for priority prediction
- `predictions/predictions.csv` – final predicted results
- `visuals/category_confusion_matrix.png` – confusion matrix for category classification
- `visuals/priority_confusion_matrix.png` – confusion matrix for priority classification

### Best Results (from latest run)

- Category model: Random Forest (Bag of Words)
- Category macro F1: ~0.25  
- Priority model: Naive Bayes (Bag of Words)  
- Priority macro F1: ~0.34  

The scores aren’t very high since the dataset is quite synthetic and many ticket categories have similar wording. I’ve kept the results as they are to reflect the actual model performance instead of over-tuning or filtering outputs.

## Visual Outputs

### Category Class Distribution

![Category Class Distribution](outputs/visuals/category_class_distribution.png)

### Priority Class Distribution

![Priority Class Distribution](outputs/visuals/priority_class_distribution.png)

### Category Confusion Matrix

![Category Confusion Matrix](outputs/visuals/category_confusion_matrix.png)

### Priority Confusion Matrix

![Priority Confusion Matrix](outputs/visuals/priority_confusion_matrix.png)

## How to Run

1. Install required libraries:
   python -m pip install pandas numpy scikit-learn nltk matplotlib joblib

2. Run the training script:
   python -m src.model_training

3. (Optional) Try prediction on a sample ticket:
   python -m src.predict

   
## How This Helps Businesses

This project can help businesses:

- reduce ticket handling time
- send tickets to the correct queue automatically
- flag urgent complaints faster
- improve customer experience
- support agents with a simple triage system

## Notes

- Built with simplicity in mind so the workflow is easy to follow.
- Code is organized into small, clear steps for better understanding.
- A notebook version is also included for easier explanation of the project.
- Focus is more on clarity and understanding rather than over-optimization.
