import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

with open('true_labels.json', 'r', encoding='utf-8') as true_file:
    true_data = json.load(true_file)
with open('labels_XX.json', 'r', encoding='utf-8') as pred_file:
    pred_data = json.load(pred_file)
# Function to calculate metrics
def calculate_metrics(true_labels, pred_labels):
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='micro', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='micro', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='micro', zero_division=0)
    return accuracy, precision, recall, f1

# Function to parse dates
def parse_date(date_str):
    if date_str == "Present":
        return pd.to_datetime("today")
    if date_str == "Not found":
        return None
    try:
        return pd.to_datetime(date_str, format="%m/%Y")
    except ValueError:
        return None

def clean_dates(dates_str):
    return dates_str.strip('[]').replace(' ', '')

def calculate_scores(true_experiences, pred_experiences):
    min_length = min(len(true_experiences), len(pred_experiences))
    true_experiences = true_experiences[:min_length]
    pred_experiences = pred_experiences[:min_length]
    true_start_dates = []
    true_end_dates = []
    pred_start_dates = []
    pred_end_dates = []
    for true_ex, pred_ex in zip(true_experiences, pred_experiences):
        true_dates = clean_dates(true_ex['dates']).split("-")
        pred_dates = clean_dates(pred_ex['dates']).split("-")
        if len(true_dates) == 2:
            true_start_dates.append(parse_date(true_dates[0]))
            true_end_dates.append(parse_date(true_dates[1]))
        else:
            true_start_dates.append(None)
            true_end_dates.append(None)
        if len(pred_dates) == 2:
            pred_start_dates.append(parse_date(pred_dates[0]))
            pred_end_dates.append(parse_date(pred_dates[1]))
        else:
            pred_start_dates.append(None)
            pred_end_dates.append(None)
    true_dates = [clean_dates(ex['dates']).replace('-', ' - ') for ex in true_experiences]
    pred_dates = [clean_dates(ex['dates']).replace('-', ' - ') for ex in pred_experiences]
    true_company = [ex['company'].lower() for ex in true_experiences]

    def has_partial_match(true_value, pred_value, threshold=2):
        true_words = set(true_value.lower().split())
        pred_words = set(pred_value.lower().split())
        common_words = true_words.intersection(pred_words)
        return len(common_words) >= threshold  # Match if common words >= threshold

    updated_pred_labels = []
    for true_exp, pred_exp in zip(true_experiences, pred_experiences):
        true_companys = true_exp.get('company', '')
        pred_company = pred_exp.get('company', '')
        if has_partial_match(true_companys, pred_company):
            updated_pred_labels.append(true_companys.lower())
        else:
            updated_pred_labels.append(pred_company)
    updated_pred_position = []

    true_position = [ex['position'].lower() for ex in true_experiences]
    for true_exp, pred_exp in zip(true_experiences, pred_experiences):
        true_positions = true_exp.get('position', '')
        pred_position = pred_exp.get('position', '')
        if has_partial_match(true_positions, pred_position):
            updated_pred_position.append(true_positions.lower())
        else:
            updated_pred_position.append(pred_position)

    accuracy_dates, precision_dates, recall_dates, f1_dates = calculate_metrics(true_dates, pred_dates)
    accuracy_company, precision_company, recall_company, f1_company = calculate_metrics(true_company,                                                                                   updated_pred_labels)
    accuracy_position, precision_position, recall_position, f1_position = calculate_metrics(true_position,
                                                                                            updated_pred_position)
    return accuracy_dates, precision_dates, recall_dates, f1_dates, accuracy_company, precision_company, recall_company, f1_company, accuracy_position, precision_position, recall_position, f1_position


results = []
for pdf in true_data.keys():
    true_experiences = true_data[pdf]
    pred_experiences = pred_data[pdf]
    scores = calculate_scores(true_experiences, pred_experiences)
    results.append(scores)
average_scores = [sum(x) / len(x) for x in zip(*results)]
# Print average scores
print(f"Moyenne des scores pour les PDF :\n"
      f"Dates - Precision: {average_scores[1]:.2f}, Recall: {average_scores[2]:.2f}, F1 Score: {average_scores[3]:.2f}, Accuracy: {average_scores[0]:.2f}\n"
      f"Company - Precision: {average_scores[5]:.2f}, Recall: {average_scores[6]:.2f}, F1 Score: {average_scores[7]:.2f}, Accuracy: {average_scores[4]:.2f}\n"
      f"Position - Precision: {average_scores[9]:.2f}, Recall: {average_scores[10]:.2f}, F1 Score: {average_scores[11]:.2f}, Accuracy: {average_scores[8]:.2f}")

# Plotting the results (Accuracy,precision,  Recall, F1-Score)
categories = ['Dates', 'Company', 'Position']
accuracies = [average_scores[0], average_scores[4], average_scores[8]]
precisions = [average_scores[1], average_scores[5], average_scores[9]]
recalls = [average_scores[2], average_scores[6], average_scores[10]]
f1_scores = [average_scores[3], average_scores[7], average_scores[11]]

# Create subplots for better visualization
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle('GPT-3.5-turbo Results', fontsize=16, fontweight='bold')

# Accuracy Bar Plot
axes[0].bar(categories, accuracies, color=['blue', 'green', 'orange'])
axes[0].set_title('Accuracy by Category')
axes[0].set_ylim(0, 1)
axes[0].set_ylabel('Accuracy')
for i, acc in enumerate(accuracies):
    axes[0].text(i, acc + 0.02, f'{acc:.2f}', ha='center', fontsize=10)
axes[1].bar(categories, recalls, color=['blue', 'green', 'orange'])
axes[1].set_title('Recall by Category')
axes[1].set_ylim(0, 1)
axes[1].set_ylabel('Recall')
for i, rec in enumerate(recalls):
    axes[1].text(i, rec + 0.02, f'{rec:.2f}', ha='center', fontsize=10)

axes[2].bar(categories, f1_scores, color=['blue', 'green', 'orange'])
axes[2].set_title('F1-Score by Category')
axes[2].set_ylim(0, 1)
axes[2].set_ylabel('F1-Score')
for i, f1 in enumerate(f1_scores):
    axes[2].text(i, f1 + 0.02, f'{f1:.2f}', ha='center', fontsize=10)
axes[3].bar(categories, precisions, color=['blue', 'green', 'orange'])
axes[3].set_title('Precision by Category')
axes[3].set_ylim(0, 1)
axes[3].set_ylabel('Precision')
for i, f1 in enumerate(precisions):
    axes[3].text(i, f1 + 0.02, f'{f1:.2f}', ha='center', fontsize=10)

plt.tight_layout()
plt.show()
