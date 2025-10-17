import pandas as pd
import numpy as np

n_records = 100

# Helper function to generate correlated grades
def correlated_grade(base, noise=2.0):
    return np.clip(base + np.random.normal(0, noise, n_records), 10, 20)

# Generate base data
data = {
    "Roll_No": [f"{np.random.randint(100000,999999)}" for _ in range(n_records)],
    "Marital Status": np.random.randint(0, 2, n_records),
    "Application mode": np.random.randint(0, 2, n_records),
    "Application order": np.random.randint(1, 20, n_records),
    "Course": np.random.randint(9000, 10000, n_records),
    "Daytime/evening attendance": np.random.randint(0, 2, n_records),
    "Previous qualification": np.random.randint(0, 2, n_records),
    "Previous qualification (grade)": np.round(np.random.uniform(100, 180, n_records), 1),
    "Nacionality": np.random.randint(0, 2, n_records),
    "Mother's qualification": np.random.randint(0, 5, n_records),
    "Father's qualification": np.random.randint(0, 5, n_records),
    "Mother's occupation": np.random.randint(0, 5, n_records),
    "Father's occupation": np.random.randint(0, 5, n_records),
    "Admission grade": None,  # will compute below
    "Displaced": np.random.randint(0, 2, n_records),
    "Educational special needs": np.random.randint(0, 2, n_records),
    "Debtor": np.random.randint(0, 2, n_records),
    "Tuition fees up to date": None,  # will compute below
    "Gender": np.random.randint(0, 2, n_records),
    "Scholarship holder": np.random.randint(0, 2, n_records),
    "Age at enrollment": np.random.randint(18, 30, n_records),
    "International": np.random.randint(0, 2, n_records),
    "Curricular units 1st sem (credited)": np.random.randint(1, 8, n_records),
    "Curricular units 1st sem (enrolled)": np.random.randint(1, 8, n_records),
    "Curricular units 1st sem (evaluations)": np.random.randint(1, 8, n_records),
    "Curricular units 1st sem (approved)": np.random.randint(1, 8, n_records),
    "Curricular units 1st sem (grade)": None,  # will compute below
    "Curricular units 1st sem (without evaluations)": np.random.randint(0, 3, n_records),
    "Curricular units 2nd sem (credited)": np.random.randint(1, 8, n_records),
    "Curricular units 2nd sem (enrolled)": np.random.randint(1, 8, n_records),
    "Curricular units 2nd sem (evaluations)": np.random.randint(1, 8, n_records),
    "Curricular units 2nd sem (approved)": np.random.randint(1, 8, n_records),
    "Curricular units 2nd sem (grade)": None,  # will compute below
    "Curricular units 2nd sem (without evaluations)": np.random.randint(0, 3, n_records),
    "Unemployment rate": np.round(np.random.uniform(5, 15, n_records), 2),
    "Inflation rate": np.round(np.random.uniform(0, 10, n_records), 2),
    "GDP": np.round(np.random.uniform(1, 5, n_records), 2)
}

# Admission grade correlates with previous qualification grade
prev_grade = data["Previous qualification (grade)"]
data["Admission grade"] = np.round(prev_grade * 0.7 + np.random.uniform(20, 60, n_records), 1)

# Tuition fees up to date slightly correlates with debtor status
debtor = data["Debtor"]
data["Tuition fees up to date"] = np.where(debtor == 1, np.random.randint(0, 2, n_records), 1)

# Curricular grades correlated with admission grade
adm_grade = data["Admission grade"]
data["Curricular units 1st sem (grade)"] = np.round(correlated_grade(adm_grade / 10, noise=1.5), 3)
data["Curricular units 2nd sem (grade)"] = np.round(correlated_grade(adm_grade / 10, noise=1.5), 3)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("synthetic_students_realistic.csv", index=False)
print("CSV file 'synthetic_students_realistic.csv' created successfully!")
