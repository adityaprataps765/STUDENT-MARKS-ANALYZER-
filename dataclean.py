import numpy as np
import pandas as pd

# Load raw examination dataset
student_df = pd.read_csv('student.csv')
print("Initial dataset preview:")
print(student_df.head(10))

# Remove duplicate admission IDs (keeping the verified first entry)
student_df = student_df.drop_duplicates(subset=['Student_ID'], keep='first').copy()
print("\nAfter removing duplicate Student_ID rows:")
print(student_df.head(10))

# Standardize Gender labels to 'M' and 'F'
student_df['Gender'] = student_df['Gender'].astype(str).str.strip().str.upper()
gender_lookup = {
    'FEMALE': 'F',
    'F': 'F',
    'MALE': 'M',
    'M': 'M'
}
student_df['Gender'] = student_df['Gender'].map(gender_lookup).fillna('Unknown')
print("\nAfter standardizing Gender values:")
print(student_df.head(10))

# Normalize Semester values to first-year cohort ('1')
student_df['Semester'] = student_df['Semester'].astype(str).str.strip().str.upper()
semester_lookup = {
    '4': '1',
    '3': '1',
    '2': '1',
    '1': '1'
}
student_df['Semester'] = student_df['Semester'].map(semester_lookup).fillna('Unknown')

# Extract numeric suffix from Student_ID, sort ascending, and rebuild ID format
student_df['Student_ID'] = student_df['Student_ID'].str[5:]
student_df['Student_ID'] = pd.to_numeric(student_df['Student_ID'])
student_df = student_df.sort_values(by='Student_ID')
student_df['Student_ID'] = '2026B0' + student_df['Student_ID'].astype(str)

# Replace 'Absent' text entries in Electrical_Marks with NaN
student_df['Electrical_Marks'] = student_df['Electrical_Marks'].astype(str).str.strip()
student_df.loc[student_df['Electrical_Marks'] == 'Absent', 'Electrical_Marks'] = np.nan

# Convert attendance and subject mark columns to float
numeric_cols = [
    'Attendance_Pct',
    'Maths_Marks',
    'Physics_Marks',
    'Programming_Marks',
    'Electrical_Marks',
    'AI_Marks'
]
for col in numeric_cols:
    student_df[col] = pd.to_numeric(student_df[col], errors='coerce')

# Validate Attendance_Pct within [0, 100] and impute out-of-range values with median
student_df.loc[(student_df['Attendance_Pct'] < 0) | (student_df['Attendance_Pct'] > 100), 'Attendance_Pct'] = np.nan
student_df['Attendance_Pct'] = student_df['Attendance_Pct'].fillna(student_df['Attendance_Pct'].median())

# Validate subject marks within [0, 100] and replace invalid/missing scores with 0
subject_cols = [
    'Maths_Marks',
    'Physics_Marks',
    'Programming_Marks',
    'Electrical_Marks',
    'AI_Marks'
]
for subj in subject_cols:
    student_df.loc[(student_df[subj] < 0) | (student_df[subj] > 100), subj] = np.nan
    student_df[subj] = student_df[subj].fillna(0)

print("\nFinal cleaned dataset preview:")
print(student_df.head(10))

# Save sanitized dataset
student_df.to_csv('cleaned_student.csv', index=False)
print("\nSaved cleaned records to 'cleaned_student.csv'.")