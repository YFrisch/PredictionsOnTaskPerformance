""" Execute this file to read and evaluate the demographic data of the subjects.

Project: Predictions on Task Performance
"""

__author__ = 'Yannik Frisch, Maximilian A. Gehrke'
__date__ = '10-03-2020'

import pandas as pd
import numpy as np

import src.utils

# Set working directory to be in top level folder
src.utils.set_working_directory()

# Get the data
df = pd.read_csv(f'assets/demographics.csv', sep=',')

print(f'----- Demographics -----')

# Number of subjects
print(f'# Subjects: {df.shape[0]}')

# Evaluate Gender
gender = df['sex'].to_numpy()
unique, counts = np.unique(gender, return_counts=True)
print(f'Gender: {counts[0]} {unique[0]}, {counts[1]} {unique[1]}')

# Evaluate Age
age = df['age'].to_numpy()
age_sorted = np.sort(age)
age_avg = np.average(age)
print(f'Age: min = {age_sorted[0]}, max = {age_sorted[-1]}, mean = {age_avg}')

# Evaluate Nationality
nat = df['nationality'].to_numpy()
unique, counts = np.unique(nat, return_counts=True)
print(f'Nationality: {counts[0]} x {unique[0]}', end='')
for u, c in zip(unique[1:], counts[1:]):
    print(f', {c} x {u}', end='')
print()

# Evaluate jobs
jobs = df['job'].to_numpy()
unique, counts = np.unique(jobs, return_counts=True)

print(f'Jobs: {counts[0]} x {unique[0]}', end='')
for u, c in zip(unique[1:], counts[1:]):
    print(f', {c} x {u}', end='')
print()

# Evaluate course of studies
study = df['study_subject'].to_numpy()
unique, counts = np.unique(study, return_counts=True)

print(f'Study subjects: {counts[0]} x {unique[0]}', end='')
for u, c in zip(unique[1:], counts[1:]):
    print(f', {c} x {u}', end='')
print()

print(f'------------------------')