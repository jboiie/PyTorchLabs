import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Programming\Projects\Learn\pytorch\sample_data.csv')

plt.hist(df["Marks"], bins=5, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel("Marks")
plt.ylabel("Count")
plt.title("Distribution of Marks")
plt.show()

colors = df["Passed"].map({"Yes": "green", "No": "red"})
plt.scatter(df["StudyHours"], df["Marks"], c=colors)
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks (Passed=Green, Failed=Red)")
plt.show()
