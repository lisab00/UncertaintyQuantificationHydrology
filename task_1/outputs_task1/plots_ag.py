import pandas as pd
import matplotlib.pyplot as plt
import re
import os

input_file_path = "/Users/agnes_dchn/PycharmProjects/UncertaintyQuantificationHydrology/task_1/outputs_task1/csv_outputs/urr/output_detailed.csv"

output_folder = os.path.dirname(input_file_path)

with open(input_file_path, "r") as f:
    lines = f.readlines()

print("\nFirst few raw lines from CSV:")
for i in range(min(5, len(lines))):
    print(lines[i])

cleaned_lines = []
current_line = ""

for line in lines:
    line = line.strip()
    if not line:
        continue
    if re.match(r'^\d+\.\d+,', line):
        if current_line:
            cleaned_lines.append(current_line)
        current_line = line
    else:
        current_line += " " + line

if current_line:
    cleaned_lines.append(current_line)

try:
    df = pd.DataFrame([line.split(",", 1) for line in cleaned_lines], columns=["Obj_fct_values", "prm_values"])
except ValueError as e:
    print(" ERROR: Could not split lines correctly! Check CSV formatting.")
    print(e)
    exit()

print("\nFirst rows of DataFrame after processing:")
print(df.head())

if "prm_values" not in df.columns:
    print("\n ERROR: 'prm_values' column not found in DataFrame!")
    exit()

df["Obj_fct_values"] = pd.to_numeric(df["Obj_fct_values"], errors="coerce")


def safe_literal_eval(value):
    try:
        value = re.sub(r'["\[\]]', '', value)
        value = re.sub(r'\s+', ' ', value).strip()
        return [float(v) for v in value.split()]
    except ValueError:
        return None


df["prm_values"] = df["prm_values"].apply(safe_literal_eval)

df = df.dropna(subset=["prm_values"])

prm_names = [
    'snw_dth', 'snw_ast', 'snw_amt', 'snw_amf', 'snw_pmf',
    'sl0_mse', 'sl1_mse',
    'sl0_fcy', 'sl0_bt0',
    'sl1_pwp', 'sl1_fcy', 'sl1_bt0',
    'urr_dth', 'lrr_dth',
    'urr_rsr', 'urr_tdh', 'urr_tdr', 'urr_cst', 'urr_dro', 'urr_ulc',
    'lrr_tdh', 'lrr_cst', 'lrr_dro'
]

df = df[df["prm_values"].apply(lambda x: isinstance(x, list) and len(x) == len(prm_names))]

if df.empty:
    print("\n ERROR: No valid data after cleaning! Check CSV formatting.")
    exit()

df[prm_names] = pd.DataFrame(df["prm_values"].tolist(), index=df.index)

df = df.drop(columns=["prm_values"])

for prm in prm_names:
    plt.figure(figsize=(8, 6))
    plt.scatter(df[prm], df["Obj_fct_values"], alpha=0.5, s=10)
    plt.grid()
    plt.xlabel(f'Parameter Value of {prm}')
    plt.ylabel('Log of Objective Function Value')
    plt.yscale('log')
    plt.title(f'Parameter {prm} vs. Objective Function')

    output_file = os.path.join(output_folder, f'scatter_nse_{prm}.svg')
    plt.savefig(output_file, format='svg')
    plt.close()
