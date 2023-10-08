import pandas as pd
import math
import sys

proccess_files = [
"data-01",
"data-05",
"data-06",
"data-07",
"data-08",
"data-09",
"data-11",
"data-13",
"data-14",
"data-15",
"data-16",
"data-17",
"data-18",
"data-19",
"data-20",
"data-21",
"data-23",
"data-24",
"data-25",
"data-26",
"data-27",
"data-28",
"data-29",
"data-30",
"data-31",
"data-32",
"data-33",
"data-34",
"data-35",
"data-36",
"data-37",
"data-38",
"data-39",
"data-40",
"data-41",
"data-42",
"data-43",
"data-46",
"data-47",
"data-48",
"data-53",
"data-67",
"data-70",
]

""" default
proccess_files = [
"data-01",
"data-02",
"data-05",
"data-06",
"data-07",
"data-08",
"data-09",
"data-10",
"data-11",
"data-12",
"data-13",
"data-14",
"data-15",
"data-16",
"data-17",
"data-18",
"data-19",
"data-20",
"data-21",
"data-23",
"data-24",
"data-25",
"data-26",
"data-27",
"data-28",
"data-29",
"data-30",
"data-31",
"data-32",
"data-33",
"data-34",
"data-35",
"data-36",
"data-37",
"data-38",
"data-39",
"data-40",
"data-41",
"data-42",
"data-43",
"data-46",
"data-47",
"data-48",
"data-53",
"data-67",
"data-70",
]
"""


def process_and_combine_rows(df):
    # Create an empty list to store the new processed rows
    new_rows = []

    # Use a flag to skip rows when needed
    skip_next = False

    process_start = False
    
    for i in range(len(df) - 1):
        if not process_start and (df.iloc[i, 2] == 33 or df.iloc[i, 2] == 34):
            continue
        else:
            process_start = True

        if not process_start:
            continue

        if skip_next:
            skip_next = False
            continue
        
        if i < len(df) - 1 and df.iloc[i, 2] == 0:
            skip_next = True
            continue

        if i < len(df) - 1 and \
            (df.iloc[i, 2] == 33 or df.iloc[i, 2] == 34) and \
            (df.iloc[i + 1, 2] == 33 or df.iloc[i + 1, 2] == 34):

            combined_row = [df.iloc[i, 0], df.iloc[i, 1], 33, f"{df.iloc[i, 3]},{df.iloc[i + 1, 3]}"]
            new_rows.append(combined_row)
            skip_next = True
        else:
            new_rows.append(df.iloc[i].tolist())

        if i < len(df) - 1 and \
            (df.iloc[i, 2] != 33 and df.iloc[i, 2] != 34) and \
            (df.iloc[i + 1, 2] != 33 and df.iloc[i + 1, 2] != 34):
            zero_insulin_row = [df.iloc[i, 0], df.iloc[i, 1], 33, 0]
            new_rows.append(zero_insulin_row)

    # Convert the list of rows to a DataFrame
    new_df = pd.DataFrame(new_rows)
    
    return new_df

def delete_only_dose(df):
        # Create an empty list to store the new processed rows
    new_rows = []

    # Use a flag to skip rows when needed
    skip_next = False
    
    for i in range(len(df) - 1):
        if skip_next:
            skip_next = False
            continue

        if i < len(df) - 1 and \
            (df.iloc[i, 2] == 33) and \
            (df.iloc[i + 1, 2] == 33 or df.iloc[i + 1, 2] == 34):

            skip_next = True

        new_rows.append(df.iloc[i].tolist())

    # Convert the list of rows to a DataFrame
    new_df = pd.DataFrame(new_rows)
    
    return new_df

def delete_continous_dose(df):
        # Create an empty list to store the new processed rows
    new_rows = []

    # Use a flag to skip rows when needed
    skip_next = False
    
    for i in range(len(df) - 1):
        if skip_next:
            skip_next = False
            continue

        if i < len(df) - 1 and \
            (df.iloc[i, 2] == 33 or df.iloc[i, 2] == 33) and \
            (df.iloc[i + 1, 2] == 33 or df.iloc[i + 1, 2] == 34):

            skip_next = True

        new_rows.append(df.iloc[i].tolist())

    # Convert the list of rows to a DataFrame
    new_df = pd.DataFrame(new_rows)
    
    return new_df

def check_error(df):
    issues_found_new = []

    for i in range(len(df) - 1):
        current_value = df.iloc[i, 2]
        next_value = df.iloc[i + 1, 2]
        
        if (current_value == 33 or current_value == 34) and (next_value == 33 or next_value == 34):
            issues_found_new.append((i, df.iloc[i], df.iloc[i + 1]))
    return issues_found_new

def calculate_risk(blood_glucose):
    return 10 * math.pow((3.5506 * (math.pow(math.log(max(1, blood_glucose)), 0.8353) - 3.7932)), 2)   

def sum_values(val):
    val_str = str(val)
    if "," in val_str:
        return int(sum(float(x) for x in val.split(",")))
    else:
        return int(val)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        data_file_paths = [sys.argv[1]]
    else:
        data_file_paths = proccess_files

    cnt = 0
    for data_path in data_file_paths:
        file_name = data_path
        data_path = "./diabetes_dataset/" + data_path
        data = pd.read_csv(data_path, sep="\t", header=None)

        # ~ は反転(!, 0 -> 1)
        data_filtered = data[~data[2].between(65, 72)]
        processed_data1 = process_and_combine_rows(data_filtered)
        processed_data2 = delete_only_dose(processed_data1)
        processed_data3 = delete_continous_dose(processed_data2)
        df = processed_data3

        # 新しいデータフレームを作成する
        new_data = []
        for i in range(0, len(df) - 1, 2):
            s1 = df.iloc[i][3]
            a1 = df.iloc[i + 1][3]
            r1 = -calculate_risk(float(s1))
            s2 = df.iloc[i + 2][3] if i + 2 < len(df) else None
            
            new_data.append([s1, a1, r1, s2])

        new_df = pd.DataFrame(new_data, columns=["s1", "a1", "r1", "s2"])
        new_df['a1'] = new_df['a1'].apply(sum_values)
        
        new_df.to_csv("./proprocessed_diabetes_dataset/" + file_name + ".csv")