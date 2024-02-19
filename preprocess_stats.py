import csv
import numpy as np


def get_proportion_populated_columns(csv_file):
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        header = next(reader)
        num_columns = len(header)
        column_array = np.zeros(num_columns)
        num_rows = 0
        # test_set = set()
        for row in reader:
            num_rows += 1
            # if row[2] in test_set:
            #     print(row[2])
            # else:
            #     test_set.add(row[2])
            for i in range(num_columns):
                if row[i] != "":
                    column_array[i] += 1

    proportion = column_array / num_rows
    print(f"Processed {num_rows} rows")
    return proportion


def gen_cooccurence_matrix(csv_file):
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        header = next(reader)
        num_columns = len(header)
        cooccurance_matrix = np.zeros((num_columns, num_columns))
        num_rows = 0
        for row in reader:
            if num_rows % 10000 == 0:
                print(f"Processing row {num_rows}")
            num_rows += 1
            for i in range(num_columns):
                if row[i] != "":
                    cooccurance_matrix[i, i] += 1
                    for j in range(i + 1, num_columns):
                        if row[j] != "":
                            cooccurance_matrix[i, j] += 1
    return cooccurance_matrix / num_rows


# Usage example
data_csv = "csv_data/HWSD2_LAYERS.csv"
proportion = get_proportion_populated_columns(data_csv)

with open("csv_data/HWSD2_LAYERS_METADATA.csv", "r") as file:
    reader = csv.reader(file)
    header = next(reader)
    num_columns = len(header)
    row_num = 0
    for row in reader:
        row_num += 1
        print(
            f"Col #{row[0]}, {row[3]} proportion populated: {proportion[row_num] * 100:.2f}%"
        )

cooccurance_matrix = gen_cooccurence_matrix(data_csv)
np.save("cooccurance_matrix.npy", cooccurance_matrix)
print("cooccurance_matrix.npy saved")

# print(f"The proportion of populated columns is: {proportion}")
