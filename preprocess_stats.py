import csv
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import util


def get_proportion_populated_columns(csv_file):
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        header = next(reader)
        num_columns = len(header)
        column_array = np.zeros(num_columns)
        num_rows = 0
        for row in reader:
            num_rows += 1
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
if __name__ == "__main__":

    matrix = np.load("cooccurance_matrix.npy", allow_pickle=True)
    labels = util.get_labels()

    print(matrix.shape, len(labels))

    displayed_labels = labels[4:]  # Adjust this based on the part of the matrix you're showing
    displayed_labels = [label.replace("_", "") for label in displayed_labels]

    # Display the matrix
    plt.imshow(matrix[5:, 5:].T, aspect='auto')  # Use aspect='auto' for automatic aspect ratio
    plt.colorbar()

    # Set the tick labels
    plt.xticks(ticks=range(len(displayed_labels)), labels=displayed_labels, rotation='vertical')
    plt.yticks(ticks=range(len(displayed_labels)), labels=displayed_labels)

    # Optional: Improve layout to make room for label rotation
    plt.tight_layout()

    plt.show()
    # data_csv = "csv_data/HWSD2_LAYERS_CLEANED.csv"
    # proportion = get_proportion_populated_columns(data_csv)

    # with open("csv_data/HWSD2_LAYERS_METADATA.csv", "r") as file:
    #     reader = csv.reader(file)
    #     header = next(reader)
    #     num_columns = len(header)
    #     row_num = 0
    #     for row in reader:
    #         row_num += 1
    #         if row_num <= 9:
    #             continue
    #         print(
    #             f"Col #{row[0]}, {row[3]} proportion populated: {proportion[row_num - 9] * 100:.2f}%"
    #         )

    # # cooccurance_matrix = gen_cooccurence_matrix(data_csv)
    # # np.save("cooccurance_matrix.npy", cooccurance_matrix)
    # # print("cooccurance_matrix.npy saved")
