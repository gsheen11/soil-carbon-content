import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import csv

def get_labels():
# def load_dataset():
#     data_type_to_np_dtype = {
#         "String": str,
#         "Number": float
#     }

    labels = []
    with open('csv_data/HWSD2_LAYERS_METADATA.csv', 'r') as file:
        for line in file.readlines()[1:]:
            values = line.strip().split(',')
            slug = values[1]
            name = values[3]
            # data_type = values[4]
            # np_dtype = data_type_to_np_dtype[data_type]
            # labels += [(name, slug, data_type, np_dtype)]
            labels += [slug]
            # labels += [name]

    return labels

    


#     # Extract labels from the first line, assuming the first row contains headers
#     data = np.genfromtxt('../csv_data/HWSD2_LAYERS.csv', delimiter=',', skip_header=1, dtype=[(label[1], label[3]) for label in labels])
#     data = data[:] # remove row number from very left


#     return labels, data
def pre_process_categorical_feature(df):
    temp = pd.read_csv("csv_data/D_WRB_PHASES.csv")
    mapping_10 = temp.set_index("1").to_dict()["0"]
    mapping_10[""] = ""
    df["WRB_PHASES"] = df["WRB_PHASES"].map(mapping_10)

    temp = pd.read_csv("csv_data/D_WRB4.csv")
    mapping_11 = temp.set_index("2").to_dict()["0"]
    mapping_11[""] = ""
    df["WRB4"] = df["WRB4"].map(mapping_11)

    mapping_12 = dict()
    with open("csv_data/D_WRB2.csv", "r") as file:
        reader = csv.reader(file)
        header = next(reader)
        row_num = 1
        for row in reader:
            mapping_12[row[0]] = row_num
            row_num += 1
    mapping_12[""] = ""
    df["WRB2"] = df["WRB2"].map(mapping_12)

    temp = pd.read_csv("csv_data/D_FAO90.csv")
    mapping_13 = temp.set_index("1").to_dict()["0"]
    mapping_13[""] = ""
    df["FAO90"] = df["FAO90"].map(mapping_13)

    temp = pd.read_csv("csv_data/D_ROOT_DEPTH.csv")
    mapping_14 = temp.set_index("1").to_dict()["0"]
    mapping_14[""] = ""
    df["FAO90"] = df["FAO90"].map(mapping_14)

    temp = pd.read_csv("csv_data/D_PHASE.csv")
    mapping_15_16 = temp.set_index("1").to_dict()["0"]
    mapping_15_16[""] = ""
    df["PHASE1"] = df["PHASE1"].map(mapping_15_16)
    df["PHASE2"] = df["PHASE2"].map(mapping_15_16)

    temp = pd.read_csv("csv_data/D_ROOTS.csv")
    mapping_17 = temp.set_index("1").to_dict()["0"]
    mapping_17[""] = ""
    df["ROOTS"] = df["ROOTS"].map(mapping_17)

    temp = pd.read_csv("csv_data/D_IL.csv")
    mapping_18 = temp.set_index("1").to_dict()["0"]
    mapping_18[""] = ""
    df["IL"] = df["IL"].map(mapping_18)

    temp = pd.read_csv("csv_data/D_SWR.csv")
    mapping_19 = temp.set_index("1").to_dict()["0"]
    mapping_19[""] = ""
    df["SWR"] = df["SWR"].map(mapping_19)

    temp = pd.read_csv("csv_data/D_DRAINAGE.csv")
    mapping_20 = temp.set_index("1").to_dict()["0"]
    mapping_20[""] = ""
    df["DRAINAGE"] = df["DRAINAGE"].map(mapping_20)

    temp = pd.read_csv("csv_data/D_AWC.csv")
    mapping_21 = temp.set_index("1").to_dict()["0"]
    mapping_21[""] = ""
    df["AWC"] = df["AWC"].map(mapping_21)

    temp = pd.read_csv("csv_data/D_ADD_PROP.csv")
    mapping_22 = temp.set_index("1").to_dict()["0"]
    mapping_22[""] = ""
    df["ADD_PROP"] = df["ADD_PROP"].map(mapping_22)

    temp = pd.read_csv("csv_data/D_DEPTH_LAYER.csv")
    mapping_23 = temp.set_index("1").to_dict()["0"]
    mapping_23[""] = ""
    df["LAYER"] = df["LAYER"].map(mapping_23)

    # temp = pd.read_csv("csv_data/D_TEXTURE_USDA.csv")
    # mapping_30 = temp.set_index("1").to_dict()["0"]
    # mapping_30[""] = ""
    # df["TEXTURE_USDA"] = df["TEXTURE_USDA"].map(mapping_30)

    temp = pd.read_csv("csv_data/D_TEXTURE_SOTER.csv")
    mapping_31 = temp.set_index("1").to_dict()["0"]
    mapping_31[""] = ""
    df["TEXTURE_SOTER"] = df["TEXTURE_SOTER"].map(mapping_31)

    return df


def pre_process_data(df):
    df = pre_process_categorical_feature(df)

    features = ["WRB_PHASES","WRB4","WRB2","FAO90","ROOT_DEPTH","PHASE1","PHASE2","ROOTS","IL","SWR","DRAINAGE","AWC","ADD_PROP","LAYER","TOPDEP","BOTDEP","COARSE","SAND","SILT","CLAY","TEXTURE_USDA","TEXTURE_SOTER","BULK","REF_BULK","PH_WATER","TOTAL_N","CN_RATIO","CEC_SOIL","CEC_CLAY","CEC_EFF","TEB","BSAT","ALUM_SAT","ESP","TCARBON_EQ","GYPSUM","ELEC_COND"]

    float_features = df[features].astype(float)

    y = df["ORG_CARBON"].astype(float).values

    # categorical_feature = df[[""]]
    # encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')  # Initialize encoder, sparse=False returns a 2D array
    # one_hot_encoded_feature = encoder.fit_transform(categorical_feature)

    # x = np.concatenate([float_features.values, one_hot_encoded_feature], axis=1)
    x = np.concatenate([float_features])

    mask = ~np.isnan(y) & (y >= 0)

    x_filtered = x[mask]
    y_filtered = y[mask]


    return x_filtered, y_filtered


def load_training_data():
    df = pd.read_csv("data/train_set.csv")
    x, y = pre_process_data(df)
    return x, y


def load_test_data():
    df = pd.read_csv("data/test_set.csv")
    x, y = pre_process_data(df)
    return x, y


def main():
    x, y = load_training_data()


if __name__ == "__main__":
    main()
