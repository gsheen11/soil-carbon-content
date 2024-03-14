import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
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
    df["ROOT_DEPTH"] = df["ROOT_DEPTH"].map(mapping_14)

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


def pre_process_one_hot_encoding(df):
    categorical_features = ["WRB_PHASES","WRB4","WRB2","FAO90","PHASE1","PHASE2","ROOTS","IL","SWR","DRAINAGE","AWC","ADD_PROP","LAYER","TEXTURE_SOTER"]
    
    categorical_mapping = {
        # "WRB_PHASES": np.arange(1, 557),
        "WRB4": np.arange(1, 192),
        "WRB2": np.arange(1, 36),
        "FAO90": np.arange(1, 186),
        "ROOT_DEPTH": np.arange(1, 5),
        "PHASE1": np.arange(0, 31),
        "PHASE2": np.arange(0, 31),
        "ROOTS": np.arange(0, 7),
        "IL": np.arange(0, 5),
        "SWR": np.arange(0, 5),
        "DRAINAGE": np.arange(1, 8),
        "AWC": np.arange(1, 8),
        "ADD_PROP": [0, 2, 3],
        "LAYER": np.arange(0, 7),
        "TEXTURE_SOTER": ["C", "F", "M", "V", "Z"]
    }

    for feature, range_ in categorical_mapping.items():
        # df[feature] = df[feature].astype("category", categories=range_)
        cat_dtype = CategoricalDtype(
            categories=range_, ordered=True)
        df[feature].astype(cat_dtype)
    ohe_df = pd.get_dummies(df[categorical_features])
    df = pd.concat([df, ohe_df], axis=1).drop(columns=categorical_features)

    # encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').set_output(transform="pandas")
    # categorical_df = df[categorical_features]
    # ohe_df = encoder.fit_transform(categorical_df)
    # df = pd.concat([df, ohe_df], axis=1).drop(columns=categorical_features)
    return df

def pre_process_data(df):
    df = pre_process_categorical_feature(df)
    df = pre_process_one_hot_encoding(df)
    # print("Post OHE")
    # pd.set_option('display.max_columns', None) 
    # print(df.head(1))
    # print(df.columns.values)

    # Original features pre-OHE
    # features = ["WRB_PHASES","WRB4","WRB2","FAO90","ROOT_DEPTH","PHASE1","PHASE2","ROOTS","IL","SWR","DRAINAGE","AWC","ADD_PROP","LAYER","TOPDEP","BOTDEP","COARSE","SAND","SILT","CLAY","TEXTURE_USDA","TEXTURE_SOTER","BULK","REF_BULK","PH_WATER","TOTAL_N","CN_RATIO","CEC_SOIL","CEC_CLAY","CEC_EFF","TEB","BSAT","ALUM_SAT","ESP","TCARBON_EQ","GYPSUM","ELEC_COND"]

    # Excludes ['ID', 'HWSD2_SMU_ID', 'WISE30s_SMU_ID', 'HWSD1_SMU_ID', 'COVERAGE', 'SEQUENCE', 'SHARE', 'NSC_MU_SOURCE1', 'NSC_MU_SOURCE2', 'ROOT_DEPTH', 'ORG_CARBON']
    features = [
        'TOPDEP', 'BOTDEP', 'COARSE', 'SAND', 'SILT', 'CLAY', 'TEXTURE_USDA',
        'BULK', 'REF_BULK', 'PH_WATER', 'TOTAL_N', 'CN_RATIO', 'CEC_SOIL', 'CEC_CLAY',
        'CEC_EFF', 'TEB', 'BSAT', 'ALUM_SAT', 'ESP', 'TCARBON_EQ', 'GYPSUM', 'ELEC_COND',
        'WRB_PHASES_1', 'WRB_PHASES_2', 'WRB_PHASES_3', 'WRB_PHASES_5',
        'WRB_PHASES_6', 'WRB_PHASES_8', 'WRB_PHASES_9', 'WRB_PHASES_10',
        'WRB_PHASES_11', 'WRB_PHASES_12', 'WRB_PHASES_13', 'WRB_PHASES_14',
        'WRB_PHASES_15', 'WRB_PHASES_16', 'WRB_PHASES_18', 'WRB_PHASES_19',
        'WRB_PHASES_20', 'WRB_PHASES_22', 'WRB_PHASES_25', 'WRB_PHASES_26',
        'WRB_PHASES_27', 'WRB_PHASES_28', 'WRB_PHASES_29', 'WRB_PHASES_31',
        'WRB_PHASES_32', 'WRB_PHASES_33', 'WRB_PHASES_34', 'WRB_PHASES_35',
        'WRB_PHASES_37', 'WRB_PHASES_38', 'WRB_PHASES_39', 'WRB_PHASES_40',
        'WRB_PHASES_41', 'WRB_PHASES_42', 'WRB_PHASES_43', 'WRB_PHASES_44',
        'WRB_PHASES_45', 'WRB_PHASES_46', 'WRB_PHASES_47', 'WRB_PHASES_48',
        'WRB_PHASES_49', 'WRB_PHASES_50', 'WRB_PHASES_51', 'WRB_PHASES_52',
        'WRB_PHASES_53', 'WRB_PHASES_54', 'WRB_PHASES_56', 'WRB_PHASES_57',
        'WRB_PHASES_58', 'WRB_PHASES_59', 'WRB_PHASES_60', 'WRB_PHASES_61',
        'WRB_PHASES_62', 'WRB_PHASES_63', 'WRB_PHASES_64', 'WRB_PHASES_65',
        'WRB_PHASES_67', 'WRB_PHASES_68', 'WRB_PHASES_69', 'WRB_PHASES_71',
        'WRB_PHASES_72', 'WRB_PHASES_73', 'WRB_PHASES_74', 'WRB_PHASES_76',
        'WRB_PHASES_78', 'WRB_PHASES_80', 'WRB_PHASES_82', 'WRB_PHASES_83',
        'WRB_PHASES_84', 'WRB_PHASES_86', 'WRB_PHASES_87', 'WRB_PHASES_88',
        'WRB_PHASES_89', 'WRB_PHASES_90', 'WRB_PHASES_92', 'WRB_PHASES_93',
        'WRB_PHASES_94', 'WRB_PHASES_97', 'WRB_PHASES_98', 'WRB_PHASES_99',
        'WRB_PHASES_100', 'WRB_PHASES_101', 'WRB_PHASES_102', 'WRB_PHASES_103',
        'WRB_PHASES_104', 'WRB_PHASES_105', 'WRB_PHASES_106', 'WRB_PHASES_107',
        'WRB_PHASES_109', 'WRB_PHASES_110', 'WRB_PHASES_111', 'WRB_PHASES_112',
        'WRB_PHASES_113', 'WRB_PHASES_114', 'WRB_PHASES_115', 'WRB_PHASES_116',
        'WRB_PHASES_117', 'WRB_PHASES_118', 'WRB_PHASES_119', 'WRB_PHASES_120',
        'WRB_PHASES_121', 'WRB_PHASES_122', 'WRB_PHASES_123', 'WRB_PHASES_124',
        'WRB_PHASES_125', 'WRB_PHASES_127', 'WRB_PHASES_128', 'WRB_PHASES_129',
        'WRB_PHASES_130', 'WRB_PHASES_131', 'WRB_PHASES_132', 'WRB_PHASES_133',
        'WRB_PHASES_134', 'WRB_PHASES_135', 'WRB_PHASES_136', 'WRB_PHASES_137',
        'WRB_PHASES_138', 'WRB_PHASES_139', 'WRB_PHASES_140', 'WRB_PHASES_141',
        'WRB_PHASES_142', 'WRB_PHASES_143', 'WRB_PHASES_144', 'WRB_PHASES_145',
        'WRB_PHASES_146', 'WRB_PHASES_147', 'WRB_PHASES_149', 'WRB_PHASES_150',
        'WRB_PHASES_151', 'WRB_PHASES_152', 'WRB_PHASES_153', 'WRB_PHASES_154',
        'WRB_PHASES_155', 'WRB_PHASES_156', 'WRB_PHASES_157', 'WRB_PHASES_158',
        'WRB_PHASES_160', 'WRB_PHASES_163', 'WRB_PHASES_164', 'WRB_PHASES_166',
        'WRB_PHASES_167', 'WRB_PHASES_168', 'WRB_PHASES_169', 'WRB_PHASES_170',
        'WRB_PHASES_171', 'WRB_PHASES_172', 'WRB_PHASES_173', 'WRB_PHASES_174',
        'WRB_PHASES_175', 'WRB_PHASES_176', 'WRB_PHASES_177', 'WRB_PHASES_178',
        'WRB_PHASES_179', 'WRB_PHASES_182', 'WRB_PHASES_183', 'WRB_PHASES_184',
        'WRB_PHASES_185', 'WRB_PHASES_186', 'WRB_PHASES_187', 'WRB_PHASES_188',
        'WRB_PHASES_189', 'WRB_PHASES_190', 'WRB_PHASES_191', 'WRB_PHASES_192',
        'WRB_PHASES_193', 'WRB_PHASES_194', 'WRB_PHASES_195', 'WRB_PHASES_196',
        'WRB_PHASES_197', 'WRB_PHASES_198', 'WRB_PHASES_199', 'WRB_PHASES_200',
        'WRB_PHASES_201', 'WRB_PHASES_202', 'WRB_PHASES_203', 'WRB_PHASES_204',
        'WRB_PHASES_206', 'WRB_PHASES_207', 'WRB_PHASES_208', 'WRB_PHASES_209',
        'WRB_PHASES_210', 'WRB_PHASES_211', 'WRB_PHASES_212', 'WRB_PHASES_213',
        'WRB_PHASES_214', 'WRB_PHASES_215', 'WRB_PHASES_216', 'WRB_PHASES_217',
        'WRB_PHASES_218', 'WRB_PHASES_220', 'WRB_PHASES_221', 'WRB_PHASES_222',
        'WRB_PHASES_223', 'WRB_PHASES_224', 'WRB_PHASES_225', 'WRB_PHASES_226',
        'WRB_PHASES_227', 'WRB_PHASES_228', 'WRB_PHASES_229', 'WRB_PHASES_230',
        'WRB_PHASES_234', 'WRB_PHASES_235', 'WRB_PHASES_236', 'WRB_PHASES_237',
        'WRB_PHASES_238', 'WRB_PHASES_239', 'WRB_PHASES_240', 'WRB_PHASES_242',
        'WRB_PHASES_243', 'WRB_PHASES_244', 'WRB_PHASES_245', 'WRB_PHASES_246',
        'WRB_PHASES_248', 'WRB_PHASES_249', 'WRB_PHASES_250', 'WRB_PHASES_251',
        'WRB_PHASES_252', 'WRB_PHASES_253', 'WRB_PHASES_254', 'WRB_PHASES_255',
        'WRB_PHASES_256', 'WRB_PHASES_257', 'WRB_PHASES_259', 'WRB_PHASES_260',
        'WRB_PHASES_261', 'WRB_PHASES_262', 'WRB_PHASES_263', 'WRB_PHASES_264',
        'WRB_PHASES_265', 'WRB_PHASES_266', 'WRB_PHASES_267', 'WRB_PHASES_270',
        'WRB_PHASES_271', 'WRB_PHASES_272', 'WRB_PHASES_274', 'WRB_PHASES_275',
        'WRB_PHASES_276', 'WRB_PHASES_277', 'WRB_PHASES_278', 'WRB_PHASES_279',
        'WRB_PHASES_280', 'WRB_PHASES_281', 'WRB_PHASES_282', 'WRB_PHASES_284',
        'WRB_PHASES_285', 'WRB_PHASES_286', 'WRB_PHASES_287', 'WRB_PHASES_289',
        'WRB_PHASES_290', 'WRB_PHASES_292', 'WRB_PHASES_293', 'WRB_PHASES_294',
        'WRB_PHASES_295', 'WRB_PHASES_298', 'WRB_PHASES_299', 'WRB_PHASES_300',
        'WRB_PHASES_301', 'WRB_PHASES_302', 'WRB_PHASES_303', 'WRB_PHASES_304',
        'WRB_PHASES_305', 'WRB_PHASES_306', 'WRB_PHASES_307', 'WRB_PHASES_308',
        'WRB_PHASES_310', 'WRB_PHASES_311', 'WRB_PHASES_312', 'WRB_PHASES_314',
        'WRB_PHASES_315', 'WRB_PHASES_316', 'WRB_PHASES_317', 'WRB_PHASES_318',
        'WRB_PHASES_319', 'WRB_PHASES_320', 'WRB_PHASES_321', 'WRB_PHASES_322',
        'WRB_PHASES_323', 'WRB_PHASES_324', 'WRB_PHASES_325', 'WRB_PHASES_326',
        'WRB_PHASES_327', 'WRB_PHASES_328', 'WRB_PHASES_329', 'WRB_PHASES_330',
        'WRB_PHASES_332', 'WRB_PHASES_333', 'WRB_PHASES_334', 'WRB_PHASES_335',
        'WRB_PHASES_336', 'WRB_PHASES_337', 'WRB_PHASES_338', 'WRB_PHASES_339',
        'WRB_PHASES_340', 'WRB_PHASES_341', 'WRB_PHASES_342', 'WRB_PHASES_343',
        'WRB_PHASES_344', 'WRB_PHASES_345', 'WRB_PHASES_346', 'WRB_PHASES_348',
        'WRB_PHASES_349', 'WRB_PHASES_350', 'WRB_PHASES_351', 'WRB_PHASES_353',
        'WRB_PHASES_354', 'WRB_PHASES_355', 'WRB_PHASES_356', 'WRB_PHASES_357',
        'WRB_PHASES_358', 'WRB_PHASES_359', 'WRB_PHASES_360', 'WRB_PHASES_361',
        'WRB_PHASES_362', 'WRB_PHASES_363', 'WRB_PHASES_367', 'WRB_PHASES_368',
        'WRB_PHASES_371', 'WRB_PHASES_372', 'WRB_PHASES_373', 'WRB_PHASES_374',
        'WRB_PHASES_376', 'WRB_PHASES_377', 'WRB_PHASES_378', 'WRB_PHASES_379',
        'WRB_PHASES_381', 'WRB_PHASES_382', 'WRB_PHASES_383', 'WRB_PHASES_384',
        'WRB_PHASES_386', 'WRB_PHASES_387', 'WRB_PHASES_388', 'WRB_PHASES_389',
        'WRB_PHASES_390', 'WRB_PHASES_391', 'WRB_PHASES_392', 'WRB_PHASES_393',
        'WRB_PHASES_394', 'WRB_PHASES_395', 'WRB_PHASES_396', 'WRB_PHASES_397',
        'WRB_PHASES_398', 'WRB_PHASES_399', 'WRB_PHASES_400', 'WRB_PHASES_401',
        'WRB_PHASES_402', 'WRB_PHASES_403', 'WRB_PHASES_405', 'WRB_PHASES_406',
        'WRB_PHASES_407', 'WRB_PHASES_408', 'WRB_PHASES_409', 'WRB_PHASES_410',
        'WRB_PHASES_411', 'WRB_PHASES_413', 'WRB_PHASES_415', 'WRB_PHASES_417',
        'WRB_PHASES_418', 'WRB_PHASES_419', 'WRB_PHASES_420', 'WRB_PHASES_422',
        'WRB_PHASES_424', 'WRB_PHASES_426', 'WRB_PHASES_427', 'WRB_PHASES_428',
        'WRB_PHASES_429', 'WRB_PHASES_430', 'WRB_PHASES_431', 'WRB_PHASES_432',
        'WRB_PHASES_433', 'WRB_PHASES_434', 'WRB_PHASES_435', 'WRB_PHASES_436',
        'WRB_PHASES_437', 'WRB_PHASES_438', 'WRB_PHASES_439', 'WRB_PHASES_440',
        'WRB_PHASES_441', 'WRB_PHASES_442', 'WRB_PHASES_443', 'WRB_PHASES_444',
        'WRB_PHASES_445', 'WRB_PHASES_446', 'WRB_PHASES_447', 'WRB_PHASES_448',
        'WRB_PHASES_449', 'WRB_PHASES_450', 'WRB_PHASES_451', 'WRB_PHASES_453',
        'WRB_PHASES_454', 'WRB_PHASES_455', 'WRB_PHASES_456', 'WRB_PHASES_457',
        'WRB_PHASES_458', 'WRB_PHASES_459', 'WRB_PHASES_460', 'WRB_PHASES_461',
        'WRB_PHASES_462', 'WRB_PHASES_463', 'WRB_PHASES_464', 'WRB_PHASES_466',
        'WRB_PHASES_467', 'WRB_PHASES_471', 'WRB_PHASES_472', 'WRB_PHASES_474',
        'WRB_PHASES_475', 'WRB_PHASES_476', 'WRB_PHASES_477', 'WRB_PHASES_478',
        'WRB_PHASES_479', 'WRB_PHASES_480', 'WRB_PHASES_481', 'WRB_PHASES_482',
        'WRB_PHASES_483', 'WRB_PHASES_484', 'WRB_PHASES_485', 'WRB_PHASES_486',
        'WRB_PHASES_487', 'WRB_PHASES_488', 'WRB_PHASES_489', 'WRB_PHASES_490',
        'WRB_PHASES_491', 'WRB_PHASES_492', 'WRB_PHASES_493', 'WRB_PHASES_494',
        'WRB_PHASES_495', 'WRB_PHASES_496', 'WRB_PHASES_497', 'WRB_PHASES_498',
        'WRB_PHASES_499', 'WRB_PHASES_500', 'WRB_PHASES_501', 'WRB_PHASES_502',
        'WRB_PHASES_503', 'WRB_PHASES_504', 'WRB_PHASES_506', 'WRB_PHASES_508',
        'WRB_PHASES_510', 'WRB_PHASES_511', 'WRB_PHASES_512', 'WRB_PHASES_513',
        'WRB_PHASES_514', 'WRB_PHASES_515', 'WRB_PHASES_516', 'WRB_PHASES_517',
        'WRB_PHASES_518', 'WRB_PHASES_519', 'WRB_PHASES_520', 'WRB_PHASES_521',
        'WRB_PHASES_522', 'WRB_PHASES_523', 'WRB_PHASES_524', 'WRB_PHASES_525',
        'WRB_PHASES_526', 'WRB_PHASES_527', 'WRB_PHASES_528', 'WRB_PHASES_529',
        'WRB_PHASES_530', 'WRB_PHASES_531', 'WRB_PHASES_532', 'WRB_PHASES_533',
        'WRB_PHASES_534', 'WRB_PHASES_535', 'WRB_PHASES_536', 'WRB_PHASES_537',
        'WRB_PHASES_538', 'WRB_PHASES_539', 'WRB_PHASES_540', 'WRB_PHASES_541',
        'WRB_PHASES_542', 'WRB_PHASES_543', 'WRB_PHASES_544', 'WRB_PHASES_546',
        'WRB_PHASES_548', 'WRB_PHASES_549', 'WRB_PHASES_551', 'WRB_PHASES_552',
        'WRB_PHASES_553', 'WRB_PHASES_555', 'WRB_PHASES_556', 'WRB4_1', 'WRB4_2',
        'WRB4_3', 'WRB4_4', 'WRB4_6', 'WRB4_8', 'WRB4_9', 'WRB4_10', 'WRB4_11',
        'WRB4_12', 'WRB4_13', 'WRB4_16', 'WRB4_17', 'WRB4_18', 'WRB4_19', 'WRB4_20',
        'WRB4_24', 'WRB4_27', 'WRB4_28', 'WRB4_29', 'WRB4_30', 'WRB4_31', 'WRB4_32',
        'WRB4_33', 'WRB4_34', 'WRB4_35', 'WRB4_36', 'WRB4_37', 'WRB4_38', 'WRB4_39',
        'WRB4_40', 'WRB4_41', 'WRB4_42', 'WRB4_43', 'WRB4_44', 'WRB4_45', 'WRB4_46',
        'WRB4_47', 'WRB4_48', 'WRB4_49', 'WRB4_50', 'WRB4_51', 'WRB4_52', 'WRB4_53',
        'WRB4_54', 'WRB4_55', 'WRB4_56', 'WRB4_57', 'WRB4_58', 'WRB4_59', 'WRB4_60',
        'WRB4_61', 'WRB4_62', 'WRB4_63', 'WRB4_64', 'WRB4_65', 'WRB4_66', 'WRB4_67',
        'WRB4_68', 'WRB4_70', 'WRB4_71', 'WRB4_73', 'WRB4_74', 'WRB4_75', 'WRB4_76',
        'WRB4_79', 'WRB4_80', 'WRB4_81', 'WRB4_82', 'WRB4_84', 'WRB4_85', 'WRB4_86',
        'WRB4_88', 'WRB4_89', 'WRB4_90', 'WRB4_91', 'WRB4_92', 'WRB4_93', 'WRB4_94',
        'WRB4_96', 'WRB4_97', 'WRB4_98', 'WRB4_99', 'WRB4_100', 'WRB4_101', 'WRB4_102',
        'WRB4_103', 'WRB4_104', 'WRB4_105', 'WRB4_106', 'WRB4_107', 'WRB4_108',
        'WRB4_109', 'WRB4_110', 'WRB4_111', 'WRB4_112', 'WRB4_113', 'WRB4_114',
        'WRB4_115', 'WRB4_116', 'WRB4_117', 'WRB4_118', 'WRB4_119', 'WRB4_121',
        'WRB4_123', 'WRB4_125', 'WRB4_126', 'WRB4_127', 'WRB4_128', 'WRB4_129',
        'WRB4_130', 'WRB4_131', 'WRB4_134', 'WRB4_135', 'WRB4_136', 'WRB4_137',
        'WRB4_138', 'WRB4_139', 'WRB4_140', 'WRB4_141', 'WRB4_142', 'WRB4_143',
        'WRB4_144', 'WRB4_145', 'WRB4_146', 'WRB4_148', 'WRB4_149', 'WRB4_150',
        'WRB4_152', 'WRB4_153', 'WRB4_154', 'WRB4_155', 'WRB4_156', 'WRB4_157',
        'WRB4_158', 'WRB4_159', 'WRB4_160', 'WRB4_161', 'WRB4_162', 'WRB4_164',
        'WRB4_165', 'WRB4_167', 'WRB4_168', 'WRB4_169', 'WRB4_170', 'WRB4_171',
        'WRB4_172', 'WRB4_173', 'WRB4_174', 'WRB4_175', 'WRB4_177', 'WRB4_178',
        'WRB4_179', 'WRB4_180', 'WRB4_181', 'WRB4_184', 'WRB4_185', 'WRB4_187',
        'WRB4_188', 'WRB4_189', 'WRB4_190', 'WRB4_191', 'WRB2_1', 'WRB2_2', 'WRB2_3',
        'WRB2_4', 'WRB2_5', 'WRB2_6', 'WRB2_7', 'WRB2_8', 'WRB2_9', 'WRB2_10', 'WRB2_11',
        'WRB2_12', 'WRB2_13', 'WRB2_14', 'WRB2_15', 'WRB2_16', 'WRB2_17', 'WRB2_18',
        'WRB2_19', 'WRB2_20', 'WRB2_21', 'WRB2_22', 'WRB2_23', 'WRB2_24', 'WRB2_25',
        'WRB2_26', 'WRB2_27', 'WRB2_28', 'WRB2_29', 'WRB2_30', 'WRB2_31', 'WRB2_32',
        'WRB2_33', 'WRB2_34', 'WRB2_35', 'FAO90_1', 'FAO90_2', 'FAO90_3', 'FAO90_4',
        'FAO90_5', 'FAO90_6', 'FAO90_7', 'FAO90_8', 'FAO90_9', 'FAO90_10', 'FAO90_11',
        'FAO90_12', 'FAO90_14', 'FAO90_15', 'FAO90_16', 'FAO90_17', 'FAO90_18',
        'FAO90_19', 'FAO90_20', 'FAO90_21', 'FAO90_22', 'FAO90_23', 'FAO90_25',
        'FAO90_26', 'FAO90_27', 'FAO90_28', 'FAO90_29', 'FAO90_30', 'FAO90_31',
        'FAO90_32', 'FAO90_33', 'FAO90_34', 'FAO90_35', 'FAO90_37', 'FAO90_38',
        'FAO90_39', 'FAO90_40', 'FAO90_41', 'FAO90_42', 'FAO90_43', 'FAO90_44',
        'FAO90_45', 'FAO90_46', 'FAO90_47', 'FAO90_48', 'FAO90_49', 'FAO90_51',
        'FAO90_52', 'FAO90_53', 'FAO90_54', 'FAO90_55', 'FAO90_56', 'FAO90_57',
        'FAO90_58', 'FAO90_59', 'FAO90_60', 'FAO90_61', 'FAO90_62', 'FAO90_63',
        'FAO90_64', 'FAO90_65', 'FAO90_66', 'FAO90_67', 'FAO90_68', 'FAO90_69',
        'FAO90_70', 'FAO90_71', 'FAO90_72', 'FAO90_73', 'FAO90_74', 'FAO90_75',
        'FAO90_76', 'FAO90_77', 'FAO90_79', 'FAO90_81', 'FAO90_82', 'FAO90_83',
        'FAO90_84', 'FAO90_85', 'FAO90_86', 'FAO90_87', 'FAO90_88', 'FAO90_89',
        'FAO90_90', 'FAO90_91', 'FAO90_92', 'FAO90_93', 'FAO90_94', 'FAO90_95',
        'FAO90_97', 'FAO90_98', 'FAO90_99', 'FAO90_100', 'FAO90_101', 'FAO90_102',
        'FAO90_103', 'FAO90_104', 'FAO90_105', 'FAO90_106', 'FAO90_107', 'FAO90_108',
        'FAO90_109', 'FAO90_110', 'FAO90_111', 'FAO90_112', 'FAO90_113', 'FAO90_114',
        'FAO90_115', 'FAO90_116', 'FAO90_117', 'FAO90_118', 'FAO90_119', 'FAO90_120',
        'FAO90_121', 'FAO90_122', 'FAO90_123', 'FAO90_124', 'FAO90_126', 'FAO90_127',
        'FAO90_128', 'FAO90_129', 'FAO90_131', 'FAO90_132', 'FAO90_133', 'FAO90_134',
        'FAO90_135', 'FAO90_136', 'FAO90_137', 'FAO90_138', 'FAO90_139', 'FAO90_140',
        'FAO90_141', 'FAO90_143', 'FAO90_144', 'FAO90_145', 'FAO90_146', 'FAO90_147',
        'FAO90_148', 'FAO90_149', 'FAO90_150', 'FAO90_151', 'FAO90_152', 'FAO90_153',
        'FAO90_154', 'FAO90_155', 'FAO90_156', 'FAO90_157', 'FAO90_158', 'FAO90_159',
        'FAO90_160', 'FAO90_161', 'FAO90_162', 'FAO90_163', 'FAO90_164', 'FAO90_165',
        'FAO90_166', 'FAO90_167', 'FAO90_168', 'FAO90_170', 'FAO90_171', 'FAO90_172',
        'FAO90_173', 'FAO90_174', 'FAO90_175', 'FAO90_176', 'FAO90_177', 'FAO90_178',
        'FAO90_179', 'FAO90_180', 'FAO90_181', 'FAO90_182', 'FAO90_183', 'FAO90_184',
        'FAO90_186', 'FAO90_187', 'FAO90_188', 'FAO90_189', 'FAO90_190', 'FAO90_193',
        'DRAINAGE_2', 'DRAINAGE_4', 'DRAINAGE_5', 'DRAINAGE_6', 'DRAINAGE_7', 'AWC_1',
        'AWC_2', 'AWC_3', 'AWC_4', 'AWC_5', 'AWC_7', 'LAYER_0', 'LAYER_1', 'LAYER_2',
        'LAYER_3', 'LAYER_4', 'LAYER_5', 'LAYER_6'
    ]

    # float_features = df[features].astype(float)

    y = df["ORG_CARBON"].astype(float).values

    # categorical_feature = df[[""]]
    # encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')  # Initialize encoder, sparse=False returns a 2D array
    # one_hot_encoded_feature = encoder.fit_transform(categorical_feature)

    # x = np.concatenate([float_features.values, one_hot_encoded_feature], axis=1)
    # x = np.concatenate([float_features])
    exclude = ['ID', 'HWSD2_SMU_ID', 'WISE30s_SMU_ID', 'HWSD1_SMU_ID', 'COVERAGE', 'SEQUENCE', 'SHARE', 'NSC_MU_SOURCE1', 'NSC_MU_SOURCE2', 'ROOT_DEPTH', 'ORG_CARBON']
    x = df[features].astype(float).to_numpy()
    #x = df.drop(columns=exclude).astype(float).to_numpy()

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
