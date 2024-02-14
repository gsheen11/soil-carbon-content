from mdb_parser import MDBParser
import pandas as pd

db = MDBParser(file_path='HWSD2.mdb')
print(db.tables)

table = db.get_table('HWSD2_LAYERS_METADATA')
table_df = pd.DataFrame(table)
print(table_df)
# table_df.to_csv(f'{name}.csv', index=False)

table = db.get_table('HWSD2_LAYERS')
table_df = pd.DataFrame(table)
print(table_df.head(50))


# # Convert tables to CSV files
# for name in db.tables:
#     table = db.get_table(name)
#     table_df = pd.DataFrame(table)
#     table_df.to_csv(f'{name}.csv', index=False)