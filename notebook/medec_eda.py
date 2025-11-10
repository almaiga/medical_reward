import pandas as pd

medec_train_path = "data_copy/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv"
medec_train = pd.read_csv(medec_train_path)
print("Shape")
print(medec_train.shape)
print("Columns")
print(medec_train.columns)
print(medec_train['Error Type'].value_counts())