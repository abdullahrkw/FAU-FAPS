import pandas as pd
from sklearn.model_selection import train_test_split

# Load CSV file into a DataFrame
df = pd.read_csv('train_multiview_img_labels_paths.csv')

# Split data into train and test (80% train, 20% test)
train_data, test_data = train_test_split(df, test_size=0.20, random_state=42)

# Split train data into train and validation (80% train, 20% validation)
train_data, val_data = train_test_split(train_data, test_size=0.20, random_state=42)

# Save the split data into separate CSV files
train_data.to_csv('train.csv', index=False, header=["view1", "view2", "label", "motor"])
test_data.to_csv('test.csv', index=False, header=["view1", "view2", "label", "motor"])
val_data.to_csv('val.csv', index=False, header=["view1", "view2", "label", "motor"])