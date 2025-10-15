import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf
from tensorflow import keras

# Adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# READ CSV
rice_dataset_raw = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv")

# Address these properties
rice_dataset = rice_dataset_raw[[
    'Area',
    'Perimeter',
    'Major_Axis_Length',
    'Minor_Axis_Length',
    'Eccentricity',
    'Convex_Area',
    'Extent',
    'Class',
]]

rice_dataset.describe()

print(
    f'The shortest grain is {rice_dataset.Major_Axis_Length.min():.1f}px ,'
    f'the longest grain is {rice_dataset.Major_Axis_Length.max():.1f}px ,'
)
print(
    f'The smallest rice grain has an area of {rice_dataset.Area.min()} px '
    f'the largest rice grain has an area of {rice_dataset.Area.max()} px '
)
print(
    f'The largest rice grain, with a perimeter of {rice_dataset.Perimeter.max():.1f} px,'
    f'is ~{(rice_dataset.Perimeter.max() - rice_dataset.Perimeter.min())/rice_dataset.Perimeter.std():.1f} standard '
    f'deviations {rice_dataset.Perimeter.std():.1f} from the mean {rice_dataset.Perimeter.mean():.1f}'
)
print(
    f'This calculated as {rice_dataset.Perimeter.max():.1f} - {rice_dataset.Perimeter.min():.1f} / {rice_dataset.Perimeter.std():.1f} '
    f'equals {(rice_dataset.Perimeter.max() - rice_dataset.Perimeter.min()) / rice_dataset.Perimeter.std():.1f}'
)

print('\n')
print("DATA GATHERING PRINTED!✅")

# Get A 3D data scatter
# Rice eccentricity = numerical measure of the elongation of a rice grain
# A higher value indicates a more elongated, less-circular shape

for x_axis_data, y_axis_data in [
    ('Area', 'Eccentricity'),
    ('Convex_Area', 'Perimeter'),
    ('Major_Axis_Length', 'Minor_Axis_Length'),
    ('Perimeter', 'Extent'),
    ('Eccentricity', 'Major_Axis_Length'),
]:
    px.scatter(
        rice_dataset,
        x=x_axis_data,
        y=y_axis_data,
        color='Class'
    ).show()

x_axis_data = 'Eccentricity'
y_axis_data = 'Area'
z_axis_data = 'Extent'

px.scatter_3d(
    rice_dataset,
    x=x_axis_data,
    y=y_axis_data,
    z=z_axis_data,
    color='Class'
).show()

# Normalizing the data
# into a new DataFrame named normalized_dataset
normalized_dataset = rice_dataset.copy()
feature_mean = rice_dataset.mean(numeric_only=True)
feature_std = rice_dataset.std(numeric_only=True)
numerical_features = rice_dataset.select_dtypes('number').columns
normalized_dataset[numerical_features] = (rice_dataset[numerical_features] - feature_mean) / feature_std
normalized_dataset['Class'] = rice_dataset['Class']

keras.utils.set_random_seed(42)

# Training the model
# Creating a boolean column for Cammeo = 1, Osmancik = 0
normalized_dataset['Class_Bool'] = (
    normalized_dataset['Class'] == 'Cammeo'
).astype(int)

# each time sample 10 rice grains
normalized_dataset.sample(10)

# Splitting dataset into train/validation/test
number_samples = len(normalized_dataset)
index_80th = round(number_samples * 0.8)
index_90th = round(number_samples * 0.9)

# In order to get the same data in different order
shuffled_dataset = normalized_dataset.sample(frac=1, random_state=100)

# Used to train the model
train_data = shuffled_dataset.iloc[0:index_80th]

# Used to tune hyperparameters and monitor overfitting
validation_data = shuffled_dataset.iloc[index_80th:index_90th]

# Used to evaluate final model performance
test_data = shuffled_dataset.iloc[index_90th:]

# Show the first five rows of the last split
test_data.head()

# Preventing label leakage
label_columns = ['Class', 'Class_Bool']

# Removing the 'Class' and 'Class_Bool' labels from the table
train_features = train_data.drop(columns=label_columns)
# The training labels should be 1 for Cammeo 0 for Osmancik
train_labels = train_data['Class_Bool'].to_numpy()

validation_features = validation_data.drop(columns=label_columns)
validation_labels = validation_data['Class_Bool'].to_numpy()

test_features = test_data.drop(columns=label_columns)
test_labels = test_data['Class_Bool'].to_numpy()

# Name of the features we'll train our model on.
input_features = [
    'Eccentricity',
    'Major_Axis_Length',
    'Area',
]

# Creating the model
def create_model(learning_rate=0.001):
    """Create and compile a simple classification model"""
    model = keras.Sequential([
        keras.layers.Input(shape=(len(input_features),)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
            keras.metrics.Precision(name='precision', thresholds=0.5),
            keras.metrics.Recall(name='recall', thresholds=0.5),
            keras.metrics.AUC(name='auc'),
        ],
    )

    return model

# Prepare the data
X_train = train_features[input_features].to_numpy()
X_val = validation_features[input_features].to_numpy()
X_test = test_features[input_features].to_numpy()

# Create the model
model = create_model(learning_rate=0.001)

# Train the model
history = model.fit(
    X_train,
    train_labels,
    batch_size=100,
    epochs=60,
    validation_data=(X_val, validation_labels),
    verbose=1
)

# Plot metrics vs. epochs
import plotly.graph_objects as go

metrics_df = pd.DataFrame(history.history)
fig = go.Figure()
for metric in ['accuracy', 'precision', 'recall']:
    fig.add_trace(go.Scatter(y=metrics_df[metric], mode='lines', name=metric))
    fig.add_trace(go.Scatter(y=metrics_df[f'val_{metric}'], mode='lines', name=f'val_{metric}'))
fig.update_layout(title="Training vs Validation Metrics", xaxis_title="Epoch", yaxis_title="Value")
fig.show()

# Plot AUC
px.line(metrics_df, y=['auc', 'val_auc'], title="AUC (Area Under Curve)").show()

# Evaluate on test data
test_results = model.evaluate(X_test, test_labels, verbose=0)
print("\nFinal Test Results:")
for name, value in zip(model.metrics_names, test_results):
    print(f"{name}: {value:.4f}")

print("\n✅ Model training complete and results plotted successfully!")
