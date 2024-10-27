import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Optional: for scaling numerical data
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_csv('Content_recommend.csv')

# Display the first few rows to check the data
df.head()

import joblib  # For saving and loading models and scalers

# Encode categorical columns
label_encoder_subject = LabelEncoder()
label_encoder_recommendation = LabelEncoder()

df['Subject'] = label_encoder_subject.fit_transform(df['Subject'])
df['Recommendation'] = label_encoder_recommendation.fit_transform(df['Recommendation'])

# Optional: Scale numerical features
scaler = StandardScaler()
df[['Course Score', 'Learning Score', 'Quiz Score']] = scaler.fit_transform(df[['Course Score', 'Learning Score', 'Quiz Score']])

# Save the encoders and scaler
joblib.dump(label_encoder_subject, 'label_encoder_subject.pkl')
joblib.dump(label_encoder_recommendation, 'label_encoder_recommendation.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Display the processed data
print(df.head())


# Separate features and target variable
X = df.drop('Recommendation', axis=1)
y = df['Recommendation']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data: {X_train.shape}, Testing data: {X_test.shape}")

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print(classification_report(y_test, y_pred))

# Optional: Confusion matrix
print(confusion_matrix(y_test, y_pred))


# Save the model
joblib.dump(model, 'classification_model.pkl')

