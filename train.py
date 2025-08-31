import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

# --- Main Training Function ---
def train_model():
    """
    Loads data, trains a RandomForestClassifier, evaluates its accuracy,
    and saves the trained model to a file.
    """
    print("Starting model training process...")
    start_time = time.time()

    # 1. Load the Dataset
    try:
        df = pd.read_csv("dataset.csv")
        print(f"✅ Dataset 'dataset.csv' loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print("\n[ERROR] --- 'dataset.csv' not found! ---")
        print("Please make sure your phishing dataset is in the same folder.")
        return

    # 2. Prepare the Data
    # Assumes the last column is the target variable ('Result')
    X = df.iloc[:, :-1]  # Features (all columns except the last one)
    y = df.iloc[:, -1]   # Target (the last column)

    if X.shape[1] != 30:
        print(f"\n[ERROR] --- Incorrect number of features! ---")
        print(f"Expected 30 feature columns, but got {X.shape[1]}. Please check 'dataset.csv'.")
        return

    print(f"✅ Data prepared: {X.shape[1]} features and 1 target variable.")

    # 3. Split Data into Training and Testing Sets
    # stratify=y ensures the same proportion of classes in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("✅ Data split into 80% training and 20% testing sets.")

    # 4. Train the RandomForestClassifier Model
    print("⏳ Training the RandomForestClassifier model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("✅ Model training completed.")

    # 5. Evaluate the Model's Performance
    print("📊 Evaluating model performance on the test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")

    # 6. Save the Trained Model
    joblib.dump(model, "phishing_model.pkl")
    end_time = time.time()
    print("-" * 50)
    print(f"✅ Model successfully trained and saved as 'phishing_model.pkl'")
    print(f"   Total training time: {end_time - start_time:.2f} seconds")
    print("-" * 50)

if __name__ == "__main__":
    train_model()

