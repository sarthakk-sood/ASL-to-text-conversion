import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle

def main():
    try:
        df = pd.read_csv("asl_landmarks.csv")
    except FileNotFoundError:
        print("Dataset not found. Please run extract_features.py first.")
        return

    print("Class distribution before balancing:")
    print(df['label'].value_counts())

    # 1. Balance the dataset via Oversampling!
    # M and N have very few samples because MediaPipe struggles with closed fists.
    # We will duplicate their rows so the model doesn't ignore them.
    max_count = df['label'].value_counts().max()
    dfs = []
    for label, group in df.groupby('label'):
        # Oversample each class to have the 'max_count' number of samples
        oversampled_group = group.sample(n=max_count, replace=True, random_state=42)
        dfs.append(oversampled_group)
    
    df_balanced = pd.concat(dfs, ignore_index=True)
    print("\nClass distribution after balancing:")
    print(df_balanced['label'].value_counts())

    X = df_balanced.drop('label', axis=1)
    y = df_balanced['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nTraining Advanced MLP Classifier...")
    # 2. Add more layers and parameters to capture complex nuances between G, M, and N
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64), 
        max_iter=1000, 
        activation='relu', 
        alpha=0.001,         # L2 penalty to prevent overfitting on the oversampled distinct data
        learning_rate_init=0.001,
        random_state=42
    )
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nValidation Accuracy: {acc:.4f}\n")
    
    with open("mlp_asl_model.pkl", "wb") as f:
        pickle.dump(mlp, f)
    print("Model saved to mlp_asl_model.pkl")

if __name__ == "__main__":
    main()
