import streamlit as st
import cv2
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt


MODEL_FOLDER = "/home/f4h1m/Documents/CSE 3200/Binary_classification/project-2/real_time_model"


def main():
    st.title("ML Real-time Fungus Detection")

    option = st.selectbox("Choose an option", ("Train", "Test"))
    
    if option == "Train":
        uploaded_files = st.file_uploader("Upload images for training...", accept_multiple_files=True)
        labels = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                label = st.radio(f"Label for {uploaded_file.name}", (0, 1), key=uploaded_file.name)
                labels.append(label)
            if st.button("Train Model"):
                train_model(uploaded_files, labels)
                st.write("Training completed!")

    if option == "Test":
        uploaded_image = st.file_uploader("Upload an image for testing...")
        if st.button("Test Model") and uploaded_image:
            if "model.npz" in os.listdir(MODEL_FOLDER):
                model = load_model(MODEL_FOLDER)
                if model is not None:
                    label = test_image(model, uploaded_image)
                    st.write(f"The image indicates of {'H5 category' if label > 0.5 else 'not H5 category'}.")
                    show_metrics(model)
            else:
                st.write("Model not trained yet. Please train the model first.")

def train_model(uploaded_files, labels):
    # Preprocess and prepare data
    X_train, y_train = preprocess_data(uploaded_files, labels)
    
    # Train neural network
    model = train_neural_network(X_train, y_train)
    
    # Save model
    save_model(model, MODEL_FOLDER)

def preprocess_data(uploaded_files, labels):
    X_train = []
    y_train = []

    for i, uploaded_file in enumerate(uploaded_files):
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(image, (179, 179))
        normalized_image = resized_image.astype(np.float32) / 255.0
        flattened_image = normalized_image.flatten()
        X_train.append(flattened_image)
        y_train.append(labels[i])

    X_train = np.array(X_train)
    y_train = np.array(y_train).reshape(-1, 1)

    return X_train, y_train

def train_neural_network(X_train, y_train):
    input_size = X_train.shape[1]
    hidden_size = 128
    output_size = 1
    learning_rate = 0.001
    num_epochs = 100

    # Initialize weights and biases
    np.random.seed(0)
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))

    # Training loop
    for epoch in range(num_epochs):
        # Forward propagation
        Z1 = np.dot(W1, X_train.T) + b1
        A1 = np.maximum(0, Z1)  # ReLU activation
        Z2 = np.dot(W2, A1) + b2
        A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation

        epsilon = 1e-10  # Small epsilon value to prevent division by zero
        loss = -np.mean(y_train.T * np.log(A2 + epsilon) + (1 - y_train.T) * np.log(1 - A2 + epsilon))

        # Backpropagation
        dZ2 = A2 - y_train.T
        dW2 = np.dot(dZ2, A1.T) / X_train.shape[0]
        db2 = np.sum(dZ2, axis=1, keepdims=True) / X_train.shape[0]
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = dA1 * (Z1 > 0)
        dW1 = np.dot(dZ1, X_train) / X_train.shape[0]
        db1 = np.sum(dZ1, axis=1, keepdims=True) / X_train.shape[0]

        # Gradient descent
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        # Print loss every 10 epochs
        if epoch % 10 == 0:
            st.write(f"Epoch {epoch}, Loss: {loss}")

    return (W1, b1, W2, b2)

def save_model(model, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.savez(os.path.join(folder, "model.npz"), W1=model[0], b1=model[1], W2=model[2], b2=model[3])
    st.write("Model saved successfully!")

def load_model(folder):
    try:
        model = np.load(os.path.join(folder, "model.npz"))
        return model['W1'], model['b1'], model['W2'], model['b2']
    except:
        st.write("Failed to load model.")
        return None

def test_image(model, uploaded_image):
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (179, 179))
    normalized_image = resized_image.astype(np.float32) / 255.0
    flattened_image = normalized_image.flatten()

    W1, b1, W2, b2 = model

    Z1 = np.dot(W1, flattened_image) + b1
    A1 = np.maximum(0, Z1)
    Z2 = np.dot(W2, A1) + b2
    prediction = 1 / (1 + np.exp(-Z2))
    return np.mean(prediction)

def show_metrics(model):
    # Dummy data for demonstration purposes
    y_true = np.array([0, 0, 1, 1]) 
    y_scores = np.array([0.1, 0.4, 0.35, 0.8]) 

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_scores > 0.5)
    st.write("Confusion Matrix:")
    st.write(cm)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    st.write("ROC Curve:")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
    