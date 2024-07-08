# Real-time Fungus Detection App using a Simple Neural Network from scratch

This is a part of my CSE 3200 course. Current project involves creating a web application for real-time fungus detection using a simple neural network. The application is built with Streamlit, an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science.

## Project Structure

1. **User Interface (UI) with Streamlit:**
   - The main user interface allows users to choose between training a new model or testing an existing one.
   - Users can upload images for training or testing directly through the web interface.
   - Labels for training images can be assigned using radio buttons, and the training process can be initiated with a button click.

2. **Dataset:** 
    - The original dataset can be found in this link from [UCI](https://www.archive.ics.uci.edu/dataset/773/defungi).
    - But the dataset used in this project is from in [this kaggle link](https://www.kaggle.com/datasets/anshtanwar/microscopic-fungi-images).
    - The dataset is preprocessed by running ```preprocess.py```.

3. **Training the Model:**
   - Uploaded images are preprocessed by converting them to grayscale, resizing, normalizing, and flattening.
   - A simple neural network is trained on these preprocessed images. The network consists of:
     - An input layer.
     - A hidden layer with ReLU activation.
     - An output layer with sigmoid activation.
   - The training process includes forward propagation, loss calculation, and backpropagation with gradient descent optimization.
   - The trained model is saved to a specified folder.

4. **Testing the Model:**
   - Users can upload an image for testing, and the application will predict whether the image indicates the presence of fungus (specifically the 'H5 category').
   - The model's performance is evaluated using a confusion matrix and an ROC curve, which are displayed in the app.

5. **Store Model(s):**
   - The ```run.ipynb``` notebook will save the model in each epoch in ```models-store``` folder.
   - The ```app.py``` file will save the model for real time training and testing in ```real_time_model``` folder.


## Key Features

- **File Uploads:**
  - Allows multiple file uploads for training and single file upload for testing.
- **Preprocessing:**
  - Converts images to grayscale, resizes them to 179x179 pixels, normalizes pixel values, and flattens the images.
  - By running ```preprocess.py``` the dataset is preprocessed into ```preprocessed_dataset``` folder.
- **Model Training:**
  - Trains a neural network with one hidden layer.
  - Uses ReLU activation for the hidden layer and sigmoid activation for the output layer.
  - Implements gradient descent optimization.
- **Model Evaluation:**
  - Displays a confusion matrix and an ROC curve to evaluate the model's performance.
- **Model Persistence:**
  - Saves the trained model to disk and loads it for testing.

## Libraries Used

- **Streamlit:** For creating the web application.
- **OpenCV (cv2):** For image processing.
- **NumPy:** For numerical operations.
- **scikit-learn:** For calculating the confusion matrix and ROC curve.
- **matplotlib:** For plotting the ROC curve.

## How to Use
- To install the required dependencies just run the following command:

```bash
pip install -r requirements.txt
```

- To run these apps, you must install Streamlit on your device. Once installed, navigate to the directory containing the ```app.py``` and run it using the following command:

```bash
python -m streamlit run app.py
```

- or simply by running the MakeFile:
```bash
make
```

1. **Training:**
   - Select "Train" from the dropdown menu.
   - Upload images and assign labels (0 for H5 and 1 for H6) for each image.
   - Click the "Train Model" button to start training. The progress and loss will be displayed during training.
2. **Testing:**
   - Select "Test" from the dropdown menu.
   - Upload an image for testing.
   - Click the "Test Model" button to see the prediction and performance metrics.

This project provides a simple yet effective framework for real-time fungus detection using a neural network, making it easy for users to train and test models with minimal setup.
