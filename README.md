# Arabic Sign Language Recognition

A deep learning project for recognizing Arabic sign language from video sequences using hand and body landmarks.

## Project Overview

This project aims to recognize Arabic sign language expressions by extracting spatial and temporal features from sign language videos and feeding them into a deep learning model.

The preprocessing pipeline uses MediaPipe Holistic to extract hand and body landmarks. The resulting sequences are normalized, resampled to a fixed number of frames, and prepared for model training.

## Features

- Hand and body landmark extraction using MediaPipe Holistic.
- Spatial normalization of hand and pose landmarks.
- Temporal resampling to a fixed sequence length.
- Data augmentation to increase training data diversity.
- Deep learning model combining CNN, Bidirectional LSTM, and Multi-Head Attention.
- Training callbacks for early stopping, learning rate reduction, and best-model checkpointing.
- Model evaluation using test accuracy and a confusion matrix.

## Model Architecture

The model combines several components to learn spatial and temporal patterns from sign language sequences:

- **1D Convolutional Neural Networks (CNN):** Extract local patterns from sequential landmark features.
- **Bidirectional LSTM (BiLSTM):** Learn temporal relationships across frames.
- **Multi-Head Attention:** Capture relationships between different time steps.
- **Dense Layers:** Produce the final classification output.

The model is implemented using TensorFlow and Keras.

## Data Preprocessing

Each video sequence is represented using 225 features per frame:

- Left hand: 21 landmarks × 3 coordinates.
- Right hand: 21 landmarks × 3 coordinates.
- Body pose: 33 landmarks × 3 coordinates.

Each sequence is resampled to 40 frames, producing an input shape of:

`40 × 225`

The preprocessing pipeline includes landmark extraction, normalization, temporal resampling, dataset splitting, and training data augmentation.

## Dataset Organization

The project uses a signer-based split:

- **Training:** Person 01 and Person 02.
- **Validation:** A subset of the training data.
- **Testing:** Person 03.

This setup is intended to evaluate how well the model generalizes to a signer not included in the training data.

The dataset files are not included in this repository.

## Project Structure

```text
SLT-BETWEEN-DEAF-MUTE-MODEL/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── extract_keypoints.py
│   ├── preprocessing.py
│   ├── build_dataset.py
│   ├── augmentation.py
│   └── augment_dataset.py
├── notebooks/
│   └── Training.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```



## Training

Model training is performed in the `Training.ipynb` notebook using TensorFlow and Keras.

The notebook loads the prepared training, validation, and test arrays, converts class labels to categorical format, builds and compiles the model, and trains it for up to 50 epochs with a batch size of 32.

The training process uses:

- Early stopping based on validation loss.
- Model checkpointing to save the best model.
- Learning rate reduction when validation performance stops improving.

## Evaluation

The model is evaluated on the test set using:

- Test loss.
- Test categorical accuracy.
- A confusion matrix to compare predicted and true classes.

No performance results are reported here because numerical evaluation results have not been included in this repository description.

## Technologies Used

- Python
- NumPy
- MediaPipe
- OpenCV
- TensorFlow / Keras
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook / Google Colab

## Running the Project

1. Clone the repository.
2. Install the required Python packages listed in `requirements.txt`.
3. Prepare the video dataset in the expected directory structure.
4. Extract hand and body landmarks.
5. Run the preprocessing and dataset-building pipeline.
6. Apply data augmentation to the training data.
7. Open `Training.ipynb` and run the training and evaluation cells.

The prepared dataset files must be available in the locations expected by the scripts and notebook before running the corresponding stages.

## Future Improvements

- Improve generalization across different signers and recording conditions.
- Explore additional temporal modeling techniques.
- Expand the dataset with more signers and sign classes.
- Investigate real-time sign recognition.

## Author

**Mary Hussain**

GitHub: [maryhussain](https://github.com/maryhussain)
