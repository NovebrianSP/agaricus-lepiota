# Mushroom Streamlit App

This project is a Streamlit application that predicts mushroom edibility based on various features. It utilizes a trained Random Forest model to provide predictions based on user input.

## Project Structure

- `src/app.py`: Contains the main application logic using Streamlit. It loads the trained Random Forest model and label encoders, processes user input, and displays predictions based on the input data.
- `agaricus-lepiota-mapped.csv`: The mapped dataset used for training the model, including features and their corresponding classes.
- `rf_mushroom.pkl`: Stores the trained Random Forest model for making predictions on user input.
- `le_dict.pkl`: Contains the label encoders for categorical features, ensuring consistent encoding of user input.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd mushroom-streamlit-app
   ```

2. Install the required packages:
   ```
   pip install streamlit scikit-learn pandas
   ```

3. Run the Streamlit application:
   ```
   streamlit run src/app.py
   ```

## Usage

- Open the application in your web browser (usually at `http://localhost:8501`).
- Fill in the input fields with the appropriate values for the mushroom features.
- Submit the form to receive a prediction on whether the mushroom is edible or poisonous.

## Additional Information

- Ensure that the `agaricus-lepiota-mapped.csv`, `rf_mushroom.pkl`, and `le_dict.pkl` files are present in the project directory for the application to function correctly.
- The model is trained on the Agaricus and Lepiota mushroom dataset, which includes various features related to mushroom characteristics.