# Ferranalytics

# Predicting % Silica Concentrate Using Machine Learning and Deep Learning  

This project aims to predict the **% Silica Concentrate** in mineral processing using advanced machine learning (Decision Tree Regressor) and deep learning (Artificial Neural Network) models.  

## Problem Statement  
Silica concentrate is a critical metric in mineral processing, and accurately predicting it can improve efficiency and quality control in the production process. This project tackles this regression problem using both traditional machine learning and modern deep learning approaches.

## Workflow Overview  
1. **Data Preprocessing**  
   - Loaded datasets and handled missing values.  
   - Dropped unnecessary columns (e.g., `date`) and standardized numerical formats.  
   - Split the data into features (`X`) and target (`y`).

2. **Model Development**  
   - **Artificial Neural Network (ANN):**  
     - A fully connected neural network with:
       - Input Layer: 22 neurons (features)
       - Hidden Layer: 14 neurons with ReLU activation
       - Output Layer: 1 neuron with linear activation (regression).  
     - Trained with Adam optimizer and Mean Squared Error (MSE) as the loss function.  

   - **Decision Tree Regressor (DTR):**  
     - A tree-based machine learning model trained on the same data for comparison with ANN.  

3. **Evaluation**  
   - Metrics used: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R² score.  
   - Predictions were exported to CSV files for comparison and analysis.  

## File Structure  
- `train.csv`: Training dataset containing features and target variable (% Silica Concentrate).  
- `test.csv`: Test dataset for which predictions were made.  
- `main.py`: Python script containing the complete code for preprocessing, training, and predictions.  
- `requirements.txt`: List of dependencies required to run the project.  
- `README.md`: This file, documenting the project.  
- `answer2.csv`: Predictions made using the ANN model.  
- `answer3_updated.csv`: Predictions made using the Decision Tree Regressor.  

## How to Run the Code  
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/silica-concentration-prediction.git
   cd silica-concentration-prediction

2. Install dependencies:
pip install -r requirements.txt

3. Run the main script:
python main.py

4. View the predictions in the answer2.csv and answer3_updated.csv files.

## PS - train.csv and test.csv are very large files so, pls use this link and download from here :) https://www.kaggle.com/competitions/Ferralytics/data
