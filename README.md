# Predicting-net-rents-per-square-foot 
## Canadian Real Estate Net Rent Prediction
This project demonstrates a machine learning approach to predict the net rent per square foot for commercial real estate properties in Canada. It uses a synthetic dataset to showcase the process of data preparation, model training using RandomForestRegressor, hyperparameter tuning with GridSearchCV, and model evaluation.

## Project Goal
The primary goal of this project is to build a predictive model that can estimate the 'net rent' of a commercial property based on various features such as postal code, building class, building size, lease duration, and additional rent per square foot.

## Dataset
The dataset used in this project is a dummy dataset generated to simulate Canadian commercial real estate properties. It includes the following features:

* postal_code: Categorical, representing different Canadian postal codes (e.g., 'M5V', 'V6B').
* b_class: Categorical, representing the building class ('A', 'B', 'C').
* b_size: Numerical, representing the building size in square feet.
* lease_months: Categorical, representing the lease duration in months.
* add_rent_sq_ft: Numerical, representing additional rent per square foot.
* net_rent: Numerical (target variable), the estimated net rent per square foot, derived from base values adjusted by location, class, size, and random noise.

## Modeling Approach
1. Data Preprocessing: Categorical features (postal_code, b_class) are converted into numerical format using one-hot encoding (pd.get_dummies).
2. Data Splitting: The dataset is split into training and testing sets (75% training, 25% testing) to evaluate the model's performance on unseen data.
3. Model Selection: A RandomForestRegressor from sklearn.ensemble is chosen for its robustness and ability to handle various data types.
4. Hyperparameter Tuning: GridSearchCV is used to find the optimal hyperparameters for the RandomForestRegressor. The grid search explores different values for n_estimators, max_depth, min_samples_split, and min_samples_leaf, optimizing for neg_mean_squared_error.
5. Model Training: The RandomForestRegressor is trained with the best hyperparameters found by GridSearchCV.

## Results
After training, the model's performance is evaluated on the test set:

* Mean Absolute Error (MAE): 2.45
* R2 Score: 0.84

The feature importance analysis reveals that b_class and postal_code are the most influential features in predicting net rent.

| Feature | Importance |
| :--- | ---: |
| `b_class_B` | 0.3843 |
| `b_class_C` | 0.2895 |
| `postal_code_K1P` | 0.1684 |
| `postal_code_T2P` | 0.0766 |
| `postal_code_V6B` | 0.0523 |
| `b_size` | 0.0189 |
| `add_rent_sq_ft` | 0.0073 |
| `lease_months` | 0.0018 |
| `postal_code_M5V` | 0.0009 |

## How to Use
To run this project:

1. **Dependencies**: Ensure you have the necessary Python libraries installed:

    - numpy
    - pandas
    - scikit-learn
    - matplotlib (for potential visualizations, though not explicitly used in the final output)

You can install them using pip:

*pip install numpy pandas scikit-learn matplotlib*

2. **Run the Notebook**: Execute the cells in the provided Jupyter/Colab notebook sequentially. The notebook covers:

    - Loading libraries.
    - Generating the dummy dataset.
    - Estimating the target variable (net_rent).
    - Preprocessing the data (one-hot encoding).
    - Splitting data into training and testing sets.
    - Instantiating and tuning the RandomForestRegressor.
    - Training the model with optimal hyperparameters.
    - Evaluating the model's performance.
    - Demonstrating a prediction for a new property entry.

## Example Prediction
A new property with specific features can be used to predict its net rent:

> new_lease_features = [[45000, 36, 12.50, 0, 0, 0, 0, 1, 0]]  # Example features for a Class B building

> predicted_price = rf_best.predict(new_lease_features)
> print(f"Predicted Net Rent: ${predicted_price[0]:.2f} / sqft")

*Output: Predicted Net Rent: $38.30 / sqft*
