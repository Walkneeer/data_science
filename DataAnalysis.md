# Data wrangling:

- Strategies for missing data:
    - Drop missing values:

        1. drop row: df.dropna(subset=["price"], axis=0, inplace = True)
        2. drop variable

    - Replace the missing values:

        1. replace with avg
        2. replace by frequency
        3. replace by interpolation
    - Leave it as missing data
- Data formating:
    - conventions
    - df.dtypes()
    - df.astype()
        - df['price'] = df["price"].astype("int")
- Data Normalization:
    - value range (0-1)
        - Simple Feature scaling: 

            - $x_\text{new} = \frac{x_\text{old}}{x_\text{max}}$
            - df["length"] = df["length"] / df["length"].max()

        - Min-max: 

            - $x_\text{new} = \frac{x_\text{old} - x_\text{min}}{x_\text{max} - x_\text{min}}$
            - df["length"] = (df["length"] - df["length"].min()) / (df["length"].max - df["length"].min(())

    - mean of the data set becomes 0 and standard deviation becomes 1:

        - Z-score: 

            - $x_\text{new} = \frac{x_\text{old} - \mu}{\sigma}$
            - df["length"] = (df["length"]-df["length"].mean()) / df["length"].std()
- Binning:
    - group values into bin
    - bins = np.linspace(min(df["price"])),max(df["price"], 4) # 3 groups
    - group_names = ["low", "Medium", "High"]
    - df["price-binned"] = pd.cut(df["price"], bins, labels= group_names, include_lowest = True)
- Categorical to Quantitative Variables:

    - Object or strings can be taken as an input for statistical models.
        
    - Car_Fuel = {
        "car_a" : gas
        "car_b" : diesel
        "car_d" : gas 
    }

    - solution: creating unique features a = [gas, diesel] 

            | car | gas | diesel |
            |-----|-----|--------|
            |  a  |  1  |   0    |
            |  b  |  0  |   1    |


# Exploratory Data Analysis (EDA) :

- EDA:
    - Summarize main characteristics of the data
        - Gain better understanding of the data set
        - Uncover ralationships between variables
        - Extract important variables
        - Descriptive stats
        - GroupBy
        - ANOVA
        - Correlarion
        - Correlarion - Sttistics
- Descriptive Statistics:

    - Explore data
    - Calculate descripive statistics of a data
    - Describe basic feature
    - Giving short summaries about the sample and measures of the data
    
    - function:
        - df.describe()
        - value_counts()
        - box plots: median, upper quartile, lower quartile, extremes, outliers
        - Scatter plot
            1. predictor variables on x-axis eg: engine size
            2. target variable eg: price

               - how to do this ?
                    - y=df["price"]
                    - x=df["engine-size"]
                    - plt.scatter(x,y)
                    - plt.xlabel("Engine Size")
                    - plt.ylabel("Price")
- GroupBy in Python:
    - df_test = df[['drive-weels', 'body-style', 'price']]
    - df_grp = df_test.groupby(['drive-weels', 'body-style'], as_index=False).mean()
        - pivot()
        - Heatma
- Correlation:
    - correlation between two features (engine-size and price)
    - Pearson Correlation:
        - linear rellations
        - measure strength of the correlation
            - Close to 1: Large positive relationship
            - Close to -1: LargeNegative relationship
            - close to 0: No relationship
            - P-value < 0.001 strong certainty in the result

        - pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
            
            - Pearson correlation: 0.81
            - P-value: 9.35 e-48

        heatmap

# Model Development :

- goal:
    1. Simple and Multiple Linear Regression
    2. Model evaluation
    3. Polynomial Rgression and Pipelines

- simple linear regression: the predictor (independent variable x) and the the target (dependent variable y)
- multiple linear regression: multiple independent variable, and target variable y

- Simple linear regression:
    - predictor: variable x
    - target: variable y
        - linear relationship
            - $y=b_0 + b_1 \ x $
                - $b_0$ the intercept
                - $b_1$ the slope

- store x -> in np.array x
- store y -> in np.array y

- Process:
    
    - Simple Linear Regression:

        ```python
        # Import necessary libraries
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np

        # 'df' with columns 'highway-mpg' and 'price'

        # Step 1: Define predictor variable (X) and target variable (Y)
        X = df[['highway-mpg']]
        Y = df['price']

        # Step 2: Perform train-test split
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

        # Step 3: Create a linear regression object
        lm = LinearRegression()

        # Step 4: Fit the model on the training data
        lm.fit(x_train, y_train)

        # Step 5: Obtain predictions on the test data
        predictions = lm.predict(x_test)

        # Step 6: Evaluate the model
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        # Display evaluation metrics
        print(f'Mean Absolute Error: {mae}')
        print(f'Mean Squared Error: {mse}')
        print(f'Root Mean Squared Error: {rmse}')
        print(f'R-squared: {r2}')
        ```

    - Multiple Linear Regression:

        ```python
        # Assuming you have additional independent variables in the DataFrame

        X_multiple = df[['highway-mpg', 'engine-size', 'horsepower']]
        Y_multiple = df['price']

        x_train_multiple, x_test_multiple, y_train_multiple, y_test_multiple = train_test_split(X_multiple, Y_multiple, test_size=0.3, random_state=0)

        lm_multiple = LinearRegression()
        lm_multiple.fit(x_train_multiple, y_train_multiple)
        predictions_multiple = lm_multiple.predict(x_test_multiple)

        mae_multiple = mean_absolute_error(y_test_multiple, predictions_multiple)
        mse_multiple = mean_squared_error(y_test_multiple, predictions_multiple)
        rmse_multiple = np.sqrt(mse_multiple)
        r2_multiple = r2_score(y_test_multiple, predictions_multiple)

        print("\nMultiple Linear Regression Metrics:")
        print(f'Mean Absolute Error: {mae_multiple}')
        print(f'Mean Squared Error: {mse_multiple}')
        print(f'Root Mean Squared Error: {rmse_multiple}')
        print(f'R-squared: {r2_multiple}')
        ```
    - 