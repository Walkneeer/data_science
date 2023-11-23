# data wrangling:

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
        - measure strength of the correlation
            - Close to 1: Large positive relationship
            - Close to -1: LargeNegative relationship
            - close to 0: No relationship
            - P-value < 0.001 strong certainty in the result

        - pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
            
            - Pearson correlation: 0.81
            - P-value: 9.35 e-48

        heatmap

# 


                




        