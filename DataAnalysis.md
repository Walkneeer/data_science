data wrangling:

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
            ...
        
