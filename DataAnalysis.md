
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
    - Simple Feature scaling: $$
    - 
