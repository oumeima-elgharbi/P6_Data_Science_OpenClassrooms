def split_categories_tree(df):
    """

    :param df:
    :return:
    """
    dataframe = df.copy()
    # we cut the sentences at every >>
    # .str.strip() # strip to remove blank space
    dataframe["category_1"] = dataframe["product_category_tree"].str.split(">>").str[0].str.slice(start=2)
    dataframe["category_2"] = dataframe["product_category_tree"].str.split(">>").str[1]  # .strip()
    dataframe["category_3"] = dataframe["product_category_tree"].str.split(">>").str[2]  # .strip()
    return dataframe
