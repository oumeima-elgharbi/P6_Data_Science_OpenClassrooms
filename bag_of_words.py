from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer


def create_bag_of_words_vectors(data, feature_fit, feature_transform):
    """
    # création du bag of words (CountVectorizer et Tf-idf)
    fit/transform can choose features
    more than one features yes

    :param data: (DataFrame)
    :param feature_fit: (list)
    :param feature_transform: (list)
    :return:
    """
    assert len(feature_fit) == len(feature_transform), "The length of features to fit and transform must be the same"

    # if the column do not have the same names we need to rename them
    if feature_fit != feature_transform:
        col_renamed_fit = {col: str(i) for i, col in enumerate(feature_fit)}
        df_fit = data[feature_fit].copy().rename(columns=col_renamed_fit)

        col_renamed_transform = {col: str(i) for i, col in enumerate(feature_transform)}
        df_transform = data[feature_transform].copy().rename(columns=col_renamed_transform)
    else:
        df_fit = data[feature_fit]
        df_transform = data[feature_transform]

    print("Count Vector")
    # data[feat] needs to be a Pandas Series
    cv_transform = ColumnTransformer([(x, CountVectorizer(stop_words='english', max_df=0.95, min_df=1), x) for x in
                                      df_fit.columns.tolist()]).fit(df_fit)
    cv_transform = cv_transform.transform(df_transform)

    print("TF-IDF")
    ctf_transform = ColumnTransformer([(x, TfidfVectorizer(stop_words='english', max_df=0.95, min_df=1), x) for x in
                                       df_fit.columns.tolist()]).fit(df_fit)
    ctf_transform = ctf_transform.transform(df_transform)

    return cv_transform, ctf_transform


"""


def create_bag_of_words_vectors(data, feat_fit, feature_transform):

assert len(feat_fit) == len(feature_transform), "The length of features to fit and transform must be the same"

print("Separate vocabulary")
# data[feat] needs to be a Pandas Series
cvect = CountVectorizer(stop_words='english', max_df=0.95, min_df=1)
ctf = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=1)

if len(feat_fit) == 1:  # we only have on word
    cv_fit = cvect.fit(data[feat_fit])
    ctf_fit = ctf.fit(data[feat_fit])

    cv_transform = cvect.transform(data[feature_transform])
    ctf_transform = ctf.transform(data[feature_transform])


elif len(feat_fit) > 1:
    cv_transform = ColumnTransformer([(x, CountVectorizer(stop_words='english', max_df=0.95, min_df=1), x) for x in
                                      feat_fit]).fit(data[feat_fit])
    cv_transform = cv_transform.transform(data[feature_transform])

    # ctf_transform = ColumnTransformer([(x, TfidfVectorizer(stop_words='english', max_df=0.95, min_df=1), x) for x in
    #                                   feat_fit]).fit_transform(data[feat_fit])

    ctf_transform = ColumnTransformer([(x, TfidfVectorizer(stop_words='english', max_df=0.95, min_df=1), x) for x in
                                       feat_fit]).fit(data[feat_fit])
    ctf_transform = ctf_transform.transform(data[feature_transform])

return cv_transform, ctf_transform
"""
