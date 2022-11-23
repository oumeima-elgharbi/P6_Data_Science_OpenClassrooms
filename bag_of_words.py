from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer


def create_bag_of_words_vectors(data, feat_fit, feature_transform):
    """
    # création du bag of words (CountVectorizer et Tf-idf)

    :param data:
    :param feat_fit:
    :param feature_transform:
    :return:
    """
    assert len(feat_fit) == len(feature_transform), "The length of features to fit and transform must be the same"

    print("Separate vocabulary")

    cvect = CountVectorizer(stop_words='english', max_df=0.95, min_df=1)
    ctf = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=1)

    if len(feat_fit) == 1:  # we only have on word
        cv_fit = cvect.fit(data[feat_fit])
        ctf_fit = ctf.fit(data[feat_fit])

        cv_transform = cv_fit.transform(data[feature_transform])
        ctf_transform = ctf_fit.transform(data[feature_transform])


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
