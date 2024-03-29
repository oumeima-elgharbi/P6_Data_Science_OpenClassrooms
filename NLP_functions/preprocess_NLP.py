import warnings
warnings.filterwarnings(action="ignore")

import pandas as pd

# NLP

from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize

# visualization
import matplotlib.pyplot as plt
from wordcloud import WordCloud  ### WARNINGS !!

global stopwords
stop_words = set(stopwords.words('english'))

global stop_w
stop_w = list(set(stopwords.words('english'))) + ['[', ']', ',', '.', ':', '?', '(', ')']


def display_tokens_info(tokens):
    """
    Displays information about corpus
    :param tokens:
    :return:
    """
    print(f"nb tokens {len(tokens)}, nb tokens uniques {len(set(tokens))}")
    print(tokens[:30])
    print(list(set(tokens[:30])))


def display_wordcloud(corpus_list):
    """

    :param corpus_list:
    :return:
    """
    if len(corpus_list) > 2:  # else figure too big
        plt.figure(figsize=(20, 20))
    else:
        plt.figure()
    # so that we can display a lot of images in the same cell
    rows = len(corpus_list)
    columns = 1
    for i, corpus in enumerate(corpus_list):
        wordcloud = WordCloud(background_color='white',
                              stopwords=[],
                              max_words=50).generate(" ".join(corpus))
        plt.subplot(rows, columns, i + 1)
        plt.imshow(wordcloud)
        plt.axis("off")
    # plt.show;


def process_text_1(doc, rejoin=False):
    """
    basic function of text processing

    :param doc:
    :param rejoin:
    :return:
    """
    # lower
    doc = doc.lower().strip()

    # tokenize
    tokenizer = RegexpTokenizer(r"\w+")
    raw_tokens_list = tokenizer.tokenize(doc)

    # stop words
    cleaned_tokens_list = [w for w in raw_tokens_list if w not in stop_words]

    if rejoin:
        return " ".join(cleaned_tokens_list)  # return a string with each token separated by a blank space
    return cleaned_tokens_list


def process_text_2(doc,
                   rejoin=False,
                   list_rare_words=None,
                   min_len_word=3,
                   force_is_alpha=True):
    """
    cf process_text_1 but with list_unique_words, min_len_word and force_is_alpha

    positional arguments :
    -----------------------
    :param doc: (string) the document (aka a text in str format) to process

    opt args :
    -----------------------
    :param rejoin: (bool) if True return a string else return the list of tokens
    :param list_rare_words: (list) a list of rare words to exclude
    :param min_len_word: (int) the minimum length if words to not exclude
    :param force_is_alpha: (int) if 1, exclude all tokens with a numeric character
    :return: (string) if rejoin is True or (list) a list of tokens
    """

    # list_unique_words
    if not list_rare_words:  # if not None == True
        list_rare_words = []

    # lower
    doc = doc.lower().strip()

    # tokenize
    tokenizer = RegexpTokenizer(r"\w+")
    raw_tokens_list = tokenizer.tokenize(doc)

    # classic stopwords
    cleaned_tokens_list = [w for w in raw_tokens_list if w not in stop_words]

    #############################

    # no rare tokens
    non_rare_tokens = [w for w in cleaned_tokens_list if w not in list_rare_words]

    # no more len words
    more_than_N = [w for w in non_rare_tokens if len(w) >= min_len_word]

    # only alpha chars
    if force_is_alpha:
        alpha_tokens = [w for w in more_than_N if w.isalpha()]
    else:
        alpha_tokens = more_than_N

    #############################

    # manage return type
    if rejoin:
        return " ".join(alpha_tokens)
    return alpha_tokens


def process_text_3(doc,
                   rejoin=False,
                   lem_or_stem="stem",
                   list_rare_words=None,
                   min_len_word=3,
                   force_is_alpha=True):
    """
    cf process_text_2 but with stem or lem

    positional arguments :
    -----------------------
    :param doc: (string) the document (aka a text in str format) to process

    opt args :
    -----------------------
    :param rejoin: (bool) if True return a string else return the list of tokens
    :param lem_or_stem: (str) if lem do lemmatise else stem
    :param list_rare_words: (list) a list of rare words to exclude
    :param min_len_word: (int) the minimum length if words to not exclude
    :param force_is_alpha: (int) if 1, exclude all tokens with a numeric character
    :return: (string) if rejoin is True or (list) a list of tokens
    """

    # list_unique_words
    if not list_rare_words:  # if not None == True
        list_rare_words = []

    # lower
    doc = doc.lower().strip()

    # tokenize
    tokenizer = RegexpTokenizer(r"\w+")
    raw_tokens_list = tokenizer.tokenize(doc)

    # classic stopwords
    cleaned_tokens_list = [w for w in raw_tokens_list if w not in stop_words]

    # no rare tokens
    non_rare_tokens = [w for w in cleaned_tokens_list if w not in list_rare_words]

    # no more len words
    more_than_N = [w for w in non_rare_tokens if len(w) >= min_len_word]

    # only alpha chars
    if force_is_alpha:
        alpha_tokens = [w for w in more_than_N if w.isalpha()]
    else:
        alpha_tokens = more_than_N

    #################################

    # stem or lem
    if lem_or_stem == "lem":
        trans = WordNetLemmatizer()
        trans_text = [trans.lemmatize(i) for i in alpha_tokens]
    else:
        trans = PorterStemmer()
        trans_text = [trans.stem(i) for i in alpha_tokens]

    #################################

    # manage return type
    if rejoin:
        return " ".join(trans_text)
    return trans_text


def process_text_4(doc,
                   rejoin=False,
                   lem_or_stem="stem",
                   list_rare_words=None,
                   min_len_word=3,
                   force_is_alpha=True,
                   eng_words=None):
    """
    cf process_text_3 but with selection of only english words

    positional arguments :
    -----------------------
    :param doc: (string) the document (aka a text in str format) to process

    opt args :
    -----------------------
    :param rejoin: (bool) if True return a string else return the list of tokens
    :param lem_or_stem: (str) if lem do lemmatise else stem
    :param list_rare_words: (list) a list of rare words to exclude
    :param min_len_word: (int) the minimum length if words to not exclude
    :param force_is_alpha: (int) if 1, exclude all tokens with a numeric character
    :param eng_words: (list) list of english words
    :return: (string) if rejoin is True or (list) a list of tokens
    """

    # list_unique_words
    if not list_rare_words:  # if not None == True
        list_rare_words = []

    # lower
    doc = doc.lower().strip()

    # tokenize
    tokenizer = RegexpTokenizer(r"\w+")
    raw_tokens_list = tokenizer.tokenize(doc)

    # classic stopwords
    cleaned_tokens_list = [w for w in raw_tokens_list if w not in stop_words]

    # no rare tokens
    non_rare_tokens = [w for w in cleaned_tokens_list if w not in list_rare_words]

    # no more len words
    more_than_N = [w for w in non_rare_tokens if len(w) >= min_len_word]

    # only alpha chars
    if force_is_alpha:
        alpha_tokens = [w for w in more_than_N if w.isalpha()]
    else:
        alpha_tokens = more_than_N

    # stem or lem
    if lem_or_stem == "lem":
        trans = WordNetLemmatizer()
        trans_text = [trans.lemmatize(i) for i in alpha_tokens]
    else:
        trans = PorterStemmer()
        trans_text = [trans.stem(i) for i in alpha_tokens]

    #################################

    # in english
    if eng_words:
        english_text = [i for i in trans_text if i in eng_words]
    else:
        english_text = trans_text

    #################################

    # manage return type
    if rejoin:
        return " ".join(english_text)
    return english_text


def process_text_5(doc,
                   rejoin=False,
                   lem_or_stem="stem",
                   list_rare_words=None,
                   min_len_word=3,
                   force_is_alpha=True,
                   eng_words=None,
                   extra_words=None):
    """
    df v4 but exclude an extra list

    positional arguments :
    -----------------------
    :param doc: (string) the document (aka a text in str format) to process

    opt args :
    -----------------------
    :param rejoin: (bool) if True return a string else return the list of tokens
    :param lem_or_stem: (str) if lem do lemmatise else stem
    :param list_rare_words: (list) a list of rare words to exclude
    :param min_len_word: (int) the minimum length if words to not exclude
    :param force_is_alpha: (int) if 1, exclude all tokens with a numeric character
    :param eng_words: (list) list of english words
    :param extra_words: (list) list of duplicate values
    :return: (string) if rejoin is True or (list) a list of tokens
    """

    # list_unique_words
    if not list_rare_words:  # if not None == True
        list_rare_words = []

    # extra_words
    if not extra_words:
        extra_words = []

    # lower and strip
    doc = doc.lower().strip()

    # tokenize
    tokenizer = RegexpTokenizer(r"\w+")
    raw_tokens_list = tokenizer.tokenize(doc)

    # remove stop words
    cleaned_tokens_list = [w for w in raw_tokens_list if w not in stop_words]

    # drop rare tokens
    non_rare_tokens = [w for w in cleaned_tokens_list if w not in list_rare_words]

    # keep only len word > N
    more_than_N = [w for w in non_rare_tokens if len(w) >= min_len_word]

    # keep only alpha not num
    if force_is_alpha:
        alpha_num = [w for w in more_than_N if w.isalpha()]
    else:
        alpha_num = more_than_N

    # stem or lem
    if lem_or_stem == "lem":
        trans = WordNetLemmatizer()
        trans_text = [trans.lemmatize(i) for i in alpha_num]
    else:
        trans = PorterStemmer()
        trans_text = [trans.stem(i) for i in alpha_num]

    # in english
    if eng_words:
        english_text = [i for i in trans_text if i in eng_words]
    else:
        english_text = trans_text

    ####################

    # drop extra words tokens
    final = [w for w in english_text if w not in extra_words]

    ####################

    # manage return type
    if rejoin:
        return " ".join(final)
    return final


def final_clean(doc, fct_processing, stem_or_lem, list_rare_words, english_words, duplicated_words):
    """
    Performs our final cleaning
    Uses process_text_5
    :param doc:
    :return:
    """
    new_doc = fct_processing(doc,
                             rejoin=True,
                             lem_or_stem=stem_or_lem,
                             list_rare_words=list_rare_words,
                             eng_words=english_words,
                             extra_words=duplicated_words)
    return new_doc


def generate_duplicated_words_list(corpus_list, n=20):
    """
    Complexity : O(n²)
    2 comb 7 = 21 possibilities to check : n = 7 ; n*(n-1) / 2 = 7 * 6 / 2
    {0, 1}
    {0, 2}
    ...
    {0, 6}
    [1, 2}
    ...
    {5, 6}

    6 + 5 + 4 + 3 + 2 + 1 + 0

    :param corpus_list: (list) a list of corpus
    :param n:
    :return:
    """
    duplicated_words = []
    step = 1

    for i in range(0, 7):
        for j in range(i + 1, 7):  # i + 1 instead of i != j
            print("__Step_{}__".format(step))
            duplicated_ij = [i for i in pd.Series(corpus_list[i]).value_counts().head(n).index if
                             i in pd.Series(corpus_list[j]).value_counts().head(n).index]

            duplicated_words.extend(duplicated_ij)
            step += 1

    print("The length of the list of duplicated words is", len(duplicated_words))
    duplicated_words_set = list(set(duplicated_words))

    print("The length of the set of duplicated words is", len(duplicated_words_set))
    return duplicated_words_set


# Other cleaning functions for text preprocessing
def tokenizer_fct(sentence):
    """
    Tokenizer
    :param sentence:
    :return:
    """
    # print(sentence)
    sentence_clean = sentence.replace('-', ' ').replace('+', ' ').replace('/', ' ').replace('#', ' ')
    word_tokens = word_tokenize(sentence_clean)
    return word_tokens


def stop_word_filter_fct(list_words):
    """

    :param list_words:
    :return:
    """
    filtered_w = [w for w in list_words if w not in stop_w]
    filtered_w2 = [w for w in filtered_w if len(w) > 2]
    return filtered_w2


def lower_start_fct(list_words):
    """
    lower case and alpha
    :param list_words:
    :return:
    """
    lw = [w.lower() for w in list_words]
    return lw


def lemma_fct(list_words):
    """
    Lemmatizer (words without prefix or suffix)
    :param list_words:
    :return:
    """
    lemmatizer = WordNetLemmatizer()
    lem_w = [lemmatizer.lemmatize(w) for w in list_words]
    return lem_w


def transform_bow_fct(desc_text):
    """
    Prepares text for bag of words (Countvectorizer et Tf_idf, Word2Vec)
    :param desc_text:
    :return:
    """
    word_tokens = tokenizer_fct(desc_text)
    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(sw)
    # lem_w = lemma_fct(lw)
    transf_desc_text = ' '.join(lw)
    return transf_desc_text


def transform_bow_lem_fct(desc_text):
    """
    Prepares text for bag of words with lemmatisation
    :param desc_text:
    :return:
    """
    word_tokens = tokenizer_fct(desc_text)
    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(sw)
    lem_w = lemma_fct(lw)
    transf_desc_text = ' '.join(lem_w)
    return transf_desc_text


def transform_dl_fct(desc_text):
    """
    Prepares text for Deep Learning (USE et BERT)
    :param desc_text:
    :return:
    """
    word_tokens = tokenizer_fct(desc_text)
    #    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(word_tokens)
    # lem_w = lemma_fct(lw)
    transf_desc_text = ' '.join(lw)
    return transf_desc_text
