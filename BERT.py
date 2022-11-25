from transformers import AutoTokenizer
import numpy as np
from time import time
from tqdm import tqdm
# from transformers.models.bert.modeling_bert import BertModel, BertForMaskedLM


# common functions

def bert_inp_fct(sentences, bert_tokenizer, max_length):
    """
    # Fonction de préparation des sentences

    :param sentences: (list) list of sentences (string)
    :param bert_tokenizer:
    :param max_length:
    :return:
    :rtype: (tuple)
    """
    input_ids = []
    token_type_ids = []
    attention_mask = []
    bert_inp_tot = []

    for sent in sentences:
        bert_inp = bert_tokenizer.encode_plus(sent,
                                              add_special_tokens=True,
                                              max_length=max_length,
                                              padding='max_length',
                                              return_attention_mask=True,
                                              return_token_type_ids=True,
                                              truncation=True,
                                              return_tensors="tf")

        input_ids.append(bert_inp['input_ids'][0])
        token_type_ids.append(bert_inp['token_type_ids'][0])
        attention_mask.append(bert_inp['attention_mask'][0])
        bert_inp_tot.append((bert_inp['input_ids'][0],
                             bert_inp['token_type_ids'][0],
                             bert_inp['attention_mask'][0]))

    input_ids = np.asarray(input_ids)
    token_type_ids = np.asarray(token_type_ids)
    attention_mask = np.array(attention_mask)

    return input_ids, token_type_ids, attention_mask, bert_inp_tot


def feature_BERT_fct(model, model_type, sentences, max_length, b_size, mode='HF'):
    """
    # Fonction de création des features

    :param model:
    :param model_type:
    :param sentences: (list)
    :param max_length:
    :param b_size: (int)
    :param mode:
    :return:
    :rtype: (tuple)
    """
    batch_size = b_size
    batch_size_pred = b_size
    bert_tokenizer = AutoTokenizer.from_pretrained(model_type)
    time1 = time()

    for step in tqdm(range(len(sentences) // batch_size)):
        idx = step * batch_size
        input_ids, token_type_ids, attention_mask, bert_inp_tot = bert_inp_fct(sentences[idx:idx + batch_size],
                                                                               bert_tokenizer, max_length)

        if mode == 'HF':  # Bert HuggingFace
            outputs = model.predict([input_ids, attention_mask, token_type_ids], batch_size=batch_size_pred)
            last_hidden_states = outputs.last_hidden_state

        if mode == 'TFhub':  # Bert Tensorflow Hub
            text_preprocessed = {"input_word_ids": input_ids,
                                 "input_mask": attention_mask,
                                 "input_type_ids": token_type_ids}
            outputs = model(text_preprocessed)
            last_hidden_states = outputs['sequence_output']

        if step == 0:
            last_hidden_states_tot = last_hidden_states
            last_hidden_states_tot_0 = last_hidden_states
        else:
            last_hidden_states_tot = np.concatenate((last_hidden_states_tot, last_hidden_states))

    features_bert = np.array(last_hidden_states_tot).mean(axis=1)

    time2 = np.round(time() - time1, 0)
    print("temps traitement : ", time2)

    return features_bert, last_hidden_states_tot
