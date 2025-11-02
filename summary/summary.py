import torch
import torch.nn as nn
import gzip
import json
import numpy as np
import spacy
import time
import os
from warnings import filterwarnings
from pprint import pprint
from joblib import Parallel, delayed, parallel_backend
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity
import score_model.score_model as score_model

filterwarnings("ignore", category = UserWarning)

# given a list of texts, and a list of selected sentences,
# use the model to predict the scores for each one
def predict_score_list(model, tokenizer, text_list = None, selected_list = None):
    if isinstance(text_list, str):
        text_list = [text_list]

    if text_list is None or text_list == []:
        input = selected_list
    elif selected_list is None or selected_list == []:
        input = text_list
    else:
        input = []
        selected_sentences = ' '.join(selected_list)
        for i, sentence in enumerate(text_list):
            input.append(selected_sentences + ' ' + sentence)

    model.eval()
    with torch.inference_mode():
        input_dict = score_model.collate_fn(tokenizer, input)
        score_model.dict_to_device(input_dict)
        return model(input_dict).squeeze().cpu().numpy()

# clean up the sentence 
def clean_sentence(sentence):
    return ''.join(char for char in sentence.lower() if char.isalpha())

# split text into sentences
def split_text(texts, nlp, win_size = 3, min_tokens = 3):
    raw_sentences = [sentence.text for sentence in nlp(texts).sents]
    sentences = []

    assert len(raw_sentences) >= win_size, "Sentence is too short"
    for i in range(len(raw_sentences) - win_size + 1):
        chunk = " ".join(raw_sentences[i: i + win_size])
        if len(chunk) >= min_tokens:
            sentences.append(chunk)
    return sentences

# given the list of scores of all sentence combinations, return the cost list
def faith_score(model, tokenizer, candidate_list, selected_list, score, full_text_list = None):
    # convert list of indices to list of text
    if full_text_list:
        if selected_list is None:
            selected_list = []
        else:
            selected_list = [full_text_list[i] for i in selected_list]

    # if no candidate, just return the total score for selected list combined
    if candidate_list.size == 0:
        candidate_predictions = score_model.predict_score(model, tokenizer, selected_list)
    else:
        if full_text_list is not None:
            candidate_list = [full_text_list[i] for i in candidate_list]
        candidate_predictions = predict_score_list(model, tokenizer, candidate_list, selected_list)
    return np.log(np.abs(candidate_predictions - score) + epsilon)

# given a list of candidate sentences and the distribution for the full text, return the cost list
def repre_score(model, tokenizer, candidate_list, selected_list, sentence_score_list):
    if selected_list.size == 0:
        selected_list_distribution = np.empty(0)
    else:
        selected_list_distribution = sentence_score_list[selected_list]
    output = np.zeros(len(candidate_list))

    if candidate_list.size == 0:
        selected_list_distribution = sentence_score_list[selected_list]
        output = wasserstein_distance(selected_list_distribution, sentence_score_list)
    else: 
        selected_list_distribution = np.append(selected_list_distribution, 0.0)
        for i, sentence_index in enumerate(candidate_list):
            # update the distribution
            selected_list_distribution[-1] = sentence_score_list[sentence_index]
            output[i] = wasserstein_distance(selected_list_distribution, sentence_score_list)
    return np.log(output + epsilon)
'''
# a lightweight version that uses average to predict the score
def faith_score_lightweight(model, tokenizer, candidate_list, selected_list, score, sentence_score_list, full_text_list):
    if candidate_list.size > 0:
        candidate_list_score = sentence_score_list[candidate_list]
        selected_list_score = sentence_score_list[selected_list]
        candidate_predictions = (candidate_list_score + np.sum(selected_list_score)) / (selected_list.size + 1)
    else:
        selected_text_list = [full_text_list[i] for i in selected_list]
        candidate_predictions = score_model.predict_score(model, tokenizer, selected_text_list)
    return np.log(np.abs(candidate_predictions - score) + epsilon)
'''

# generate the cosine similarity matrix and a text index dictionary
def get_similarity_matrix(full_sentence_list, sent_embedder):
    embedding_matrix = sent_embedder.encode(full_sentence_list, convert_to_numpy = True)
    assert not np.isnan(embedding_matrix).any(), "Embedding matrix has nan. Check input"
    return cosine_similarity(embedding_matrix, embedding_matrix)

# evalute non-reduncancy; use cached distance matrix
def redun_score(text_list, selected_indices, similarity_matrix):
    num_selected = len(selected_indices)

    if text_list.size == 0:
        cos_sim = similarity_matrix[np.ix_(selected_indices, selected_indices)]
        np.fill_diagonal(cos_sim, 0)
        return np.sum(np.max(cos_sim, axis = 1))
    else:
        output = np.zeros(len(text_list))
        cos_sim = np.zeros((num_selected + 1, num_selected + 1))
        cos_sim[:num_selected, :num_selected] = similarity_matrix[np.ix_(selected_indices, selected_indices)]
        np.fill_diagonal(cos_sim, 0)
        for i, candidate in enumerate(text_list):
            candidate_selected_sim = similarity_matrix[candidate][selected_indices]
            cos_sim[-1, :-1] = candidate_selected_sim
            cos_sim[:-1, -1] = candidate_selected_sim
            cos_sim[-1, -1] = 0
            output[i] = np.sum(np.max(cos_sim, axis = 1))
        return output  

# create a new candidate list such that each element does not overlap with selected (dist >= win_size)
def reduce_candidate(selected, candidate):
    mask = np.min(np.abs(candidate[np.newaxis, :] - selected[:, np.newaxis]), axis = 0) >= win_size
    return candidate[mask]

'''
# in-place modification to remove all replication in the search space
def remove_perm(selected_list_list, candidate_list_list):
    unique_set = set()

    for i in range(len(selected_list_list) - 1, -1, -1):
        selected_list = selected_list_list[i]
        candidate_list = candidate_list_list[i]
        mask = np.ones(len(candidate_list), dtype = bool)

        combinations = np.tile(selected_list, (len(candidate_list), 1))
        combinations = np.hstack((combinations, candidate_list.reshape(-1, 1)))
        combinations = np.sort(combinations, axis = 1)
        for j, row in enumerate(combinations):
            key = tuple(row)
            if key not in unique_set:
                unique_set.add(key)
            else:
                mask[j] = False

        new_candidate_list = candidate_list[mask]
        if new_candidate_list.size == 0:
            selected_list_list.pop(i)
            candidate_list_list.pop(i)
        else:
            candidate_list_list[i] = new_candidate_list
'''

# normalize a numpy array
def normalize_vector(vector):
    mean = np.mean(vector)
    std_dev = np.std(vector)
    return (vector - mean) / (std_dev + epsilon)

# return the loss list from the three components cost lists
def get_loss_list(faith, repre, redun, norm = False):
    if norm:
        return alpha * normalize_vector(faith) + beta * normalize_vector(repre) + gamma * normalize_vector(redun)
    else:
        return alpha * faith + beta * repre + gamma * redun

# given a chunck of text, returns a list of summarized sentences
def decsum_search(model, tokenizer, full_text, len_summary = 6, beam = 4, nlp = None, sent_embedder = None, curr_level = 0, candidate_list_list = None, num_candidates = None,
                  selected_list_list = None, full_text_score = None,
                  similarity_matrix = None, sentence_score_list = None, full_text_list = None):

    # initialize the lists for first level
    if curr_level == 0:
        filterwarnings("ignore", category = UserWarning)
        if not nlp:
            nlp = spacy.load(nlp_name, disable = ["ner", "tagger"])
        if not sent_embedder:
            sent_embedder = SentenceTransformer(sent_embedder_name).half()
        full_text_list = split_text(full_text, nlp)
        num_candidates = len(full_text_list)
        candidate_list_list = [np.arange(len(full_text_list))]
        selected_list_list = [np.array([], dtype = int)]
        full_text_score = score_model.predict_score(model, tokenizer, full_text)
        similarity_matrix = get_similarity_matrix(full_text_list, sent_embedder)
        sentence_score_list = predict_score_list(model, tokenizer, full_text_list)
    else:
        assert len(selected_list_list[-1]) == curr_level, "Fatal error: wrong selected sentence length"

    # reach max depth or no sentences remaining
    if (curr_level == len_summary) or (num_candidates == 0):
        faith, repre, redun = [np.zeros(len(selected_list_list)) for _ in range(3)]
        for i, selected_list in enumerate(selected_list_list):
            if alpha > 0:
                faith[i] = faith_score(model, tokenizer, np.empty(0, dtype = int), selected_list, full_text_score, full_text_list)
            if beta > 0:
                repre[i] = repre_score(model, tokenizer, np.empty(0, dtype = int), selected_list, sentence_score_list)
            if gamma > 0:
                redun[i] = redun_score(np.empty(0, dtype = int), selected_list, similarity_matrix)
            
        loss = get_loss_list(faith, repre, redun)
        best_list = selected_list_list[np.argmin(loss)]
        return ([full_text_list[index] for index in best_list], full_text_score)
    
    assert len(candidate_list_list) == len(selected_list_list), "Fatal error: candidate / selected list length mismatch"
    full_candidate_list_list = []
    full_selected_list_list = []
    new_loss_list = []
    combined_candidate_selected_list_list = zip(candidate_list_list, selected_list_list)

    # iterate through all possibilities so far and find the top choices
    for candidate_list, selected_list in combined_candidate_selected_list_list:
        num_candidate = len(candidate_list)
        if beta > 0 and curr_level == 0:
            repre = repre_score(model, tokenizer, candidate_list, selected_list, sentence_score_list)
            assert repre.size == num_candidate
            loss_list = repre
        else:
            faith, repre, redun = 0.0, 0.0, 0.0
            if alpha > 0:
                faith = faith_score(model, tokenizer, candidate_list, selected_list, full_text_score, full_text_list)
            if beta > 0:
                repre = repre_score(model, tokenizer, candidate_list, selected_list, sentence_score_list)
            if gamma > 0:
                redun = redun_score(candidate_list, selected_list, similarity_matrix)
            loss_list = get_loss_list(faith, repre, redun)
        
        assert loss_list.size == len(candidate_list), "Fatal error: loss_list length mismatch"
        top_indices = np.argsort(loss_list)[: beam]
        for index in top_indices:
            new_selected_list = np.append(selected_list, candidate_list[index])
            full_selected_list_list.append(new_selected_list)

            new_candidate_list = np.delete(candidate_list, index)
            new_candidate_list = reduce_candidate(new_selected_list, new_candidate_list)
            full_candidate_list_list.append(new_candidate_list)

            new_loss_list.append(loss_list[index])
    
    new_loss_list = np.array(new_loss_list)
    sorted_indices = np.argsort(new_loss_list)
    new_candidate_list_list = []
    new_selected_list_list = []
    seen = set()
    count = 0

    # remove permuations
    for i in sorted_indices:
        if count == beam:
            break
        
        selected_list = tuple(np.sort(full_selected_list_list[i]))
        if selected_list in seen:
            continue
        else:
            seen.add(selected_list)
            new_candidate_list_list.append(full_candidate_list_list[i])
            new_selected_list_list.append(full_selected_list_list[i])
            count += 1

    #remove_perm(new_selected_list_list, new_candidate_list_list)
    
    return decsum_search(model, tokenizer, full_text, len_summary, beam, None, None, curr_level + 1, new_candidate_list_list, num_candidates - 1,
                         new_selected_list_list, full_text_score, similarity_matrix, sentence_score_list, full_text_list)

# run decsum on a single restaurant from the test set
def run_decsum(model, tokenizer, index, len_summary = 6, beam = 4):
    full_text = test_set[index][0]
    business_id = test_set.business_id[index]
    summary, full_text_score = decsum_search(model, tokenizer, full_text, len_summary, beam)
    summary_score = score_model.predict_score(model, tokenizer, summary)
    return (business_id, summary, full_text_score, summary_score)

# print decsum results on a restraunt from the texst set
def run_decsum_display(model, tokenizer, index, len_summary = 6, beam = 4, verbose = True):
    _, summary, full_text_score, summary_score = run_decsum(model, tokenizer, index, len_summary, beam)

    if verbose:
        print("The first 10 reviews are: ")
        full_text = test_set[index][0]
        text = full_text.split("\n")
        for line in text:
            if line.strip():
                print(line.strip())
        
        print("\nThe {} sentences decsum summary with beam {} is: ".format(len_summary, beam))
        for sentence in summary:
            print(sentence.strip())
    print("\nScore from summary is", summary_score)
    print("Score from full text is", full_text_score)
    print("Actual score is", test_set[index][1])

# run decsum on the entire test set
def run_decsum_test(model, tokenizer, len_summary = 6, beam = 4, start = None, end = None):
    start_time = time.time()
    if start is None:
        start = 0
    if end is None:
        end = len(test_set) - 1
    
    with parallel_backend("threading"):
        results = Parallel(n_jobs = 4)(delayed(run_decsum)(model, tokenizer, i, len_summary, beam) for i in range(start, end + 1))

    _, _, full_text_score_list, summary_score_list = zip(*results)
    full_text_score_list = np.array(full_text_score_list)
    summary_score_list = np.array(summary_score_list)
    mse = np.sum((full_text_score_list - summary_score_list) ** 2) / (end - start)

    name = f"result_{start}_to_{end}_with_{len_summary}_sentences_{beam}_beam_alpha{alpha:.1f}_beta{beta:.1f}_gamma{gamma:.1f}.txt"
    with open(os.path.join(out_dir, name), 'w') as f:
        for business_id, summary, full_text_score, summary_score in results:
            cleaned_summary = [sentence.replace('\n', '') for sentence in summary]
            out = {"business_id": business_id, "full_text_score": full_text_score, "summary_score": summary_score, "summary": cleaned_summary}
            pprint(out, stream = f)
        f.write("MSE error: {:.8f}\n".format(mse))
        f.write("Time: {:.2f} mins\n".format((time.time() - start_time) / 60))

if __name__ == "__main__":
    score_model_name = "model_out/pretrained"
    tokenizer_name = score_model_name
    nlp_name = "en_core_web_sm"  # to split combined reviews into sentences
    sent_embedder_name = "distilbert-base-nli-stsb-mean-tokens"  # for semantics
    test = "preprocess_out/test.jsonl.gz"
    max_tokens = 3000
    num_sent = 5 # number of sentences in a summary
    win_size = 3 # sliding window to make a chunk
    min_tokens = 3 # shortest tokens for a sentence
    out_dir = "decsum_out"
    epsilon = 1e-6
    alpha = 1.0
    beta = 1.0
    gamma = 1.0

    #global device, model, tokenizer, nlp, sent_embeder, test_set
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    model_0 = score_model.linear_model(model_name = score_model_name)
    model_0.to(device)
    model_0.model = model_0.model.half()
    print("Loaded score model from", score_model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    global_nlp = spacy.load(nlp_name, disable = ["ner", "tagger"]) # only works for en_core_web_sm
    print("Loaded nlp (review segmenting model) from", nlp_name)
    global_sent_embedder = SentenceTransformer(sent_embedder_name).half()
    print("Loaded sentiment BERT", sent_embedder_name)

    '''
    try:
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.disable()
        model_0.model = torch.compile(model_0.model)
        print("Compiled score model")
    except AttributeError:
        print("Compilation not available on this version")
    '''    

    os.makedirs(out_dir, exist_ok = True)
    test_set = score_model.Yelp_dataset(path = test, tokenizer = tokenizer, max_tokens = max_tokens)
    print("Generated test set")

    # run_decsum_display(model_0, tokenizer, 128, len_summary = 6, beam = 4)
    run_decsum_test(model_0, tokenizer, len_summary = 5, beam = 4, start = 0, end = 31)
    
    