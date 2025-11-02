import pickle
import gzip
import json
import os
import pandas as pd
import random
from datetime import datetime
from tqdm import tqdm

# return a set of business ids that are restruants
def extract_rest_ids(bus_list_path):
    rest_ids = set()

    with open(bus_list_path, 'r') as f:
        for line in f:  
            if line:    
                json_content = json.loads(line)
                if json_content["categories"] != None: 
                    categories = [val.lower().strip() for val in json_content["categories"].split(",")]
                    if "restaurants" in categories:
                        rest_ids.add(json_content["business_id"])
    
    print("Extracted restaurants ids from", bus_list_path)
    return rest_ids

# convert str time to a number
def get_created_time(text):
    return int(datetime.strptime(text, "%Y-%m-%d %H:%M:%S").strftime("%s"))

# return a list of (business_id, [reviews], scores) where reviews are in time order
# calculate score based on average and only keeps first num_sum_reviews scores
def get_review_score_list(review_list_path, rest_ids, num_sum_reviews, num_score_review):
    # a temp dict {business_id: [(time, review, score)]}
    bus_review_dict = {}
    with open(review_list_path, 'r') as f:
        print("Starting to read reviews from", review_list_path)
        for line in tqdm(f):
            data = json.loads(line)
            bus_id = data["business_id"]
            if bus_id in rest_ids:
                if bus_id not in bus_review_dict:
                    bus_review_dict[bus_id] = []
                
                bus_review_dict[bus_id].append((data["date"], data["text"], data["stars"]))

        print("Making a list of reviews")
        bus_review_score_list = []
        for business_id in tqdm(bus_review_dict):
            if len(bus_review_dict[business_id]) < num_score_review:
                continue
            bus_review_dict[business_id] = sorted(bus_review_dict[business_id], key = lambda x: get_created_time(x[0]))
            review_list = []
            score_list = []
            
            for count, review_tuple in enumerate(bus_review_dict[business_id]):
                if count < num_sum_reviews:
                    review_list.append(review_tuple[1])
                if count >= num_score_review:
                    break
                score_list.append(review_tuple[2])


            bus_review_score_list.append({"business": business_id,
                                          "reviews": review_list,
                                          "scores": score_list[: num_sum_reviews],
                                          "avg_score": sum(score_list) / num_score_review
                                         })
    return bus_review_score_list

# given a review_score_list and a name, dump it as jsonl.gz
def write_json(review_score_list, out_path, name):
    out_file = os.path.join(out_path, name + ".jsonl.gz")
    with gzip.open(out_file, "wt", encoding = "utf-8") as f:
        for line in review_score_list:
            f.write(json.dumps(line) + '\n')
    print("Written {} review list with {} lines".format(name, len(review_score_list)))

def main():
    bus_list_path = "yelp_dataset/business.json"
    review_list_path = "yelp_dataset/review.json"
    out_path = "preprocess_out"
    os.makedirs(out_path, exist_ok = True)

    percent_train = 0.64
    percent_valid = 0.16
    num_sum_reviews = 10  # number of reviews saved for summary
    num_score_review = 50  # number of reviews for average rating

    rest_ids = extract_rest_ids(bus_list_path)
    bus_review_score_list = get_review_score_list(review_list_path, rest_ids, num_sum_reviews, num_score_review)
    
    random.seed(403)
    random.shuffle(bus_review_score_list)

    num_bus = len(bus_review_score_list)
    num_train = int(percent_train * num_bus)
    num_valid = int(percent_valid * num_bus)
    num_test = num_bus - num_train - num_valid

    write_json(bus_review_score_list[: num_train], out_path, "train")
    write_json(bus_review_score_list[num_train: num_train+num_valid], out_path, "valid")
    write_json(bus_review_score_list[num_train+num_valid: ], out_path, "test")
    print("Done")

if __name__ == "__main__":
    main()