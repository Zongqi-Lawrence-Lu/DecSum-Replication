# DecSum Replication (Hsu & Tan, 2021)
This is a project that replicates the DecSum decision-focused summarization framework using the Yelp restaurant dataset:

Hsu, Chao-Chun, and Chenhao Tan. "Decision-focused summarization." *arXiv preprint arXiv:2109.06896* (2021).

The replication reimplements the **DecSum** pipeline — from data preprocessing to model training and summary generation — using the original **Yelp restaurant reviews** dataset.  

It achieves comparable **MSE performance** and runtime as reported in the paper, and introduces several implementation optimizations for reproducibility and efficiency.

The DecSum pipeline consists of three main stages:

1. **Data Preprocessing**  
   - Filter Yelp dataset to include only restaurants with ≥50 reviews.  
   - Extract first 10 reviews per restaurant and compute the mean rating over the first 50 reviews.

2. **Model Training (Longformer)**  
   - Fine-tune a `Longformer-base-4096` model to predict restaurant ratings from text reviews.  
   - The trained model simulates a human decision-maker’s rating process.

3. **Decision-Focused Summarization (DecSum Search)**  
   - Generate summaries optimizing three objectives:  
     - *Faithfulness* — how well the summary preserves decision.  
     - *Representativeness* — how similar the score distribution is to the full reviews.  
     - *Non-redundancy* — penalizes overlap and repetition.  
   - Search implemented via recursive **beam search** over sentence candidates.
  
# Environment Setup
Create and activate a conda environment:

```bash
conda create -n decsum python=3.10
conda activate decsum
python3 -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
Change the three scripts in `scripts` to match the env name.

# Folder Structure
- **`preprocess/`**: Contains script for preprocessing and cleaning Yelp data.
- **`model/`**: Includes the Longformer model training code and the rating prediction logic.
- **`summary/`**: Implements the DecSum algorithm for summarization and its evaluation.
- **`scripts/`**: Contains shell scripts that runs training and inference.
- **`sample_output/`**: Provides sample outputs and results from the model for reference.
- **`requirements.txt`**: Lists all dependencies required to run the project.
- **`README.md`**: This file.

## Data Preprocessing

Download raw dataset from https://www.yelp.com/dataset/download and uncompress it to `yelp_dataset`.
The files need to be renamed to `business.json` and `review.json`.

Set the correct env variables by changing the global variables in `preprocess/preprocess.py`.

Then, run script at the base directory. By default the output will go to `preprocess_out`.
```bash
bash scripts/preprocess.sh
```

## Train Longformer model
Set the correct env variables by changing the global variables in `model/model.py`

Then, run script at the base directory. By default the input directory is `preprocess_out` and the output goes to `model_out`.

```bash
bash scripts/model.sh
```

This takes about 1.5 hours on one A100.

## Run DecSum
Set the correct env variables by changing the global variables in `summary/summary.py`.
Change `score_model_name` to the model output directory like `model_out/pretrained`
and change `test` to the test set file like `preprocess_out/test.jsonl.gz`.

Then, run script at the base directory. Default output directory is `decsum_out`.

```bash
bash scripts/summary.sh
```

By default this will use $(\alpha, \beta, \gamma) = (1, 1, 1)$, sentence length = 5, beam size = 4 on the whole test set (n = 4205). They can be changed in the global variables.

This will generate a log file in the output directory with the summaries and MSE with full.

Running the whole set takes about 9 hours on one A100.

