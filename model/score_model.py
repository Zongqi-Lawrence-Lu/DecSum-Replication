import os
import argparse
import json
import gzip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# Parse Yelp dataset inputs. Uses autotokenizer and expects the file "yelp_10reviews_50avg.jsonl.gz"
class Yelp_dataset(Dataset):
    def __init__(self, path, tokenizer = "allenai/longformer-base-4096", max_tokens = 2048):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.path = path
        self.reviews = []  
        self.scores = []
        self.business_id = []
        
        # get all the reviews and the scores for restraunts
        with gzip.open(path, "rt") as f:
            for line in f:
                raw_line = json.loads(line)
                reviews = " ".join(raw_line.get("reviews", []))  # join reviews into a single line
                score = float(raw_line.get("avg_score", float("nan")))
                self.reviews.append(reviews)
                self.scores.append(score)
                self.business_id.append(raw_line["business"])
            self.len_data = len(self.reviews)

    def __len__(self):
        return self.len_data
        
    def __repr__(self):
        return "a Yelp dataset with {} stores".format(self.len_data)
        
    def __getitem__(self, index):
        return (self.reviews[index], self.scores[index])
        
    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()
    
    def to(self, arg):
        self.model.to(arg)

    # Called by DataLoader, given a list of (text, score) tuple, returns tensors for training
    def collate_fn(self, batch):
        reviews, scores = zip(*batch)  # each is a list
        encoding = self.tokenizer(text = reviews,
                                  padding = True,
                                  truncation = True,
                                  max_length = self.max_tokens,
                                  return_tensors = "pt"
                                 )

        # CLS token mask
        global_mask = torch.zeros_like(encoding["input_ids"])
        global_mask[:, 0] = 1 

        #return (encoding["input_ids"], encoding["attention_mask"], global_mask)
        return {"input_ids": encoding["input_ids"],
                "mask": encoding["attention_mask"],
                "global_mask": global_mask,
                "scores": torch.tensor(scores).unsqueeze(1)}

# generate the model input to a piece of text or a list of text
def collate_fn(tokenizer, text_list, max_tokens = 3000):
    if not isinstance(text_list, list):
        text_list = [text_list]
    encoding = tokenizer(text = text_list,
                         padding = True,
                         truncation = True,
                         max_length = max_tokens,
                         return_tensors = "pt"
                         )

    # CLS token mask
    global_mask = torch.zeros_like(encoding["input_ids"])
    global_mask[:, 0] = 1 

    #return (encoding["input_ids"], encoding["attention_mask"], global_mask)
    return {"input_ids": encoding["input_ids"],
            "mask": encoding["attention_mask"],
            "global_mask": global_mask
            }

# the score prediction model
class linear_model(pl.LightningModule):
    def __init__(self, model_name = "test", lr = 1e-5, grad_accum = 5, weight_decay = 0, warmup_steps = 500, total_steps = 10000):
        super().__init__()
        self.model_name = model_name
        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(model_name, num_labels = 1)
        self.config.return_dict = True
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config = self.config)
        self.loss_fn = nn.MSELoss()
        self.no_decay_list = ["bias", "LayerNorm.weight"]

    def __repr__(self):
        return "a linear prediction model"

    def forward(self, input_dict):
        return self.model(input_ids = input_dict["input_ids"], attention_mask = input_dict["mask"], global_attention_mask = input_dict["global_mask"]).logits

    # get the loss of a batch
    def get_loss(self, logits, labels):
        return self.loss_fn(logits, labels)

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.loss_fn(logits, batch["scores"])
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.loss_fn(logits, batch["scores"])
        self.log("val_loss", loss)
        return {"val_loss": loss}
    
    # given the name of a parameter, return true if it should have weight decay
    def should_param_decay(self, name):
        for no_decay in self.no_decay_list:
            if no_decay in name:
                return False
        return True

    # construct optimizer
    def configure_optimizers(self):
        grouped_params = [
        {
            "params": [param for name, param in self.model.named_parameters() if self.should_param_decay(name)],
            "weight_decay": self.hparams.weight_decay
        },
        {
            "params": [param for name, param in self.model.named_parameters() if not self.should_param_decay(name)],
            "weight_decay": 0.0,
        }]
        optimizer = AdamW(grouped_params, lr = self.hparams.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = self.hparams.warmup_steps, num_training_steps = self.hparams.total_steps)
        
        #return (optimizer, scheduler)  # for manual training only
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

# write trained model
def write_model(out_dir, model):
    save_dir = os.path.join(out_dir, "pretrained")
    os.makedirs(save_dir, exist_ok = True)
    model.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Saved final model to", save_dir)

# training cycle that uses pl automatic Trainers
def train_auto(model, out_dir = ".", max_epochs = 3):
    assert torch.cuda.is_available(), "CUDA must be available"
    checkpoint_fn = pl.callbacks.ModelCheckpoint(dirpath = out_dir, save_top_k = 1, monitor = "val_loss", mode = "min")
    logger = pl.loggers.TensorBoardLogger(save_dir = out_dir, name = "training_log")
    trainer = pl.Trainer(accelerator = device,
                         devices = 1,
                         max_epochs = max_epochs,
                         precision = 16,
                         callbacks = [checkpoint_fn],
                         logger = logger,
                         default_root_dir = out_dir)
    print("Finished constructing trainers")
    
    # automatic training
    trainer.fit(model, train_loader, val_loader)
    print("Trained model for {} epochs".format(max_epochs))
    write_model(out_dir, model)

# load all dictionary values to a device
def dict_to_device(dct, device = "cuda"):
    for key, value in dct.items():
        dct[key] = value.to(device)

'''
# return the MSE loss of the valid set
def get_val_loss():
    loss = 0.0
    model.eval()
    
    with torch.inference_mode():
        for batch in val_loader:
            dict_to_device(batch)
            logits = model(batch)
            scores = batch["scores"]
            loss += model.get_loss(logits, scores)
    
    return loss / len(val_set)

# standard pytorch trainer
def train_manual(max_epochs = 3):
    assert torch.cuda.is_available(), "CUDA must be available"
    model.train()
    optimizer, scheduler = model.optimizer_cons()
    optimizer.zero_grad()
    best_valid_MSE = torch.inf
    save_dir = out_dir, "finetuned"
    os.makedirs(save_dir, exist_ok=True)
    print("")

    for epoch in range(max_epochs):
        for i, batch in enumerate(train_loader):
            dict_to_device(batch)
            logits = model(batch)
            scores = batch["scores"]
            loss = nn.functional.mse_loss(logits, scores) / grad_accum
        
            loss.backward()

            len_set = len(train_loader)
            if (i % grad_accum == grad_accum - 1):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # save lowest valid MSE model
        valid_MSE = get_val_loss()
        print("")
        if valid_MSE < best_valid_MSE:
            model.model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print("")
    
    print("")
'''

# given a chunk of text, return the rating the model predicts
def predict_score(model, tokenizer, text):
    if isinstance(text, list):
        text = ' '.join(text)
    
    model.eval()
    with torch.inference_mode():
        input_dict = collate_fn(tokenizer, text)
        dict_to_device(input_dict)
        return model(input_dict).item()

# given an index of validation set, print the performance
def predict_score_index(model, tokenizer, index, verbose = True):
    text = val_set[index][0]
    score = predict_score(model, tokenizer, text)
    
    if verbose:
        print("The first 10 reviews are: ")
        text = text.split("\n")
        for i, line in enumerate(text):
            if line.strip():
                print(line.strip())
        print("\nThe actual 50-review score is", val_set[index][1])
        print("The predicted score is", score)
    return score

# model construction
if __name__ == "__main__":
    #model_name = "allenai/longformer-base-4096"
    model_name = "/scratch/midway3/zongqi/Workshop/NLP_Project/decsum/longformer-model"
    train = "preprocess_out/train.jsonl.gz"
    val = "preprocess_out/valid.jsonl.gz"
    tokenizer_name = model_name
    max_tokens = 3000
    batch_size = 4  # effective batch_size = batch_size * grad_accum
    grad_accum = 1
    lr = 5e-5
    warmup_steps = 500
    weight_decay = 0
    max_epochs = 3
    out_dir = "model_out"
    num_workers = 8
    train_model = True # train or load

    #global device, tokenizer, model_0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    os.makedirs(out_dir, exist_ok = True)
    #global train_set, val_set
    train_set = Yelp_dataset(path = train, tokenizer = tokenizer, max_tokens = max_tokens)
    val_set = Yelp_dataset(path = val, tokenizer = tokenizer, max_tokens = max_tokens)
    print("Successfully constructed training and validation sets")
    
    #global train_loader, val_loader
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = num_workers, collate_fn = train_set.collate_fn)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = False, num_workers = num_workers, collate_fn = val_set.collate_fn) 

    if train_model:
        total_steps = len(train_loader) * max_epochs // grad_accum
        model_0 = linear_model(model_name = model_name, lr = lr, grad_accum = grad_accum, weight_decay = weight_decay, warmup_steps = warmup_steps, total_steps = total_steps)
        print("Finished model construction, starting training")
        train_auto(model_0, out_dir, max_epochs)
        #train_manual(max_epochs = max_epochs)
        print("Execution complete")
    else:
        load_path = os.path.join(out_dir, "pretrained")
        model_0 = linear_model(model_name = load_path)
        tokenizer = AutoTokenizer.from_pretrained(load_path)
        print("Loaded fine-tuned model")