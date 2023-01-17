import torch
import numpy as np
from dstc import DstcError
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Importing libraries
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console

"""
===================================================================================
SETUP
===================================================================================
"""

console = Console(record=True)

# to display dataframe in ASCII format
def display_df(df):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)

# training logger to log training progress
training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Loss", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)

# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

print(device)


"""
===================================================================================
HELPER FUNCS
===================================================================================
"""

def validate(epoch, tokenizer, model, device, loader):
  """
  Function to evaluate model for predictions

  """
  model.eval()
  predictions = []
  actuals = []
  templates = []
  with torch.no_grad():
      for _, data in enumerate(loader, 0):
          y = data['target_ids'].to(device, dtype = torch.long)
          ids = data['source_ids'].to(device, dtype = torch.long)
          mask = data['source_mask'].to(device, dtype = torch.long)

          generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=512,
              num_beams=4,
              repetition_penalty=2.5, 
              length_penalty=0.6, 
              early_stopping=True
              )

          preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
          target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
          template = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in ids]

          if _%10==0:
              console.print(f'Completed {_}')

          predictions.extend(preds)
          actuals.extend(target)
          templates.extend(template)

  return predictions, actuals, templates

def train(epoch, tokenizer, model, device, loader, optimizer):
    """
    Function to be called for training with the parameters passed from main function

    """
    model.train()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]

        if _ % 10 == 0:
            training_logger.add_row(str(epoch), str(_), str(loss))
            console.print(training_logger)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

"""
===================================================================================
TRAINER FUNC
===================================================================================
"""

def T5Trainer(
    model_params, output_dir="./outputs/"
):

    """
    T5 trainer

    """

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)

    # logging
    console.log(f"[Data]: Reading data...\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = DstcError(
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        is_training=True,
        schema=None,
    )

    val_set = DstcError(
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        is_training=False,
        schema=None,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    eval_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **eval_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )

    # Training loop
    console.log(f"[Initiating Fine Tuning]...\n")

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    console.log(f"[Saving Model]...\n")
    # Saving the model after training
    path = os.path.join(output_dir, "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    # Print test and train predictions
    console.log(f"[Initiating Validation]...\n")
    for epoch in range(model_params["VAL_EPOCHS"]):
        test_predictions, test_actuals, test_templates = validate(epoch, tokenizer, model, device, val_loader)
        test_final_df = pd.DataFrame({"Generated Text": test_predictions, "Actual Text": test_actuals, "Orig. Template": test_templates})
        test_final_df.to_csv(os.path.join(output_dir, "test_predictions.csv"))

        train_predictions, train_actuals, train_templates = validate(epoch, tokenizer, model, device, training_loader)
        train_final_df = pd.DataFrame({"Generated Text": train_predictions, "Actual Text": train_actuals, "Orig. Template" : train_templates})
        train_final_df.to_csv(os.path.join(output_dir, "train_predictions.csv"))

        
    console.save_text(os.path.join(output_dir, "logs.txt"))

    console.log(f"[Validation Completed.]\n")
    console.print(
        f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    )
    console.print(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n"""
    )
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")
