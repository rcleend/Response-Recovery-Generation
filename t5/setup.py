""" 
NOTES:
    hyper parameters:
        - learning_rate = 0.001
        - batch_size = 256
        - steps = 5000

# During inference, use beam search with a width of 4 and lenght penalty of 0.6 (?)

"""

"""
===================================================================================
SETUP
===================================================================================
"""

# let's define model parameters specific to T5
from trainer import T5Trainer


model_params = {
    "MODEL": "t5-base",  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": 8,  # training batch size
    "VALID_BATCH_SIZE": 8,  # validation batch size
    "TRAIN_EPOCHS": 3,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 0.001,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 512,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
}


T5Trainer(
    model_params=model_params,
    output_dir="outputs",
)

