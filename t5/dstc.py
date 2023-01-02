import torch
from torch.utils.data import Dataset

import glob
import os
import json
import pandas as pd

from utterance_generator import TemplateUtteranceGenerator

class DstcError(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, 
        tokenizer, 
        source_len, 
        target_len, 
        is_training=True, 
        schema=None
    ):
        """
        Initializes a Dataset class

        Args:
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            template_dir_path (str): Path to template directory
            dialogue_dir_path (str): Path to dialogue directory
        """
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.target_len = target_len
        self.is_training = is_training
        self.schema = schema

        self.template_utterance_generator = TemplateUtteranceGenerator('../data/utterance_templates/')
        self.utterance_df = self.__get_utterance_df()

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.utterance_df)


    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        source_text = self.utterance_df['source'][index]
        target_text = self.utterance_df['target'][index]

        source = self.tokenizer.batch_encode_plus(
            [source_text], max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.target_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }

    def __get_utterance_df(self):
        if self.is_training:
            turns = self.__get_all_training_turns()
        else:
            turns = self.__get_all_test_turns()

        utterance_df = pd.DataFrame(columns=['source', 'target'])
        for turn in turns:
            source = self.template_utterance_generator.get_robot_utterance(turn, self.schema)
            target = turn['utterance']
            utterance_df.loc[len(utterance_df)] = [source , target]

        return utterance_df

    def __get_all_test_turns(self):
        error_dialogues = self.__concatenate_dialogue_files('../data/test/error/')
        test_error_turns = self.__get_relevant_error_turns(error_dialogues)
        return test_error_turns
        
    def __get_all_training_turns(self):
        error_dialogues = self.__concatenate_dialogue_files('../data/train/error/')
        train_error_turns = self.__get_relevant_error_turns(error_dialogues)
        all_system_turns = self.__get_all_system_turns('../data/train/5_shot.json')
        all_system_turns.extend(train_error_turns)
        return all_system_turns

    def __concatenate_dialogue_files(self, path):
        all_dialogues = []
        for filename in glob.glob(os.path.join(path, '*.json')): 
            with open(filename) as currentFile:
                dialogues_json = json.load(currentFile)
                all_dialogues.extend(dialogues_json)

        return all_dialogues

    def __get_relevant_error_turns(self, dialogues):

        return [dialogue['turns'][2] for dialogue in dialogues]
    
    def __get_all_system_turns(self, dialogue_file_path):
        with open(dialogue_file_path) as f:
            dialogues = json.load(f)
    
        return [turn for dialogue in dialogues for turn in dialogue['turns'] if turn['speaker'] == 'SYSTEM']

    




