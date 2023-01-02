import json
import os
import glob

"""
==============================================
Create new dialogues file using x_splits
==============================================
"""
def save_dialogues(splitted_dialogue_ids_dict):
    filtered_dialogues = []
    for domain_id, dialogue_ids in splitted_dialogue_ids_dict.items():
        with open('../data/error/dialogues_' + domain_id + '.json') as f:
            dialogues_json = json.load(f)

        filtered_dialogues.extend([dialogue for dialogue in dialogues_json if dialogue['dialogue_id'] in dialogue_ids])

    # Save filtered_dialogues to json file in ../data/error/domain_id.json
    with open('../data/splits/5_shot/dialogues.json', 'w') as f:
        json.dump(filtered_dialogues, f)

    
def split_dialogue_ids_by_file_id(dialogue_ids):
    dialogue_id_dict = {}
    for dialogue_id in dialogue_ids:
        file_id = dialogue_id.split("_")[0]

        if file_id in dialogue_id_dict:
            dialogue_id_dict[file_id].append(dialogue_id)
        else:
            dialogue_id_dict[file_id] = [dialogue_id]

    return dialogue_id_dict
        
def get_dialogue_ids_from_txt(path):
    with open(path) as f:
        dialogue_ids = f.read().splitlines()

    return dialogue_ids

"""
Uncomment to use the functions above
"""
# dialogue_ids = get_dialogue_ids_from_txt('../data/splits/5_shot/5_shot.txt')

# splitted_dialogue_ids_dict = split_dialogue_ids_by_file_id(dialogue_ids)

# save_dialogues(splitted_dialogue_ids_dict)

# with open('../data/error/dialogues_10_shot.json', 'w') as f:
#     json.dump(filtered_dialogues, f)


"""
==============================================
Combine all dialogues file into a single large one
==============================================
"""

def concatenate_dialogue_files(path):
    all_dialogues = []
    for filename in glob.glob(os.path.join(path, '*.json')): #only process .JSON files in folder.      
        with open(filename) as currentFile:
            dialogues_json = json.load(currentFile)
            all_dialogues.extend(dialogues_json)

    return all_dialogues

def save_dialogues_to_file(path, dialogues):
    with open(path, 'w') as f:
        json.dump(dialogues, f)
        print('Successfully created: ' + path)

"""
Uncomment to use the functions above
"""
# all_train_dialogues = concatenate_dialogue_files('../data/error/')
# all_test_dialogues = concatenate_dialogue_files('../data/test/')

# save_dialogues_to_file('../data/error/all_dialogues.json', all_train_dialogues)
# save_dialogues_to_file('../data/test/all_dialogues.json', all_test_dialogues)

