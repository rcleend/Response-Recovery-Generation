"""
Discription of the issue I want to fix:

Possible system acts (the acts which require a user response are indicated with an X):
    - REQUEST X -> (W-questions: what city/time etc.)
    - CONFIRM X -> (please confirm x; yes/no)
    - OFFER X -> (system makes suggestion)
    - OFFER_INTENT X -> ("would you like to perform this intent?"; AFFIRM_INTENT/NEGATE_INTENT/INFORM_INTENT)
    - REQ_MORE X -> ("is there anything else I can help you with?"; yes/no/other)

    - INFORM
    - NOTIFY_SUCCESS
    - NOTIFY_FAILURE
    - INFORM_COUNT 
    - GOODBYE


Two possible solutions regarding the generation of templates:

Solution 1: generic error templates:
    - ERROR_AREP!!@ -> "could you please repeat @?" (Maybe AREP is applicable for all acts?)
    - ERROR_RP!!@ -> repeat previous utterance (should also be applicable for all acts)
    - ERROR_TYCS!!@@ -> "you can say @ or @"
    - ERROR_REQUEST_ARPH!!@ -> "could you please rephrase @?" (Not applicable for yes/no question)
    - ERROR_OFFER_INTENT_ARPH!!@ -> "could you please rephrase @?" (Not applicable for yes/no question)
    - ERROR_REQ_MORE_ARPH!!@ -> "could you please rephrase @?" (Not applicable for yes/no question)

Solution 2: domain specific templates:
    - restaurant: ERROR_REQUEST!!MOVE!!time -> "They still have a table for 6PM. Is that okay?"

Solution 3: Combination of both generic and domain specific error templates
    - only use domain specific error templates for MOVE strategy
"""


"""
Update 5_shot dataset with error recovery dialog according to the paper
Paper guidelines:
    - per domain: examples for every dialogue act and slot type

TODO: alter the dataset with the templates described in solution 1

Per domain following:
    - AREP
    - RP
    - TYCS
    - DRP
    - MOVE


Homes
    [] AREP
    [] RP
    [] TYCS

Buses
    [] AREP
    [] RP
    [] TYCS

Media:
    [] AREP
    [] RP
    [] TYCS

RideShare
    [] AREP
    [] RP
    [] TYCS

Movies
    [] AREP
    [] RP
    [] TYCS

Flights
    [] AREP
    [] RP
    [] TYCS

Music
    [] AREP
    [] RP
    [] TYCS

Services
    [] AREP
    [] RP
    [] TYCS

RentalCars
    [] AREP
    [] RP
    [] TYCS

Restaurant:
    [] AREP
    [] RP
    [] TYCS

Events
    [] AREP
    [] RP
    [] TYCS

Hotels
    [] AREP
    [] RP
    [] TYCS


"""

import json
import random


class ErrorDialogueHelper():
    def __init__(self, dialogues_path = '../data/error/all_dialogues.json', save_path='../data/error/error/', system_dialogue_acts=[], error_dialogue_acts=[]):
        self.__dialogues = self.__get_dialogues_file(dialogues_path)
        self.__save_path = save_path
        self.__system_dialogue_acts = system_dialogue_acts
        self.__error_dialogue_acts = error_dialogue_acts 
        self.__error_dialogue_acts_copy = error_dialogue_acts.copy()


    def create_random_error_dialogues(self, domain, amount=5, seed=0):
        try:
            self.__error_dialogue_acts_copy = self.__error_dialogue_acts.copy()
            domain_dialogues = self.__filter_domain_dialogues(domain)
            qa_list = self.__get_qa_list_from_dialogues(domain_dialogues)
            filtered_qa_list = self.__get_qa_for_each_dialogue_act(qa_list)
            random_qa_list_subset = self.__get_random_qa_list_subset(filtered_qa_list, subset_size=amount, seed=seed)
            non_understanding_qar_list = self.__create_non_understandings_and_system_response(random_qa_list_subset)

            if self.__confirm_random_selection(non_understanding_qar_list):
                self.__save_qar_list_to_json(domain, non_understanding_qar_list)
            else:
                seed+=1
                self.create_random_error_dialogues(domain, amount, seed=seed)
        except:
            print('Max options reached. No other error dialogues could be generated')

    def __filter_domain_dialogues(self, domain):
        domain_dialogues = []
        for dialogue in self.__dialogues:
            if 'services' in dialogue and domain in str(dialogue['services']):
                domain_dialogues.append(dialogue)

        return domain_dialogues

    def __get_dialogues_file(self, dialogues_path):
        with open(dialogues_path, 'r') as f:
           dialogues = json.load(f)

        return dialogues 

    # Gets an list with touples os a System Questions and User answers
    def __get_qa_list_from_dialogues(self, dialogues):
        qa_list = []
        for dialogue in dialogues:
            qa = [(dialogue['turns'][i - 1], turn) for i, turn in enumerate(dialogue['turns']) if turn['speaker'] == 'USER']
            qa_list.extend(qa)

        return qa_list


    def __get_random_qa_list_subset(self, qa_list, subset_size=5, seed=0):
        random.seed(seed)
        random.shuffle(qa_list)

        if len(qa_list) < subset_size:
            raise Exception('Subset size must be smaller than the total amount of user turns.')

        return qa_list[:subset_size]
    

    def __get_qa_for_each_dialogue_act(self, qa_list):
        dialogue_acts_copy = self.__system_dialogue_acts.copy()
        filtered_qa_list = []
        for system_question, user_answer in qa_list:
            # Check if the system question contains any of the dialogue acts that allow for a non_understanding
            system_actions = system_question['frames'][0]['actions']
            found_system_actions = [action for action in system_actions if action['act'] in dialogue_acts_copy]

            if len(found_system_actions) > 0:
                filtered_qa_list.append((system_question, user_answer))
                dialogue_acts_copy.remove(found_system_actions[0]['act'])
                    
        return filtered_qa_list
    

    def __create_non_understandings_and_system_response(self, qa_list):
        non_understanding_qar_list = []
        for system_question, user_answer in qa_list:
            non_understanding_action_list = [action for action in system_question['frames'][0]['actions'] if action['act'] in self.__system_dialogue_acts]
            misunderstood_system_action = random.choice(non_understanding_action_list)
            
            # iterate over the actions in user_answers and return the action that has the same slot value as the random_action
            misunderstood_user_actions, understood_user_actions = self.__get_misunderstood_user_actions(user_answer, misunderstood_system_action)

            if len(misunderstood_user_actions) > 0:
                user_answer['utterance'] = self.__add_non_understanding_to_utterance(user_answer['utterance'], misunderstood_user_actions[0]['values'])
            else:
                user_answer['utterance'] = '[NON UNDERSTANDING]'
                
            system_response = self.__create_response_utterance(misunderstood_system_action, understood_user_actions)

            non_understanding_qar_list.append((system_question, user_answer, system_response))

        return non_understanding_qar_list

    def __get_misunderstood_user_actions(self, user_answer, misunderstood_system_action):
        understood_actions = []
        misunderstood_actions = []

        for action in user_answer['frames'][0]['actions']:
            if not action['slot']: 
                continue

            if action['slot'] ==  misunderstood_system_action['slot']:
                misunderstood_actions.append(action)
            else:
                understood_actions.append(action)

        return (misunderstood_actions, understood_actions)


    def __add_non_understanding_to_utterance(self, utterance, value_list):
        for value in value_list:
            utterance = utterance.replace(value, '[NON UNDERSTANDING]')

        return utterance

    def __create_response_utterance(self, misunderstood_system_action, understood_user_actions):

        # create actions dict list
        misunderstood_system_action['act'] = self.__get_error_act(misunderstood_system_action)
        actions_dict = [misunderstood_system_action]

        for action in understood_user_actions:
            action['act'] = 'INFORM'
            actions_dict.append(action)

        return {
                'frames': [{'actions': actions_dict}],
                'speaker': 'SYSTEM',
                'utterance': 'TODO'
                }


    def __get_error_act(self, action):
        # Get error tsv file
        random_error_act = random.choice(self.__error_dialogue_acts_copy)
        self.__error_dialogue_acts_copy.remove(random_error_act)
        return  random_error_act

    def __confirm_random_selection(self, qar_list):
        for question, answer, response in qar_list:
            print('====================================')
            print('-----------------------------------')
            print(question['utterance'])
            print('-----------------------------------')
            self.__print_actions(question)
            print('-----------------------------------')
            print(answer['utterance'])
            print('-----------------------------------')
            self.__print_actions(answer)
            print('-----------------------------------')
            print(response['utterance'])
            print('-----------------------------------')
            self.__print_actions(response)


        answer = input("Confirm Selection? y/n")

        if answer.lower() in ["y", "yes"]:
            return True
        elif answer.lower() in ["n", "no"]:
            return False
        else:
            print('Wrong input. Try again.')
            self.__confirm_random_selection(qar_list)
        

    def __print_actions(self, turn):
        for action in turn['frames'][0]['actions']:
            print('ACT: ' + action['act'], end=' ')
            print(' SLOT: ' + action['slot'], end=' ')
            print(' VALUES: ' + str(action['values']))

    def __save_qar_list_to_json(self, domain, qar_list):
        json_list = []
        for (question, answer, response) in qar_list:
            dict = {}
            dict['services'] = [domain]
            dict['turns'] = [question, answer, response]
            json_list.append(dict)

        with open(self.__save_path + domain + '.json', 'w') as f:
            json.dump(json_list, f, indent=4)


            



system_dialogue_acts = ['REQUEST', 'CONFIRM', 'OFFER', 'OFFER_INTENT', 'REQ_MORE']         
error_dialogue_acts = ['ERROR_AREP', 'ERROR_RP', 'ERROR_TYCS', 'ERROR_DRP', 'ERROR_MOVE']

# train_helper = ErrorDialogueHelper(dialogues_path = '../data/error/all_dialogues.json', save_path='../data/error/error/', system_dialogue_acts=system_dialogue_acts, error_dialogue_acts=error_dialogue_acts)
test_helper = ErrorDialogueHelper(dialogues_path = '../data/test/all_dialogues.json', save_path='../data/error/test/', system_dialogue_acts=system_dialogue_acts, error_dialogue_acts=error_dialogue_acts)

# train_helper.create_random_error_dialogues(domain='Buses', seed=1)
# train_helper.create_random_error_dialogues(domain='Events', seed=1)
# train_helper.create_random_error_dialogues(domain='Flights', seed=1)
# train_helper.create_random_error_dialogues(domain='Homes', seed=1)
# train_helper.create_random_error_dialogues(domain='Hotels', seed=1)
# train_helper.create_random_error_dialogues(domain='Media', seed=1)
# train_helper.create_random_error_dialogues(domain='Movies', seed=1)
# train_helper.create_random_error_dialogues(domain='Music', seed=1)
# train_helper.create_random_error_dialogues(domain='RentalCars', seed=1)
# train_helper.create_random_error_dialogues(domain='Restaurant', seed=1)
# train_helper.create_random_error_dialogues(domain='Service', seed=1)
# train_helper.create_random_error_dialogues(domain='Ride', seed=1)

# test_helper.create_random_error_dialogues(domain='Buses', seed=1, amount=2)
# test_helper.create_random_error_dialogues(domain='Events', seed=1, amount=2)
# test_helper.create_random_error_dialogues(domain='Flights', seed=1, amount=2)
# test_helper.create_random_error_dialogues(domain='Homes', seed=1, amount=2)
# test_helper.create_random_error_dialogues(domain='Hotels', seed=1, amount=2)
# test_helper.create_random_error_dialogues(domain='Media', seed=1, amount=2)
# test_helper.create_random_error_dialogues(domain='Movies', seed=1, amount=2)
# test_helper.create_random_error_dialogues(domain='Music', seed=1, amount=2)
# test_helper.create_random_error_dialogues(domain='RentalCars', seed=1, amount=2)
# test_helper.create_random_error_dialogues(domain='Restaurant', seed=1, amount=2)
# test_helper.create_random_error_dialogues(domain='Service', seed=1, amount=2)
test_helper.create_random_error_dialogues(domain='Ride', seed=1, amount=2)


        
