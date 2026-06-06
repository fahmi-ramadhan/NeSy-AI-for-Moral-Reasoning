import os

from models.WeakUnification import WeakUnification


class DataProcessor:
    def __init__(self):
        self.chars = ["'s", '?', ',', '(', ')', '[', ']', '-', "n't", "'d", "'m"]
        self.prolog_builtins = {'see', 'tell', 'put', 'get', 'write', 'read', 'nl', 'tab', 'fail', 'true', 'false', 'repeat', 'close', 'open'}

    def process_string(self, string: str) -> str:
        import unicodedata
        string = ''.join(
            ' ' if unicodedata.category(c) in ('Zs', 'Cf') else c
            for c in string
        )
        import re
        string = re.sub(r'[‐-―−﹘﹣－]', '-', string)
        string = string.encode('ascii', errors='ignore').decode('ascii')
        for char in self.chars:
            if char == "n't":
                string = string.replace(char, 'not')
            else:
                string = string.replace(char, '')
        return ' '.join(string.lstrip().lower().split())

    def split_string(self, string) -> list:
        if isinstance(string, float):
            string = str(string)
        return string.split(',')

    def process_data(self, agents_a: list, actions_a: list, patients_a: list, arguments_a: list, agents_b: list, actions_b: list, patients_b: list, arguments_b: list):
        """
        Cleans and normalises lists from semantic role extraction.
        """
        agent_list_a = [
            self.process_string(a) for a in agents_a
            if a and a.lower() not in ('i', '')
        ]
        action_list_a = [
            self.process_string(a) for a in actions_a if a and a.lower() not in self.prolog_builtins
        ]
        patient_list_a = [
            self.process_string(p) for p in patients_a if p
        ]
        argument_list_a = [
            self.process_string(a) for a in arguments_a if a
        ]
        agent_list_b = [
            self.process_string(a) for a in agents_b
            if a and a.lower() not in ('i', '')
        ]
        action_list_b = [
            self.process_string(a) for a in actions_b if a and a.lower() not in self.prolog_builtins
        ]
        patient_list_b = [
            self.process_string(p) for p in patients_b if p
        ]
        argument_list_b = [
            self.process_string(a) for a in arguments_b if a
        ]
        return agent_list_a, action_list_a, patient_list_a, argument_list_a, agent_list_b, action_list_b, patient_list_b, argument_list_b


class WeakRulesGenerator:

    def __init__(self):
        self.data_processor = DataProcessor()
        self.weak_unification = WeakUnification()
        print("WeakRulesGenerator initialised.")

    def get_weak_rules(self, agents_a: list, actions_a: list, patients_a: list,
                       arguments_a: list, agents_b: list, actions_b: list, patients_b: list,
                       arguments_b: list, goal_predicate: str, q_id: str, iteration: int):
        """
        Reads autoformalized rules from kb/rules/question_{q_id}/{iteration}it.txt,
        then calls WeakUnification.create_rule() to build the full KB.
        """
        agent_list_a, action_list_a, patient_list_a, argument_list_a, agent_list_b, action_list_b, patient_list_b, argument_list_b = self.data_processor.process_data(
            agents_a, actions_a, patients_a, arguments_a, agents_b, actions_b, patients_b, arguments_b
        )

        kb_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'kb')
        rules_path = os.path.join(kb_base, 'rules', f'question_{q_id}', f'{iteration}it.txt')
        prolog_transferred_rule_list = []
        with open(rules_path, 'r') as f:
            prolog_transferred_rule_list = [
                item.split('. =')[0].strip()
                for item in f.readlines()
                if item.strip()
            ]

        self.weak_unification.create_rule(
            prolog_rules=prolog_transferred_rule_list,
            goal_predicate=goal_predicate,
            q_id=q_id,
            iteration=iteration,
            agent_list_a=agent_list_a,
            action_list_a=action_list_a,
            patient_list_a=patient_list_a,
            argument_list_a=argument_list_a,
            agent_list_b=agent_list_b,
            action_list_b=action_list_b,
            patient_list_b=patient_list_b,
            argument_list_b=argument_list_b,
        )
