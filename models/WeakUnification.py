import os
import re

import gensim.downloader as api
import numpy as np
from scipy import spatial


def sanitize_prolog_rule(rule: str) -> str:
    # Remove or replace special characters that are invalid in Prolog
    # Remove dollar signs (replace with empty string, not allowed in Prolog)
    rule = re.sub(r'\$', '', rule)
    # Percent signs (%) are used for comments in Prolog, so replace with "percent"
    rule = rule.replace('%', 'percent')
    # Remove single quotes and apostrophes
    rule = rule.replace("'", '').replace("'", '')
    # Remove other problematic characters
    rule = re.sub(r'[<>{}\[\]|\\^`~]', '', rule)
    # Handle hyphens within predicate names
    rule = re.sub(r'(?<=[a-zA-Z0-9_])-(?=[a-zA-Z0-9_])', '_', rule)
    # Remove dots in predicate names (but not in numbers like 0.25 in arguments)
    # Handle cases like $8.5_million -> 8_5_million after $ removal
    rule = re.sub(r'(?<=[a-z])(\.)(?=[a-z_])', '_', rule)
    rule = re.sub(r'\.+(?=[a-zA-Z_(])', '', rule)
    rule = re.sub(r'(^|[(:-])(\d\w*)', r'\1p_\2', rule)
    rule = rule.replace('"', '')
    return rule.strip()


def sanitize_string_for_prolog_predicate(s: str) -> str:
    """Sanitize individual role strings for use in Prolog facts."""
    # Remove single quotes and apostrophes first
    s = s.replace("'", '').replace("'", '')
    # Replace percent with word first (before number separation)
    s = s.replace('%', 'percent')
    # Remove other special characters
    s = re.sub(r'[<>{}\[\]|\\^`~:*]', '', s)
    # Replace hyphens with underscores
    s = re.sub(r'-', '_', s)
    # Replace dots in numbers with underscores (e.g., 8.5 -> 8_5, 0.25 -> 0_25)
    s = re.sub(r'(\d)\.(\d)', r'\1_\2', s)
    # Replace spaces with underscores
    s = s.replace(' ', '_')
    # Insert underscore between number and letter (e.g., 80percent -> 80_percent, 5year -> 5_year)
    s = re.sub(r'(\d)([a-zA-Z])', r'\1_\2', s)
    s = re.sub(r'([a-zA-Z])(\d)', r'\1_\2', s)
    # Remove non-ASCII characters
    s = s.encode('ascii', errors='ignore').decode('ascii')
    # Clean up multiple underscores
    s = re.sub(r'_+', '_', s)
    return s.strip().lower()

class WeakUnification:
    def __init__(self, model_name: str = "glove-wiki-gigaword-300", threshold: float = 0.5):
        self.threshold = threshold
        self.model = None
        self.model = api.load(model_name)

    def split_into_words(self, sentence: str) -> list:
        return sentence.lower().split()

    def get_sentence_vector(self, sentence):
        words = self.split_into_words(sentence)
        vectors = []
        for word in words:
            # Skip tokens that don't exist in GloVe vocabulary
            # Also sanitize: replace % with percent, separate number-letter, remove quotes
            word = word.replace('%', 'percent')
            word = word.replace("'", '').replace("'", '')
            word = re.sub(r'(\d)([a-zA-Z])', r'\1_\2', word)
            word = re.sub(r'([a-zA-Z])(\d)', r'\1_\2', word)
            if word in self.model: #type: ignore
                vectors.append(self.model[word]) #type: ignore
        if not vectors:
            return None
        return np.mean(vectors, axis=0)

    def calculate_cosine_similarity(self, s1: str, s2: str) -> float:
        if self.model is None:
            return 0.0
        try:
            vec1 = self.get_sentence_vector(s1)
            vec2 = self.get_sentence_vector(s2)
            if vec1 is None or vec2 is None:
                return 0.0
            return float(1 - spatial.distance.cosine(vec1, vec2))
        except (ValueError, TypeError):
            return 0.0

    def _process_list(self, list_to_process: list, predicate_phrase: str):
        for role_phrase in list_to_process:
            if role_phrase in ('None', '', 'none'):
                continue
            sim = self.calculate_cosine_similarity(role_phrase, predicate_phrase)
            if sim >= self.threshold:
                # Sanitize for Prolog output
                sanitized_role = sanitize_string_for_prolog_predicate(role_phrase)
                sanitized_predicate = sanitize_string_for_prolog_predicate(predicate_phrase)
                sanitized_role = re.sub(r'^(\d)', r'p_\1', sanitized_role)
                sanitized_predicate = re.sub(r'^(\d)', r'p_\1', sanitized_predicate)
                head = sanitized_predicate + '(X) :- '
                body = sanitized_role + '(X)'
                rule = head + body + '. = ' + str(sim)
                return rule
        return None

    def create_rule(self, prolog_rules: list, goal_predicate: str,
                    q_id: str, iteration: int,
                    agent_list_a: list = [],
                    action_list_a: list = [],
                    patient_list_a: list = [],
                    argument_list_a: list = [],
                    agent_list_b: list = [],
                    action_list_b: list = [],
                    patient_list_b: list = [],
                    argument_list_b: list = []):

        semantic_roles_a = [agent_list_a, action_list_a, patient_list_a, argument_list_a]
        semantic_roles_b = [agent_list_b, action_list_b, patient_list_b, argument_list_b]

        transferred_rules_to_write = []
        facts_to_write             = []
        weak_rules_to_write        = []
        similarity_to_write        = []
        head_body_dic              = {}
        predicates_to_calculate    = []

        for rule in prolog_rules:
            rule = sanitize_prolog_rule(rule)
            if not rule:
                continue

            if ':-' in rule:
                core = re.sub(r'\.\s*=\s*[\d.]+\s*$', '', rule).strip()
                transferred_rules_to_write.append(core + '. = 1.0')

                head_part = rule.split(':-')[0]
                body_part = re.sub(r'\.\s*=\s*[\d.]+\s*$', '',
                                   rule.split(':-')[1]).strip()
                x_idx = head_part.find('(')
                head_pred = head_part[:x_idx].strip() if x_idx != -1 else head_part.strip()
                
                head_pred_sanitized = sanitize_string_for_prolog_predicate(head_pred)
                head_phrase = head_pred_sanitized.replace('_', ' ')

                body_preds = re.findall(r'\b([a-z0-9][a-z0-9_]*)\s*\(', body_part)

                def add_p_prefix(predicate):
                    return re.sub(r'^(\d)', r'p_\1', predicate)

                if body_preds:
                    body_pred_sanitized = add_p_prefix(sanitize_string_for_prolog_predicate(body_preds[0]))
                    head_body_dic[head_phrase] = body_pred_sanitized.replace('_', ' ')
                    if len(body_preds) > 1:
                        for bp in body_preds[1:]:
                            bp_sanitized = add_p_prefix(sanitize_string_for_prolog_predicate(bp))
                            clean = bp_sanitized.replace('_', ' ')
                            predicates_to_calculate.append(clean)
                        predicates_to_calculate.append(head_phrase)
            else:
                core = re.sub(r'\.\s*=\s*[\d.]+\s*$', '', rule).strip().rstrip('.')
                is_goal_fact = re.match(
                    rf'^{re.escape(goal_predicate)}\(option_[a-z]+\)\.?$', core
                )
                if not is_goal_fact:
                    facts_to_write.append(core + '. = 1.0')

        transferred_rules_to_write = list(set(transferred_rules_to_write))
        facts_to_write             = list(set(facts_to_write))

        for pred_idx in range(len(predicates_to_calculate) - 1):
            pred_phrase = predicates_to_calculate[pred_idx]
            try:
                for semantic_role_list in semantic_roles_a + semantic_roles_b:
                    sim_temp = self._process_list(semantic_role_list, pred_phrase)
                    if sim_temp:
                        weak_rules_to_write.append(sim_temp)
            except KeyError:
                continue

        principles_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'kb', f'principles_{goal_predicate}.txt'
        )

        principles_phrases = []
        if os.path.exists(principles_path):
            with open(principles_path) as f:
                for line in f:
                    if ':-' in line:
                        body = line.split(':-')[1]
                        if ',' in body:
                            parts = body.split(',')
                            for part in parts:
                                pred = part.strip().split('(')[0].strip()
                                if pred:
                                    phrase = pred.replace('_', ' ')
                                    if phrase not in principles_phrases:
                                        principles_phrases.append(phrase)
                        else:
                            pred = body.strip().split('(')[0].strip()
                            if pred:
                                phrase = pred.replace('_', ' ')
                                if phrase not in principles_phrases:
                                    principles_phrases.append(phrase)

        for head_phrase, body_phrase in head_body_dic.items():
            
            head_sanitized = sanitize_string_for_prolog_predicate(head_phrase.replace('_', ' '))
            body_sanitized = sanitize_string_for_prolog_predicate(body_phrase.replace('_', ' '))
            
            head_sanitized = re.sub(r'^(\d)', r'p_\1', head_sanitized)
            body_sanitized = re.sub(r'^(\d)', r'p_\1', body_sanitized)
            
            try:
                for principle_phrase in principles_phrases:
                    sim = self.calculate_cosine_similarity(head_sanitized, principle_phrase)
                    if self.threshold <= sim < 1.0:
                        arg1 = principle_phrase.replace(' ', '_') + '(X) :- '
                        arg2 = head_sanitized + '(X)'
                        rule = arg1 + arg2 + '. = ' + str(sim)
                        sim_entry = (principle_phrase.replace(' ', '_') + ' ~ ' +
                                     head_sanitized + ' = ' + str(sim))
                        weak_rules_to_write.append(rule)
                        similarity_to_write.append(sim_entry)

                for other_head, other_body in head_body_dic.items():
                    if head_phrase == other_head:
                        continue
                    other_body_sanitized = sanitize_string_for_prolog_predicate(other_body.replace('_', ' '))
                    sim = self.calculate_cosine_similarity(head_sanitized, other_body_sanitized)
                    if self.threshold <= sim < 1.0:
                        dup = (other_body_sanitized + '(X) :- ' +
                               head_sanitized + '(X)')
                        if any(dup in r for r in transferred_rules_to_write):
                            pass
                        else:
                            weak_rules_to_write.append(dup + '. = ' + str(sim))

                if head_sanitized in predicates_to_calculate:
                    pass
                else:
                    for pred_phrase in predicates_to_calculate:
                        pred_sanitized = sanitize_string_for_prolog_predicate(pred_phrase.replace('_', ' '))
                        pred_sanitized = re.sub(r'^(\d)', r'p_\1', pred_sanitized)
                        sim = self.calculate_cosine_similarity(head_sanitized, pred_sanitized)
                        if self.threshold <= sim < 1.0:
                            arg1 = pred_sanitized + '(X) :- '
                            arg2 = head_sanitized + '(X)'
                            rule = arg1 + arg2 + '. = ' + str(sim)
                            weak_rules_to_write.append(rule)

                for semantic_role_list in semantic_roles_a + semantic_roles_b:
                    sim_temp = self._process_list(semantic_role_list, body_sanitized)
                    if sim_temp:
                        weak_rules_to_write.append(sim_temp)
                    sim_temp = self._process_list(semantic_role_list, body_phrase)
                    if sim_temp:
                        weak_rules_to_write.append(sim_temp)

            except (KeyError, ValueError):
                continue

        for semantic_role_list in semantic_roles_a:
            for role_phrase in semantic_role_list:
                if role_phrase not in ('None', '', 'none'):
                    sanitized_role = sanitize_string_for_prolog_predicate(role_phrase)
                    if not sanitized_role:
                        continue
                    
                    opt_fact = sanitized_role + '(option_a). = 1.0'
                    facts_to_write.append(opt_fact)

        for semantic_role_list in semantic_roles_b:
            for role_phrase in semantic_role_list:
                if role_phrase not in ('None', '', 'none'):
                    sanitized_role = sanitize_string_for_prolog_predicate(role_phrase)
                    if not sanitized_role:
                        continue
                    
                    opt_fact = sanitized_role + '(option_b). = 1.0'
                    facts_to_write.append(opt_fact)
        
        facts_to_write = list(set(facts_to_write))

        VALID_LINE = re.compile(r"^.+\.\s*=\s*[\d.]+\s*$")

        principles_lines = []
        if os.path.exists(principles_path):
            with open(principles_path) as f:
                principles_lines = f.readlines()

        directory = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'kb', 'prolog_kb', f'question_{q_id}'
        )
        os.makedirs(directory, exist_ok=True)

        seen = set()
        ordered_lines = []

        def add_line(line):
            key = sanitize_prolog_rule(line)
            if not key or key in seen:
                return
            if not VALID_LINE.match(key):
                return
            seen.add(key)
            ordered_lines.append(key + '\n')

        for line in principles_lines:
            add_line(line.strip())
        for rule in transferred_rules_to_write + facts_to_write:
            add_line(rule)
        for rule in weak_rules_to_write:
            add_line(rule)

        if not ordered_lines:
            return

        output_path = os.path.join(directory, f'{iteration}it.txt')
        with open(output_path, 'w') as outfile:
            outfile.writelines(ordered_lines)

        kb_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kb"
        )
        os.makedirs(kb_dir, exist_ok=True)
        sims_path = os.path.join(kb_dir, 'sims.txt')

        cluster_lines = []
        for entry in similarity_to_write:
            try:
                left, right = entry.split(' ~ ')
                pred2, score = right.split(' = ')
                cluster = f"{left}_{pred2}"
                line_a = f"{left} ~ {cluster} = {score}"
                line_b = f"{pred2} ~ {cluster} = {score}"
                cluster_lines.append(line_a)
                cluster_lines.append(line_b)
            except (ValueError, AttributeError):
                continue

        with open(sims_path, 'w') as f:
            for line in cluster_lines:
                f.write(line + '\n')
