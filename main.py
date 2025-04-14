from nervaluate import Evaluator
from collections import defaultdict
from collections import Counter
import difflib
import os
#import krippendorff



def compute_kappa_from_nervaluate(nervaluate_values):
    # Extract the required counts
    correct = nervaluate_values['correct']
    incorrect = nervaluate_values['incorrect']
    missed = nervaluate_values['missed']
    spurious = nervaluate_values['spurious']

    # Total number of instances
    total = correct + incorrect + missed + spurious

    # Observed agreement (p_o)
    p_o = correct / total if total > 0 else 0

    # Expected agreement (p_e)
    p_e = ((correct + spurious + incorrect) / total) * ((correct + missed) / total) if total > 0 else 0

    # Cohen's Kappa (κ)
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else 0

    print(f"Cohen's Kappa: {kappa:.4f}")
    return kappa
def read_conll_tags(filepath):
    sequences = []
    tokens = []
    tags = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                if tokens:
                    sequences.append((tokens, tags))
                    tokens, tags = [], []
            else:
                parts = stripped.split()
                tokens.append(parts[0])
                tags.append(parts[-1])
        if tokens:
            sequences.append((tokens, tags))

    return sequences


def extract_entities(seq):
    """
    Convert BIO-tagged sequence into list of (start_idx, end_idx, label) spans.
    End index is inclusive.
    """
    spans = []
    start = None
    label = None

    for i, tag in enumerate(seq):
        if tag.startswith("B-"):
            if start is not None:
                spans.append((start, i - 1, label))
            start = i
            label = tag[2:]
        elif tag.startswith("I-") and start is not None and tag[2:] == label:
            continue
        else:
            if start is not None:
                spans.append((start, i - 1, label))
                start = None
                label = None

    if start is not None:
        spans.append((start, len(seq) - 1, label))

    return spans


def highlight_disagreements(file1, file2):
    data1 = read_conll_tags(file1)
    data2 = read_conll_tags(file2)

    assert len(data1) == len(data2), "Sentence count mismatch"

    disagreements = []

    for idx, ((tokens1, tags1), (tokens2, tags2)) in enumerate(zip(data1, data2)):
        assert tokens1 == tokens2, f"Token mismatch in sentence {idx + 1}"
        tokens = tokens1

        spans1 = set(extract_entities(tags1))
        spans2 = set(extract_entities(tags2))

        false_neg = spans1 - spans2  # missed by annotator 2
        false_pos = spans2 - spans1  # added by annotator 2
        true_pos = spans1 & spans2   # agreement

        if false_neg or false_pos:
            print(f"\n--- Sentence {idx + 1} ---")
            print("Tokens:", ' '.join(tokens))
            print("Annotator 1 tags:", ' '.join(tags1))
            print("Annotator 2 tags:", ' '.join(tags2))
            print("✅ Agreement:", sorted(true_pos))
            print("❌ False Negatives (missed by A2):", sorted(false_neg))
            print("⚠️ False Positives (added by A2):", sorted(false_pos))

def validate_tags(tag_seqs):
    for i, tags in enumerate(tag_seqs):
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            print(f"[Error] Invalid tag format at sentence {i}: {tags}")
            return False
    return True
def get_entities_from_tags(tag_lists):
    """Extract spans (start, end, type) per sentence."""
    def extract_spans(tags):
        spans = []
        start, entity_type = None, None
        for i, tag in enumerate(tags):
            if tag.startswith('B-'):
                if start is not None:
                    spans.append((start, i, entity_type))
                start = i
                entity_type = tag[2:]
            elif tag.startswith('I-') and entity_type == tag[2:]:
                continue
            else:
                if start is not None:
                    spans.append((start, i, entity_type))
                    start, entity_type = None, None
        if start is not None:
            spans.append((start, len(tags), entity_type))
        return spans

    all_spans = []
    for tags in tag_lists:
        all_spans.append(extract_spans(tags))
    return all_spans



def evaluate_nervaluate(file1, file2):
    tags1 = [tags for _, tags in read_conll_tags(file1)]
    tags2 = [tags for _, tags in read_conll_tags(file2)]
    #tags1 = [['B-PER', 'I-PER', 'O', 'B-LOC']]
    #tags2 = [['B-PER', 'I-PER', 'O', 'B-LOC']]
    tag_set = set(tag[2:] for seq in tags1 + tags2 for tag in seq if tag != 'O')
    label_set = sorted(set(
        tag[2:] for sent in (tags1 + tags2) for tag in sent if tag.startswith(('B-', 'I-'))
    ))
    tag_set.discard('')

    if not validate_tags(tags1) or not validate_tags(tags2):
        print("Fix the input format before calling nervaluate.")
        exit()
    #compute_confusion_matrix(tags1, tags2)
    evaluator = Evaluator(tags1, tags2, tags=list(label_set), loader="list")
    results, results_by_tag, result_indices, result_indices_by_tag  = evaluator.evaluate()

    print("\n=== Overall Scores ===")
    #cmethods: 'ent_type', partial, strict, exact, emt_type
    cmethod = 'ent_type'
    printout(cmethod,results_by_tag, results, tag_set)



    #print(results)
    print (results)

def printout( cmethod, results_by_tag, results, tag_set):
    metrics= ['correct', 'incorrect', 'partial', 'spurious','missed',"f1", 'recall', 'precision']
    nervaluate_values = {}
    for m in metrics:
        print("Total : " + str(m) + ":" + str(results[cmethod][m]) + ";")
        nervaluate_values[m] = results[cmethod][m]
    #compute_kappa_from_nervaluate(nervaluate_values)
    for tag in tag_set :
        nervaluate_values={}
        for m in metrics:
            print(tag +" : " +str(m)+":"+str(results_by_tag[tag][cmethod][m])+";")
            nervaluate_values[m] = results_by_tag[tag][cmethod][m]
        #compute_kappa_from_nervaluate(nervaluate_values)

if __name__ == "__main__":
    local= "C:\\Users\\schei008\\surfdrive - Scheider, S. (Simon)@surfdrive.surf.nl\\Exchange\\Exchange\\thesis\\2025\\ADS\\Annotations\\Annotation experiment"
    import sys
    if len(sys.argv) != 3:
        print("Usage: python compare_conll_disagreements.py annotator1.conll annotator2.conll")

        #file1 = os.path.join(local,"Michiel\\1973_Harts_Jan_Migratie_UU.conll")
        #file2 = os.path.join(local, "1973_Harts\\1973_Harts_Jan_Migratie_UU.conll")
        #file1 = os.path.join(local, "Michiel\\1981_Vulto_Marlies_Vrouwen_in_een_nieuwbouwwijk_UU.conll")
        #file2 = os.path.join(local, "Tarik\\1981_Vulto_Marlies_Vrouwen_in_een_nieuwbouwwijk_UU.conll")
        #file2 = os.path.join(local, "1981_Vulto_Marlies_Vrouwen_in_een_nieuwbouwwijk_UU\\1981_Vulto_Marlies_Vrouwen_in_een_nieuwbouwwijk_UU.conll")
        file1 = os.path.join(local, "Michiel\\1988_H~1.conll")
        #file2 = os.path.join(local, "Tarik\\1988_H_1.conll")
        file2 = os.path.join(local, "1988_Hassink_Robert_Innovatiebevordering in Baden-Wuerttemburg\\1988_H~1.conll")

        #POSSIBLE(POS) = COR + INC + PAR + MIS = TP + FN
        #ACTUAL(ACT) = COR + INC + PAR + SPU = TP + FP
        #Precision = (COR + 0.5 × PAR) / ACT = TP / (TP + FP)
        #Recall = (COR + 0.5 × PAR)/POS = COR / ACT = TP / (TP + FN)

        highlight_disagreements(file1, file2)
        evaluate_nervaluate(file1, file2)

    else:
        file1, file2 = sys.argv[1], sys.argv[2]
        highlight_disagreements(file1, file2)
        evaluate_nervaluate(file1, file2)
