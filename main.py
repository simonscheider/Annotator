from nervaluate import Evaluator
import os



#Reads conll files as token tag sequences
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

#This code is just used to visually highlight disagreements in annotations
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


#This is the main method to measure agreements by comparing tag sequences
def evaluate_nervaluate(file1, file2, cmethod = 'ent_type'):
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

    #print("\n=== Overall Scores ===")
    #cmethods: 'ent_type', partial, strict, exact, emt_type
    #cmethod = 'ent_type'
    #printout(cmethod,results_by_tag, results, tag_set)
    #print(results)
    #print (results)
    return (cmethod,results_by_tag, results, tag_set)

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


def get_conll_files(folder_path):
    conll_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.conll'):
                conll_files.append( file)
    return conll_files

def safe_format(value, missing='--'):
    return f"{value:.2f}" if isinstance(value, (int, float)) else missing

def escape_latex(s):
    special_chars = {
        '\\': r'\textbackslash{}',
        '{': r'\{',
        '}': r'\}',
        '#': r'\#',
        '$': r'\$',
        '%': r'\%',
        '&': r'\&',
        '_': r'\_',
        '^': r'\^{}',
        '~': r'\~{}',
    }
    for char, escape in special_chars.items():
        s = s.replace(char, escape)
    return s

#This is just to produce a latex table of the resulting inter annotation agreements between annotators.
def generate_latex_table(cmethod, doc_name, agreement,tagset):
    rows = []
    header = ["Pair", "Precision", "Recall", "F1"]

    # Start with overall scores
    rows.append("\\multicolumn{4}{l}{\\textbf{Overall}} \\\\ \\hline")
    for pair in agreement:
        results= agreement[pair][0]
        f1 = results[cmethod]["f1"]
        p = results[cmethod]["precision"]
        r = results[cmethod]["recall"]
        rows.append(f"{pair} & {safe_format(p)} & {safe_format(r)} & {safe_format(f1)} \\\\ \\hline")

    # Then add per-tag scores
    for tag in tagset:
        rows.append(f"\\multicolumn{{4}}{{l}}{{\\textbf{{Tag: {tag}}}}} \\\\ \\hline")
        for pair in agreement:
            results_by_tag = agreement[pair][1]
            if tag in results_by_tag:
                f1 = results_by_tag[tag][cmethod]["f1"]
                p = results_by_tag[tag][cmethod]["precision"]
                r = results_by_tag[tag][cmethod]["recall"]
            else:
                f1 = '--'
                p = '--'
                r = '--'
            rows.append(f"{pair} & {safe_format(p)} & {safe_format(r)} & {safe_format(f1)} \\\\ \\hline")

    # Compose full LaTeX table
    table = "\\begin{table}[ht]\n\\centering\n"
    table += f"\\caption{{Inter-annotator agreement for {escape_latex(doc_name)}}}\n"
    table += "\\begin{tabular}{lccc}\n\\toprule\n"
    table += " & ".join(header) + " \\\\ \\midrule\n"
    table += "\n".join(rows) + "\n"
    table += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    return table

if __name__ == "__main__":
    local= "C:\\Users\\schei008\\surfdrive - Scheider, S. (Simon)@surfdrive.surf.nl\\Exchange\\Exchange\\thesis\\2025\\ADS\\Annotations\\Annotation experiment\\Pre-annotation-experiment"
    import sys
    if len(sys.argv) != 3:
        me= os.path.join(local,"Simon")
        tarik = os.path.join(local,"Tarik")
        michiel= os.path.join(local,"Michiel")
        myfiles=get_conll_files(me)
        print(myfiles)
        cmethod = 'ent_type'
        for thesis in myfiles:
            print(thesis)
            agreement = {}
            tag_set = []
            file1 = os.path.join(me,thesis)
            tarikfile =os.path.join(tarik, thesis)
            if os.path.isfile(tarikfile):
                file2 = tarikfile
                (cmethod, results_by_tag, results, tag_set) = evaluate_nervaluate(file1, file2,cmethod)
                agreement["Simon-Tarik"]=[results,results_by_tag]
                print("Simon-Tarik")
                #highlight_disagreements(file1, file2)
            michielfile = os.path.join(michiel,thesis)
            if os.path.isfile(michielfile):
                file2 = michielfile
                (cmethod, results_by_tag, results, tag_set) = evaluate_nervaluate(file1, file2,cmethod)
                agreement["Simon-Michiel"]=[results,results_by_tag]
                print("Simon-Michiel")
                #highlight_disagreements(file1, file2)
            print(thesis)
            print(generate_latex_table(cmethod, thesis, agreement, tag_set))


        # see details in : https://pypi.org/project/nervaluate/
        #POSSIBLE(POS) = COR + INC + PAR + MIS = TP + FN
        #ACTUAL(ACT) = COR + INC + PAR + SPU = TP + FP
        #Precision = (COR + 0.5 × PAR) / ACT = TP / (TP + FP)
        #Recall = (COR + 0.5 × PAR)/POS = COR / ACT = TP / (TP + FN)

        #highlight_disagreements(file1, file2)
        #(cmethod, results_by_tag, results, tag_set) = evaluate_nervaluate(file1, file2)


    else:
        file1, file2 = sys.argv[1], sys.argv[2]
        highlight_disagreements(file1, file2)
        evaluate_nervaluate(file1, file2)
