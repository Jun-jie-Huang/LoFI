# import warnings
# warnings.filterwarnings("ignore")
from rouge import Rouge
import evaluate
from evaluation.my_squad_v2 import SquadV2

# from nltk.translate.bleu_score import sentence_bleu
rouge = Rouge()


class EvalUnigram:
    def __init__(self):
        self.result = {'precision': 0, 'recall': 0, 'f_score': 0, 'compression_rate': 0}
        self.count = 0

    def evaluate_unigram(self, references: str, hypothesis: str):
        score = dict()
        self.count += 1
        # score['BLEU'] = 0
        score['ROUGE'] = {'f': 0, 'p': 0, 'r': 0}
        if len(hypothesis) == 0:
            return score
        # score['BLEU'] = sentence_bleu([references.split()], hypothesis.split(), weights=(1,0,0,0))
        score['ROUGE'] = rouge.get_scores(references, hypothesis)[0]['rouge-1']
        self.result['precision'] += score['ROUGE']['p']
        self.result['recall'] += score['ROUGE']['r']
        self.result['f_score'] += score['ROUGE']['f']
        return score

    def record_compression_rate(self, hypothesis_length: int, doc_length: int):
        self.result['compression_rate'] += hypothesis_length / doc_length

    def get_result(self):
        if self.count == 0:
            return self.result
        temp = self.result.copy()
        temp['precision'] /= self.count
        temp['recall'] /= self.count
        temp['f_score'] /= self.count
        temp['compression_rate'] /= self.count
        return temp

    def clean_data(self):
        self.count = 0
        self.result = {'precision': 0, 'recall': 0, 'f_score': 0, 'compression_rate': 0}


def evaluate_generations(reference, hypotheses, logs=None):
    logs = ["n"]*len(hypotheses) if not logs else logs
    eval_log_summary = EvalUnigram()
    ems, rouge1 = [], []
    # Rouge score
    for hypo, log in zip(hypotheses, logs):
        ems.append(reference==hypo)
        _ = eval_log_summary.evaluate_unigram(reference, hypo)
        eval_log_summary.record_compression_rate(len(hypo), len(log))
        rouge1.append(_)
    result = eval_log_summary.get_result()
    final_score = {'rouge_precision': round(result['precision']*100, 2),
                   'rouge_recall': round(result['recall']*100, 2),
                   'rouge_f1': round(result['f_score']*100, 2),
                   'rouge_cr': round(result['compression_rate']*100, 2),
                   'rouge_em': round(sum(ems)/(len(ems)+1e-5)*100, 2),
                   'num': len(rouge1),
                   }
    # QA F1 and EM
    # metric = evaluate.load("squad_v2")
    metric = SquadV2()
    qa_results = {'f1': [], 'exact_match': [], 'recall': [], 'precision': []}
    for hypo, log in zip(hypotheses, logs):
        metrics = metric.compute(predictions=[{'id': '0', 'prediction_text': hypo, 'no_answer_probability': 0.0}],
                                 references=[{'id': '0', 'answers': {'answer_start': [100], 'text': [reference]}}])
        qa_results['f1'].append(metrics['f1'])
        qa_results['exact_match'].append(metrics['exact'])
        qa_results['precision'].append(metrics.get('precision', 0))
        qa_results['recall'].append(metrics.get('recall', 0))
    final_score.update({'squad_em': round(sum(qa_results["exact_match"])/len(qa_results["exact_match"])*100, 2),
                        'squad_f1': round(sum(qa_results["f1"])/len(qa_results["f1"])*100, 2),
                        'squad_pre': round(sum(qa_results["precision"]) / len(qa_results["precision"]) * 100, 2),
                        'squad_rec': round(sum(qa_results["recall"]) / len(qa_results["recall"]) * 100, 2),
                        })
    if logs==["n"]*len(hypotheses):
        final_score.update({'cr': 0})
    return final_score

