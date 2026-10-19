"""Prepare answer-only CoT SFT from pinned GSM8K/MATH training splits."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def statistics(rows):
    lengths = [len(row['input_ids']) for row in rows]
    return {'examples':len(rows),'by_dataset':dict(Counter(row['dataset'] for row in rows)),
            'input_tokens':sum(lengths),
            'supervised_tokens':sum(len(row['input_ids'])-row['prompt_length'] for row in rows),
            'length_percentiles':{str(p):float(np.percentile(lengths,p)) for p in (0,50,90,95,99,100)}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--vendor',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    sources=json.loads(a.sources.read_text())
    tokenizer=AutoTokenizer.from_pretrained(a.checkpoint,local_files_only=True)
    eos=tokenizer.eos_token_id
    tests=set()
    for name in ('gsm8k','math'):
        for line in (a.vendor/'evaluation/datasets'/name/'test.jsonl').read_text().splitlines():
            row=json.loads(line);tests.add(normalized(row.get('question',row.get('problem'))))
    seen=set();groups={'gsm8k':[],'math':[]};rejected=[];raw_counts=Counter()
    for source in sources:
        path=Path(source['path']);assert digest(path)==source['sha256']
        name='gsm8k' if source['repo_id']=='openai/gsm8k' else 'math'
        for index,row in enumerate(pq.read_table(path).to_pylist()):
            raw_counts[name]+=1
            question=row.get('question',row.get('problem')).strip()
            key=normalized(question)
            identifier=f"{source['filename']}:{index}"
            reason=None
            if key in tests:reason='normalized_question_matches_test'
            elif key in seen:reason='duplicate_normalized_training_question'
            if reason:
                rejected.append({'dataset':name,'id':identifier,'reason':reason});continue
            if name=='gsm8k':
                assert '####' in row['answer']
                body,answer=row['answer'].rsplit('####',1)
                body=re.sub(r'<<[^<>]*>>','',body).strip()
                answer=answer.strip().replace(',','')
                if not re.fullmatch(r'-?\d+(?:\.\d+)?',answer):
                    rejected.append({'dataset':name,'id':identifier,'reason':'non_numeric_gsm_answer'});continue
                prompt=f'Q: {question}\nA:'
                completion=f' {body}\nThe answer is {answer}.'
            else:
                prompt=f'Problem:\n{question}\n\nSolution:'
                completion='\n'+row['solution'].strip()
                if not ('\\boxed' in completion or '\\fbox' in completion):
                    rejected.append({'dataset':name,'id':identifier,'reason':'math_solution_without_boxed_answer'});continue
            prompt_ids=tokenizer.encode(prompt,add_special_tokens=False)
            answer_ids=tokenizer.encode(completion,add_special_tokens=False)
            assert eos not in prompt_ids and eos not in answer_ids
            ids=prompt_ids+answer_ids+[eos]
            if len(ids)>4096:
                rejected.append({'dataset':name,'id':identifier,'reason':'complete_example_exceeds_4096','tokens':len(ids)});continue
            assert len(prompt_ids)>=2 and len(answer_ids)>0
            seen.add(key)
            groups[name].append({'dataset':name,'source_id':identifier,'question':question,
                'prompt':prompt,'completion':completion,'input_ids':ids,'prompt_length':len(prompt_ids),
                'question_sha256':hashlib.sha256(key.encode()).hexdigest()})
    assert raw_counts=={'gsm8k':7473,'math':7500}, raw_counts
    train=[];validation=[]
    for offset,(name,rows) in enumerate(groups.items()):
        rng=random.Random(42+offset);indices=set(rng.sample(range(len(rows)),256))
        validation.extend(row for index,row in enumerate(rows) if index in indices)
        train.extend(row for index,row in enumerate(rows) if index not in indices)
    random.Random(42).shuffle(train)
    keys=[{row['question_sha256'] for row in rows} for rows in (train,validation)]
    assert not keys[0]&keys[1] and len(keys[0])==len(train) and len(keys[1])==len(validation)
    assert all(normalized(row['question']) not in tests for row in train+validation)
    artifacts={}
    for name,rows in (('train',train),('validation',validation),('rejected',rejected)):
        path=a.output/f'{name}.jsonl'
        with path.open('w') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        artifacts[path.name]={'sha256':digest(path),'bytes':path.stat().st_size}
    manifest={'status':'ready','created_utc':datetime.now(timezone.utc).isoformat(),
        'sources':sources,'source_checkpoint':str(a.checkpoint.resolve()),
        'source_cpt_input_tokens':10000000000,'max_sequence_length':4096,'eos_token_id':eos,
        'bos_policy':'No inserted BOS; matches existing base-model benchmark prompts',
        'formats':{'gsm8k':'Q: {question}\\nA: {solution}\\nThe answer is {answer}.',
                   'math':'Problem:\\n{problem}\\n\\nSolution:\\n{original_boxed_solution}'},
        'training_loss':'Assistant completion including EOS only; all prompt and padding labels are -100',
        'packing':False,'overlong_policy':'Drop complete example; never truncate answer/EOS',
        'raw_examples':dict(raw_counts),'rejections':dict(Counter(row['reason'] for row in rejected)),
        'train':statistics(train),'validation':statistics(validation),'artifacts':artifacts,
        'split_seed':42,'test_isolation':{'benchmarks':['GSM8K test 1319','MATH test 5000'],
            'normalized_question_overlap_after_filter':0,'train_validation_question_overlap':0,
            'scope':'Exact NFKC/casefold/whitespace question matching for SFT only; not semantic decontamination of CPT'},
        'preparation_script_sha256':digest(Path(__file__))}
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:manifest[key] for key in ('status','raw_examples','rejections','train','validation')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
