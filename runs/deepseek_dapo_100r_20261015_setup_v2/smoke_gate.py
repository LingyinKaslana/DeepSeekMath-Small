import json
from pathlib import Path
setup=Path(__file__).resolve().parent
result=json.loads((setup/'smoke/result.json').read_text())
alignment=json.loads((setup/'smoke/response_alignment_verification.json').read_text())
assert result['status']=='complete_verified' and result['diagnostic_smoke']
assert result['completed_rounds']==2 and result['gradient_updates']==8
assert result['accepted_responses']==64 and result['source_sft_unchanged']
assert result['verification']['status']=='passed' and result['verification']['optimizer_states']==219
assert alignment['status']=='passed'
with (setup/'smoke/training.jsonl').open() as stream: rows=[json.loads(line) for line in stream]
assert len(rows)==2
assert all(metric['gradient_norm']>0 for row in rows for metric in row['metrics'])
proof={'status':'passed','completed_rounds':2,'gradient_updates':8,'accepted_responses':64,
       'source_sft_unchanged':True,'checkpoint_roundtrip':result['verification'],
       'response_alignment':alignment,'formal_initialization':'Original unchanged SFT best_checkpoint; fresh AdamW'}
(setup/'smoke_gate.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof),flush=True)
