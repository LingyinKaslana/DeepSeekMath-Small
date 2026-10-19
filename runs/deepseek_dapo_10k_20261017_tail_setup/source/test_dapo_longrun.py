"""Guard unchanged training settings, group accounting, and immutable retention."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from dapo_longrun import validate_extension,add_group_totals,retain_weights,seed_history

class LongRunTests(unittest.TestCase):
    def test_extension_allows_only_budget_and_bookkeeping(self):
        old=dict(rounds=100,groups=32,samples=8,lr=1e-6,save_every=5,validation_every=10,script_sha256='old')
        new={**old,'rounds':10000,'save_every':100,'validation_every':100,'script_sha256':'new','checkpoint_retention':'weights'}
        validate_extension(old,new,100)
        for bad in ({**new,'lr':3e-6},{**new,'samples':16},{**new,'rounds':100},{**new,'save_every':0}):
            with self.assertRaises(AssertionError):validate_extension(old,bad,100)

    def test_used_is_distinct_from_discovered_surplus(self):
        state={'accepted_responses':16,'config':{'samples':8}}
        add_group_totals(state,{'mixed':3,'all_wrong':4,'all_correct':1},2)
        self.assertEqual(state['effective_groups_found'],3)
        self.assertEqual(state['effective_groups_used'],2)
        state['accepted_responses']+=16
        add_group_totals(state,{'mixed':2,'all_wrong':2},2)
        self.assertEqual(state['effective_groups_found'],5)
        self.assertEqual(state['effective_groups_used'],4)
        self.assertEqual(state['all_wrong_groups'],6)

    def test_archive_survives_latest_replacement_without_adam(self):
        with tempfile.TemporaryDirectory() as directory,patch('torch.cuda.get_rng_state',return_value=torch.tensor([1],dtype=torch.uint8)):
            root=Path(directory);full=root/'latest';full.mkdir()
            (full/'model.safetensors').write_bytes(b'original weights')
            (full/'training_state.pt').write_bytes(b'optimizer moments')
            (full/'checkpoint_manifest.json').write_text('{}')
            state={'completed_rounds':100,'training_order_cursor':20,'gradient_updates':400,
                   'effective_groups_found':3300,'effective_groups_used':3200}
            archive=root/'checkpoints/round000100';retain_weights(full,archive,state)
            self.assertFalse((archive/'training_state.pt').exists())
            self.assertEqual((archive/'model.safetensors').stat().st_ino,(full/'model.safetensors').stat().st_ino)
            (full/'model.safetensors').unlink();(full/'model.safetensors').write_bytes(b'new weights')
            self.assertEqual((archive/'model.safetensors').read_bytes(),b'original weights')
            self.assertFalse(json.loads((archive/'checkpoint_manifest.json').read_text())['checkpoint_storage']['exact_optimizer_resume'])

    def test_history_import_counts_and_independent_append_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'old';output=root/'new';source.mkdir();output.mkdir()
            row={'round':1,'candidate_groups':5,'response_tokens':100,'counts':{'mixed':3,'all_wrong':1,'all_correct':1}}
            (source/'training.jsonl').write_text(json.dumps(row)+'\n');(source/'rollouts.jsonl').write_text('{}\n')
            (source/'final_checkpoint').mkdir();(source/'final_checkpoint/model').write_text('immutable')
            state={'completed_rounds':1,'candidate_groups':5,'accepted_responses':16,'trained_response_tokens':100,'config':{'groups':2,'samples':8}}
            seed_history(source,output,state)
            self.assertEqual(state['effective_groups_found'],3);self.assertEqual(state['effective_groups_used'],2)
            with (output/'training.jsonl').open('a') as stream:stream.write('{}\n')
            self.assertEqual(len((source/'training.jsonl').read_text().splitlines()),1)

if __name__=='__main__':unittest.main()
