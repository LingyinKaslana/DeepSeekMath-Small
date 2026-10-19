import unittest
from group_length_shadow import GroupLengthShadow,replay_group


def sample(length,correct=False,finish='eos'):
    return {'tokens':[1]*length,'correct':correct,'finish_reason':finish}


class GuardTests(unittest.TestCase):
    def test_no_reference_from_future_correct_answers(self):
        values=[sample(300,True),sample(400,True),sample(1000)]+[sample(10)]*5
        flags=replay_group(values)
        self.assertEqual(len(flags),1);self.assertEqual(flags[0]['sample_index'],2)
        self.assertEqual(flags[0]['would_stop_at_token'],701)

    def test_floor_and_strict_boundary_eos_first(self):
        values=[sample(20,True),sample(30,True),sample(257,True),sample(258)]+[sample(10)]*4
        flags=replay_group(values)
        # A response ending on the check step is already finished, never cancelled.
        self.assertEqual([f['sample_index'] for f in flags],[3])
        self.assertEqual(flags[0]['would_stop_at_token'],257)
        self.assertEqual(flags[0]['potential_saved_tokens'],1)

    def test_single_correct_never_arms(self):
        self.assertEqual(replay_group([sample(10,True)]+[sample(2048,False,'length')]*7),[])

    def test_length_capped_answer_not_a_completed_reference(self):
        self.assertEqual(replay_group([sample(10,True),sample(20,True,'length')]+[sample(1000)]*6),[])

    def test_virtual_cancelled_correct_cannot_become_later_reference(self):
        values=[sample(20,True),sample(30,True),sample(300,True),sample(1000)]+[sample(10)]*4
        flags=replay_group(values)
        self.assertEqual([f['sample_index'] for f in flags],[2,3])
        self.assertTrue(all(f['reference_samples']==[0,1] for f in flags))
        self.assertTrue(flags[0]['actual_final_correct'])

    def test_event_replay_equals_per_token_stream(self):
        values=[sample(17,True),sample(130,True),sample(300,True),sample(2048,False,'length')]+[sample(220)]*4
        replay=replay_group(values);guard=GroupLengthShadow()
        for step in range(1,2049):
            events=[{'index':i,'length':step,'correct':s['correct'],'complete_eos':s['finish_reason']=='eos'}
                    for i,s in enumerate(values) if len(s['tokens'])==step]
            guard.observe(step,events)
        self.assertEqual({f['sample_index']:f['would_stop_at_token'] for f in replay},
                         {i:f['would_stop_at_token'] for i,f in guard.would_stop.items()})


if __name__=='__main__':unittest.main()
