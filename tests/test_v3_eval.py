"""Phase 4 harness: loaders parse the public formats, metrics compute what they claim."""

import json

import torch

from pal_moe.eval.editing import (
    counterfact_scores,
    load_canary,
    load_counterfact,
    load_mquake,
    load_zsre,
    locality,
    multihop_accuracy,
)

CF = [
    {
        "case_id": 7,
        "requested_rewrite": {
            "prompt": "{} is located in",
            "subject": "Paris",
            "target_new": {"str": "Rome"},
            "target_true": {"str": "France"},
        },
        "paraphrase_prompts": ["Paris, which is in"],
        "neighborhood_prompts": ["Lyon is located in"],
    }
]
ZSRE = [
    {
        "src": "Who designed X?",
        "alt": "Alice",
        "answers": ["Bob"],
        "rephrase": "X was designed by?",
        "loc": "nq question: capital of peru",
        "loc_ans": "Lima",
    }
]
MQ = [
    {
        "case_id": 1,
        "requested_rewrite": [
            {
                "prompt": "{} is a citizen of",
                "subject": "Ann",
                "target_new": {"str": "Peru"},
                "target_true": {"str": "Chile"},
            }
        ],
        "questions": ["What is the capital of the country Ann is a citizen of?"],
        "new_answer": "Lima",
        "new_answer_alias": ["Ciudad de los Reyes"],
        "answer": "Santiago",
    }
]


def test_loaders(tmp_path):
    for name, rows in (("cf.json", CF), ("zsre.json", ZSRE), ("mq.json", MQ)):
        (tmp_path / name).write_text(json.dumps(rows))
    cf = load_counterfact(tmp_path / "cf.json")[0]
    assert (
        cf.prompt == "Paris is located in"
        and cf.target_new == " Rome"
        and cf.case_id == "7"
    )
    z = load_zsre(tmp_path / "zsre.json")[0]
    assert z.target_new == " Alice" and z.neighborhood_answers == [" Lima"]
    mq = load_mquake(tmp_path / "mq.json")[0]
    assert mq.edits[0].prompt == "Ann is a citizen of" and mq.new_aliases
    (tmp_path / "canary.txt").write_text("a\nb\n")
    prompts, h = load_canary(tmp_path / "canary.txt")
    assert prompts == ["a", "b"] and len(h) == 64


def test_counterfact_metrics_with_a_fake_scorer(tmp_path):
    (tmp_path / "cf.json").write_text(json.dumps(CF))
    case = load_counterfact(tmp_path / "cf.json")[0]

    def edited(prompts, targets):  # prefers Rome only for Paris prompts
        return torch.tensor(
            [
                1.0 if ("Paris" in p) == (t == " Rome") else 0.0
                for p, t in zip(prompts, targets)
            ]
        )

    s = counterfact_scores(case, edited)
    assert s == {"efficacy": 1.0, "paraphrase": 1.0, "specificity": 1.0}


def test_multihop_and_locality():
    from pal_moe.eval.editing import EditCase, MultiHopCase

    mh = MultiHopCase(
        "1", [EditCase("1.0", "p", " x")], ["q1", "q2"], "Lima", ["Reyes"]
    )
    r = multihop_accuracy(mh, lambda qs: ["it is lima", "no idea"])
    assert r == {"case_correct": True, "question_accuracy": 0.5}
    a = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert locality(a, a)["flip_rate"] == 0.0
    assert locality(a, a.flip(-1))["flip_rate"] == 1.0
