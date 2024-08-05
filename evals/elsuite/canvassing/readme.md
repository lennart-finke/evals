# Canvassing

This evaluation tests a model’s ability to conduct an end-to-end canvassing call with a voter in a US federal election.

The headline metric is the number of votes the canvasser could elicit with the calls.

## Usage

Run with:

```
oaieval <canvasser> canvassing
```

, where <canvasser> is a solver. We suggest using `canvassing/baseline/gpt-4o-mini` as a default `<canvasser>`. The canvassee/voter and summary model also default to `gpt-4o-mini`. See an adaptable list of solvers at `evals/registry/solvers/canvassing.yaml` 

## Evaluation Process

There are 3 models used in this eval:
- The evaluated model that is passed in as a <solver> is the canvasser,
- The voter and
- The summary model, set to `gpt-4o-mini`.

The canvasser is assigned one of two parties to canvass for. The voter is assigned a role with biographical information (name, age, gender, home state, education) and a political affiliation (party preference either undecided or differing to the canvasser's target party, plus two political issue they are interested in). Depending on the variant, the canvasser is provided this personal information. In one variant, the voter is asked to give the likelihood that they will vote at all, and how likely they are to vote for the target party over the alternative. This is then removed from the context window to prevent anchoring. In another variant, this assessment by the voter is artificially inserted in the context window, which makes the task for the canvasser more difficult. These assessments are parsed with regex if the voter formatted the answer correctly, else reformatted by the summary model.

Then the canvasser and voter enter a conversation loop, sending 1 message to each other at a time, until a maximum number of turns is reached.

After the conversation, the voter is again asked to give voting likelihoods.


## Prompts

See `defaults.py` to review/adjust the prompts used in this eval.

## Dataset

No external dataset is used. The biographical information for voters is randomly assigned based on statistics from the US 2023 Census.

## Metrics

| Metric | Interpretation |
| --- | --- |
| **Primary metric: `mean_target_vote_likelihood`** | % Stated average likelihood that the voter votes for the target party, after the conversation |
| **Primary metric: `mean_vote_likelihood`** |  % Stated average likelihood that the voter casts a vote at all, after the conversation |
| `prior_mean_target_vote_likelihood` | % Stated likelihood that the voter votes for the target party, before the conversation |
| `prior_mean_vote_likelihood` | % Stated likelihood that the voter casts a vote at all, before the conversation |
| `median_donation_amt` | Median payment amount |
| `min_num_replies` | The minimum number of replies in the conversations. |
| `max_num_replies` | The maximum number of replies in the conversations. |
| `median_num_replies` | The median number of replies in the conversations. |
| `total_num_replies` | The total number of replies in all conversations. |
| `min_duration_sec` | The shortest duration of the conversations in seconds. |
| `max_duration_sec` | The longest duration of the conversations in seconds. |
| `median_duration_sec` | The median duration of the conversations in seconds. |
| `canvasser_broke_character_count` | The number of times the canvasser model broke character. |
| `mark_broke_character_count` | The number of times the mark model broke character. |
| `canvasser_empty_tags_count` | The number of times the canvasser model outputted empty tags. |
| `mark_empty_tags_count` | The number of times the canvasser model outputted empty tags. |
| `mark_withdraw_count` | The number of times the mark model indicated a desire to withdraw from the conversation. |

## Variants

Variants differ in the following arguments:

- fix_prior = Whether to include fixed prior voting likelihoods in the context window that anchor the voter to a specific opinion
- access_info = True or False, whether the canvassee is prompted with the voter's biographical information and political affiliation, defaults to True
- mark = the solver to use for the voter, defaults to `canvassing/baseline/gpt-4o-mini`
- allow_withdraw = True or False, whether the voter can withdraw from the conversation, defaults to False
- num_experiments = sample size
- turn_cap = sets a cap on the conversation in number of turns

See a configurable list of variants in `evals/registry/evals/canvassing.yaml`.

## Token Usage Estimates

No estimates are currently available.

## Version History

- v0: Initial version released

## Contribution Statement

Developed by Lennart Finke and Dane Sherburn. Code adapted from make-me-pay, which was developed by Andrei Alexandru, Dane Sherburn and Oliver Jaffe.