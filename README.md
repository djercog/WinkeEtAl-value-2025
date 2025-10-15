# WinkeEtAl-value-2025

This repository contains data and codes for analysis associated with the manuscript
"Prefrontal neural geometry of associated cues guides learned motivated behaviors"
% Winke N, Luthi A, Herry C, Jercog D.

## /data content:

- __Fig1behData.mat__: <br> Individual mice (n=10) behavioral data from the instrumental approach-avoidance task.
  - _CSShutProb_: Shuttle Probabilities for each CS-type.
  - _CSShutTraj_: Animals’ trajectories during shuttle trials for each CS-type.
  - _CSnosePoke_:
    - .Prob = Probability to poke during 12 sec after CS onset (by CS-type).
    - .cdf = Cumulative distribution function of poking by CS-type (0.5 sec bins).
  - _motorFeature_ = Motor predictors for pairwise ridge-regularized logistic regression.
    
- __Fig1dmPFCInactivation.xlsx__:<br>
  Pharmacological experiments data (Excel file).

- __Fig2neuralRec1.mat__: <br>Individual mice (n=10) imaging from the instrumental approach-avoidance task.
  - heatmaps = z-scored trial-average activity aligned at CS onset for different CS types.
  - imData(i) = CS onset aligned activity for Subject i (input for decoding).
    - .tm: time bin
    - .CSx_Ca: dimensions [nr Trials x nr. of neuron x time bin]

- __Fig2neuralRec2_RewardDevaluationPre.mat__:
	<br> Individual mice (n=9) imaging from the pre reward devaluation.
  - CSShutProbRevDev_Pre = Shuttle Probabilities for each CS-type.
  - imDataRewDev_Pre(i) = CS onset aligned activity for Subject i (Pre reward devaluation).

- __Fig2neuralRec2_RewardDevaluationPost.mat__:
  <br> Individual mice (n=9) imaging from the post reward devaluation.
  - CSShutProbRevDev_Post = Shuttle Probabilities for each CS-type.
  - imDataRewDev_Post(i) = CS onset aligned activity for Subject i (Post reward devaluation).

- __Fig2neuralRec3_AversiveRevaluationPre.mat__:
  <br> Individual mice (n=8) imaging from the pre aversive revaluation.
  - CSShutProbAverRev = Shuttle Probabilities for each CS-type.
  - imDataAverRev_Pre(i) = CS onset aligned activity for Subject i (Pre aversive revaluation).

- __Fig2neuralRec3_AversiveRevaluationPost.mat__:
  <br> Individual mice (n=8) imaging from the pre aversive revaluation.
  - imDataAverRev_Post(i) = CS onset aligned activity for Subject i (Post aversive revaluation).
<br>

## /code content:


## License
- MIT License (see LICENSE file)
- Please cite our paper if you use this repository.
