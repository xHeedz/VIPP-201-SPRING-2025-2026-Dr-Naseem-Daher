# j. context aware driving behaviour and retrieval

four references for part 4 of the fall 2026 plan: scoring a window relative to what is normal in its situation, retrieved from stored examples, instead of with one fixed threshold.

**[35]** J. Yuan, S. Sun, D. Omeiza, B. Zhao, P. Newman, L. Kunze and M. Gadd, "RAG-Driver: Generalisable driving explanations with retrieval-augmented in-context learning in multi-modal large language model," in *Proc. Robotics: Science and Systems (RSS)*, Delft, 2024. arXiv:2402.10828.

*used in this project:* closest precedent for retrieving similar driving situations from a memory of labelled examples and using them to explain a decision. RAG-Driver retrieves driving experiences to condition a multimodal LLM that outputs control signals and explanations. here retrieval only picks the reference set of normal windows for a percentile score, and the explanation is template text that cites a rules file; no LLM sits in the scoring loop (it cannot run in the 10 Hz ROS2 node).

**[36]** F. Marchetti, F. Becattini, L. Seidenari and A. Del Bimbo, "MANTRA: Memory augmented networks for multiple trajectory prediction," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2020.

*used in this project:* example of a non-parametric memory of trajectories queried at run time: once trained, the memory can grow with new patterns without retraining. the same property motivates `model/rag.py` `WindowIndex`: new labelled or simulated windows (for example SUMO planted normal drivers in a jam) are added to the reference set without refitting any weight.

**[37]** C. M. Martinez, M. Heucke, F. Wang, B. Gao and D. Cao, "Driving style recognition for intelligent vehicle control and advanced driver assistance: A survey," *IEEE Trans. Intelligent Transportation Systems*, vol. 19, pp. 666 to 676, 2018.

*used in this project:* survey of driving style characterisation and recognition (rule based, model based, learning based). supports the point that driving style is not separable from the conditions it is measured in, and that the inputs (speed, acceleration, headway, lateral behaviour) and their normalisation decide what a style label means.

**[38]** M. Nasr Azadani and A. Boukerche, "Driving behavior analysis guidelines for intelligent transportation systems," *IEEE Trans. Intelligent Transportation Systems*, vol. 23, no. 7, pp. 6027 to 6045, 2022. doi:10.1109/TITS.2021.3076140.

*used in this project:* roadmap of driving behaviour analysis by data type, goal and modelling technique. used for the dataset side of the fall work: which sensors each source has (phone IMU only, camera, drone trajectories) limits which terms of the index can be computed, which is how the candidate datasets in `docs/datasets.md` were screened.
