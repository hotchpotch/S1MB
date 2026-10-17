# Evaluation source checklist

Use this checklist before training-data disclosure and before an S1MB submission.
The tables map every active English/Japanese/vision benchmark to its public source
repository or named source family. They contain no evaluation rows or targets.

S1MB is a mosaic of existing research. We thank the authors, annotators and
maintainers of the linked datasets and evaluation projects. The individual
Dataset cards also retain source links and attribution notices; citing S1MB
does not replace crediting those original works.

**Not using Bekko System One or Saihai does not establish separation from S1MB.**
Those are collections, not the complete set of possible training sources. Training
directly on an original dataset, a mirror, a translated/converted version, or a
mixture containing it still requires disclosure. Check pretraining, fine-tuning,
distillation, calibration and model/configuration selection data to the best of
your knowledge; explicitly state inventories you cannot audit.

For each matching family, record the affected benchmark IDs, actual training
revision/config/split and filtering when known, and pinned evidence in
`training_exposure`. Different tasks or splits and duplicate filtering do not
erase source-family exposure. Declare `evaluation_examples` when actual evaluation
inputs were used; distinguish confirmed examples from approximate-match candidates.
Do not claim “no S1MB inputs used” when a source audit confirms matching inputs.
See [training exposure](training_exposure.md) and the
[test-set policy](evaluation.md#test-set-policy). Exposure disclosures do not adjust
scores; deliberate test-set tuning remains ineligible.

This is a source-discovery checklist, not a complete training inventory or an
overlap audit. Dataset aliases and derived collections must be traced to their
original families. For example, ContractNLI may be included through LegalBench,
and NanoDBPedia through a retrieval wrapper. A repository link alone does not
prove that a particular training corpus retained its examples. Unknown is not
“not used.” Review release provenance and source-specific terms for exact lineage.

The active `evaluator/data/categories/*.json` and benchmark definitions control
membership. The public links below follow
[`benchmark-sources.json`](../viewer/src/lib/benchmark-sources.json); update this
checklist when active membership or source mappings change. Links are discovery
references, not assertions about a submitted model’s training revision.

## english-v1

137 active benchmark definitions. HF identifiers are shown as `owner/dataset`;
GitHub links identify sources distributed outside HF.

| Original dataset / source family | Affected benchmark IDs |
| --- | --- |
| [nvidia/Aegis-AI-Content-Safety-Dataset-2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) | `aegis2-noul-test-v1` |
| [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news) | `laya--ag-news-choice-test-v1` |
| [google-deepmind/AQuA](https://github.com/google-deepmind/AQuA)<br>[deepmind/aqua_rat](https://huggingface.co/datasets/deepmind/aqua_rat) | `aqua-rat-choice-test-v1` |
| [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | `arc-choice-test-v1` |
| [tasksource/arct](https://huggingface.co/datasets/tasksource/arct) | `arct-choice-test-v1` |
| [ibm-research/argument_quality_ranking_30k](https://huggingface.co/datasets/ibm-research/argument_quality_ranking_30k) | `argument-quality-choice-test-v1`, `argument-quality-noul-test-v1` |
| [tasksource/babi_nli](https://huggingface.co/datasets/tasksource/babi_nli) | `babi-nli-noul-test-v1` |
| [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) | `banking77-choice-test-v1` |
| [lighteval/bbq_helm](https://huggingface.co/datasets/lighteval/bbq_helm) | `bbq-choice-test-v1` |
| [tasksource/Boardgame-QA](https://huggingface.co/datasets/tasksource/Boardgame-QA) | `boardgameqa-choice-test-v1` |
| [nvidia/CantTalkAboutThis-Topic-Control-Dataset](https://huggingface.co/datasets/nvidia/CantTalkAboutThis-Topic-Control-Dataset) | `canttalk-noul-test-v1` |
| [google/civil_comments](https://huggingface.co/datasets/google/civil_comments) | `civil-comments-noul-test-v1` |
| [causalNLP/cladder](https://github.com/causalNLP/cladder) | `cladder-noul-test-v1` |
| [clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos) | `clinc-choice-test-v1` |
| [stanfordnlp/contract-nli](https://github.com/stanfordnlp/contract-nli) | `contract-nli-choice-test-v1` |
| [tasksource/corr2cause](https://huggingface.co/datasets/tasksource/corr2cause) | `corr2cause-choice-test-v1` |
| [amydeng2000/CREAK](https://huggingface.co/datasets/amydeng2000/CREAK) | `creak-noul-test-v1` |
| [nyu-mll/crows-pairs](https://github.com/nyu-mll/crows-pairs) | `crows-pairs-choice-test-v1` |
| [fancyzhx/dbpedia_14](https://huggingface.co/datasets/fancyzhx/dbpedia_14) | `dbpedia-choice-test-v1` |
| [tasksource/defeasible-nli](https://huggingface.co/datasets/tasksource/defeasible-nli) | `defeasible-nli-choice-test-v1` |
| [SetFit/enron_spam](https://huggingface.co/datasets/SetFit/enron_spam) | `laya--enron-spam-noul-test-v1` |
| [tasksource/esci](https://huggingface.co/datasets/tasksource/esci) | `esci-choice-test-v1` |
| [hendrycks/ethics](https://huggingface.co/datasets/hendrycks/ethics) | `ethics-choice-test-v1`, `ethics-noul-test-v1` |
| [SetFit/ethos_binary](https://huggingface.co/datasets/SetFit/ethos_binary) | `ethos-noul-test-v1` |
| [DFKI-SLT/few-nerd](https://huggingface.co/datasets/DFKI-SLT/few-nerd) | `few-nerd-choice-test-v1` |
| [tasksource/FOL-nli](https://huggingface.co/datasets/tasksource/FOL-nli) | `fol-nli-choice-test-v1` |
| [mteb/Core17InstructionRetrieval](https://huggingface.co/datasets/mteb/Core17InstructionRetrieval) | `followir-core17-noul-test-v1` |
| [mteb/News21InstructionRetrieval](https://huggingface.co/datasets/mteb/News21InstructionRetrieval) | `followir-news21-noul-test-v1` |
| [mteb/Robust04InstructionRetrieval](https://huggingface.co/datasets/mteb/Robust04InstructionRetrieval) | `followir-robust04-noul-test-v1` |
| [google-research-datasets/go_emotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) | `go-emotions-noul-test-v1` |
| [gretelai/gretel-pii-masking-en-v1](https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1) | `gretel-pii-choice-test-v1` |
| [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | `gsm8k-choice-test-v1` |
| [tasksource/hans](https://huggingface.co/datasets/tasksource/hans) | `hans-noul-test-v1` |
| [Paul/hatecheck](https://huggingface.co/datasets/Paul/hatecheck) | `hatecheck-noul-test-v1` |
| [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) | `hh-rlhf-choice-test-v1` |
| [FastFit/hwu_64](https://huggingface.co/datasets/FastFit/hwu_64) | `hwu64-choice-test-v1` |
| [UKPLab/acl2022-impli](https://github.com/UKPLab/acl2022-impli) | `impli-noul-test-v1` |
| [tasksource/lexcomp-nc-relation](https://huggingface.co/datasets/tasksource/lexcomp-nc-relation) | `lexcomp-noul-test-v1` |
| [tasksource/logical-entailment](https://huggingface.co/datasets/tasksource/logical-entailment) | `logical-entailment-noul-test-v1` |
| [tasksource/lonli](https://huggingface.co/datasets/tasksource/lonli) | `lonli-choice-test-v1` |
| [AmazonScience/massive](https://huggingface.co/datasets/AmazonScience/massive) | `massive-choice-test-v1` |
| [tasksource/miqa](https://huggingface.co/datasets/tasksource/miqa) | `miqa-choice-test-v1` |
| [tasksource/mtop](https://huggingface.co/datasets/tasksource/mtop) | `mtop-choice-test-v1` |
| [tasksource/nlsat](https://huggingface.co/datasets/tasksource/nlsat) | `nlsat-noul-test-v1` |
| [ZefanCai/Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev) | `open-jev--amount-extraction-control-v1-choice-test-v1`, `open-jev--amount-extraction-control-v1-noul-test-v1`, `open-jev--browser-control-v1-choice-test-v1`, `open-jev--citation-control-v1-choice-test-v1`, `open-jev--context-retention-control-v1-noul-test-v1`, `open-jev--customer-control-v1-choice-test-v1`, `open-jev--customer-control-v1-noul-test-v1`, `open-jev--customer-control-v1-score-test-v1`, `open-jev--drone-control-v1-choice-test-v1`, `open-jev--drone-control-v1-noul-test-v1`, `open-jev--drone-control-v1-score-test-v1`, `open-jev--email-selection-control-v1-choice-test-v1`, `open-jev--email-selection-control-v1-noul-test-v1`, `open-jev--entity-alignment-control-v1-noul-test-v1`, `open-jev--entity-alignment-control-v1-score-test-v1`, `open-jev--ir-control-v1-choice-test-v1`, `open-jev--ir-control-v1-noul-test-v1`, `open-jev--ir-control-v1-score-test-v1`, `open-jev--mailroom-control-v1-choice-test-v1`, `open-jev--mailroom-control-v1-noul-test-v1`, `open-jev--painting-geometry-v1-choice-test-v1`, `open-jev--painting-geometry-v1-noul-test-v1`, `open-jev--painting-geometry-v1-score-test-v1`, `open-jev--phone-extraction-control-v1-choice-test-v1`, `open-jev--phone-extraction-control-v1-noul-test-v1`, `open-jev--reasoning-control-v1-choice-test-v1`, `open-jev--reasoning-control-v1-noul-test-v1`, `open-jev--reasoning-control-v1-score-test-v1`, `open-jev--silent-failure-control-v1-noul-test-v1`, `open-jev--snake-v1-choice-test-v1`, `open-jev--snake-v1-noul-test-v1`, `open-jev--sponsor-segment-control-v1-choice-test-v1`, `open-jev--tic-tac-toe-v1-choice-test-v1`, `open-jev--vizdoom-basic-v1-noul-test-v1`, `open-jev--vizdoom-basic-v1-score-test-v1`, `open-jev--workflow-controls-v1--agent-trace-observability-noul-test-v1`, `open-jev--workflow-controls-v1--customer-service-noul-test-v1`, `open-jev--workflow-controls-v1--invoice-processing-noul-test-v1`, `open-jev--workflow-controls-v1--security-incidents-noul-test-v1` |
| [allenai/openbookqa](https://huggingface.co/datasets/allenai/openbookqa) | `openbookqa-choice-test-v1` |
| [tasksource/patent-phrase-similarity](https://huggingface.co/datasets/tasksource/patent-phrase-similarity) | `patent-similarity-score-test-v1` |
| [google-research-datasets/paws](https://huggingface.co/datasets/google-research-datasets/paws) | `paws-noul-test-v1` |
| [tasksource/PLANE-ood](https://huggingface.co/datasets/tasksource/PLANE-ood) | `plane-noul-test-v1` |
| [google-research-datasets/poem_sentiment](https://huggingface.co/datasets/google-research-datasets/poem_sentiment) | `poem-sentiment-choice-test-v1` |
| [allenai/qasper](https://huggingface.co/datasets/allenai/qasper) | `qasper-noul-test-v1` |
| [allenai/quartz](https://huggingface.co/datasets/allenai/quartz) | `quartz-choice-test-v1` |
| [tasksource/robustLR](https://huggingface.co/datasets/tasksource/robustLR) | `robust-lr-choice-test-v1` |
| [tasksource/ruletaker](https://huggingface.co/datasets/tasksource/ruletaker) | `ruletaker-noul-test-v1` |
| S1MB-authored synthetic sets; no separate upstream dataset | `s1mb-generalization-contextual-choice-choice-test-v1`, `s1mb-generalization-contextual-noul-noul-test-v1`, `s1mb-generalization-contextual-score-score-test-v1`, `s1mb-generalization-diverse-choice-choice-test-v1`, `s1mb-generalization-diverse-noul-noul-test-v1`, `s1mb-generalization-diverse-score-score-test-v1` |
| [tasksource/scicite](https://huggingface.co/datasets/tasksource/scicite) | `scicite-choice-test-v1` |
| [allenai/scitail](https://huggingface.co/datasets/allenai/scitail) | `scitail-noul-test-v1` |
| [tasksource/scone](https://huggingface.co/datasets/tasksource/scone) | `scone-choice-test-v1` |
| [tasksource/SDOH-NLI](https://huggingface.co/datasets/tasksource/SDOH-NLI) | `sdoh-nli-noul-test-v1` |
| [google-research-datasets/dstc8-schema-guided-dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) | `sgd-choice-test-v1` |
| [stanfordnlp/snli](https://huggingface.co/datasets/stanfordnlp/snli) | `snli-choice-test-v1` |
| [tasksource/spartqa-mchoice](https://huggingface.co/datasets/tasksource/spartqa-mchoice) | `spartqa-choice-test-v1` |
| [tasksource/stepgame](https://huggingface.co/datasets/tasksource/stepgame) | `stepgame-choice-test-v1` |
| [hakari-bench/NanoBEIR-en](https://huggingface.co/datasets/hakari-bench/NanoBEIR-en); named subsets: `NanoArguAna`, `NanoDBPedia`, `NanoFiQA2018`, `NanoHotpotQA`, `NanoMSMARCO`, `NanoNFCorpus`, `NanoNQ`, `NanoSCIDOCS`, `NanoTouche2020` | `synthetic-relevance--nanobeir--nanoarguana-noul-test-v1`, `synthetic-relevance--nanobeir--nanoarguana-score-test-v1`, `synthetic-relevance--nanobeir--nanodbpedia-noul-test-v1`, `synthetic-relevance--nanobeir--nanodbpedia-score-test-v1`, `synthetic-relevance--nanobeir--nanofiqa2018-noul-test-v1`, `synthetic-relevance--nanobeir--nanofiqa2018-score-test-v1`, `synthetic-relevance--nanobeir--nanohotpotqa-noul-test-v1`, `synthetic-relevance--nanobeir--nanohotpotqa-score-test-v1`, `synthetic-relevance--nanobeir--nanomsmarco-noul-test-v1`, `synthetic-relevance--nanobeir--nanomsmarco-score-test-v1`, `synthetic-relevance--nanobeir--nanonfcorpus-noul-test-v1`, `synthetic-relevance--nanobeir--nanonfcorpus-score-test-v1`, `synthetic-relevance--nanobeir--nanonq-noul-test-v1`, `synthetic-relevance--nanobeir--nanonq-score-test-v1`, `synthetic-relevance--nanobeir--nanoscidocs-noul-test-v1`, `synthetic-relevance--nanobeir--nanoscidocs-score-test-v1`, `synthetic-relevance--nanobeir--nanotouche2020-noul-test-v1`, `synthetic-relevance--nanobeir--nanotouche2020-score-test-v1` |
| [hakari-bench/NanoCoIR](https://huggingface.co/datasets/hakari-bench/NanoCoIR); named subsets: `nanocoir` | `synthetic-relevance--nanocoir-noul-test-v1`, `synthetic-relevance--nanocoir-score-test-v1` |
| [tasksource/temporal-nli](https://huggingface.co/datasets/tasksource/temporal-nli) | `temporal-nli-choice-test-v1` |
| [tasksource/tracie](https://huggingface.co/datasets/tasksource/tracie) | `tracie-noul-test-v1` |
| [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | `laya--typed-decisions-choice-test-v1`, `laya--typed-decisions-noul-test-v1`, `laya--typed-decisions-score-test-v1` |
| [universal-dependencies/universal_dependencies](https://huggingface.co/datasets/universal-dependencies/universal_dependencies) | `ud-ewt-choice-test-v1` |
| [uclanlp/corefBias](https://github.com/uclanlp/corefBias) | `winobias-choice-test-v1` |
| [tasksource/winowhy](https://huggingface.co/datasets/tasksource/winowhy) | `winowhy-noul-test-v1` |
| [allenai/wiqa](https://huggingface.co/datasets/allenai/wiqa) | `wiqa-choice-test-v1` |

Original partitions are not uniformly held out by their original authors. See
[language scopes and source partitions](language_scopes.md#scores-and-source-partitions).

## japanese-v1

27 active benchmark definitions. HF identifiers are shown as `owner/dataset`;
GitHub links identify sources distributed outside HF.

| Original dataset / source family | Affected benchmark IDs |
| --- | --- |
| [Language-Media-Lab/commonsense-moral-ja](https://github.com/Language-Media-Lab/commonsense-moral-ja/tree/1d232e86b0448a2686939a5e7bc1a3731637fe37) | `commonsensemoralja-noul` |
| [sbintuitions/JamC-QA](https://huggingface.co/datasets/sbintuitions/JamC-QA/tree/ec2f8a34b1cfb9dcd8c50583eb688ef2ee3fb671) | `jamc-qa-choice` |
| [verypluming/JaNLI](https://github.com/verypluming/JaNLI/tree/2bbb63622570f7b20abc118146141b501efe5183) | `janli-choice` |
| [yahoojapan/JGLUE](https://github.com/yahoojapan/JGLUE/tree/6791a5003e6deaa6fb2458bcb3aa1f58dc52ef33) | `jcommonsenseqa-choice`, `jnli-choice` |
| [llm-jp/llm-jp-instructions-jculture-mcq](https://huggingface.co/datasets/llm-jp/llm-jp-instructions-jculture-mcq/tree/c23f8a1a427786812d66cb5b6946cc8ff417f260) | `jculture-mcq-noul` |
| [ku-nlp/KUCI](https://github.com/ku-nlp/KUCI/tree/79afa1d63d1f262e6709e9be714d9ad5dae8d6f6) | `kuci-choice` |
| [amazon-science/mintaka](https://github.com/amazon-science/mintaka/tree/b6146ffb41562803c45df3c028b4cdad8c09dd4d) | `mintaka-ja-boolean-noul` |
| [li-lab/MMLU-ProX](https://huggingface.co/datasets/li-lab/MMLU-ProX/tree/8e6106a6c6ce1c5027e66cc338143cf997b2aa09) | `mmlu-prox-ja-choice` |
| [openai/MMMLU](https://huggingface.co/datasets/openai/MMMLU/tree/325a01dc3e173cac1578df94120499aaca2e2504) | `mmmlu-choice` |
| [verypluming/JSICK](https://github.com/verypluming/JSICK/tree/b3034994192fae2f41b5937bcf69544e4282fc39) | `jsick-sts-score` |
| [yahoojapan/JGLUE](https://github.com/yahoojapan/JGLUE) | `jsts-score` |
| S1MB-authored synthetic sets; no separate upstream dataset | `generalization-japanese-knowledge-hard-choice`, `generalization-japanese-knowledge-hard-noul`, `generalization-japanese-knowledge-hard-score`, `generalization-japanese-math-formal-hard-choice`, `generalization-japanese-math-formal-hard-noul`, `generalization-japanese-math-formal-hard-score`, `generalization-japanese-diverse-noul`, `generalization-japanese-diverse-choice` |
| [hakari-bench/NanoJMTEB-v2](https://huggingface.co/datasets/hakari-bench/NanoJMTEB-v2)<br>[hotchpotch/JaCWIR](https://huggingface.co/datasets/hotchpotch/JaCWIR/tree/0cd42bf06d0cde0a4c46c82d1a1d76341dc4ddd9)<br>[mteb/JaGovFaqsRetrieval](https://huggingface.co/datasets/mteb/JaGovFaqsRetrieval/tree/0726b2af53c907628067871e2e7c84e0d8e099c2)<br>[mteb/jaqket](https://huggingface.co/datasets/mteb/jaqket/tree/37edb92ba3d5debfe219f3d83e91d1979b8e6d08)<br>[miracl/miracl](https://huggingface.co/datasets/miracl/miracl/tree/5be20db9509754dadad47689368639fcec739c00)<br>[castorini/mr-tydi](https://huggingface.co/datasets/castorini/mr-tydi/tree/1d43c80218d06d0ef80f5b172ccabd848b948bc1)<br>[mteb/NLPJournalTitleAbsRetrieval.V2](https://huggingface.co/datasets/mteb/NLPJournalTitleAbsRetrieval.V2/tree/7ea085c4107e8554f92409193358790fe40516f8) | `jmteb-retrieval` |
| [osekilab/JCoLA](https://github.com/osekilab/JCoLA/tree/736d9eef3af04bb17e4cf59adf01325973234ff9) | `jcola-noul` |
| [tomo-vv/temporalNLI_dataset](https://github.com/tomo-vv/temporalNLI_dataset/tree/f578b5daac7bff3855d238c032a60fd047cddd9c) | `jamp-choice` |
| [DaisukeBekki/JSeM](https://github.com/DaisukeBekki/JSeM/tree/c2cd39168e267007b4690dbf625329bac57a8185) | `jsem-choice` |
| [llm-jp/jgpqa](https://github.com/llm-jp/jgpqa/tree/110e6d62605a3bd3f2ed49c77992e600216eb4d2) | `gpqa-ja-main-choice` |
| [nlp-waseda/JMMLU](https://github.com/nlp-waseda/JMMLU/tree/762cbf192c9e4588b574d718f95afd64c96fcbf4) | `jmmlu-japan-choice` |
| [Taise228/SOBACO](https://huggingface.co/datasets/Taise228/SOBACO/tree/77d8bb422d598d09f6b689cd09443c7027cdf11d) | `sobaco-culture-choice` |

Original partitions are not uniformly held out by their original authors. See
[language scopes and source partitions](language_scopes.md#scores-and-source-partitions).

## Vision sources

We thank [MTEB](https://github.com/embeddings-benchmark/mteb),
[MIEB](https://arxiv.org/abs/2504.10471), and
[MMEB / VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec)
([MMEB-eval](https://huggingface.co/datasets/TIGER-Lab/MMEB-eval)) for helping
guide candidate discovery and coverage. These references are distinct from the
original datasets below. S1MB uses typed decision adaptations, not the original
embedding evaluation protocols or directly comparable scores.

| Original dataset / project | S1MB benchmark IDs |
| --- | --- |
| [CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench) | `cvbench-depth-choice`, `cvbench-distance-choice`, `cvbench-relation-choice` |
| [KonIQ-10k](https://doi.org/10.18419/darus-2435) | `koniq-score` |
| [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) | `mvtec-ad-noul` |
| [NIGHTS / DreamSim](https://github.com/ssundaram21/dreamsim/tree/main/dataset) | `nights-choice` |
| [NLVR2](https://github.com/lil-lab/nlvr/tree/master/nlvr2) | `nlvr2-noul` |
| [POPE](https://github.com/RUCAIBox/POPE) | `pope-noul` |
| [RefCOCO / RefCOCO+ / RefCOCOg (REFER)](https://github.com/lichengunc/refer) | `refcoco-box`, `refcoco-plus-box`, `refcocog-box` |
| [RVL-CDIP](https://huggingface.co/datasets/aharley/rvl_cdip) | `rvlcdip-choice` |
| [ScienceQA](https://github.com/lupantech/ScienceQA) | `scienceqa-choice` |
| [ScreenSpot / SeeClick](https://github.com/njucckevin/SeeClick) | `screenspot-box` |
| [SugarCrepe](https://github.com/RAIVNLab/sugar-crepe) | `sugarcrepe-choice` |
| [TallyQA](https://github.com/manoja328/tallyqa) | `tallyqa-score` |
| [VisA](https://github.com/amazon-science/spot-diff) | `visa-noul` |
| [VSR](https://huggingface.co/datasets/cambridgeltl/vsr_zeroshot) | `vsr-noul` |
| S1MB-authored synthetic tasks using [WIT](https://github.com/google-research-datasets/wit) and the [generalization image sources](vision-source-licenses.md#generalization-image-sources) | `generalization-vision-easy-noul`, `generalization-vision-easy-choice`, `generalization-vision-easy-score`, `generalization-vision-easy-box`, `generalization-vision-hard-noul`, `generalization-vision-hard-choice`, `generalization-vision-hard-score`, `generalization-vision-hard-box` |

See the [Vision Dataset card](https://huggingface.co/datasets/hotchpotch/S1MB-vision-v1#sources-and-licenses)
and [license inventory](vision-source-licenses.md) for source-specific terms and
image attribution. The eight authored sets use synthetic teachers, not human
gold. Shared source images, original split lineage and record-level training
exposure remain separate considerations.

## Disclosure to include in a submission

- Source families reviewed, including direct originals and derived mixtures:
- Sources actually used and affected benchmark IDs, with `training_exposure` entries:
- Confirmed evaluation-input use and approximate-match candidates, separately:
- Actual training revisions, configs, splits, filtering and pinned evidence:
- Unavailable pretraining/fine-tuning inventories or other unknown details:

“Bekko and Saihai were not used” alone does not answer this checklist.
