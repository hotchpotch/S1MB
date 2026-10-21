# Vision model training-source review

Reviewed 2026-10-21 for the 38 measured public-weight model releases. This is a public-source review, not an independent retained-row overlap audit. No inference API was used.

11/38 models have supported registered source-family disclosures; 27/38 remain unknown. This review adds 45 benchmark-source entries. All new entries concern source-family training, not established use of S1MB evaluation examples. Missing task coverage remains unknown.

The machine-readable disclosures live in [`model-training-exposure.json`](../evaluator/model-training-exposure.json). They affect disclosure badges only, never scores, ranking or model inputs. Public corpus availability does not establish the exact historical training revision. Evidence concerns the measured release; later materials are supplementary unless release lineage is explicit.

| Measured model | Registered benchmark-source entries | Finding |
| --- | ---: | --- |
| [internlm/Intern-Decision-0.8B](https://huggingface.co/internlm/Intern-Decision-0.8B/blob/85a0cc5a99d67ea8d56dfe98115689212867171d/README.md) | 0 | Unknown / not sufficiently disclosed |
| [internlm/Intern-Decision-2B](https://huggingface.co/internlm/Intern-Decision-2B/blob/8797836c65fc91a2435b1fb6850b5f0aabd75cc3/README.md) | 0 | Unknown / not sufficiently disclosed |
| [kirp/jpt-4b](https://huggingface.co/kirp/jpt-4b/blob/78312f855b8bebf83ae7e9e8f05b5a0f73f519a9/README.md) | 4 | Supported source-family disclosure |
| [kirp/jpt-0.8b](https://huggingface.co/kirp/jpt-0.8b/blob/1431c0509bbc10772cd59964cc7af5835c8720d6/README.md) | 4 | Supported source-family disclosure |
| [OmniJev/OneJev1.5-0.8B](https://huggingface.co/OmniJev/OneJev1.5-0.8B/blob/99d60969338977d3f23a023b1dd7a0fd51946069/README.md) | 0 | Unknown / not sufficiently disclosed |
| [OmniJev/OneJev1.5-9B](https://huggingface.co/OmniJev/OneJev1.5-9B/blob/ec2f276eab207810f576fb1f2d68c2acabd3eb71/README.md) | 0 | Unknown / not sufficiently disclosed |
| [TokenRhythm/NeoHorse-Jev-4B](https://huggingface.co/TokenRhythm/NeoHorse-Jev-4B/blob/434cb21d3a994a953d3ae5788405fcb2c4970554/README.md) | 0 | Unknown / not sufficiently disclosed |
| [shgao/rsi-jev-v5.0-vl-3b](https://huggingface.co/shgao/rsi-jev-v5.0-vl-3b/blob/66eeaee47dce202c4d802e469ad9b806e0ff6df2/README.md) | 14 | Supported source-family disclosure |
| [akhilaaa3/Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni/blob/fee89b076e2263bfa8bce8d34142caa4ca733a48/README.md) | 0 | Unknown / not sufficiently disclosed |
| [EldanRing/Winnow-12B](https://huggingface.co/EldanRing/Winnow-12B/blob/0b5b7f98e718cf0faffd0acea3093b86ebf00b56/README.md) | 0 | Unknown / not sufficiently disclosed |
| [EldanRing/Winnow-E4B](https://huggingface.co/EldanRing/Winnow-E4B/blob/17871fd27a880b3f8dad94841774db70e59fca47/README.md) | 0 | Unknown / not sufficiently disclosed |
| [lihe07/span-02](https://huggingface.co/lihe07/span-02/blob/63cbdd774391f1b7d9bc3829179efe996d0191ce/README.md) | 0 | Unknown / not sufficiently disclosed |
| [surogate/rune-26b-a4b-GGUF](https://huggingface.co/surogate/rune-26b-a4b-GGUF/blob/c6b360d47895bb77bdf3805a13ee5a5557ab1921/README.md) | 0 | Unknown / not sufficiently disclosed |
| [Maincode/matilda-jev-v1.5](https://huggingface.co/Maincode/matilda-jev-v1.5/blob/9a732d3d635e519ab06e21f9c60db3b2acd1eed1/README.md) | 0 | Unknown / not sufficiently disclosed |
| [OmniJev/OneJev1.5-27B](https://huggingface.co/OmniJev/OneJev1.5-27B/blob/0742f8d16dd42c97643eec095cfc252acc4db18f/README.md) | 0 | Unknown / not sufficiently disclosed |
| [startlux-models/StartLux-Decision-v1.1-27B-Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant/blob/32e7d35d74db615c4a059181efbf21c5421f3561/README.md) | 1 | Supported source-family disclosure |
| [vllm-sr/d3](https://huggingface.co/vllm-sr/d3/blob/cbff3a3cb0b074b6585fe9f747b9f9f5557a0e75/README.md) | 0 | Unknown / not sufficiently disclosed |
| [juspay/Xor-26B-A4B](https://huggingface.co/juspay/Xor-26B-A4B/blob/e4d0e99442def315d0c8fc9430cfd78367be6466/README.md) | 0 | Unknown / not sufficiently disclosed |
| [shisa-ai/shisa-de-2](https://huggingface.co/shisa-ai/shisa-de-2/blob/fd9f487763b4d5b7dc4c719f2704290330a65d62/README.md) | 0 | Unknown / not sufficiently disclosed |
| [autotrust/JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL/blob/9e53b8a809d2ab98e3e6a55b2b2dbf94b4c1e84a/README.md) | 20 | Supported source-family disclosure |
| [Cloudflare/clef-omni](https://huggingface.co/Cloudflare/clef-omni/blob/0db1cd2607d76a7bdb2a382f659e7b313079f84b/README.md) | 0 | Unknown / not sufficiently disclosed |
| [empiriolabsai/aplomb-1](https://huggingface.co/empiriolabsai/aplomb-1/blob/60b3a1c67d6fee193d1543b6530b4cc9c57c7fae/README.md) | 4 | Supported source-family disclosure |
| [PelaAI/KnowLine-35B-A3B-Gen1](https://huggingface.co/PelaAI/KnowLine-35B-A3B-Gen1/blob/e800968d38761c190476da5c69c6a5087a3c9b29/README.md) | 2 | Supported source-family disclosure |
| [juspay/xor](https://huggingface.co/juspay/xor/blob/a286a461e6daedee1e2400bdbe81cbb741cd592f/README.md) | 0 | Unknown / not sufficiently disclosed |
| [DoccyHealth/Solomon](https://huggingface.co/DoccyHealth/Solomon/blob/6b9bcc4a517bdbcecd4493f962446436c368ccb5/README.md) | 0 | Unknown / not sufficiently disclosed |
| [kirp/jpt-9b](https://huggingface.co/kirp/jpt-9b/blob/9fda2c7f123d36c4789c46c93737e80fc3a6c9fe/README.md) | 4 | Supported source-family disclosure |
| [internlm/Intern-Decision-4B](https://huggingface.co/internlm/Intern-Decision-4B/blob/1e65a988dacbcf00b84aee7366f9e47563407000/README.md) | 0 | Unknown / not sufficiently disclosed |
| [vllm-sr/d3-flash](https://huggingface.co/vllm-sr/d3-flash/blob/13dc1cc8ef23caefa765907e1d764751e5629f7d/README.md) | 0 | Unknown / not sufficiently disclosed |
| [vllm-sr/d3-nano](https://huggingface.co/vllm-sr/d3-nano/blob/462efc366229ea268cf4eed3738b3fc4415e4817/README.md) | 0 | Unknown / not sufficiently disclosed |
| [vllm-sr/d3-mini](https://huggingface.co/vllm-sr/d3-mini/blob/bd0090dd758d32ba59ef5c749910a8e87fc4391f/README.md) | 0 | Unknown / not sufficiently disclosed |
| [vllm-sr/d3-lite](https://huggingface.co/vllm-sr/d3-lite/blob/03560c26f5199c3c04129a2ccb38960c913635e0/README.md) | 0 | Unknown / not sufficiently disclosed |
| [vllm-sr/d3-edge](https://huggingface.co/vllm-sr/d3-edge/blob/c454ee39b862248d70abd7f84195d2f54ffe6d43/README.md) | 0 | Unknown / not sufficiently disclosed |
| [shgao/rsi-jev-v6.1-vl-27b](https://huggingface.co/shgao/rsi-jev-v6.1-vl-27b/blob/d0dccc052ee844ab7481ddfacad1b7335ab394e0/README.md) | 13 | Supported source-family disclosure |
| [shgao/rsi-jev-v6.1-vl-4b](https://huggingface.co/shgao/rsi-jev-v6.1-vl-4b/blob/7f93b1926dc23c62635538f5d40f1dabbddee5b6/README.md) | 22 | Supported source-family disclosure |
| [e12ex2/jev-vision-27b](https://huggingface.co/e12ex2/jev-vision-27b/blob/4084f7f755fd16f6c54a87c9a4534c6b877d02e4/README.md) | 1 | Supported source-family disclosure |
| [perplexity-ai/pplx-decider-v1.2](https://huggingface.co/perplexity-ai/pplx-decider-v1.2/blob/225c913ddc8761a8264f04f64548e1f97e443eb3/README.md) | 0 | Unknown / not sufficiently disclosed |
| [perplexity-ai/pplx-decider-flash-v1.2](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2/blob/6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d/README.md) | 0 | Unknown / not sufficiently disclosed |
| [mukaddamzaid/amos-1-large](https://huggingface.co/mukaddamzaid/amos-1-large/blob/d67e051c763dc88a1e3662a5003c1e1ea5ffaa74/README.md) | 0 | Unknown / not sufficiently disclosed |

## Evidence and limitations

### internlm/Intern-Decision-0.8B

Measured revision: `85a0cc5a99d67ea8d56dfe98115689212867171d`. [Release evidence](https://huggingface.co/internlm/Intern-Decision-0.8B/blob/85a0cc5a99d67ea8d56dfe98115689212867171d/README.md).

Pinned card discloses fine-tuning and benchmark scores, not actual training sources. Linked primary docs/DATA.md at GitHub 3572c8a68b5df5dafe02d0e093989ba8ec0183bc explicitly says training sources, composition, quantities and proportions are not disclosed. Evaluation AG News and Typed Decisions scores do not establish training exposure.

- [Supporting source](https://github.com/internlm/Intern-Decision/blob/3572c8a68b5df5dafe02d0e093989ba8ec0183bc/docs/DATA.md)

### internlm/Intern-Decision-2B

Measured revision: `8797836c65fc91a2435b1fb6850b5f0aabd75cc3`. [Release evidence](https://huggingface.co/internlm/Intern-Decision-2B/blob/8797836c65fc91a2435b1fb6850b5f0aabd75cc3/README.md).

Pinned card discloses fine-tuning and benchmark scores, not actual training sources. Linked primary docs/DATA.md at GitHub 3572c8a68b5df5dafe02d0e093989ba8ec0183bc explicitly says training sources, composition, quantities and proportions are not disclosed. Evaluation AG News and Typed Decisions scores do not establish training exposure.

- [Supporting source](https://github.com/internlm/Intern-Decision/blob/3572c8a68b5df5dafe02d0e093989ba8ec0183bc/docs/DATA.md)

### kirp/jpt-4b

Measured revision: `78312f855b8bebf83ae7e9e8f05b5a0f73f519a9`. [Release evidence](https://huggingface.co/kirp/jpt-4b/blob/78312f855b8bebf83ae7e9e8f05b5a0f73f519a9/README.md).

Existing disclosures verified against pinned cards. AG News on the two smaller siblings uses explicit same-data lineage from JPT-9B. Generic BEIR and community typed sets do not identify retained families; no expansion from those terms.

- [Supporting source](https://huggingface.co/kirp/jpt-9b/blob/9fda2c7f123d36c4789c46c93737e80fc3a6c9fe/README.md)

### kirp/jpt-0.8b

Measured revision: `1431c0509bbc10772cd59964cc7af5835c8720d6`. [Release evidence](https://huggingface.co/kirp/jpt-0.8b/blob/1431c0509bbc10772cd59964cc7af5835c8720d6/README.md).

Existing disclosures verified against pinned cards. AG News on the two smaller siblings uses explicit same-data lineage from JPT-9B. Generic BEIR and community typed sets do not identify retained families; no expansion from those terms.

- [Supporting source](https://huggingface.co/kirp/jpt-4b/blob/78312f855b8bebf83ae7e9e8f05b5a0f73f519a9/README.md)
- [Supporting source](https://huggingface.co/kirp/jpt-9b/blob/9fda2c7f123d36c4789c46c93737e80fc3a6c9fe/README.md)

### OmniJev/OneJev1.5-0.8B

Measured revision: `99d60969338977d3f23a023b1dd7a0fd51946069`. [Release evidence](https://huggingface.co/OmniJev/OneJev1.5-0.8B/blob/99d60969338977d3f23a023b1dd7a0fd51946069/README.md).

Pinned 1.5 card states actual full fine-tune on about 320,000 questions from real agent runs, videos, images and public decision-dataset training splits, without naming those sources. Linked GitHub README at 8b097886bbeb8f9ef6fe1f749a88f99d55b596ea (1.5 release) still describes original 99,193-question corpus and reusable recipe, not an explicit 320,000-question 1.5 retained inventory. Public OneJev-Data card/licenses.csv at 1028292856b8bc3029e4e423df78d0637c735dc0 identifies original corpus, with tallyqa 441, nlvr2 438, scienceqa 437, vsr 385, koniq10k 192, dbpedia 207, clinc 206, banking77 142, civil_comments 116. These are strong follow-up candidates but do not safely establish measured 1.5 retained-source use; no invented exposure entries. Follow-up: inspected 1.5 release tree, training config, current tree, website and model file inventory. Config references generic data/onejev/train.jsonl; website has no explicit 1.5-to-original-corpus bridge. No source carry-over assumed. Website is mutable supplementary evidence.

- [Supporting source](https://github.com/OmniJev/OneJev/blob/8b097886bbeb8f9ef6fe1f749a88f99d55b596ea/README.md)
- [Supporting source](https://github.com/OmniJev/OneJev/blob/8b097886bbeb8f9ef6fe1f749a88f99d55b596ea/train/configs/onejev_08b_full.yaml)
- [Supporting source](https://huggingface.co/datasets/OmniJev/OneJev-Data/blob/1028292856b8bc3029e4e423df78d0637c735dc0/README.md)
- [Supporting source](https://huggingface.co/datasets/OmniJev/OneJev-Data/blob/1028292856b8bc3029e4e423df78d0637c735dc0/licenses.csv)
- [Supporting source](https://github.com/OmniJev/OneJev/blob/6ce62b3a57a75deb60fde3bcc3dbaa4068c73650/README.md)
- [Supporting source](https://omnijev.github.io/OneJev/)

### OmniJev/OneJev1.5-9B

Measured revision: `ec2f276eab207810f576fb1f2d68c2acabd3eb71`. [Release evidence](https://huggingface.co/OmniJev/OneJev1.5-9B/blob/ec2f276eab207810f576fb1f2d68c2acabd3eb71/README.md).

Pinned 1.5 card states actual full fine-tune on about 320,000 questions from real agent runs, videos, images and public decision-dataset training splits, without naming those sources. Linked GitHub README at 8b097886bbeb8f9ef6fe1f749a88f99d55b596ea (1.5 release) still describes original 99,193-question corpus and reusable recipe, not an explicit 320,000-question 1.5 retained inventory. Public OneJev-Data card/licenses.csv at 1028292856b8bc3029e4e423df78d0637c735dc0 identifies original corpus, with tallyqa 441, nlvr2 438, scienceqa 437, vsr 385, koniq10k 192, dbpedia 207, clinc 206, banking77 142, civil_comments 116. These are strong follow-up candidates but do not safely establish measured 1.5 retained-source use; no invented exposure entries. Follow-up: inspected 1.5 release tree, training config, current tree, website and model file inventory. Config references generic data/onejev/train.jsonl; website has no explicit 1.5-to-original-corpus bridge. No source carry-over assumed. Website is mutable supplementary evidence.

- [Supporting source](https://github.com/OmniJev/OneJev/blob/8b097886bbeb8f9ef6fe1f749a88f99d55b596ea/README.md)
- [Supporting source](https://github.com/OmniJev/OneJev/blob/8b097886bbeb8f9ef6fe1f749a88f99d55b596ea/train/configs/onejev_08b_full.yaml)
- [Supporting source](https://huggingface.co/datasets/OmniJev/OneJev-Data/blob/1028292856b8bc3029e4e423df78d0637c735dc0/README.md)
- [Supporting source](https://huggingface.co/datasets/OmniJev/OneJev-Data/blob/1028292856b8bc3029e4e423df78d0637c735dc0/licenses.csv)
- [Supporting source](https://github.com/OmniJev/OneJev/blob/6ce62b3a57a75deb60fde3bcc3dbaa4068c73650/README.md)
- [Supporting source](https://omnijev.github.io/OneJev/)

### TokenRhythm/NeoHorse-Jev-4B

Measured revision: `434cb21d3a994a953d3ae5788405fcb2c4970554`. [Release evidence](https://huggingface.co/TokenRhythm/NeoHorse-Jev-4B/blob/434cb21d3a994a953d3ae5788405fcb2c4970554/README.md).

Pinned card and model_manifest.json describe decision tuning/runtime but disclose no retained training-source families. Benchmark comparison datasets and Kev-derived inference code are not evidence that those datasets trained this release.

- [Supporting source](https://huggingface.co/TokenRhythm/NeoHorse-Jev-4B/blob/434cb21d3a994a953d3ae5788405fcb2c4970554/model_manifest.json)

### shgao/rsi-jev-v5.0-vl-3b

Measured revision: `66eeaee47dce202c4d802e469ad9b806e0ff6df2`. [Release evidence](https://huggingface.co/shgao/rsi-jev-v5.0-vl-3b/blob/66eeaee47dce202c4d802e469ad9b806e0ff6df2/README.md).

Existing entries supported by release-specific source inventories/card. Additions use retained source names and explicit release lineage, not generic loader availability. Do not blanket-map cov_open_jev or cov_tasksource_jev to every task: selected subtasks remain unspecified. No image exposure inherited for 27B.

- [Supporting source](https://github.com/Shanghua-Gao/RSI-Jev/blob/fa70834769374c9ef38b41fb3ef7a3a69da26d76/data/v3.0_corpus_manifest.json)
- [Supporting source](https://github.com/Shanghua-Gao/RSI-Jev/blob/fa70834769374c9ef38b41fb3ef7a3a69da26d76/data/v5.0-vl_recipe/b4-exit20.json)
- [Supporting source](https://github.com/Shanghua-Gao/RSI-Jev/blob/fa70834769374c9ef38b41fb3ef7a3a69da26d76/data/v5.0-vl_recipe/vis-v4k.json)
- [Supporting source](https://github.com/Shanghua-Gao/RSI-Jev/blob/fa70834769374c9ef38b41fb3ef7a3a69da26d76/data/v5.0-vl_recipe/README.md)
- [Supporting source](https://github.com/Shanghua-Gao/RSI-Jev/blob/fa70834769374c9ef38b41fb3ef7a3a69da26d76/scripts/build_vision_v1.py)

### akhilaaa3/Jev-Omni

Measured revision: `fee89b076e2263bfa8bce8d34142caa4ca733a48`. [Release evidence](https://huggingface.co/akhilaaa3/Jev-Omni/blob/fee89b076e2263bfa8bce8d34142caa4ca733a48/README.md).

Pinned card discloses 30,000-question fine-tuning but no source inventory. Linked DecisionBench is evaluation data, not an asserted training corpus.


### EldanRing/Winnow-12B

Measured revision: `0b5b7f98e718cf0faffd0acea3093b86ebf00b56`. [Release evidence](https://huggingface.co/EldanRing/Winnow-12B/blob/0b5b7f98e718cf0faffd0acea3093b86ebf00b56/README.md).

Pinned card explicitly keeps training corpus/pipeline private. Lists generic synthetic/semantic task categories but no same-source identities. Development use of Jev/Kev panels is disclosed; it does not establish S1MB evaluation-example training.


### EldanRing/Winnow-E4B

Measured revision: `17871fd27a880b3f8dad94841774db70e59fca47`. [Release evidence](https://huggingface.co/EldanRing/Winnow-E4B/blob/17871fd27a880b3f8dad94841774db70e59fca47/README.md).

Pinned card explicitly keeps training corpus/pipeline private. Lists generic synthetic/semantic task categories but no same-source identities. Development use of Jev/Kev panels is disclosed; it does not establish S1MB evaluation-example training.


### lihe07/span-02

Measured revision: `63cbdd774391f1b7d9bc3829179efe996d0191ce`. [Release evidence](https://huggingface.co/lihe07/span-02/blob/63cbdd774391f1b7d9bc3829179efe996d0191ce/README.md).

Pinned card and linked Respan announcement describe architecture, inference and Decision Index scores; no actual retained training source-family inventory found.

- [Supporting source](https://www.respan.ai/blog/introducing-span-2)

### surogate/rune-26b-a4b-GGUF

Measured revision: `c6b360d47895bb77bdf3805a13ee5a5557ab1921`. [Release evidence](https://huggingface.co/surogate/rune-26b-a4b-GGUF/blob/c6b360d47895bb77bdf3805a13ee5a5557ab1921/README.md).

Live HF endpoint returned 401 (repository no longer publicly accessible). Reviewed previously pinned local card at cited immutable revision and live primary launch blog https://invergent.ai/blog/rune/; both describe training engine and evaluation, with no actual retained training-source inventory. Image JevBench ScreenSpot evaluations do not establish training use.

- [Supporting source](https://invergent.ai/blog/rune/)

### Maincode/matilda-jev-v1.5

Measured revision: `9a732d3d635e519ab06e21f9c60db3b2acd1eed1`. [Release evidence](https://huggingface.co/Maincode/matilda-jev-v1.5/blob/9a732d3d635e519ab06e21f9c60db3b2acd1eed1/README.md).

Pinned card explicitly acknowledges earlier lineage training used benchmark-related material, without identifying source families. This supports a broad caution but cannot map to stable S1MB benchmark IDs. Decision Index scores and saved predictions are evaluation evidence only.


### OmniJev/OneJev1.5-27B

Measured revision: `0742f8d16dd42c97643eec095cfc252acc4db18f`. [Release evidence](https://huggingface.co/OmniJev/OneJev1.5-27B/blob/0742f8d16dd42c97643eec095cfc252acc4db18f/README.md).

Pinned 1.5 card states actual full fine-tune on about 320,000 questions from real agent runs, videos, images and public decision-dataset training splits, without naming those sources. Linked GitHub README at 8b097886bbeb8f9ef6fe1f749a88f99d55b596ea (1.5 release) still describes original 99,193-question corpus and reusable recipe, not an explicit 320,000-question 1.5 retained inventory. Public OneJev-Data card/licenses.csv at 1028292856b8bc3029e4e423df78d0637c735dc0 identifies original corpus, with tallyqa 441, nlvr2 438, scienceqa 437, vsr 385, koniq10k 192, dbpedia 207, clinc 206, banking77 142, civil_comments 116. These are strong follow-up candidates but do not safely establish measured 1.5 retained-source use; no invented exposure entries. Follow-up: inspected 1.5 release tree, training config, current tree, website and model file inventory. Config references generic data/onejev/train.jsonl; website has no explicit 1.5-to-original-corpus bridge. No source carry-over assumed. Website is mutable supplementary evidence.

- [Supporting source](https://github.com/OmniJev/OneJev/blob/8b097886bbeb8f9ef6fe1f749a88f99d55b596ea/README.md)
- [Supporting source](https://github.com/OmniJev/OneJev/blob/8b097886bbeb8f9ef6fe1f749a88f99d55b596ea/train/configs/onejev_08b_full.yaml)
- [Supporting source](https://huggingface.co/datasets/OmniJev/OneJev-Data/blob/1028292856b8bc3029e4e423df78d0637c735dc0/README.md)
- [Supporting source](https://huggingface.co/datasets/OmniJev/OneJev-Data/blob/1028292856b8bc3029e4e423df78d0637c735dc0/licenses.csv)
- [Supporting source](https://github.com/OmniJev/OneJev/blob/6ce62b3a57a75deb60fde3bcc3dbaa4068c73650/README.md)
- [Supporting source](https://omnijev.github.io/OneJev/)

### startlux-models/StartLux-Decision-v1.1-27B-Instant

Measured revision: `32e7d35d74db615c4a059181efbf21c5421f3561`. [Release evidence](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant/blob/32e7d35d74db615c4a059181efbf21c5421f3561/README.md).

Final Stage 2 Lux-Rounds explicitly includes reconstructed GSM8K training problems. Appendix A.3 source-by-benchmark table applies only to earlier preview models, so its BANKING77/CLINC150/ContractNLI/ESCI list is not accepted as final-checkpoint evidence.

### vllm-sr/d3

Measured revision: `cbff3a3cb0b074b6585fe9f747b9f9f5557a0e75`. [Release evidence](https://huggingface.co/vllm-sr/d3/blob/cbff3a3cb0b074b6585fe9f747b9f9f5557a0e75/README.md).

Card reports multimodal decision capabilities and training hardware, but no actual training dataset/source inventory. Linked semantic-router training utilities do not establish release use. Measured checkpoint card reviewed at this pinned revision; no specific actual-training source inventory established. Newer card in evidence_urls was also reviewed as supplementary release-family context, not evidence that the measured weights changed.

- [Supporting source](https://huggingface.co/vllm-sr/d3/blob/cf15549a78fc52598ddd1951f321c58372bbd64f/README.md)

### juspay/Xor-26B-A4B

Measured revision: `e4d0e99442def315d0c8fc9430cfd78367be6466`. [Release evidence](https://huggingface.co/juspay/Xor-26B-A4B/blob/e4d0e99442def315d0c8fc9430cfd78367be6466/README.md).

Card discloses POP909 test-row training, 26 exact and ~3500 near-matching Decision Index contexts, but POP909 is not an active S1MB source and the matching context sources are unidentified. Do not map tool-context similarity to S1MB.

### shisa-ai/shisa-de-2

Measured revision: `fd9f487763b4d5b7dc4c719f2704290330a65d62`. [Release evidence](https://huggingface.co/shisa-ai/shisa-de-2/blob/fd9f487763b4d5b7dc4c719f2704290330a65d62/README.md).

Card lists 28 training families in broad groups and a second classifier suite of 49 tasks, without identifying original datasets. Financial intents, sentiment, emotion or logic similarity does not establish BANKING77 or other S1MB source use.

### autotrust/JEV-27B-VL

Measured revision: `9e53b8a809d2ab98e3e6a55b2b2dbf94b4c1e84a`. [Release evidence](https://huggingface.co/autotrust/JEV-27B-VL/blob/9e53b8a809d2ab98e3e6a55b2b2dbf94b4c1e84a/README.md).

VL card says it is JEV-27B with vision and the same text decisions. Pinned JEV-27B card confirms actual training on SargeDev/jev-distill-corpus-v3, including Open-Jev and HotpotQA-derived memory relevance. Underlying release-v2 Open-Jev OOD tasks are held out, so family disclosures do not claim every task-specific training item was present. No image training. On scrutiny, restricted Open-Jev mappings from all39 to18 benchmark IDs whose original task families are actually present in release-v2 provenance, consistent with existing Open-Jev-2B/9B registry treatment. This avoids attributing later Open-Jev task additions merely from collection identity. Public corpus itself is released; exact historical retained snapshot remains unknown.

- [Supporting source](https://huggingface.co/autotrust/JEV-27B/blob/14b425269ed24452e0a5ad5a8f3e51d139647a88/README.md)
- [Supporting source](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3/blob/75111a13a8455032f0435050bfc7e0c9026ecfe8/README.md)
- [Supporting source](https://huggingface.co/datasets/ZefanCai/Open-Jev/blob/c67699e13d0ae25e35b77165a4b6b079bedc8aba/provenance/original-manifests/release-v2.json)

### Cloudflare/clef-omni

Measured revision: `0db1cd2607d76a7bdb2a382f659e7b313079f84b`. [Release evidence](https://huggingface.co/Cloudflare/clef-omni/blob/0db1cd2607d76a7bdb2a382f659e7b313079f84b/README.md).

Card identifies the post-trained base and evaluation workflows, without specific training-source inventory. Measured checkpoint card reviewed at this pinned revision; no specific actual-training source inventory established. Newer card in evidence_urls was also reviewed as supplementary release-family context, not evidence that the measured weights changed.

- [Supporting source](https://huggingface.co/Cloudflare/clef-omni/blob/df2d7c9505e62ba8a527b147513a4fa66a77dc5d/README.md)

### empiriolabsai/aplomb-1

Measured revision: `60b3a1c67d6fee193d1543b6530b4cc9c57c7fae`. [Release evidence](https://huggingface.co/empiriolabsai/aplomb-1/blob/60b3a1c67d6fee193d1543b6530b4cc9c57c7fae/README.md).

Existing four entries supported by locally preserved pinned README/release.json. Current HF API and pinned README now return 401 Unauthorized; no claim of live verification. Local card names only ContractNLI, WinoGrande and typed-decisions train; WinoGrande has no same-source S1MB benchmark (do not map WinoBias). No additions.


### PelaAI/KnowLine-35B-A3B-Gen1

Measured revision: `e800968d38761c190476da5c69c6a5087a3c9b29`. [Release evidence](https://huggingface.co/PelaAI/KnowLine-35B-A3B-Gen1/blob/e800968d38761c190476da5c69c6a5087a3c9b29/README.md).

Final card explicitly names GSM8K and Amazon ESCI train splits. Other source names in example lists not matching active sources are not mapped. Decontamination removes near duplicates but retains shared source passages; this is source-family exposure, not established S1MB evaluation-example use.

### juspay/xor

Measured revision: `a286a461e6daedee1e2400bdbe81cbb741cd592f`. [Release evidence](https://huggingface.co/juspay/xor/blob/a286a461e6daedee1e2400bdbe81cbb741cd592f/README.md).

Card and pinned RELEASE_PROVENANCE.json describe adapter/merge/runtime, without an actual training-source inventory.

### DoccyHealth/Solomon

Measured revision: `6b9bcc4a517bdbcecd4493f962446436c368ccb5`. [Release evidence](https://huggingface.co/DoccyHealth/Solomon/blob/6b9bcc4a517bdbcecd4493f962446436c368ccb5/README.md).

Card discloses synthetic documents edited from 13 natural-text sources and a real public-government-document train panel. No active S1MB source family is established; contracts/policy subject similarity alone is insufficient.

### kirp/jpt-9b

Measured revision: `9fda2c7f123d36c4789c46c93737e80fc3a6c9fe`. [Release evidence](https://huggingface.co/kirp/jpt-9b/blob/9fda2c7f123d36c4789c46c93737e80fc3a6c9fe/README.md).

Existing disclosures verified against pinned cards. AG News on the two smaller siblings uses explicit same-data lineage from JPT-9B. Generic BEIR and community typed sets do not identify retained families; no expansion from those terms.

- [Supporting source](https://huggingface.co/kirp/jpt-4b/blob/78312f855b8bebf83ae7e9e8f05b5a0f73f519a9/README.md)

### internlm/Intern-Decision-4B

Measured revision: `1e65a988dacbcf00b84aee7366f9e47563407000`. [Release evidence](https://huggingface.co/internlm/Intern-Decision-4B/blob/1e65a988dacbcf00b84aee7366f9e47563407000/README.md).

Pinned card discloses fine-tuning and benchmark scores, not actual training sources. Linked primary docs/DATA.md at GitHub 3572c8a68b5df5dafe02d0e093989ba8ec0183bc explicitly says training sources, composition, quantities and proportions are not disclosed. Evaluation AG News and Typed Decisions scores do not establish training exposure.

- [Supporting source](https://github.com/internlm/Intern-Decision/blob/3572c8a68b5df5dafe02d0e093989ba8ec0183bc/docs/DATA.md)

### vllm-sr/d3-flash

Measured revision: `13dc1cc8ef23caefa765907e1d764751e5629f7d`. [Release evidence](https://huggingface.co/vllm-sr/d3-flash/blob/13dc1cc8ef23caefa765907e1d764751e5629f7d/README.md).

Card reports multimodal decision capabilities and training hardware, but no actual training dataset/source inventory. Linked semantic-router training utilities do not establish release use.

### vllm-sr/d3-nano

Measured revision: `462efc366229ea268cf4eed3738b3fc4415e4817`. [Release evidence](https://huggingface.co/vllm-sr/d3-nano/blob/462efc366229ea268cf4eed3738b3fc4415e4817/README.md).

Card reports multimodal decision capabilities and training hardware, but no actual training dataset/source inventory. Linked semantic-router training utilities do not establish release use.

### vllm-sr/d3-mini

Measured revision: `bd0090dd758d32ba59ef5c749910a8e87fc4391f`. [Release evidence](https://huggingface.co/vllm-sr/d3-mini/blob/bd0090dd758d32ba59ef5c749910a8e87fc4391f/README.md).

Card reports multimodal decision capabilities and training hardware, but no actual training dataset/source inventory. Linked semantic-router training utilities do not establish release use.

### vllm-sr/d3-lite

Measured revision: `03560c26f5199c3c04129a2ccb38960c913635e0`. [Release evidence](https://huggingface.co/vllm-sr/d3-lite/blob/03560c26f5199c3c04129a2ccb38960c913635e0/README.md).

Card reports multimodal decision capabilities and training hardware, but no actual training dataset/source inventory. Linked semantic-router training utilities do not establish release use.

### vllm-sr/d3-edge

Measured revision: `c454ee39b862248d70abd7f84195d2f54ffe6d43`. [Release evidence](https://huggingface.co/vllm-sr/d3-edge/blob/c454ee39b862248d70abd7f84195d2f54ffe6d43/README.md).

Card reports multimodal decision capabilities and training hardware, but no actual training dataset/source inventory. Linked semantic-router training utilities do not establish release use.

### shgao/rsi-jev-v6.1-vl-27b

Measured revision: `d0dccc052ee844ab7481ddfacad1b7335ab394e0`. [Release evidence](https://huggingface.co/shgao/rsi-jev-v6.1-vl-27b/blob/d0dccc052ee844ab7481ddfacad1b7335ab394e0/README.md).

Existing entries supported by release-specific source inventories/card. Additions use retained source names and explicit release lineage, not generic loader availability. Do not blanket-map cov_open_jev or cov_tasksource_jev to every task: selected subtasks remain unspecified. No image exposure inherited for 27B.

- [Supporting source](https://github.com/Shanghua-Gao/RSI-Jev/blob/fa70834769374c9ef38b41fb3ef7a3a69da26d76/data/v3.0_corpus_manifest.json)
- [Supporting source](https://github.com/Shanghua-Gao/RSI-Jev/blob/fa70834769374c9ef38b41fb3ef7a3a69da26d76/data/v5.0-vl_recipe/b4-exit20.json)
- [Supporting source](https://github.com/Shanghua-Gao/RSI-Jev/blob/fa70834769374c9ef38b41fb3ef7a3a69da26d76/data/v5.0-vl_recipe/vis-v4k.json)
- [Supporting source](https://github.com/Shanghua-Gao/RSI-Jev/blob/fa70834769374c9ef38b41fb3ef7a3a69da26d76/data/v5.0-vl_recipe/README.md)
- [Supporting source](https://github.com/Shanghua-Gao/RSI-Jev/blob/fa70834769374c9ef38b41fb3ef7a3a69da26d76/scripts/build_vision_v1.py)
- [Supporting source](https://huggingface.co/shgao/rsi-jev-v6.1-vl-4b/blob/7f93b1926dc23c62635538f5d40f1dabbddee5b6/README.md)
- [Supporting source](https://github.com/Shanghua-Gao/RSI-Jev/blob/fa70834769374c9ef38b41fb3ef7a3a69da26d76/data/v6.1-vl_recipe/1-member-b-domain-mix.json)
- [Supporting source](https://github.com/Shanghua-Gao/RSI-Jev/blob/fa70834769374c9ef38b41fb3ef7a3a69da26d76/data/v6.1-vl_recipe/README.md)
- [Supporting source](https://github.com/Shanghua-Gao/RSI-Jev/blob/fa70834769374c9ef38b41fb3ef7a3a69da26d76/data/v6.0-vl_recipe/1-trunk-exits.json)
- [Supporting source](https://huggingface.co/shgao/rsi-jev-v6.1-vl-27b/blob/d0dccc052ee844ab7481ddfacad1b7335ab394e0/meta.json)

### shgao/rsi-jev-v6.1-vl-4b

Measured revision: `7f93b1926dc23c62635538f5d40f1dabbddee5b6`. [Release evidence](https://huggingface.co/shgao/rsi-jev-v6.1-vl-4b/blob/7f93b1926dc23c62635538f5d40f1dabbddee5b6/README.md).

Existing entries supported by release-specific source inventories/card. Additions use retained source names and explicit release lineage, not generic loader availability. Do not blanket-map cov_open_jev or cov_tasksource_jev to every task: selected subtasks remain unspecified. No image exposure inherited for 27B.

- [Supporting source](https://github.com/Shanghua-Gao/RSI-Jev/blob/fa70834769374c9ef38b41fb3ef7a3a69da26d76/data/v3.0_corpus_manifest.json)
- [Supporting source](https://github.com/Shanghua-Gao/RSI-Jev/blob/fa70834769374c9ef38b41fb3ef7a3a69da26d76/data/v5.0-vl_recipe/b4-exit20.json)
- [Supporting source](https://github.com/Shanghua-Gao/RSI-Jev/blob/fa70834769374c9ef38b41fb3ef7a3a69da26d76/data/v5.0-vl_recipe/vis-v4k.json)
- [Supporting source](https://github.com/Shanghua-Gao/RSI-Jev/blob/fa70834769374c9ef38b41fb3ef7a3a69da26d76/data/v5.0-vl_recipe/README.md)
- [Supporting source](https://github.com/Shanghua-Gao/RSI-Jev/blob/fa70834769374c9ef38b41fb3ef7a3a69da26d76/scripts/build_vision_v1.py)
- [Supporting source](https://github.com/Shanghua-Gao/RSI-Jev/blob/fa70834769374c9ef38b41fb3ef7a3a69da26d76/data/v6.1-vl_recipe/1-member-b-domain-mix.json)
- [Supporting source](https://github.com/Shanghua-Gao/RSI-Jev/blob/fa70834769374c9ef38b41fb3ef7a3a69da26d76/data/v6.1-vl_recipe/README.md)
- [Supporting source](https://github.com/Shanghua-Gao/RSI-Jev/blob/fa70834769374c9ef38b41fb3ef7a3a69da26d76/data/v6.0-vl_recipe/1-trunk-exits.json)
- [Supporting source](https://huggingface.co/shgao/rsi-jev-v6.1-vl-4b/blob/7f93b1926dc23c62635538f5d40f1dabbddee5b6/meta.json)

### e12ex2/jev-vision-27b

Measured revision: `4084f7f755fd16f6c54a87c9a4534c6b877d02e4`. [Release evidence](https://huggingface.co/e12ex2/jev-vision-27b/blob/4084f7f755fd16f6c54a87c9a4534c6b877d02e4/README.md).

Measured-revision card fetched and checked; findings unchanged. Existing ScienceQA disclosure verified at its original pinned card; current card 4084f7f755fd16f6c54a87c9a4534c6b877d02e4 confirms it. Public SargeDev/jev-distill-corpus-v3 inventory (75111a13a8455032f0435050bfc7e0c9026ecfe8) lists Open-Jev and HotpotQA-derived streams, but the model used only 18k unspecified rows, so actual retained source-family membership cannot be inferred. COCO training does not establish RefCOCO-family training.

- [Supporting source](https://huggingface.co/e12ex2/jev-vision-27b/blob/483df76c76a96b1cc5f76e66140da925b346cef2/README.md)
- [Supporting source](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3/blob/75111a13a8455032f0435050bfc7e0c9026ecfe8/README.md)

### perplexity-ai/pplx-decider-v1.2

Measured revision: `225c913ddc8761a8264f04f64548e1f97e443eb3`. [Release evidence](https://huggingface.co/perplexity-ai/pplx-decider-v1.2/blob/225c913ddc8761a8264f04f64548e1f97e443eb3/README.md).

Measured-revision card fetched and checked; findings unchanged. Pinned public card fetched. It acknowledges tasksource and (27B) unnamed Decision Index official training splits; flash distills the 27B teachers training requests. Neither release publishes source-family retained inventory or training manifest: release-manifest.json is file checksums. Do not copy v1.1 disclosures to the reworked v1.2 data or infer all tasksource datasets were retained. All actual S1MB families remain unknown.

- [Supporting source](https://huggingface.co/perplexity-ai/pplx-decider-v1.2/blob/cce0f3f80033e9d89d80831d5281fd72124aacce/README.md)

### perplexity-ai/pplx-decider-flash-v1.2

Measured revision: `6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d`. [Release evidence](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2/blob/6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d/README.md).

Measured-revision card fetched and checked; findings unchanged. Pinned public card fetched. It acknowledges tasksource and (27B) unnamed Decision Index official training splits; flash distills the 27B teachers training requests. Neither release publishes source-family retained inventory or training manifest: release-manifest.json is file checksums. Do not copy v1.1 disclosures to the reworked v1.2 data or infer all tasksource datasets were retained. All actual S1MB families remain unknown.

- [Supporting source](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2/blob/9ff7cf9838e59df461c276b7699d56b244d2af58/README.md)

### mukaddamzaid/amos-1-large

Measured revision: `d67e051c763dc88a1e3662a5003c1e1ea5ffaa74`. [Release evidence](https://huggingface.co/mukaddamzaid/amos-1-large/blob/d67e051c763dc88a1e3662a5003c1e1ea5ffaa74/README.md).

Pinned public card fetched. It lists many evaluation benchmarks but no named training corpus/source inventory. The typed-decisions train split is used to fit answer-time priors with unchanged weights, not evidence of model training. All actual S1MB source-family training remains unknown.


