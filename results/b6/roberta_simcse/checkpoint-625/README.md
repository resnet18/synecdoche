---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:500
- loss:MultipleNegativesRankingLoss
base_model: hfl/chinese-roberta-wwm-ext-large
widget:
- source_sentence: 很意外。
  sentences:
  - 很意外。
  - 暗影中必定蛰伏着更庞大的危险。[称谓]，请务必不要与我们走散。
  - 这些档案上的加密[术语]均已解除，除了一本名册上的[术语]过于繁琐难以完全消除外，其余的均可放心翻阅。不过，[称谓]，您为何自刚才起便缄默不言？
- source_sentence: '......为我自己。'
  sentences:
  - 塔已安息，[称谓]......
  - '......为我自己。'
  - '[人名][称谓]第一次把她带到我们面前的时候，她还无比稚嫩。我还记得，她的视线在我的角羽上停留——'
- source_sentence: 只是......
  sentences:
  - 我不能说自己丝毫未受环境影响。
  - 我奉劝你仔细考虑自己的行动。
  - 只是......
- source_sentence: 并非如此，他们只是尚不甚了解我的言语规则罢了。
  sentences:
  - '[种族]，真正的迷失者恐怕是你。'
  - 并非如此，他们只是尚不甚了解我的言语规则罢了。
  - 她催动了从伤口滴下的血液。
- source_sentence: 不必当真。
  sentences:
  - 你好。
  - 战争。
  - 不必当真。
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on hfl/chinese-roberta-wwm-ext-large

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [hfl/chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large). It maps sentences & paragraphs to a 1024-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [hfl/chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) <!-- at revision a25cc9e05974bd9687e528edd516f2cfdb3f5db9 -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 1024 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    '不必当真。',
    '不必当真。',
    '战争。',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 1.0000, 0.3855],
#         [1.0000, 1.0000, 0.3855],
#         [0.3855, 0.3855, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 500 training samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 500 samples:
  |         | anchor                                                                            | positive                                                                          |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            |
  | details | <ul><li>min: 5 tokens</li><li>mean: 22.37 tokens</li><li>max: 96 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 22.37 tokens</li><li>max: 96 tokens</li></ul> |
* Samples:
  | anchor                                    | positive                                  |
  |:------------------------------------------|:------------------------------------------|
  | <code>我不知道你至今都没有陷入疯狂，是你的幸运还是不幸。</code>    | <code>我不知道你至今都没有陷入疯狂，是你的幸运还是不幸。</code>    |
  | <code>不......[人名]阁下。这里没有丧钟[组织]的主人。</code> | <code>不......[人名]阁下。这里没有丧钟[组织]的主人。</code> |
  | <code>“穿刺”。</code>                        | <code>“穿刺”。</code>                        |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Evaluation Dataset

#### Unnamed Dataset

* Size: 125 evaluation samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 125 samples:
  |         | anchor                                                                           | positive                                                                         |
  |:--------|:---------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                           |
  | details | <ul><li>min: 5 tokens</li><li>mean: 22.5 tokens</li><li>max: 77 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 22.5 tokens</li><li>max: 77 tokens</li></ul> |
* Samples:
  | anchor                                        | positive                                      |
  |:----------------------------------------------|:----------------------------------------------|
  | <code>字词的力量，不在此处。</code>                      | <code>字词的力量，不在此处。</code>                      |
  | <code>那是[人名]的职责所在，我告诫了我的同僚，不要纵容她的偷懒行为。</code> | <code>那是[人名]的职责所在，我告诫了我的同僚，不要纵容她的偷懒行为。</code> |
  | <code>暗影中必定蛰伏着更庞大的危险。[称谓]，请务必不要与我们走散。</code>  | <code>暗影中必定蛰伏着更庞大的危险。[称谓]，请务必不要与我们走散。</code>  |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 4
- `num_train_epochs`: 10
- `learning_rate`: 1e-05
- `weight_decay`: 0.1
- `fp16`: True
- `eval_strategy`: epoch
- `per_device_eval_batch_size`: 4
- `load_best_model_at_end`: True

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 4
- `num_train_epochs`: 10
- `max_steps`: -1
- `learning_rate`: 1e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.1
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1.0
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: True
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: trackio
- `eval_strategy`: epoch
- `per_device_eval_batch_size`: 4
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: True
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | Training Loss | Validation Loss |
|:-----:|:----:|:-------------:|:---------------:|
| 1.0   | 125  | 0.0038        | 0.0000          |
| 2.0   | 250  | 0.0001        | 0.0000          |
| 3.0   | 375  | 0.0000        | 0.0000          |
| 4.0   | 500  | 0.0001        | 0.0000          |
| 5.0   | 625  | 0.0000        | 0.0000          |


### Framework Versions
- Python: 3.11.9
- Sentence Transformers: 5.2.3
- Transformers: 5.3.0
- PyTorch: 2.12.0.dev20260302+cu128
- Accelerate: 1.12.0
- Datasets: 4.6.1
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->