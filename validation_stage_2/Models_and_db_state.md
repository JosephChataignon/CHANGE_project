## Embedding models avail
### raw
sentence-transformers/all-MiniLM-L6-v2
sentence-transformers/all-mpnet-base-v2
Qwen/Qwen3-Embedding-0.6B
Qwen/Qwen3-Embedding-8B
nvidia_llama-embed-nemotron-8b
microsoft/harrier-oss-v1-0.6b

### finetuned
'sentence-transformers-all-MiniLM-L6-v2_finetuned-on_education-scale0.0001_2026-02-27 12:00:13.175118'
'sentence-transformers-all-MiniLM-L6-v2_finetuned-on_education-scale0.1_2026-02-27 03:00:15.216663'
'sentence-transformers-all-MiniLM-L6-v2_finetuned-on_education-scale1_2026-06-12 15:00:16.897693'
'sentence-transformers-all-MiniLM-L6-v2_finetuned-on_education-scale10_2026-04-09 04:00:13.860417'
'sentence-transformers-all-mpnet-base-v2_finetuned-on_education-scale0.0001_2026-02-27 13:00:46.275991'
'sentence-transformers-all-mpnet-base-v2_finetuned-on_education-scale0.1_2026-02-27 04:00:13.311469'
missing: all-mpnet-base-v2 scale=1 (repeat failure)
'Qwen-Qwen3-Embedding-0.6B_finetuned-on_education-scale0.0001_2026-04-27 13:00:13.278216'
'Qwen-Qwen3-Embedding-0.6B_finetuned-on_education-scale0.1_2026-04-27 23:00:13.896472'
'Qwen-Qwen3-Embedding-0.6B_finetuned-on_education-scale1_2026-06-13 22:00:13.436906'
missing: Qwen-Embedding-8B (too big, repeat failure)
'microsoft-harrier-oss-v1-0.6b_finetuned-on_education-scale0.0001_2026-04-07 12:00:14.688578'
'microsoft-harrier-oss-v1-0.6b_finetuned-on_education-scale0.1_2026-04-07 22:00:13.682025'
'microsoft-harrier-oss-v1-0.6b_finetuned-on_education-scale1_2026-06-23 07:00:12.856399'


## Corresponding database avail
### raw
eduscience_data_2026-02-26_18-30-02_sentence-transformers_all-MiniLM-L6-v2.db
eduscience_data_2026-02-26_19-20-01_sentence-transformers_all-mpnet-base-v2.db
eduscience-data_2026-02-20_qwen3emb0-6B.db
eduscience_data_2026-02-26_19-30-01_Qwen_Qwen3-Embedding-0.6B.db
eduscience_data_2026-02-26_19-35-01_Qwen_Qwen3-Embedding-8B.db
eduscience_data_2026-03-12_19-40-01_nvidia_llama-embed-nemotron-8b.db
missing: microsoft/harrier-oss-v1-0.6b


### finetuned
eduscience_data_2026-04-15_13-30-01__home_change_finetuned-models_sentence-transformers-all-MiniLM-L6-v2_finetuned-on_education-scale0.0001_2026-02-27_12:00:13.175118.db
eduscience_data_2026-04-15_10-15-02__home_change_finetuned-models_sentence-transformers-all-MiniLM-L6-v2_finetuned-on_education-scale0.1_2026-02-27_03:00:15.216663.db
missing: scale 1 and 10
eduscience_data_2026-04-15_12-00-02__home_change_finetuned-models_sentence-transformers-all-mpnet-base-v2_finetuned-on_education-scale0.0001_2026-02-27_13:00:46.275991.db
eduscience_data_2026-04-15_14-30-02__home_change_finetuned-models_sentence-transformers-all-mpnet-base-v2_finetuned-on_education-scale0.1_2026-02-27_04:00:13.311469.db
missing: scale 1 (unavailable)
missing: Qwen models
eduscience_data_2026-04-15_10-00-02__home_change_finetuned-models_microsoft-harrier-oss-v1-0.6b_finetuned-on_education-scale0.0001_2026-04-07_12:00:14.688578.db
eduscience_data_2026-04-15_10-45-01__home_change_finetuned-models_microsoft-harrier-oss-v1-0.6b_finetuned-on_education-scale0.1_2026-04-07_22:00:13.682025.db
missing: scale 1

*Ingestion launched for missing databases on 2026-10-22*

## Validation stage 2 to run
miniLM: nonFT, TODO: scale 0.0001 , 0.1, 1, 10
mpnet: TODO: nonFT, scale 0.0001, 0.1
Qwen-0.6b: TODO: nonFT, scale 0.0001, 0.1, 1
harrier: TODO: nonFT, scale 0.0001, 0.1, 1

### To run validation
*Change serving params in dhserver02*
```
nano /home/change/rag-change/ragchange/config/local.yaml
```
*run on dhserver03 with:*
```
cd CHANGE_project
nohup ../venv/bin/python3 validation_stage_2/benchmarking.py validation_stage_2/FOLDER_NAME/ >> validation_stage_2/FOLDER_NAME/log.log&
```
