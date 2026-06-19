# Deep NER leaderboard

| rank | model | architecture | base | valid entity F1 | test entity F1 | valid token F1 | test token F1 | path |
|---:|---|---|---|---:|---:|---:|---:|---|
| 1 | roberta_crf | crf | roberta-base | 0.7985 | 0.7989 | 0.8973 | 0.8791 | `/Users/kodomotachi/specialist/NLP-test/models/deep_ner/roberta_crf` |
| 2 | roberta_token_cls | transformer_token_classification | roberta-base | 0.7561 | 0.7506 | 0.8885 | 0.8711 | `/Users/kodomotachi/specialist/NLP-test/models/deep_ner/roberta_token_cls/best` |
| 3 | bert_token_cls | transformer_token_classification | bert-base-uncased | 0.7078 | 0.7004 | 0.8838 | 0.8676 | `/Users/kodomotachi/specialist/NLP-test/models/deep_ner/bert_token_cls/best` |
| 4 | bert_global_context | global_context | bert-base-uncased | 0.6994 | 0.6998 | 0.8833 | 0.8696 | `/Users/kodomotachi/specialist/NLP-test/models/deep_ner/bert_global_context` |
| 5 | xlmroberta_token_cls | transformer_token_classification | xlm-roberta-base | 0.6557 | 0.6724 | 0.8778 | 0.8536 | `/Users/kodomotachi/specialist/NLP-test/models/deep_ner/xlmroberta_token_cls/best` |
| 6 | bert_global_pointer | global_pointer | bert-base-uncased | 0.3093 | 0.3088 | 0.3210 | 0.3169 | `/Users/kodomotachi/specialist/NLP-test/models/deep_ner/bert_global_pointer` |
