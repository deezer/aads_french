# Handling new content

This folder presents how implemented and trained models for (W)DSR can be leveraged to annotate new corpora made of raw text files. The process is illustrated in a [demo notebook](./demo_new_content.ipynb), and command lines to replicate the procedure are given below.

For the purpose of the example, two chapters of children's literature novels are gathered in [`texts` folder](./texts):
- `Les_Aventures_de_Nono_VII.txt`: *Les Aventures de Nono*, Jean Grave, 1901, chapter VII: *LE TRAVAIL À AUTONOMIE*. [Wikisource link](https://fr.wikisource.org/wiki/Les_Aventures_de_Nono/_VII._Le_travail_à_Autonomie)
- `Encore_Heidi_09.txt`: *Encore Heidi*, Johanna Spyri, 1882, chapter IX: *On se dit adieu, mais au revoir !*.[Wikisource link](https://fr.wikisource.org/wiki/Encore_Heidi/09)



# Preprocessing

## Generating `.json` corpus

The `.txt` files are compiled into a `.json` corpus that contains their raw text, their `split` is set to `test`, the `original_corpus` is called `new_content` and the `labels` field is left empty. This can be done by running the following command line:
```bash
python -c 'from new_content_helpers import make_json_from_texts; make_json_from_texts(folder_path="texts", output_dir="new_content_preprocessed")'
```

## Generating `.tsv` files

Then, for ML models, the tokenized files need to be generated (eg. with spacy tokenization):
```bash
python ../preprocessing/data_utils.py --data_dir new_content_preprocessed/new_corpus.json --output_dir new_content_preprocessed/ --do_split False --tokenizer 'spacy_tokenization'
```

# Run (W)DSR

(W)DSR can then be ran using [configurations files](./new_content_configs), leveraging trained models:
```
python ../run_experiments.py --configs_folder new_content_configs
```

The [config folder](./new_content_configs) stores configuration files from the best models (ie. similar to files in [`experiments_configs/best_configs`](../experiments_configs/best_configs)) adapted to the newly created corpus.

# Labelled files

The tokenized text files together with models' predictions can be seen in the [output folder](./output).

These files can be merged and saved in a new table with the following command line:
```bash
python -c 'from new_content_helpers import merge_predictions; merge_predictions()'
```
The [resulting `.tsv`](./output/merged_predictions.tsv) will then be stored in the output folder.
It will combine predictions from the different models in one file, as shown below:

|       | file                      | token   | sentstart   | token_idx    | pred_regex   | pred_flair   | pred_transformer   |
|------:|:--------------------------|:--------|:------------|:-------------|:-------------|:-------------|:-------------------|
| 11420 | Les_Aventures_de_Nono_VII | —       | no          | (9238, 9239) | DS           | DS           | DS                 |
| 11421 | Les_Aventures_de_Nono_VII | C’      | no          | (9240, 9242) | DS           | DS           | DS                 |
| 11422 | Les_Aventures_de_Nono_VII | est     | no          | (9242, 9245) | DS           | DS           | DS                 |
| 11423 | Les_Aventures_de_Nono_VII | pour    | no          | (9246, 9250) | DS           | DS           | DS                 |
| 11424 | Les_Aventures_de_Nono_VII | toi     | no          | (9251, 9254) | DS           | DS           | DS                 |
| 11425 | Les_Aventures_de_Nono_VII | ,       | no          | (9254, 9255) | DS           | DS           | DS                 |
| 11426 | Les_Aventures_de_Nono_VII | fit     | no          | (9256, 9259) | DS           | O            | O                  |
| 11427 | Les_Aventures_de_Nono_VII | Nono    | no          | (9260, 9264) | DS           | O            | O                  |
| 11428 | Les_Aventures_de_Nono_VII | en      | no          | (9265, 9267) | DS           | O            | O                  |
| 11429 | Les_Aventures_de_Nono_VII | la      | no          | (9268, 9270) | DS           | O            | O                  |
| 11430 | Les_Aventures_de_Nono_VII | lui     | no          | (9271, 9274) | DS           | O            | O                  |
| 11431 | Les_Aventures_de_Nono_VII | posant  | no          | (9275, 9281) | DS           | O            | O                  |
| 11432 | Les_Aventures_de_Nono_VII | sur     | no          | (9282, 9285) | DS           | O            | O                  |
| 11433 | Les_Aventures_de_Nono_VII | la      | no          | (9286, 9288) | DS           | O            | O                  |
| 11434 | Les_Aventures_de_Nono_VII | tête    | no          | (9289, 9293) | DS           | O            | O                  |
| 11435 | Les_Aventures_de_Nono_VII | .       | no          | (9293, 9294) | DS           | O            | O                  |

