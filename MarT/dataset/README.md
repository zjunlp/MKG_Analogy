# MARS


Code and datasets for the ICLR2023 paper "Multimodal Analogical Reasoning over Knowledge Graphs "

## Data File Structure

The structure of data files is as follows:


```
MKG_Analogy
 |-- MarT
 |    |-- dataset       # data
 |    |    |-- MarKG    # knowledge graph data
 |    |    |    |-- entity2text.txt         # entity_id to entity_description
 |    |    |    |-- entity2textlong.txt     # entity_id to longer entity_description
 |    |    |    |-- relation2text.txt       # relation_id to relation_description
 |    |    |    |-- relation2textlong.txt   # relation_id to longer relation_description
 |    |    |    |-- wiki_tuple_ids.txt      # knowledge triplets with (head_id, rel_id, tail_id) format
 |    |    |-- MARS     # analogical reasoning data
 |    |    |    |-- images                  # the image data
 |    |    |    |-- analogy_entities.txt    # the analogical entities
 |    |    |    |-- analogy_entity_to_wiki_qid.txt   # the ids of analogical entities
 |    |    |    |-- analogy_relations.txt   # the analogical relations
 |    |    |    |-- dev.json                # analogical reasoning data for validation
 |    |    |    |-- test.json               # analogical reasoning data for testing
 |    |    |    |-- train.json              # analogical reasoning data for training
```

## Get the Image Data
The image data can be downloaded through this [link](https://pan.baidu.com/s/1WZvpnTe8m0m-976xRrH90g) with extraction code (7hoc) and placed on `MarT/dataset/MARS/images`.

## Data Format

### 1. MarKG

The format of the knowledge triplet data (wiki_tuple_ids.txt) is as below:
```text
Q15026	P276	Q107
Q34266	P47	    Q30
Q317557	P140	Q5043
Q8686	P910	Q7214221
```
In each line, the entity ids start with `Q` and relation ids start with `P`.

### 2. MARS

The analogical reasoning data is stored in json format. For each analogical reasoning instance, we provide an analogical example and an analogical question entity to acquire the analgical answer entity, the data format is as below:
```text
{
    "example": ["Q14536140", "Q581459"], 
    "question": "Q50000", 
    "answer": "Q202875", 
    "relation": "P828", 
    "mode": 0
}
```