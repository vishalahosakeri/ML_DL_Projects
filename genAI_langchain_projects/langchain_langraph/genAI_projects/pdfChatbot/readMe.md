

# RAG evaluation

## Retrieval evaluation

### Choosing right chunk size and overlap
|size/overlap | Hit@3 | Hit@5 | MRR|
|----------------|----------|-------|-----  |
|300/30 |0.9666666666666667 |1.0 | 0.886|
|500/50 |0.9666666666666667 |0.9666666666666667 | 0.861|
|1000/100 |0.9333333333333333|1.0 | 0.861|

Note : Chunk size and overlap -> 300/30 selected.

### Choosing right embedding model
|Retriever(300/30 chunks)	|Hit@3	|Hit@5	|MRR|
|----------------|----------|-------|-----  |
|MiniLM (dense only)|	0.967	|1.0	|0.886|
|bge-small (dense only)	|0.933|	0.967	|0.901|
|OpenAI 3-small (dense only)	|0.967	|1.0	|0.975|
|BM25 only|	1.0	|1.0	|0.928|
|MiniLM + BM25 (0.7/0.3)|	0.967|	1.0	|0.903|
|bge-small + BM25 (0.7/0.3)|	0.967	|0.967|	0.922|
|OpenAI + BM25 (0.7/0.3)	|1.0|	1.0	|0.933|
|MiniLM + BM25 (0.5/0.5)	|0.967	|1.0	|0.903|
|bge-small + BM25 (0.5/0.5)	|1.0	|1.0	|0.917|
|OpenAI + BM25 (0.5/0.5)	|1.0	|1.0	|0.933|
|Any model + BM25 (0.3/0.7)	|1.0	|1.0	|0.928|

Note : bge-small + BM25 (0.5/0.5) is final choice since hit rate is 1 and free too. 

### final embedding model , run with 300/30, 500/50 to get final confirmation

|Chunks(size/overlap)	|Hit@3	|Hit@5	|MRR|
|----------------|----------|-------|-----  |
| 300/30 | 1.0 | 1.0 |0.917 |
| 500/50 | 0.9666666666666667| 1.0 |0.868 |


retrieval configuration: 300/30 chunks, bge-small + BM25, weights 0.5/0.5


### Summary :
1. Chunking: small and medium chunks beat large ones; 1000-character chunks diluted topics.
2. Embeddings: OpenAI ranked best as dense-only, but every model missed different questions.
3. Hybrid: on this terminology-heavy handbook, BM25 alone matched the best hybrid. Balanced bge + BM25 was chosen for perfect recall, local privacy and robustness to paraphrasing.