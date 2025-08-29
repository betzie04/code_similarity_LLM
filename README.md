# analyse_assembler_by_llm
Create Docker image first by using Dockerfile.

Run docker image and mount data files in running container: 

docker run --rm -it -d --gpus 'device=5' -v /raid/betzie/data-raw/:/analyse_binary_llm/data-raw/ -v /raid/betzie/data-src/:/analyse_binary_llm/data-src/ -v /raid/betzie/data-bin/:/analyse_binary_llm/data-bin/ -v ~/analyse_assembler_by_llm/:/analyse_binary_llm -v ~/improving-binary-similarity-models-using-dynamic-information/:/analyse_binary_llm/trex analyse-llm-ghidranew /bin/bash




## Getting started

'get_assembler_pairs.py' creates assembler pairs into data-src/similarity_test_llm/
'load_assembler_to_llm.py' loads assembler pairs from data-src/similarity_test_llm/ and sends the promp to a dspy llm that is seted up in this file. The promts are stored into data_src/similarity_test_llm/llm_results_sim_range.txt

'analyzy_llm_output.py' structures the output from load_assembler_to_llm.py again and contains the results, where the llm sayes a different similarity score than the label is





