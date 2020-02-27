#!/usr/bin/env bash
###########################################################
# MAX_CONTEXTS - the number of actual contexts that are
# taken into consideration every training iteration.
#
# SUBTOKEN_VOCAB_SIZE, TARGET_VOCAB_SIZE - the number of
# subtokens and target words to keep in the vocabulary
# (the top occurring words and paths will be kept).

MAX_CONTEXTS=200
SUBTOKEN_VOCAB_SIZE=186277
TARGET_VOCAB_SIZE=26347

data_dir=${1:-data}
mkdir -p "${data_dir}"
train_data_file=$data_dir/train_output_file.txt
valid_data_file=$data_dir/valid_output_file.txt
test_data_file=$data_dir/test_output_file.txt
###########################################################

echo "Extracting..."
python extract.py

echo "Creating histograms from the training data..."
target_histogram_file=$data_dir/histo.tgt.c2s
source_subtoken_histogram=$data_dir/histo.ori.c2s
node_histogram_file=$data_dir/histo.node.c2s
cut <"${train_data_file}" -d' ' -f1 | tr '|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' >"${target_histogram_file}"
cut <"${train_data_file}" -d' ' -f2- | tr ' ' '\n' | cut -d',' -f1,3 | tr ',|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' >"${source_subtoken_histogram}"
cut <"${train_data_file}" -d' ' -f2- | tr ' ' '\n' | cut -d',' -f2 | tr '|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' >"${node_histogram_file}"

echo "Preprocessing..."
python preprocess.py \
  --train_data "${train_data_file}" \
  --test_data "${test_data_file}" \
  --val_data "${valid_data_file}" \
  --max_contexts ${MAX_CONTEXTS} \
  --subtoken_vocab_size ${SUBTOKEN_VOCAB_SIZE} \
  --target_vocab_size ${TARGET_VOCAB_SIZE} \
  --target_histogram ${target_histogram_file} \
  --subtoken_histogram ${source_subtoken_histogram} \
  --node_histogram ${node_histogram_file} \
  --output_name ${data_dir}/$(basename ${data_dir})
# If all went well, the raw data files can be deleted, because preprocess.py creates new files
# with truncated and padded number of paths for each example.
rm ${train_data_file} \
  ${test_data_file} \
  ${valid_data_file} \
  ${target_histogram_file} \
  ${source_subtoken_histogram} \
  ${node_histogram_file}

# todo simplify & work on Python Corpus
# Step#1 parse AST from code
# Step#2 extract Path from AST
# Step#3 preprocess Path
