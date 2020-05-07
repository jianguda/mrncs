from typing import Any, Dict, Optional, Tuple, Iterable, List

import tensorflow as tf


from .common import Common
from .tree_leaf_encoder import TreeLeafEncoder
from .tree_path_encoder import TreePathEncoder
from .tree_raw_encoder import TreeRawEncoder
from ..encoder import Encoder, QueryType


class TreeAllEncoder(Encoder):
    def make_model(self, is_train: bool = False) -> tf.Tensor:
        attention = True
        embeddings = list()
        with tf.variable_scope("tree_encoder"):
            embedding4leaf = self.treeLeafEncoder.make_model(is_train)
            embedding4path = self.treePathEncoder.make_model(is_train)
            embeddings.append(embedding4leaf)
            embeddings.append(embedding4path)
            embeddings = tf.stack(embeddings, axis=0)
            if attention:
                embeddings = Common.self_attention_layer(embeddings)
            # "concat one-hot" is equal to "accumulate embedding"
            # [v1^T, v2^T, v3^T] * W = [v1^T, v2^T, v3^T]*[w1, w2, w3]^T = v1^T*w1+v2^T*w2+v3^T*w3
            print('*@' * 16)
            print(embeddings)
            print(tf.shape(embeddings))
            return tf.reduce_sum(embeddings, axis=0)
            # return tf.reduce_mean(embeddings, axis=0)

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        hypers = super().get_default_hyperparameters()
        hypers4leaf = cls.treeLeafEncoder.get_default_hyperparameters()
        hypers4path = cls.treePathEncoder.get_default_hyperparameters()
        hypers.update(hypers4leaf)
        hypers.update(hypers4path)
        return hypers

    treeLeafEncoder = None
    treePathEncoder = None

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        self.treeLeafEncoder = TreeLeafEncoder(label, hyperparameters, metadata)
        self.treePathEncoder = TreePathEncoder(label, hyperparameters, metadata)
        super().__init__(label, hyperparameters, metadata)

    @classmethod
    def init_metadata(cls) -> Dict[str, Any]:
        raw_metadata = super().init_metadata()
        raw_metadata4leaf = cls.treeLeafEncoder.init_metadata()
        raw_metadata4path = cls.treePathEncoder.init_metadata()
        raw_metadata.update(raw_metadata4leaf)
        raw_metadata.update(raw_metadata4path)
        return raw_metadata

    @classmethod
    def load_metadata_from_sample(cls, language: str, data_to_brew: Iterable[str], data_to_load: Iterable[str],
                                  raw_metadata: Dict[str, Any], use_subtokens: bool=False, mark_subtoken_end: bool=False) -> None:
        cls.treeLeafEncoder.load_data_from_sample(language, data_to_brew, data_to_load, raw_metadata, use_subtokens, mark_subtoken_end)
        cls.treePathEncoder.load_data_from_sample(language, data_to_brew, data_to_load, raw_metadata, use_subtokens, mark_subtoken_end)

    @classmethod
    def finalise_metadata(cls, encoder_label: str, hyperparameters: Dict[str, Any], raw_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        final_metadata = super().finalise_metadata(encoder_label, hyperparameters, raw_metadata_list)
        final_metadata4leaf = cls.treeLeafEncoder.finalise_metadata(encoder_label, hyperparameters, raw_metadata_list)
        final_metadata4path = cls.treePathEncoder.finalise_metadata(encoder_label, hyperparameters, raw_metadata_list)
        final_metadata.update(final_metadata4leaf)
        final_metadata.update(final_metadata4path)
        return final_metadata

    @classmethod
    def load_data_from_sample(cls,
                              encoder_label: str,
                              hyperparameters: Dict[str, Any],
                              metadata: Dict[str, Any],
                              language: str,
                              data_to_brew: Any,
                              data_to_load: Any,
                              function_name: Optional[str],
                              result_holder: Dict[str, Any],
                              is_test: bool = True) -> bool:
        """
        Saves two versions of both the code and the query: one using the docstring as the query and the other using the
        function-name as the query, and replacing the function name in the code with an out-of-vocab token.
        Sub-tokenizes, converts, and pads both versions, and rejects empty samples.
        """
        flag4leaf = cls.treeLeafEncoder.load_data_from_sample(
            encoder_label, hyperparameters, metadata, language,
            data_to_brew, data_to_load, function_name, result_holder, is_test)
        flag4path = cls.treePathEncoder.load_data_from_sample(
            encoder_label, hyperparameters, metadata, language,
            data_to_brew, data_to_load, function_name, result_holder, is_test)
        return flag4leaf and flag4path

    def extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any], is_train: bool=False,
                                   query_type: QueryType = QueryType.DOCSTRING.value) -> bool:
        """
        Implements various forms of data augmentation.
        """
        self.treeLeafEncoder.extend_minibatch_by_sample(batch_data, sample, is_train, query_type)
        self.treePathEncoder.extend_minibatch_by_sample(batch_data, sample, is_train, query_type)
        return False

    def get_token_embeddings(self) -> Tuple[tf.Tensor, List[str]]:
        # L41@embeddingvis.py
        # because this func is not important at all
        # I just make sure the interface work normally
        return self.treeLeafEncoder.get_token_embeddings()
        # return self.treePathEncoder.get_token_embeddings()

    def init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        super().init_minibatch(batch_data)
        self.treeLeafEncoder.init_minibatch(batch_data)
        self.treePathEncoder.init_minibatch(batch_data)

    def minibatch_to_feed_dict(self, batch_data: Dict[str, Any], feed_dict: Dict[tf.Tensor, Any], is_train: bool) -> None:
        super().minibatch_to_feed_dict(batch_data, feed_dict, is_train)
        self.treeLeafEncoder.minibatch_to_feed_dict(batch_data, feed_dict, is_train)
        self.treePathEncoder.minibatch_to_feed_dict(batch_data, feed_dict, is_train)
