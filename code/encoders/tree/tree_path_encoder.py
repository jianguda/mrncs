from typing import Dict, Any, List, Iterable, Optional, Tuple

from .tree_seq_encoder import TreeSeqEncoder
from scripts.ts import code2paths, code2identifiers
from scripts.run import get_path, load_data, paths2tokens


class TreePathEncoder(TreeSeqEncoder):
    @classmethod
    def load_metadata_from_sample(cls, language: str, data_to_brew: Any, data_to_load: Iterable[str],
                                  raw_metadata: Dict[str, Any], use_subtokens: bool=False, mark_subtoken_end: bool=False) -> None:
        # for leaf data
        # leaf_data = code2identifiers(data_to_brew, language)
        # for path data
        path_data = code2paths(data_to_brew, language)
        super().load_metadata_from_sample(language, None, path_data, raw_metadata, use_subtokens, mark_subtoken_end)

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
        # for leaf data
        # leaf_data = code2identifiers(data_to_brew, language)
        # for path data
        path_data = code2paths(data_to_brew, language)
        return super().load_data_from_sample(encoder_label, hyperparameters, metadata, language,
                                             None, path_data, function_name, result_holder, is_test)
