from typing import Any, Dict, Optional

from encoders import NBoWEncoder, TreePathEncoder
from models import Model


class TreePathModel(Model):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        hypers = {}
        label = 'code'
        hypers.update({f'{label}_{key}': value
                       for key, value in TreePathEncoder.get_default_hyperparameters().items()})
        label = 'query'
        hypers.update({f'{label}_{key}': value
                       for key, value in NBoWEncoder.get_default_hyperparameters().items()})
        model_hypers = {
            'code_use_subtokens': False,
            'code_mark_subtoken_end': False,
            'loss': 'cosine',
            'batch_size': 1000
        }
        hypers.update(super().get_default_hyperparameters())
        hypers.update(model_hypers)
        return hypers

    def __init__(self,
                 hyperparameters: Dict[str, Any],
                 run_name: str = None,
                 model_save_dir: Optional[str] = None,
                 log_save_dir: Optional[str] = None):
        super().__init__(
            hyperparameters,
            code_encoder_type=TreePathEncoder,
            query_encoder_type=NBoWEncoder,
            run_name=run_name,
            model_save_dir=model_save_dir,
            log_save_dir=log_save_dir)
