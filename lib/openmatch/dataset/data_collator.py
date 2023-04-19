from dataclasses import dataclass

from transformers import DataCollatorWithPadding
from transformers import default_data_collator


@dataclass
class QPCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        qq = [f["query"] for f in features]
        dd = [f["passages"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )

        return q_collated, d_collated


@dataclass
class EncodeCollator():
    def __call__(self, features):
        text_ids = [x["text_id"] for x in features]
        collated_features = default_data_collator(features)
        return text_ids, collated_features
