from typing import Optional, Tuple, Union

import torch
from transformers.models.clip.modeling_clip import (
    CLIP_TEXT_INPUTS_DOCSTRING,
    BaseModelOutputWithPooling,
    CLIPTextConfig,
    CLIPTextModel,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)


class CustomCLIPTextModel(CLIPTextModel):
    default_clip_skip = None

    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        clip_skip: Optional[int] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, CLIPTextModel

        >>> model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        clip_skip = clip_skip if clip_skip is not None else self.default_clip_skip

        text_encoder_outputs: BaseModelOutputWithPooling = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        if clip_skip is not None and clip_skip > 0:
            last_hidden_state = text_encoder_outputs.hidden_states[-clip_skip]
            normed_state = self.text_model.final_layer_norm(last_hidden_state)
            text_encoder_outputs.last_hidden_state = normed_state
            return text_encoder_outputs
        else:
            return text_encoder_outputs
