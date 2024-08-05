import warnings

import torch


class PromptEncoder(torch.nn.Module):
    """
    The prompt encoder network that is used to generate the virtual token embeddings for p-tuning.
    """

    def __init__(
        self,
        embedding_layer,
        token_dim,
        encoder_hidden_size,
        num_virtual_tokens,
        num_transformer_submodules,
        inference_mode,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.og_embedding = embedding_layer
        self.input_size = self.token_dim
        self.output_size = self.token_dim
        self.hidden_size = encoder_hidden_size
        self.total_virtual_tokens = num_virtual_tokens * num_transformer_submodules
        self.inference_mode = inference_mode

        # embedding
        self.embedding = torch.nn.Embedding(self.total_virtual_tokens, self.token_dim)
        if not self.inference_mode:
            layers = [
                torch.nn.Linear(self.input_size, self.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_size, self.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_size, self.output_size),
            ]
            self.mlp_head = torch.nn.Sequential(*layers)

    def forward(self, indices):
        input_embeds = self.embedding(indices)
        output_embeds = self.mlp_head(input_embeds)

        return output_embeds

    def get_prompt(self, batch_size):
        self.prompt_tokens = torch.arange(
            self.num_virtual_tokens * self.num_transformer_submodules
        ).long()

        processed_prompt_tokens = (
            self.prompt_tokens[self.active_adapter]
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(self.embedding.weight.device)
        )

        if self.inference_mode:
            final_prompts = self.embedding.weight.repeat(batch_size, 1, 1)
        else:
            final_prompts = self(processed_prompt_tokens)

        return final_prompts

    def get_input_embeds(self, input_ids):
        return self.og_embedding(input_ids)
