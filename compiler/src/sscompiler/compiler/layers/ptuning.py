# import torch
# import torch.nn as nn


# class PTuningPrompt(nn.Module):
#     def __init__(
#         self,
#         embedding_layer: nn.Embedding,
#         num_virtual_tokens: int,
#     ):
#         super().__init__()
#         self.embedding_layer = embedding_layer
#         self.num_virtual_tokens = num_virtual_tokens
#         self.virtual_prompt = nn.Parameter(
#             torch.randn(num_virtual_tokens, embedding_layer.embedding_dim)
#         )
#         self.first_token = True

#     def forward(self, indices):

#         if self.first_token:
#             embedded_tokens = self.embedding_layer(indices)

#             # Expand virtual tokens to match the batch size of the indices and concatenate
#             batch_size = indices.size(0)
#             expanded_virtual_tokens = (
#                 self.virtual_prompt.unsqueeze(0)
#                 .expand(batch_size, -1, -1)
#                 .to(torch.device("cuda:0"))
#                 .to(torch.bfloat16)
#             )

#             output = torch.cat([embedded_tokens, expanded_virtual_tokens], dim=1)
#             self.first_token = False
#             return output
#         else:
#             return self.embedding_layer(indices)

#     def get_prompt(self, batch_size):
#         self.prompt_tokens = torch.arange(
#             self.num_virtual_tokens * self.num_transformer_submodules
#         ).long()

#         processed_prompt_tokens = (
#             self.prompt_tokens[self.active_adapter]
#             .unsqueeze(0)
#             .expand(batch_size, -1)
#             .to(self.embedding.weight.device)
#         )

#         if self.inference


# def mark_only_p_tuning_as_trainable(model: nn.Module) -> None:
#     for n, p in model.named_parameters():
#         if "virtual_prompt" not in n:
#             p.requires_grad = False
#         else:
#             p.requires_grad = True
