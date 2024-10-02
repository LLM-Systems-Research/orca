
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from threading import Thread

# ORCA Execution Engine
class ORCAExecutionEngine:
    def __init__(self, input_dim: int = 64, num_heads: int = 4):
        self.num_heads = num_heads
        self.input_dim = input_dim  # Define your input dimension
        self.attn_layer = ORCAAttention(input_dim, num_heads)
        

    def execute(self, x):
        # Split the input across heads
        batch_size, seq_len, dim = x.shape
        qkv = self.attn_layer.qkv_linear(x)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.attn_layer.head_dim)
        q, k, v = qkv.split(self.attn_layer.head_dim, dim=-1)

        # Initialize attention outputs
        attention_outputs = [None] * self.num_heads

        # Define a worker for each attention head
        def worker(head_idx):
            q_i, k_i, v_i = q[:, :, head_idx, :], k[:, :, head_idx, :], v[:, :, head_idx, :]
            attn_out = self.attn_layer.attention(q_i, k_i, v_i)
            attention_outputs[head_idx] = attn_out

        # Create and start threads for each head
        threads = []
        for head in range(self.num_heads):
            thread = Thread(target=worker, args=(head,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Merge attention outputs
        merged_attn = torch.cat(attention_outputs, dim=-1)

        # Output projection after merging
        output = self.attn_layer.attn_out_linear(merged_attn)
        print(f"Output: {output}")

        return output

class ORCAAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(ORCAAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # QKV linear projections
        self.qkv_linear = nn.Linear(input_dim, 3 * input_dim)

        # Output linear projection after attention
        self.attn_out_linear = nn.Linear(input_dim, input_dim)
    
    def attention(self, q, k, v):
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        return attn_output