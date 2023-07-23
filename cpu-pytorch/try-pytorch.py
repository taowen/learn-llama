import torch

def try_embedding_1():
    my_embedding = torch.nn.Embedding(num_embeddings=4, embedding_dim=5, padding_idx=0)
    my_embedding.weight = torch.nn.Parameter(torch.tensor([
        # 就是个字典表，第一个维度是字典的key，第二个维度是查表出来的embedding
        [0.1, 0.2, 0.3, 0.4, 0.7],
        [0.9, 0.1, 0.2, 0.5, 0.5],
        [0.4, 0.2, 0.3, 0.4, 0.3],
        [0.2, 0.2, 0.1, 0.6, 0.1],
    ]))
    input_ids = torch.tensor([1, 2, 0, 3])  
    input_embeds = my_embedding(input_ids)
    print(input_embeds)
    # tensor([[0.9000, 0.1000, 0.2000, 0.5000],
    #     [0.4000, 0.2000, 0.3000, 0.4000],
    #     [0.1000, 0.2000, 0.3000, 0.4000],
    #     [0.2000, 0.2000, 0.1000, 0.6000]])

def try_embedding_2():
    my_embedding = torch.nn.Embedding(num_embeddings=4, embedding_dim=5, padding_idx=0)
    my_embedding.weight = torch.nn.Parameter(torch.tensor([
        # 就是个字典表，第一个维度是字典的key，第二个维度是查表出来的embedding
        [0.1, 0.2, 0.3, 0.4, 0.7],
        [0.9, 0.1, 0.2, 0.5, 0.5],
        [0.4, 0.2, 0.3, 0.4, 0.3],
        [0.2, 0.2, 0.1, 0.6, 0.1],
    ]))
    input_ids = torch.tensor([
        [1, 2, 0, 3],
        [3, 0, 1, 2],
    ])  
    input_embeds = my_embedding(input_ids)
    print(input_embeds)
    # tensor([[[0.9000, 0.1000, 0.2000, 0.5000, 0.5000],
    #         [0.4000, 0.2000, 0.3000, 0.4000, 0.3000],
    #         [0.1000, 0.2000, 0.3000, 0.4000, 0.7000],
    #         [0.2000, 0.2000, 0.1000, 0.6000, 0.1000]],

    #         [[0.2000, 0.2000, 0.1000, 0.6000, 0.1000],
    #         [0.1000, 0.2000, 0.3000, 0.4000, 0.7000],
    #         [0.9000, 0.1000, 0.2000, 0.5000, 0.5000],
    #         [0.4000, 0.2000, 0.3000, 0.4000, 0.3000]]])


def try_linear_1():
    my_linear = torch.nn.Linear(in_features=3, out_features=4, bias=False)
    # 3 个元素进来，4 个元素出去
    my_linear.weight = torch.nn.Parameter(torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [2, 1, 0]
    ], dtype=torch.float32))
    # 输入的 tensor 最后一个维度必须是 3 个元素的
    input = torch.tensor([[
        [1, 2, 0],
        [0, 1, 0],
    ]], dtype=torch.float32)  
    output = my_linear(input)
    print(output)
    # 输出的 tensor 最后一个维度变成了 4 个元素
    # tensor([[[ 5., 14., 23.,  4.],
    #         [ 2.,  5.,  8.,  1.]]])

def main():
    try_linear_1()

with torch.inference_mode():
    main()