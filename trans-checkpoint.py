import torch

# 假设模型 checkpoint 文件路径
checkpoint_path = './pretrained/model_best_sid_sony.pth'

# 加载 checkpoint
checkpoint = torch.load(checkpoint_path)

# 获取模型的 state_dict
state_dict = checkpoint['model']  # 如果 checkpoint 中有其他键，请根据实际情况调整

# 遍历 state_dict，找到所有符合条件的层
new_state_dict = {}
for key, value in state_dict.items():
    # 只替换 'color_correction_blocks.x.feedforward.2.weight' 中的 2 为 3
    if 'color_correction_blocks.' in key and '.feedforward.2.' in key:
        # 将 '.feedforward.2.' 替换为 '.feedforward.3.'
        new_key = key.replace('.feedforward.2.', '.feedforward.3.')
        print(key)
        print(new_key)
        # 将修改后的层添加到新的 state_dict 中
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

# 现在我们保留原 checkpoint 的其他部分，更新 model_state_dict
checkpoint['model'] = new_state_dict  # 更新 model 部分

# 保存完整的 checkpoint，包括原有的其他部分
torch.save(checkpoint, './pretrained/modified_checkpoint.pth')

print("Checkpoint updated and saved.")
