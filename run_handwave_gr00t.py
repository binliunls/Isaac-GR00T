from gr00t.model.policy import Gr00tPolicy
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import torch

def load_and_visualize_tensors(tensor_np, save_plots=True, output_dir="./plots"):
    """
    Load concatenated tensor from .pt file, squeeze size 1 dimensions, and visualize vectors.
    
    Args:
        tensor_file_path (str): Path to the concatenated tensor file (.pt)
        save_plots (bool): Whether to save plots to files
        output_dir (str): Directory to save plots
    
    Returns:
        torch.Tensor: The loaded and squeezed tensor
    """
    # Replace values outside [-3.5, 3.5] with 0.
    tensor_np = np.where((tensor_np >= -3.5) & (tensor_np <= 3.5), tensor_np, 0)
    tensor_np = tensor_np[:1200, :]
    
    
    # Create output directory if it doesn't exist.
    if save_plots:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get dimensions: [time_steps, features]
    time_steps, num_features = tensor_np.shape
    print(f"Time series data: {time_steps} time steps, {num_features} features")
    
    # Create time series plot where x-axis is time and y-axis shows all 28 features.
    plt.figure(figsize=(15, 10))
    
    # Plot each of the 28 features as separate lines.
    for feature_idx in range(num_features):
        action = tensor_np[:, feature_idx]
        if np.max(action) > 3.5:
            # Find indices where values are larger than 3.5.
            # large_indices = np.where(action > 3.5)[0]
            # print(f"Feature {feature_idx+1}: Found {len(large_indices)} values > 3.5")
            # print(f"  Indices: {large_indices}")
            # print(f"  Values: {action[large_indices]}")
            # print(f"  Max value: {np.max(action):.4f}")
            # print("-" * 50)
            breakpoint()
            
        plt.plot(range(time_steps), tensor_np[:, feature_idx], 
                alpha=0.7, linewidth=1, label=f'Feature {feature_idx+1}')
    
    plt.title(f'Time Series Plot: {time_steps} Time Steps × {num_features} Features')
    plt.xlabel('Time Step')
    plt.ylabel('Feature Value')
    plt.grid(True, alpha=0.3)
    
    # Only show legend if we have reasonable number of features.
    if num_features <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f"{output_dir}/time_series_plot.png", dpi=300, bbox_inches='tight')
        print(f"Saved time series plot to {output_dir}/time_series_plot.png")
    
    plt.close()  # Close the figure to free memory.
    
    # Create a heatmap visualization.
    plt.figure(figsize=(15, 8))
    plt.imshow(tensor_np.T, cmap='viridis', aspect='auto', interpolation='nearest')
    plt.title(f'Heatmap: Time Series Data ({time_steps} × {num_features})')
    plt.xlabel('Time Step')
    plt.ylabel('Feature Index')
    plt.colorbar(label='Feature Value')
    
    if save_plots:
        plt.savefig(f"{output_dir}/heatmap_plot.png", dpi=300, bbox_inches='tight')
        print(f"Saved heatmap plot to {output_dir}/heatmap_plot.png")
    
    plt.close()  # Close the figure to free memory.
    
    # Create feature-wise statistics plot.
    plt.figure(figsize=(15, 6))
    
    # Calculate statistics for each feature across time.
    feature_means = np.mean(tensor_np, axis=0)
    feature_stds = np.std(tensor_np, axis=0)
    feature_mins = np.min(tensor_np, axis=0)
    feature_maxs = np.max(tensor_np, axis=0)
    
    x_pos = np.arange(num_features)
    plt.errorbar(x_pos, feature_means, yerr=feature_stds, 
                fmt='o', capsize=5, capthick=2, alpha=0.7, label='Mean ± Std')
    plt.fill_between(x_pos, feature_mins, feature_maxs, alpha=0.2, label='Min-Max Range')
    
    plt.title('Feature Statistics Across Time')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_plots:
        plt.savefig(f"{output_dir}/feature_statistics.png", dpi=300, bbox_inches='tight')
        print(f"Saved feature statistics plot to {output_dir}/feature_statistics.png")
    
    plt.close()  # Close the figure to free memory.
    
    # Print statistics.
    print(f"\nTensor Statistics:")
    print(f"  Mean: {tensor_np.mean():.6f}")
    print(f"  Std:  {tensor_np.std():.6f}")
    print(f"  Min:  {tensor_np.min():.6f}")
    print(f"  Max:  {tensor_np.max():.6f}")


def get_policy_and_dataset(dataset_path: str = '/home/binliu/groot/dataset/g1_wave',
                           model_path: str = 'nvidia/GR00T-N1.5-3B-WaveHand-Dev',
                           device: str = "cuda"):
    # load the policy
    data_config = DATA_CONFIG_MAP["unitree_g1_v2"]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    EMBODIMENT_TAG = "new_embodiment"
    
    policy = Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=device,
    )

    # torch.cuda.manual_seed(42)
    # if not hasattr(policy.model.action_head, "init_actions"):
    #     print("Initializing init_actions")
    #     policy.model.action_head.init_actions = torch.randn(
    #         (1, policy.model.action_head.action_horizon,
    #             policy.model.action_head.action_dim),
    #         dtype=torch.float16,
    #         device=device,
    #     )
    modality_config = policy.modality_config
    # load the dataset
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        video_backend="torchvision_av",  # Use PyAV for better AV1 codec support
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=EMBODIMENT_TAG,
    )

    return policy, dataset


if __name__ == "__main__":
    policy, dataset = get_policy_and_dataset(
        dataset_path='/home/nvidia/binliu/dataset/g1_wave_long_prompt',
        model_path='nvidia/GR00T-N1.5-3B-WaveHand-Dev',
        device='cuda',
    )
    # video.rs_view [1, 480, 640, 3]
    # state.body [1, 29]
    # state.hands [1, 14]
    # action.upper_body [16, 14]
    # actions.hands [16, 14]
    use_random = True


    data = dataset[5]
    full_actions = []
    for i in range(100):
        if use_random:
            img = np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8)
            data = {
                "video.rs_view": img,
                "state.body": np.random.rand(1, 29) - 0.5,
                "state.hands": np.random.rand(1, 14) - 0.5,
            }
        else:
            data = dataset[i]
        action = policy.get_action(data)
        upper_action = action["action.upper_body"]
        hands_action = action["action.hands"]
        full_action = np.concatenate([upper_action, hands_action], axis=1)
        full_actions.append(full_action)
    full_actions = np.concatenate(full_actions, axis=0)
    print(full_actions.shape)
    load_and_visualize_tensors(full_actions, save_plots=True, output_dir="./plots")
    # actions = policy.get_action(dataset[0])
    # print(actions)
    # print('~~~~~~~~~~~~\n\n\n\n\~~~~~~~~~~~~~')
    # print(dataset[1])
