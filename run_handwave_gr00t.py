from gr00t.model.policy import Gr00tPolicy
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP

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
        dataset_path='/home/binliu/groot/dataset/g1_wave',
        model_path='nvidia/GR00T-N1.5-3B-WaveHand-Dev',
        device='cuda',
    )

    data = dataset[5]
    for k, v in data.items():
        print(k, v)

    print(policy.get_action(data))

    # actions = policy.get_action(dataset[0])
    # print(actions)
    # print('~~~~~~~~~~~~\n\n\n\n\~~~~~~~~~~~~~')
    # print(dataset[1])
