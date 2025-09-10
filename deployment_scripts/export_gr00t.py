import argparse
import os
from typing import Dict, Optional

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

from export_scripts.export_onnx import export_onnx
from export_scripts.export_preprocess import export_and_test_preprocess
from export_scripts.export_postprocess import export_and_test_postprocess
from export_scripts.export_denoising_subgraph_onnx import export_denoising_subgraph
from export_scripts.utils.export_utils import get_input_info


def get_policy_and_dataset(dataset_path: str, model_path: str, device: str = "cuda"):
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
        video_backend="torchvision_av",
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=EMBODIMENT_TAG,
    )

    return policy, dataset


if __name__ == "__main__":
    # Make sure you have logged in to huggingface using `huggingface-cli login` with your nvidia email.
    parser = argparse.ArgumentParser(description="Run Groot Inference")
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset",
        default=os.path.join(os.getcwd(), "demo_data/robot_sim.PickNPlace"),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model",
        default="nvidia/GR00T-N1.5-3B",
    )

    parser.add_argument(
        "--onnx_model_path",
        type=str,
        help="Path where the ONNX model will be stored",
        default=os.path.join(os.getcwd(), "saved_models"),
    )

    args = parser.parse_args()

    print(f"Dataset path: {args.dataset_path}")
    print(f"Model path: {args.model_path}")
    print(f"ONNX model path: {args.onnx_model_path}")

    policy, dataset = get_policy_and_dataset(
        args.dataset_path, args.model_path)

    # export the preprocess model
    preprocess_model_path = os.path.join(
        args.onnx_model_path, "preprocess")
    if not os.path.exists(preprocess_model_path):
        os.makedirs(preprocess_model_path)
    export_and_test_preprocess(
        data=dataset[0],
        policy=policy,
        model_path=preprocess_model_path,
    )

    # export the postprocess model
    postprocess_model_path = os.path.join(
        args.onnx_model_path, "postprocess")
    if not os.path.exists(postprocess_model_path):
        os.makedirs(postprocess_model_path)
    export_and_test_postprocess(
        data=dataset[0],
        policy=policy,
        model_path=postprocess_model_path,
    )

    inputs = get_input_info(policy, dataset[0])

    attention_mask = inputs["eagle_attention_mask"]
    state = inputs["state"]

    export_onnx(
        dataset=dataset,
        policy=policy,
        input_state=state,
        attention_mask=attention_mask,
        onnx_model_path=args.onnx_model_path,
    )

    inputs = get_input_info(policy, dataset[0])

    attention_mask = inputs["eagle_attention_mask"]
    state = inputs["state"]
    export_denoising_subgraph(
        policy=policy,
        input_state=state,
        attention_mask=attention_mask,
        save_model_path=os.path.join(
            args.onnx_model_path, "action_head"),
    )
