import os
from pathlib import Path
import numpy as np
import pytest
import torch
import onnx
import onnxruntime as ort
from gr00t.model.policy import Gr00tPolicy, unsqueeze_dict_values
from gr00t.model.gr00t_n1 import GR00T_N1_ONNX
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP

COMPUTE_DTYPE = torch.bfloat16

@pytest.fixture
def model_path():
    return "nvidia/GR00T-N1-2B"


@pytest.fixture
def dataset_path():
    return Path("demo_data/robot_sim.PickNPlace")


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def prepare_policy_input(policy, sample_input):
    # let the get_action handles both batch and single input
    is_batch = policy._check_state_is_batched(sample_input)
    if not is_batch:
        sample_input = unsqueeze_dict_values(sample_input)

    normalized_input = policy.apply_transforms(sample_input)
    return normalized_input

def test_onnx_conversion_and_accuracy(model_path, dataset_path, device, with_jit=True, with_onnx=True, rtol=1e-3, atol=1e-5):
    # Get data config for GR1 arms only configuration
    data_config = DATA_CONFIG_MAP["gr1_arms_only"]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    # Load policy
    policy = Gr00tPolicy(
        model_path=model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=EmbodimentTag.GR1,
        device=device
    )

    model_for_onnx = GR00T_N1_ONNX.from_pretrained(model_path, torch_dtype=COMPUTE_DTYPE)
    model_for_onnx.eval()  # Set model to eval mode
    model_for_onnx.to(device=device)  # type: ignore

    modality_config =  policy.modality_config
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        video_backend="decord",
        video_backend_kwargs=None,
        transforms=None,  # Load raw data first
        embodiment_tag=EmbodimentTag.GR1
    )

    # Get raw sample and prepare input
    sample_input = dataset[0]
    
    # Get prediction from policy
    # with torch.no_grad():
    #     original_output = policy.get_action(sample_input)

    # Export to ONNX
    action_onnx_path = "gr00t_model_action.onnx"
    backbone_onnx_path = "gr00t_model_backbone.onnx"
    normalized_input = prepare_policy_input(policy, sample_input)
    backbone_inputs, action_inputs = policy.model.prepare_input(normalized_input)
    # Define dynamic axes for ONNX export
    backbone_dynamic_axes = {
        "pixel_values": {0: 'batch'},
        "input_ids": {0: 'batch'},
        "attention_mask": {0: 'batch'}, # Allow dynamic batch size
        'backbone_features': {0: 'batch'},
        "backbone_attention_mask": {0: 'batch'}
    }

    action_header_dynamic_axes = {
        "backbone_features": {0: 'batch'},
        "embodiment_id": {0: 'batch'},
        "state": {0: 'batch'}
    }

    # backbone_example_inputs = (backbone_inputs.pixel_values.to(torch.float32), backbone_inputs.input_ids, backbone_inputs.attention_mask)
    backbone_example_inputs = (backbone_inputs.pixel_values, backbone_inputs.input_ids, backbone_inputs.attention_mask)
    with torch.no_grad():
        model_for_onnx.backbone.eval()
        if with_jit:
            print("Tracing backbone model with jit...")
            if os.path.exists("backbone.ts"):
                ts_backbone_model = torch.jit.load("backbone.ts")
            else:
                ts_backbone_model = torch.jit.trace(model_for_onnx.backbone, backbone_example_inputs)
                torch.jit.save(ts_backbone_model, "backbone.ts")
            backbone_features, backbone_attention_mask = ts_backbone_model(*backbone_example_inputs)
            real_output = policy.model.backbone(backbone_inputs)

            
            # Check if outputs are close
            is_close = torch.allclose(backbone_features, real_output.backbone_features, rtol=rtol, atol=atol)
            is_close = is_close and torch.allclose(backbone_attention_mask, real_output.backbone_attention_mask, rtol=rtol, atol=atol)
            if is_close:
                print(f"Backbone features and attention mask are close.")
                print("Test passed!")
            # If not close, print some statistics
            else:
                max_diff = torch.max(torch.abs(backbone_features - real_output.backbone_features))
                mean_diff = torch.mean(torch.abs(backbone_features - real_output.backbone_features))
                print(f"Maximum backbone features difference: {max_diff}")
                print(f"Mean backbone features difference: {mean_diff}")
                # Convert attention masks to float before computing differences
                backbone_attention_mask_float = backbone_attention_mask.float()
                real_attention_mask_float = real_output.backbone_attention_mask.float()
                max_diff = torch.max(torch.abs(backbone_attention_mask_float - real_attention_mask_float))
                mean_diff = torch.mean(torch.abs(backbone_attention_mask_float - real_attention_mask_float))
                print(f"Maximum backbone attention mask difference: {max_diff}")
                print(f"Mean backbone attention mask difference: {mean_diff}")
        
        if with_onnx:
            print("Exporting backbone model to ONNX using dynamo backend...")
            try:
                # First export to memory ModelProto
                if not os.path.exists(backbone_onnx_path):
                    onnx_model = torch.onnx.export(
                        model_for_onnx.backbone,
                        backbone_example_inputs,
                        input_names=['pixel_values', 'input_ids', 'attention_mask'],
                        output_names=['backbone_features', 'backbone_attention_mask'],
                        dynamic_axes=backbone_dynamic_axes,
                        opset_version=17,
                        do_constant_folding=True,
                        dynamo=True  # Enable dynamo backend for better complex number support
                    )
                    onnx_model.save(backbone_onnx_path, external_data=True)
                    print("Model converted to external data format")
                else:
                    onnx_model = onnx.load(backbone_onnx_path, load_external_data=True)
                    print("Model loaded from file")

                # Verify ONNX model
                onnx.checker.check_model(backbone_onnx_path)
                print("ONNX model is valid")

                # onnx_model_bytes = onnx_model.SerializeToString()
                # print("ONNX model serialized")
                ort_session = ort.InferenceSession(
                    backbone_onnx_path, 
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                )
                print("ONNX Runtime session created")

                # Prepare inputs for ONNX runtime
                ort_inputs = {
                    'pixel_values': backbone_inputs.pixel_values.cpu().numpy(),
                    'input_ids': backbone_inputs.input_ids.cpu().numpy(),
                    'attention_mask': backbone_inputs.attention_mask.cpu().numpy()
                }

                # Run inference with ONNX model
                ort_outputs = ort_session.run(None, ort_inputs)
                onnx_backbone_features = torch.from_numpy(ort_outputs[0]).to(device)
                onnx_backbone_attention_mask = torch.from_numpy(ort_outputs[1]).to(device)

                # Compare ONNX outputs with original model
                is_close = torch.allclose(onnx_backbone_features, real_output.backbone_features, rtol=rtol, atol=atol)
                is_close = is_close and torch.allclose(onnx_backbone_attention_mask, real_output.backbone_attention_mask, rtol=rtol, atol=atol)
                
                if is_close:
                    print("ONNX model outputs match original model outputs.")
                    print("ONNX test passed!")
                else:
                    print("ONNX model outputs differ from original model outputs:")
                    max_diff = torch.max(torch.abs(onnx_backbone_features - real_output.backbone_features))
                    mean_diff = torch.mean(torch.abs(onnx_backbone_features - real_output.backbone_features))
                    print(f"Maximum backbone features difference: {max_diff}")
                    print(f"Mean backbone features difference: {mean_diff}")
                    
                    # Convert attention masks to float for comparison
                    onnx_attention_mask_float = onnx_backbone_attention_mask.float()
                    real_attention_mask_float = real_output.backbone_attention_mask.float()
                    max_diff = torch.max(torch.abs(onnx_attention_mask_float - real_attention_mask_float))
                    mean_diff = torch.mean(torch.abs(onnx_attention_mask_float - real_attention_mask_float))
                    print(f"Maximum backbone attention mask difference: {max_diff}")
                    print(f"Mean backbone attention mask difference: {mean_diff}")

            except Exception as e:
                print(f"Export failed: {str(e)}")
                print("Please check if your PyTorch version supports dynamo backend")
                return

            # Clean up
            if os.path.exists(backbone_onnx_path):
                os.remove(backbone_onnx_path)
            if os.path.exists("model_data.bin"):
                os.remove("model_data.bin")



def debug_test():
    # Manually provide the fixture values
    model_path = "nvidia/GR00T-N1-2B"
    dataset_path = Path("/workspace/volumes/issac/Isaac-GR00T/demo_data/robot_sim.PickNPlace")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Call the test function with the parameters
    test_onnx_conversion_and_accuracy(model_path, dataset_path, device, with_jit=False, with_onnx=True)


if __name__ == "__main__":
    debug_test()
