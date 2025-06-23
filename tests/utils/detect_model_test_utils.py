import torch

from tests.utils.model_test_utils import create_test_data
from tests.utils.model_test_utils import setup_training
from tests.utils.model_test_utils import verify_model_save_load


def verify_detection_output_properties(output, model_name: str) -> None:
    """Verify standard detection model output structure (like FasterRCNN, RetinaNet, SSD)."""

    assert isinstance(output, list), f"{model_name}: Detection output should be a list."

    # Iterate over each image's detections.
    for i, detection in enumerate(output):
        assert isinstance(detection, dict), f"{model_name}: Output for image {i} should be a dict."

        for key in ["boxes", "labels", "scores"]:
            assert key in detection, f"{model_name}: Missing key '{key}' in output for image {i}."

        boxes, labels, scores = (
            detection["boxes"],
            detection["labels"],
            detection["scores"],
        )

        assert isinstance(boxes, torch.Tensor), f"{model_name}: 'boxes' for image {i} is not a torch.Tensor."
        assert isinstance(labels, torch.Tensor), f"{model_name}: 'labels' for image {i} is not a torch.Tensor."
        assert isinstance(scores, torch.Tensor), f"{model_name}: 'scores' for image {i} is not a torch.Tensor."

        # Verify expected shapes.
        assert (
            boxes.ndim == 2 and boxes.shape[1] == 4
        ), f"{model_name}: 'boxes' for image {i} should have shape (N, 4), got {boxes.shape}."
        assert labels.shape[0] == boxes.shape[0], (
            f"{model_name}: 'labels' length for image {i} ({labels.shape[0]}) "
            f"does not match number of boxes ({boxes.shape[0]})."
        )
        assert scores.shape[0] == boxes.shape[0], (
            f"{model_name}: 'scores' length for image {i} ({scores.shape[0]}) "
            f"does not match number of boxes ({boxes.shape[0]})."
        )

        # Check for invalid values.
        for tensor, name in [(boxes, "boxes"), (labels, "labels"), (scores, "scores")]:
            assert not torch.isnan(tensor).any(), f"{model_name}: {name} for image {i} contains NaN values."
            assert not torch.isinf(tensor).any(), f"{model_name}: {name} for image {i} contains Inf values."

        assert torch.all(
            (scores >= 0) & (scores <= 1)
        ), f"{model_name}: 'scores' for image {i} should be between 0 and 1."
        print(f"{model_name}: Detection output properties verified successfully.")


def verify_backward_pass_detection(model, loss_dict, model_name) -> None:
    """Verify model backward pass for standard detection models."""

    assert isinstance(loss_dict, dict), f"{model_name}: Expected loss_dict to be a dict, got {type(loss_dict)}."

    total_loss = sum(loss for loss in loss_dict.values())

    total_loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0, f"{model_name}: Gradients are not computed during backward pass (grad norm: {grad_norm})."

    print(f"{model_name}: Backward pass verified successfully, total grad norm: {grad_norm}")


def verify_detection_model_output_and_backward_pass(model, model_name, dummy_input, dummy_target, cfg):
    device = next(model.parameters()).device
    training_mode = model.training
    dummy_target = [{k: v.to(device) for k, v in t.items()} for t in dummy_target]

    dummy_input = dummy_input.to(device)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    model.train(training_mode)

    verify_detection_output_properties(output, model_name)

    loss_dict = model(dummy_input, dummy_target)
    verify_backward_pass_detection(model, loss_dict, model_name)


def run_detection_model_tests(cfg, model, model_class, model_name, input_size, batch_size):
    dummy_input, dummy_target = create_test_data(
        batch_size=batch_size,
        input_size=input_size,
        num_classes=cfg.data.num_classes,
        task="detection",
        model_name=model_name,
        bbox_format=cfg.model.bbox_format,
    )

    verify_detection_model_output_and_backward_pass(model, model_name, dummy_input, dummy_target, cfg)

    setup_training(
        model=model,
        input_size=input_size,
        num_classes=cfg.data.num_classes,
        task="detection",
        batch_size=cfg.training.batch_size,
        max_epochs=cfg.training.trainer.max_epochs,
        model_name=model_name,
        bbox_format=cfg.model.bbox_format,
    )
    print(f"{model_name}: Training integration test successful")

    verify_model_save_load(model, model_class, cfg, model_name)
    print(f"{model_name}: Model saving and loading successful")
