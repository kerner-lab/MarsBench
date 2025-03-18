import torch

from tests.utils.model_test_utils import create_test_data
from tests.utils.model_test_utils import setup_training
from tests.utils.model_test_utils import verify_model_save_load


def verify_detection_output_properties(output, model_name: str) -> None:
    """Verify standard detection model output structure (like FasterRCNN, RetinaNet, SSD)."""

    assert isinstance(output, list), f"{model_name}: Detection output should be a list."

    # Iterate over each image's detections.
    for i, detection in enumerate(output):
        assert isinstance(
            detection, dict
        ), f"{model_name}: Output for image {i} should be a dict."

        for key in ["boxes", "labels", "scores"]:
            assert (
                key in detection
            ), f"{model_name}: Missing key '{key}' in output for image {i}."

        boxes, labels, scores = (
            detection["boxes"],
            detection["labels"],
            detection["scores"],
        )

        assert isinstance(
            boxes, torch.Tensor
        ), f"{model_name}: 'boxes' for image {i} is not a torch.Tensor."
        assert isinstance(
            labels, torch.Tensor
        ), f"{model_name}: 'labels' for image {i} is not a torch.Tensor."
        assert isinstance(
            scores, torch.Tensor
        ), f"{model_name}: 'scores' for image {i} is not a torch.Tensor."

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
            assert not torch.isnan(
                tensor
            ).any(), f"{model_name}: {name} for image {i} contains NaN values."
            assert not torch.isinf(
                tensor
            ).any(), f"{model_name}: {name} for image {i} contains Inf values."

        assert torch.all(
            (scores >= 0) & (scores <= 1)
        ), f"{model_name}: 'scores' for image {i} should be between 0 and 1."
        print(f"{model_name}: Detection output properties verified successfully.")


def verify_detr_output_properties(output: dict, model_name: str) -> None:
    """
    DETR typically returns a dict with 'pred_logits' and 'pred_boxes' of shape:
    (batch_size, num_queries, num_classes) and (batch_size, num_queries, 4).
    """

    assert isinstance(output, dict), f"{model_name}: Output should be a dictionary."
    for key in ["pred_logits", "pred_boxes"]:
        assert key in output, f"{model_name}: Missing key '{key}' in output."

    pred_logits = output["pred_logits"]
    pred_boxes = output["pred_boxes"]
    assert isinstance(
        pred_logits, torch.Tensor
    ), f"{model_name}: 'pred_logits' should be a tensor."
    assert isinstance(
        pred_boxes, torch.Tensor
    ), f"{model_name}: 'pred_boxes' should be a tensor."

    # Check shapes
    assert (
        pred_logits.ndim == 3
    ), f"{model_name}: 'pred_logits' should have 3 dimensions, got {pred_logits.ndim}."
    assert (
        pred_boxes.ndim == 3
    ), f"{model_name}: 'pred_boxes' should have 3 dimensions, got {pred_boxes.ndim}."

    # Check for invalid values
    for tensor, name in [(pred_logits, "pred_logits"), (pred_boxes, "pred_boxes")]:
        assert not torch.isnan(tensor).any(), f"{model_name}: {name} has NaNs."
        assert not torch.isinf(tensor).any(), f"{model_name}: {name} has Infs."

    # DETR typically returns normalized boxes.
    if not torch.all((pred_boxes >= 0) & (pred_boxes <= 1)):
        raise AssertionError(
            f"{model_name}: 'pred_boxes' values must be between 0 and 1."
        )

    # Softmax over the last dimension of pred_logits -> sum = 1
    pred_probs = torch.softmax(pred_logits, dim=-1)
    sums = pred_probs.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5):
        raise AssertionError(f"{model_name}: Softmax along classes != 1.")

    print(
        f"{model_name}: DETR output properties verified successfully. "
        f"pred_logits shape: {pred_logits.shape}, pred_boxes shape: {pred_boxes.shape}"
    )


def verify_efficientdet_output_properties(detections, model_name: str) -> None:
    """Verify model output properties for EfficientDet."""
    assert isinstance(
        detections, torch.Tensor
    ), f"{model_name}: Detections should be a tensor, got {type(detections)}"
    assert (
        detections.ndim == 3 and detections.shape[-1] == 6
    ), f"{model_name}: Detections should have shape (batch, num_detections, 6), got {detections.shape}."

    # Unpack the detection tensor.
    x1, y1, x2, y2, score, cls = detections.unbind(-1)

    # Check bboxes and score are valid.
    assert torch.all(x1 < x2), f"{model_name}: Not all detections satisfy x1 < x2."
    assert torch.all(y1 < y2), f"{model_name}: Not all detections satisfy y1 < y2."
    assert torch.all(
        (score >= 0) & (score <= 1)
    ), f"{model_name}: Some detection scores are not in [0, 1]."

    print(
        f"{model_name}: EfficientDet detections verified successfully. Detections shape: {detections.shape}"
    )


def verify_backward_pass_detection(model, loss_dict, model_name) -> None:
    """Verify model backward pass for standard detection models."""

    assert isinstance(
        loss_dict, dict
    ), f"{model_name}: Expected loss_dict to be a dict, got {type(loss_dict)}."

    total_loss = sum(loss for loss in loss_dict.values())

    total_loss.backward()
    grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    assert (
        grad_norm > 0
    ), f"{model_name}: Gradients are not computed during backward pass (grad norm: {grad_norm})."

    print(
        f"{model_name}: Backward pass verified successfully, total grad norm: {grad_norm}"
    )


def verify_backward_pass_detection_detr(model, output, target, model_name) -> None:
    """Verify model backward pass for DETR."""
    criterion = model.criterion
    weight_dict = criterion.weight_dict
    loss_dict = criterion(output, target)
    total_loss = sum(
        loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
    )

    total_loss.backward()
    grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    assert (
        grad_norm > 0
    ), f"{model_name}: Gradients are not computed during backward pass (grad norm: {grad_norm})."

    print(
        f"{model_name}: Backward pass verified successfully, total grad norm: {grad_norm}"
    )


def verify_backward_pass_detection_efficientdet(model, loss, model_name) -> None:
    """Verify model backward pass for EfficientDET."""
    loss.backward()
    grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    assert (
        grad_norm > 0
    ), f"{model_name}: Gradients not computed (grad norm: {grad_norm})."
    print(
        f"{model_name}: Backward pass verified successfully with grad norm: {grad_norm}"
    )


def verify_detection_model_output_and_backward_pass(
    model, model_name, dummy_input, dummy_target, cfg
):
    device = next(model.parameters()).device
    training_mode = model.training
    dummy_target = [{k: v.to(device) for k, v in t.items()} for t in dummy_target]

    if model_name.lower() == "detr":
        dummy_input = dummy_input.detach()
        dummy_input = dummy_input.to(device)

        output = model(dummy_input)

        verify_detr_output_properties(output, model_name)
        verify_backward_pass_detection_detr(model, output, dummy_target, model_name)

    elif model_name.lower() == "efficientdet":
        dummy_input = dummy_input.to(device)

        dummy_target = {
            "bbox": [target["boxes"] for target in dummy_target],
            "cls": [target["labels"] for target in dummy_target],
            "img_size": torch.stack([target["img_size"] for target in dummy_target]),
            "img_scale": torch.tensor([target["img_scale"] for target in dummy_target]),
        }

        model.eval()
        with torch.no_grad():
            output = model(dummy_input, dummy_target)
        model.train(training_mode)

        expected_eval_keys = {"loss", "class_loss", "box_loss", "detections"}
        assert expected_eval_keys.issubset(
            output.keys()
        ), f"{model_name}: Evaluation output missing keys. Expected {expected_eval_keys}, got {output.keys()}"

        detections = output["detections"]
        verify_efficientdet_output_properties(detections, model_name)

        loss_dict = model(dummy_input, dummy_target)
        verify_backward_pass_detection_efficientdet(
            model, loss_dict["loss"], model_name
        )

    else:
        dummy_input = dummy_input.to(device)

        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        model.train(training_mode)

        verify_detection_output_properties(output, model_name)

        loss_dict = model(dummy_input, dummy_target)
        verify_backward_pass_detection(model, loss_dict, model_name)


def run_detection_model_tests(
    cfg, model, model_class, model_name, input_size, batch_size
):
    dummy_input, dummy_target = create_test_data(
        batch_size=batch_size,
        input_size=input_size,
        num_classes=cfg.data.num_classes,
        task="detection",
        model_name=model_name,
        bbox_format=cfg.model.bbox_format,
    )

    verify_detection_model_output_and_backward_pass(
        model, model_name, dummy_input, dummy_target, cfg
    )

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
