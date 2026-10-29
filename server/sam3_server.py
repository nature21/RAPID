"""SAM3 image segmentation service compatible with RAPID's SAM3 client."""

import numpy as np

from server.common import create_app, decode_image, encode_bytes, parser, serve


class SAM3:
    def __init__(self, checkpoint=None, device="cuda"):
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.processor_class = Sam3Processor
        self.device = device
        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint, device=device,
            enable_inst_interactivity=True,
        )

    def __call__(self, image, text, bbox, confidence):
        import torch

        with torch.inference_mode():
            processor = self.processor_class(
                self.model, device=self.device, confidence_threshold=confidence,
            )
            state = processor.set_image(image)
            processor.reset_all_prompts(state)
            if text is not None:
                state = processor.set_text_prompt(state=state, prompt=text)
            if bbox is not None:
                x, y, width, height = bbox
                box = [(x + width / 2) / image.width, (y + height / 2) / image.height,
                       width / image.width, height / image.height]
                state = processor.add_geometric_prompt(state=state, box=box, label=True)
            return {
                "scores": state["scores"].detach().cpu().tolist(),
                "bboxes": state["boxes"].detach().cpu().tolist(),
                "masks": [encode_bytes(mask.squeeze(0).cpu().numpy().astype(np.uint8))
                          for mask in state["masks"]],
            }


def build_app(predictor):
    def query(payload):
        image = decode_image(payload)
        text, bbox = payload.get("text"), payload.get("bbox")
        if text is None and bbox is None:
            raise ValueError("Provide text or a pixel xywh bbox")
        if text is not None and (not isinstance(text, str) or not text.strip()):
            raise ValueError("text must be a nonempty string")
        if bbox is not None:
            bbox = np.asarray(bbox, dtype=float)
            if bbox.shape != (4,) or not np.isfinite(bbox).all() or np.any(bbox[2:] <= 0):
                raise ValueError("bbox must be [x, y, width, height] with positive size")
        confidence = float(payload.get("confidence_threshold", 0.1))
        if not 0 <= confidence <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        return predictor(image, text, bbox, confidence)
    return create_app("sam3", {"/sam3_query": query})


def main():
    cli = parser(__doc__, 4002)
    cli.add_argument("--checkpoint", help="Local SAM3 checkpoint; otherwise use Hugging Face")
    cli.add_argument("--device", default="cuda")
    args = cli.parse_args()
    serve(build_app(SAM3(args.checkpoint, args.device)), args)


if __name__ == "__main__":
    main()
