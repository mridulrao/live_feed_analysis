"""GPU-only OneStreamer adapter. All methods run on the worker's single executor."""
import os
from pathlib import Path
import sys
import time
import uuid

from protocol import Observation, crop_frame


class OneStreamer:
    def __init__(self):
        workspace = Path(os.environ.get('ONESTREAMER_WORKSPACE', '/workspace'))
        sys.path.insert(0, str(workspace / 'OneStreamer/Eval/VLMEvalKit'))
        import torch
        from vlmeval.vlm.qwen3_vl.model import Qwen3VLChat
        from vlmeval.vlm.qwen3_ovotiming.utils import _Qwen3VLInferenceFastEngine
        if not torch.cuda.is_available():
            raise RuntimeError('A CUDA GPU is required')
        self.model = Qwen3VLChat(
            model_path=str(workspace / 'models/OneStreamer-4B'), seed=42,
            temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0,
            do_sample=True, max_new_tokens=128)
        self.engine = _Qwen3VLInferenceFastEngine(self.model)
        self.session = None

    def start(self, options):
        from vlmeval.vlm.qwen3_ovotiming.inference_fast import StreamingSession
        from vlmeval.dataset.ovo_timing import _streaming_system_prompt
        self.stop()
        self.options, self.round, self.last_end = options, 0, 0
        self.session = StreamingSession(
            self.engine, system=_streaming_system_prompt(), question=options.question,
            question_time=0, global_question=True, max_num_frames=32,
            max_tokens=128, temperature=0.7)

    def step(self, batch):
        from vlmeval.vlm.qwen3_ovotiming.utils import processor_visual_geometry, smart_video_resize
        times = [t for t, _ in batch]
        originals = [crop_frame(im, self.options.crop) for _, im in batch]
        _, _, factor = processor_visual_geometry(self.model.processor)
        images = []
        for image in originals:
            height, width = smart_video_resize(
                num_frames=1, height=image.height, width=image.width, factor=factor,
                frame_min_pixels=factor**2, frame_max_pixels=self.options.image_budget**2,
                force_resize=True)
            images.append(image.resize((width, height)))
        intervals = [(t, t + .25) for t in times]
        began = time.monotonic()
        raw = self.session.step_frames(images, intervals, self.round) or ''
        kind = 'response' if '</Response>' in raw else 'standby' if '</Standby>' in raw else 'silence'
        result = Observation(
            session_id=self.options.session_id, event_id=uuid.uuid4().hex,
            start=times[0], end=intervals[-1][1], state=kind,
            response=raw.split('</Response>', 1)[-1].strip() if kind == 'response' else None,
            inference_seconds=round(time.monotonic()-began, 3),
            skipped_seconds=round(max(0, times[0]-self.last_end), 3),
            image_budget=self.options.image_budget, input_size=list(images[-1].size))
        self.round += 1
        self.last_end = intervals[-1][1]
        return result, originals, times

    def stop(self):
        if self.session is not None:
            self.session.reset()
        self.session = None
