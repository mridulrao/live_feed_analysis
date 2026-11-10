"""GPU-only OneStreamer adapter. All methods run on the worker's single executor."""
import os
import re
from pathlib import Path
import sys
import time
import uuid

from protocol import Observation, crop_frame

CONTINUOUS_QUESTION = (
    'Continuously describe the visible scene, people, objects, their positions and actions. '
    'For every update, report concrete visible details and changes using </Observe>. '
    'Do not restrict observations to a specific target or wait for a requested event. '
    'Do not invent details outside the visible crop. If evidence is insufficient, say so. '
    'Avoid repeating unchanged details at length. Additional user interests: '
)

CONTINUOUS_SYSTEM = '''You analyze a continuous stream of sampled video frames with time intervals.
On EVERY update, write </Observe> followed by a brief factual description of the visible scene,
people, objects, positions, actions and changes. Observe broadly, even without a requested event.
When an event has completed, optionally add </Summary> with that event description.
For a noteworthy change that could merit a conversational reaction, optionally add </Response>
with a concise description. Do not repeat an unchanged event. Never invent unseen activity.
Use </Silence> only when the frames cannot support any description; silence does not mean no activity.
Descriptions are observations, not instructions to the user.'''


def parse_output(raw):
    """Accept both the pinned QA output and descriptive observation/summary tags."""
    sections = {}
    matches = list(re.finditer(r'</(Observe|Summary|Response|Standby|Silence)>', raw))
    for i, match in enumerate(matches):
        stop = matches[i+1].start() if i+1 < len(matches) else len(raw)
        sections[match.group(1)] = raw[match.end():stop].strip()[:8000] or None
    state = 'response' if 'Response' in sections else 'standby' if 'Standby' in sections else 'silence'
    return state, sections.get('Response'), sections.get('Observe'), sections.get('Summary')


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
            self.engine, system=CONTINUOUS_SYSTEM if options.vision_memory else _streaming_system_prompt(),
            question=(CONTINUOUS_QUESTION + options.question if options.vision_memory else options.question),
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
        kind, response, description, summary = parse_output(raw)
        result = Observation(
            session_id=self.options.session_id, event_id=uuid.uuid4().hex,
            start=times[0], end=intervals[-1][1], state=kind,
            response=response, description=description, summary=summary,
            sample_times=times, inference_seconds=round(time.monotonic()-began, 3),
            skipped_seconds=round(max(0, times[0]-self.last_end), 3),
            image_budget=self.options.image_budget, input_size=list(images[-1].size))
        self.round += 1
        self.last_end = intervals[-1][1]
        return result, originals, times

    def stop(self):
        if self.session is not None:
            self.session.reset()
        self.session = None
