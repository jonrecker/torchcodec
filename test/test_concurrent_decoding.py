# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import concurrent.futures
import faulthandler
import multiprocessing
import os
import subprocess
import traceback

import pytest
import torch

from .utils import all_supported_devices, make_video_decoder, needs_ffmpeg_cli

# The workload normally finishes within seconds, interpreter startup included.
# The bugs this test guards against can crash, abort or hang decoding, none of
# which can be caught in-process, hence the separate process and the timeout. A
# hang is reported with the stacks of all threads.
TIMEOUT_SECONDS = 120
STACK_DUMP_SECONDS = TIMEOUT_SECONDS - 10

NUM_THREADS = 10
NUM_TASKS = 20
# The bugs are probabilistic, so the workload is repeated to make a miss unlikely.
NUM_PASSES = 3

# 2 s of 1080p60 with a single keyframe. Large frames are needed to trigger the
# bugs reliably: smaller clips, such as the 720p resources, did not trigger them
# even when decoding thousands of frames.
CLIP_NUM_FRAMES = 120


@pytest.fixture(scope="session")
def clip_1080p(tmp_path_factory):
    path = tmp_path_factory.mktemp("clip1080p") / "mandelbrot_1080p.mp4"
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error",
            "-f", "lavfi", "-i", "mandelbrot=size=1920x1080:rate=60",
            "-frames:v", str(CLIP_NUM_FRAMES),
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-g", "600",
            str(path),
        ],
        check=True,
    )  # fmt: skip
    return path


def _decode_all_frames(path, device):
    decoder, _ = make_video_decoder(path, device=device, seek_mode="approximate")
    return decoder.get_frames_in_range(0, CLIP_NUM_FRAMES).data.cpu()


def _run_concurrent_decoding(path, device):
    # Runs in a child process: a failure may kill the process, and a hang must
    # not block the test run.
    faulthandler.dump_traceback_later(STACK_DUMP_SECONDS)
    # Every decoder must output exactly what a single decoder outputs alone.
    expected = _decode_all_frames(path, device)
    pool = concurrent.futures.ThreadPoolExecutor(NUM_THREADS)
    for _ in range(NUM_PASSES):
        futures = [
            pool.submit(_decode_all_frames, path, device) for _ in range(NUM_TASKS)
        ]
        for future in futures:
            try:
                assert torch.equal(future.result(), expected)
            except BaseException:
                # Fail immediately instead of waiting for the remaining tasks.
                traceback.print_exc()
                os._exit(1)
    pool.shutdown()


@needs_ffmpeg_cli
@pytest.mark.slow
@pytest.mark.parametrize("device", all_supported_devices())
def test_concurrent_decoders_of_large_frames(device, clip_1080p):
    # Many decoders on separate threads, each decoding every frame of a 1080p
    # clip, which must match a decode by a single decoder. Wrong output, a
    # crash, an abort or a hang in the child process fails the test.
    process = multiprocessing.get_context("spawn").Process(
        target=_run_concurrent_decoding, args=(str(clip_1080p), device)
    )
    process.start()
    process.join(TIMEOUT_SECONDS)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail(f"decoding hung for {TIMEOUT_SECONDS}s")
    assert process.exitcode == 0, f"workload failed with exit code {process.exitcode}"
