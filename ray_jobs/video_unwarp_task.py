#!/usr/bin/env python3
"""
Ray tasks for 360 video processing.

This driver defines two Ray tasks:
  - erp_unwarp_task: equirectangular (2:1) -> multiple perspective views
  - fisheye_unwarp_task: single fisheye -> multiple perspective views (GeoCalib-assisted)

How to run (local Ray):
  pip install ray
  ray start --head  # optional; for cluster use address="auto"
  python ray_tasks.py  # runs a demo erp job

Make sure the module 'fisheye_360_with_geocalib.py' is in the same directory
(or on PYTHONPATH). We set runtime_env to ship the working dir to workers.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple
from pathlib import Path

import ray
import cv2
import subprocess

from video_process.video4_unwarpERP import insv_to_4viewsERP

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

DEFAULT_VIEWS4_ERP: Sequence[Tuple[str, float, float]] = (
    ("front", 0.0, 0.0),
    ("right", 90.0, 0.0),
    ("back", 180.0, 0.0),
    ("left", -90.0, 0.0),
)

V_FOV_DEG = 90.0


def _ensure_runtime_env() -> dict:
    """Package the current working directory so Ray workers can import modules here."""
    # package the repository root (one level above ray_jobs) so top-level
    # packages like `utils` are available to workers
    wd = str(Path(__file__).resolve().parents[1])
    return {"working_dir": wd}

def opencv_tune_for_worker(num_threads: int = 1, use_opencl: bool = False):
    """Call this at the top of every Ray remote task or in the module imported by workers."""
    cv2.setUseOptimized(True)
    cv2.setNumThreads(int(num_threads))  # keep small in multi-process setting
    try:
        if use_opencl and cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Ray tasks
# ---------------------------------------------------------------------------


@ray.remote(num_cpus=1, num_gpus=1, max_retries=1)
def erp_unwarp_task(mp4_path: str,
                    views: Optional[Sequence[Tuple[str, float, float]]] = None,
                    out_size: Tuple[int, int] = (1280, 720),
                    v_fov_deg: float = 95.0,
                    out_dir: Optional[str] = None,
                    roll_deg: float = 0.0) -> Dict[str, str]:
    """Extract perspective views from an ERP 2:1 video.

    Returns a dict {view_name: output_path}.
    """
    opencv_tune_for_worker(num_threads=1)
    try:
        from video_process.video_unwarp import unwarp_equirectangular_viewsP
        if views is None:
            views = DEFAULT_VIEWS4_ERP
        return unwarp_equirectangular_viewsP(
            mp4_path=mp4_path,
            views=views,
            out_size=out_size,
            v_fov_deg=v_fov_deg,
            out_dir=out_dir,
            roll_deg=roll_deg,
        )
    except Exception as e:
        return {"__error__": f"erp_unwarp_task failed: {e}"}



@ray.remote(num_gpus=1, max_retries=1)
def insv_unwarp_task(
    insv_path: str,
    out_dir: Optional[str] = None,
    erp_w: int = 5760,
    erp_h: int = 2880,
    lens_fov_deg: float = 180.0,
    out_size: Optional[Tuple[int, int]] = None, # erp_w/4
    v_fov_deg: float = 90.0,
    h_fov_deg: float = 90.0,
    roll_deg: float = 0.0,
    views: Optional[Sequence[Tuple[float, str]]] = None,
) -> Dict[str, str]:
    """Convert INSV -> ERP -> four perspective views using the local video4_unwarp implementation.

    This wrapper imports `video_process.video4_unwarp.video4_unwarp_task` and runs it inside the Ray worker.
    Returns the same dict produced by that function: {success: bool, erp: str, views: {...}} or an error dict.
    """
    opencv_tune_for_worker(num_threads=1)
    try:
        # import here so the worker can receive the working_dir via runtime_env
        from video_process.video4_unwarp import video4views_unwarpF
        #Based on openCV slow (tested)
        # return video4views_unwarp( 
        #     insv_path,
        #     output_dir=out_dir,
        #     erp_w=erp_w,
        #     erp_h=erp_h,
        #     lens_fov_deg=lens_fov_deg,
        #     out_size=out_size,
        #     v_fov_deg=v_fov_deg,
        #     roll_deg=roll_deg,
        #     views=views,
        # )

        # The new insv -> fisheye --> 4views pipeline (tested)
        # return video4views_unwarpF(
        #     insv_path,
        #     output_dir=out_dir,
        #     out_size=out_size,
        #     h_fov=h_fov_deg,
        #     v_fov=v_fov_deg,
        #     roll=roll_deg,
        # )
     
        # The new insv -> ERP --> 4views pipeline (under testing)
        from video_process.video4_unwarpERP import insv_to_4viewsERP_one_shot
        return insv_to_4viewsERP_one_shot(
            insv_path,
            output_dir=out_dir,
            out_size=out_size,  # Pass None to let the function auto-detect
            h_fov_deg=h_fov_deg,
            v_fov_deg=v_fov_deg,
            roll_deg=roll_deg,
        )
        
    except Exception as e:
        return {"__error__": f"insv_unwarp_task failed: {e}"}


@ray.remote(num_cpus=1, num_gpus=1, max_retries=1)
def fisheye_unwarp_task(mp4_path: str,
                        views: Optional[Sequence[Tuple[str, float]]] = None,
                        out_size: Tuple[int, int] = (1280, 720),
                        model: str = "equisolid",
                        fish_e2e_fov_deg: float = 190.0,
                        border: int = 10,
                        auto_from_geo: bool = True,
                        geo_model: str = "simple_divisional",
                        sample_ts: Sequence[float] = (0.0, 1.0, 2.0),
                        manual_pitch_deg: float = 0.0,
                        manual_roll_deg: float = 0.0,
                        h_fov_deg: Optional[float] = 120.0,
                        v_fov_deg: Optional[float] = None,
                        out_dir: Optional[str] = None) -> Dict[str, str]:
    """Unwarp a single-fisheye video into multiple perspective views using GeoCalib.

    Returns a dict {view_name: output_path}.
    """
    try:
        from video_process.video_unwarp import unwarp_fisheye_views
        return unwarp_fisheye_views(
            mp4_path=mp4_path,
            out_dir=out_dir,
            out_size=out_size,
            model=model,
            fish_e2e_fov_deg=fish_e2e_fov_deg,
            border=border,
            auto_from_geo=auto_from_geo,
            geo_model=geo_model,
            sample_ts=sample_ts,
            manual_pitch_deg=manual_pitch_deg,
            manual_roll_deg=manual_roll_deg,
            h_fov_deg=h_fov_deg,
            v_fov_deg=v_fov_deg,
            views=views,
        )
    except Exception as e:
        return {"__error__": f"fisheye_unwarp_task failed: {e}"}


# ---------------------------------------------------------------------------
# Demo driver
# ---------------------------------------------------------------------------

def run_demo_erp(mp4_path: str) -> Dict[str, str]:
    """Run a single ERP job on Ray and return outputs dict."""
    # Local init; for cluster pass address="auto"
    ray.init(ignore_reinit_error=True, runtime_env=_ensure_runtime_env())
    print("[RAY] Initialized.")

    # Submit task
    job = erp_unwarp_task.remote(mp4_path=mp4_path,
                                 views=DEFAULT_VIEWS4_ERP,
                                 out_size=(1280, 720),
                                 v_fov_deg=95.0,
                                 out_dir=None,
                                 roll_deg=0.0)
    result = ray.get(job)
    print("[RAY] ERP outputs:", result)
    return result


def run_batch_erp(mp4_paths: Sequence[str], out_size=(1280, 720), v_fov_deg=95.0) -> List[Dict[str, str]]:
    """Submit a batch of ERP videos in parallel and return a list of outputs dicts."""
    ray.init(ignore_reinit_error=True, runtime_env=_ensure_runtime_env())
    jobs = [erp_unwarp_task.remote(mp4_path=p, views=DEFAULT_VIEWS4_ERP,
                                   out_size=out_size, v_fov_deg=v_fov_deg) for p in mp4_paths]
    return ray.get(jobs)


if __name__ == "__main__":
    # Replace with your ERP 2:1 video
    test_mp4 = "/home/nvcoe_admin/code/oslo/whole_pipeline_testing/kamwai_chan_input_videos/vaccum_floor_1GB.mp4"
    run_demo_erp(test_mp4)
