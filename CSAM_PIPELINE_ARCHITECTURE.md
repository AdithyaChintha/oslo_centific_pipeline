# CSAM Ray Pipeline Architecture

## 1. Overview

The CSAM pipeline is an Azure Blob-driven, session-oriented media-processing system. It discovers recording sessions in Azure, downloads their video and audio chunks, converts INSV recordings into multiple views, creates time-aligned shards, runs AI models through Ray, consolidates results, uploads artifacts, and creates Label Studio tasks.

The current implementation uses parallelism for chunk downloads, video-view sharding, and model inference. Sessions, shard indices, and views within a shard are processed sequentially in the main multi-chunk path.

```text
Azure Blob Storage
        |
        v
Session discovery and metadata parsing
        |
        v
Sequential session scheduler
        |
        v
Parallel chunk download with Ray
        |
        v
Video preparation / INSV unwarping
        |
        v
Parallel sharding across video views + audio sharding
        |
        v
Audio/video shard alignment
        |
        v
Sequential shard loop
        |
        v
Sequential view loop within each shard
        |
        v
Parallel NSFW and face-detection model tasks
        |
        v
Per-shard multi-view consolidation
        |
        v
Session-level aggregation
        |
        v
Azure upload, Label Studio import, tracking, and cleanup
```

## 2. Operating Modes

The CLI supports several execution modes:

- `multi_sessions`: discovers multiple Azure session folders and processes them sequentially.
- `blob_polling`: continuously checks Azure Blob Storage for new sessions.
- `integrated`: runs continuous polling and processing inside a Ray task.
- `standalone`: processes explicitly supplied video and audio files.
- Direct multi-chunk mode: downloads ready sessions and invokes `pipeline_main_multichunks()`.

The main multi-chunk entry point is `pipeline_main_multichunks()` in `ray_pipeline_testing_csam.py`.

## 3. Azure Session Discovery

`discover_multiple_sessions()` lists blobs beneath the configured input prefix. The first directory after that prefix is treated as the session directory and session ID.

```python
relative_path = blob.name[len(base_prefix):]
session_dir = relative_path.split('/')[0]

session_info = {
    "session_path": session_path,
    "session_id": session_dir,
    "is_walkthrough": metadata_info["is_walkthrough"],
    "metadata": metadata_info["metadata"],
    "domain": metadata_info["domain"],
    "activity": metadata_info["activity"],
    "specific_activity": metadata_info["specific_activity"]
}
```

For example:

```text
input-prefix/session-123/video_01.insv
```

produces:

```text
session_id = session-123
```

`read_session_metadata()` reads metadata JSON and determines whether the session is a walkthrough.

The repository also contains a legacy filename parser in `utils/multi_chunk_blob_utils.py`. It constructs an ID from values already embedded in the Azure filename:

```python
session_id = f"{session_index}-{uuid}-{label.lower()}"
```

## 4. Session Scheduling

`pipeline_main_multichunks_sequential_sessions()` processes complete sessions one at a time:

```text
Session 1: download -> shard -> infer -> consolidate -> upload
Session 2: download -> shard -> infer -> consolidate -> upload
Session 3: download -> shard -> infer -> consolidate -> upload
```

Before processing, the pipeline checks whether the session was already completed. Completed sessions are skipped. If a session raises an exception, it is marked failed and the scheduler proceeds to the next session.

## 5. Parallel Azure Chunk Download

`process_single_session()` calls `download_ready_sessions()` with parallel chunk downloading enabled:

```python
download_ready_sessions(
    blob_service_client=create_azure_blob_client(azure_config),
    container_name=azure_config.get('container'),
    input_prefix=session_path,
    local_download_dir=pipeline_config['local_storage']['temp_download_dir'],
    max_sessions=1,
    use_parallel_chunks=True
)
```

Each media chunk is submitted as a Ray task through `download_single_chunk_ray.remote()`. The caller waits for the group using `ray.get()`.

```text
Azure session
  |-- video chunk 1 ---- Ray download task
  |-- video chunk 2 ---- Ray download task
  |-- audio chunk 1 ---- Ray download task
  `-- metadata JSON ---- Ray download task
                              |
                              v
                         ray.get(all)
```

## 6. Video Preparation and INSV Unwarping

Downloaded video inputs are separated from audio inputs. An MP4 can be processed directly. An INSV recording is sent to `insv_unwarp_task.remote()` to generate multiple flat views.

```python
mp4_result = ray.get(
    insv_unwarp_task.remote(
        local_video_path,
        out_dir=os.path.join(output_dir, "4views")
    )
)
```

Although unwarping runs as a Ray task, the immediate `ray.get()` makes the current caller wait for that video before continuing.

## 7. Parallel Video Sharding Across Views

After unwarping, each view is submitted for video sharding before the results are collected:

```python
ref = split_video_into_shards.remote(...)
split_tasks.append((view_name, ref, output_dir))

results = ray.get([ref for _, ref, _ in split_tasks])
```

Walkthrough sessions use `split_video_into_shards_with_overlap.remote()`. Normal sessions use `split_video_into_shards.remote()`.

```text
Unwarped video
  |-- top view ----- split task -> top shards
  |-- left view ---- split task -> left shards
  |-- right view --- split task -> right shards
  `-- bottom view -- split task -> bottom shards
                              |
                              v
                         ray.get(all)
```

This is parallelism across views during sharding. It is not parallel processing of the resulting shard indices.

### Walkthrough sharding

Walkthrough recordings use overlapping windows. A representative layout is:

```text
Shard 1:   0-180 seconds
Shard 2: 120-300 seconds
Shard 3: 240-420 seconds
```

### Normal sharding

Normal recordings use non-overlapping windows:

```text
Shard 1:   0-180 seconds
Shard 2: 180-360 seconds
Shard 3: 360-540 seconds
```

Durations and overlaps are configuration-driven.

## 8. Audio Sharding

Audio is sharded separately using either `split_audio_into_shards.remote()` or the overlapping walkthrough equivalent.

Each audio chunk is submitted and immediately awaited. Consequently, separate audio chunks are effectively handled sequentially:

```text
Audio chunk 1 -> Ray split -> wait
Audio chunk 2 -> Ray split -> wait
Audio chunk 3 -> Ray split -> wait
```

## 9. Audio/Video Time Alignment

After sharding, the pipeline compares the audio shard count with every video-view shard count.

If all counts match, all shards are processed. If they differ, the pipeline uses the minimum common count:

```text
min(audio count, top count, left count, right count, bottom count)
```

This prevents indexing errors but ignores any unmatched trailing shards.

## 10. Current Shard-Level Processing

Shard indices are currently processed sequentially:

```python
for shard_idx in range(min_shards):
    shard_results = process_time_aligned_shard_multiview(...)
```

`process_time_aligned_shard_multiview()` is a normal function call, not a Ray remote call. Therefore, the next shard does not begin until the current shard completes.

```text
Shard 1 -> all views, models, and consolidation complete
    |
    v
Shard 2 -> all views, models, and consolidation complete
    |
    v
Shard 3 -> ...
```

## 11. Processing Views Within a Shard

One aligned shard contains the same time interval from all available camera views plus its corresponding audio shard.

```python
view_shard_paths = {
    "top": top_shard_n,
    "left": left_shard_n,
    "right": right_shard_n,
    "bottom": bottom_shard_n
}
```

Inside `process_time_aligned_shard_multiview()`, views are also processed sequentially:

```python
for view_name, view_shard_path in view_shard_paths.items():
    view_result = process_single_shard_through_pipeline(...)
```

A view failure is caught and stored as an error result. Remaining views are still attempted. If no view produces a result, the function raises `RuntimeError`.

## 12. Parallel Model Inference

`process_single_shard_through_pipeline()` launches the currently active model tasks for one view shard:

```python
nsfw_ref = process_video_chunks_for_nsfw.remote(...)
face_ref = process_video_chunks_for_face_detection.remote(...)

nsfw_res, face_res = ray.get([nsfw_ref, face_ref])
```

The active model parallelism is:

```text
One view shard
  |-- NSFW detection ---- Ray worker
  `-- Face detection ---- Ray worker
                    |
                    v
               ray.get(both)
```

Additional model stages exist in the source but are currently commented out, including:

- Audio diarization
- Scene detection
- Motion-energy analysis
- Clap detection
- Signal-quality analysis
- Lighting analysis
- Sensitive-information detection
- YOLO detection

## 13. Per-Shard Consolidation

After all available views are processed, the pipeline:

1. Generates a multi-view consolidated model-results JSON.
2. Combines flagged time segments across views.
3. Merges overlapping detections.
4. Assigns views to Label Studio display positions.
5. Generates one multi-view Label Studio task for the shard.
6. Merges model results into that task.

Important functions include:

- `generate_multiview_consolidated_model_results_json()`
- `consolidate_multiview_time_segment_results()`
- `assign_views_to_labelstudio_positions()`
- `generate_multiview_4view_labelstudio_task()`
- `merge_model_results_into_labelstudio_task()`

This produces one annotation unit per aligned time interval instead of one task per view.

## 14. Session-Level Aggregation

The pipeline accumulates:

```python
label_studio_tasks = []
consolidated_json_paths = []
```

After all shard indices complete, `generate_final_combined_model_results_json()` combines the per-shard model results into a session-level result.

Video-level domain-classification components exist in the repository, although parts of that flow are disabled in the active multi-chunk CSAM path.

## 15. Azure Upload and Label Studio Integration

The session output directory is uploaded through `upload_output_directory_with_sas_optimized()`.

Typical output artifacts include:

```text
session output/
  |-- original video and audio files
  |-- unwarped video views
  |-- video shards
  |-- audio shards
  |-- per-view model results
  |-- per-shard consolidated results
  |-- Label Studio task JSON files
  |-- final combined results
  `-- timing CSV
```

After upload, local media paths in Label Studio tasks are replaced with Azure URLs through `update_labelstudio_tasks_with_new_urls()`.

The downstream flow then:

1. Integrates ERP video/audio information.
2. Updates Label Studio tasks.
3. Imports consolidated tasks into Label Studio.

## 16. Tracking, Retry, Timing, and Cleanup

The pipeline tracks:

- Session completion status
- Processed sessions
- Chunk download progress
- Per-stage chunk status
- Model-processing status
- Timing information
- Failure summaries
- Cleanup results

Representative tracking stages are:

```text
download
view_sharding.<view>
view_sharding.audio
model_processing.nsfw_detection
model_processing.face_detection
```

Timing identifiers follow a hierarchy similar to:

```text
video.<session_id>.<video_id>.shard.<index>.<view>.<model>
```

After a successful upload, configuration flags determine whether downloaded temporary files and local output files are deleted.

## 17. Failure Behavior

### Session failure

A session exception is caught by the sequential session scheduler. That session is marked failed, and the next session is processed.

### Model or view failure

A view-level exception is captured in `process_time_aligned_shard_multiview()`. Other views are still attempted. If all views fail, the shard raises an exception.

### Parallel video-sharding failure

The current sharding handler logs the Ray failure but does not assign a fallback to `results`:

```python
try:
    results = ray.get([ref for _, ref, _ in split_tasks])
except Exception as e:
    logger.error(f"Parallel split_video_into_shards failed: {e}")
```

The next access to `results` can raise `UnboundLocalError`. This propagates out of the current session, which is then marked failed by the session scheduler. Other sessions can continue.

## 18. Current Concurrency Summary

| Processing level | Current behavior |
|---|---|
| Azure polling | Continuous in polling and integrated modes |
| Sessions | Sequential in multi-session mode |
| Chunk downloads | Parallel with Ray |
| INSV unwarping | Ray task, immediately awaited per video |
| Video-view sharding | Parallel with Ray |
| Separate audio chunks | Sequential Ray calls |
| Different shard indices | Sequential |
| Views within a shard | Sequential |
| Models within one view | Parallel: NSFW and face detection |
| Output upload | Optimized uploader; may use internal concurrency |

The effective execution pattern is:

```text
Session sequential
  `-- Video chunk sequential
       `-- View splitting parallel
            `-- Shard index sequential
                 `-- View processing sequential
                      `-- NSFW and face models parallel
```

## 19. Proposed True Shard-Level Parallel Architecture

For different shard indices to run concurrently, shard processing can be wrapped in a Ray task:

```python
@ray.remote
def process_shard_task(**kwargs):
    return process_time_aligned_shard_multiview(**kwargs)
```

The session pipeline could then submit aligned shards before waiting:

```python
shard_refs = []

for shard_idx in range(min_shards):
    shard_refs.append(
        process_shard_task.remote(
            shard_index=shard_idx,
            view_shard_paths=view_paths_for_shard,
            audio_shard_path=audio_shards[shard_idx],
            base_output_dir=output_dir,
            total_shard_count=min_shards
        )
    )

shard_results = ray.get(shard_refs)
```

The target flow would be:

```text
Session
  |-- Shard 1 -> views -> models -> consolidation
  |-- Shard 2 -> views -> models -> consolidation
  |-- Shard 3 -> views -> models -> consolidation
  `-- Shard N -> views -> models -> consolidation
                         |
                         v
                   session aggregation
```

Concurrency must be bounded. The approximate number of simultaneous model tasks is:

```text
concurrent shards x views per shard x active models per view
```

For four concurrent shards, four views, and two active models, as many as 32 model tasks could be submitted. Ray CPU/GPU resource annotations, a maximum in-flight shard limit, and per-model GPU-memory constraints should therefore be added before enabling full shard-level parallelism.

## 20. Final Architectural Assessment

The repository already uses Ray effectively for I/O fan-out, view splitting, and independent model execution. However, the active CSAM multi-chunk pipeline is not fully parallel at shard level.

Its current architecture is best summarized as:

> Parallel model execution inside a sequential view loop, inside a sequential shard loop, inside a sequential session scheduler.

True shard-level concurrency requires remote shard workers, bounded submission, explicit CPU/GPU resource allocation, isolated output directories, deterministic result ordering, and clearer partial-failure handling.
