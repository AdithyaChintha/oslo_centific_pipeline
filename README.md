# insta360-video-activity-segmentation
Pre-annotation Video segmentation pipeline for Insta360 video



# Ray pipeline

insta360-video-activity-segmentation/
├── config/
│   └── azure_blob_config.yaml
├── ray_jobs/
│   ├── upload_manager.py
|   ├── insv_to_mp4.py
│   ├── download_manager.py
│   ├── video_splitter.py
│   ├── scene_change.py
│   ├── vad_audio.py
│   ├── motion_energy.py
│   ├── embeddings_driver.py
│   ├── yolo_sort_tracker.py
│   ├── fusion_gap_merge.py
│   ├── quality_flagger.py
│   ├── segment_classifier.py
│   └── ray_driver.py
├── utils/
│   ├── blob_utils.py
│   └── logger.py
├── ray_pipeline.py
└── requirements.txt
