def blobs_to_send(self) -> List[Tuple[str, dict, str]]:
    all_blobs = self.source_client.list_blobs_with_props()
    if self.source_prefix:
        all_blobs = [b for b in all_blobs if b["name"].startswith(self.source_prefix)]
    
    def is_target(blob_path: str) -> bool:
        parts = blob_path.split("/")
        if len(parts) < 2:
            return False
        filename = parts[-1]
        parent_folders = parts[:-1]
        has_home = any(p.startswith(self.home_id) for p in parent_folders)
        is_media = filename.lower().endswith((".insv", ".wav"))
        return has_home and is_media
    if self.transfer_by=="home_id":
        target_blobs = [b for b in all_blobs if is_target(b["name"])]
    elif self.transfer_by=="excel":
        # Load allowed media blob names from Excel
        allowed_blobs = load_media_blob_names_from_excel(self.excel_path)
        
        target_blobs = [b for b in all_blobs if any(folder in b["name"] for folder in allowed_blobs)]

        target_blob_names = set(b["name"] for b in target_blobs)
        
        missing_blobs = allowed_blobs - target_blob_names
        if missing_blobs:
            logger.warning("The following media blobs listed in Excel were not found in source container:")
            for mb in missing_blobs:
                logger.warning("  %s", mb)
        
    # Apply safety window (keep existing logic)
    safe_blobs = []
    for b in target_blob_names:
        blob = next((blob for blob in all_blobs if blob["name"] == b), None)
        if not blob:
            logger.warning("Blob %s not found in all_blobs", b)
            continue
        if self.is_blob_older_than_safety_window(blob.get("last_modified")):
            safe_blobs.append(blob)
        else:
            logger.info("SKIP (safety window) %s", blob["name"])


                metadata_validated_blobs = []
    for b in safe_blobs:
        if self.has_metadata_file(b["name"]):
            metadata_validated_blobs.append(b)
            logger.info("METADATA FOUND for %s", b["name"])
        else:
            logger.warning("SKIP (no metadata) %s", b["name"])

        
    # Convert to format expected by state store
    candidates = []
    for b in metadata_validated_blobs:
        fingerprint = VideoFingerprintManager.create_fingerprint(b)
        candidates.append((b["name"], fingerprint, b.get("last_modified")))
    
    # Use state store to filter out already processed videos
    if not self.state_store:
        videos_to_process = candidates
    else:
        videos_to_process = self.state_store.get_videos_to_process(candidates)
    # print("all_blobs:", [b["name"] for b in all_blobs])
    # print("Allowed blobs from Excel:", allowed_blobs)
    # print("filtered target_blobs:", [b["name"] for b in target_blobs])
    # print("target_blob_names:", target_blob_names)
    # print("safe_blobs:", safe_blobs)
    # print("metadata_validated_blobs:", metadata_validated_blobs)
    # print("candidates:", candidates)
    # print("videos_to_process:", videos_to_process)

    logger.info("Selected %d videos for processing (out of %d total)",
            len(videos_to_process), len(target_blobs))
    return videos_to_process