"""
complete_post_consolidation_pipeline_testing.py

Pytest tests for consolidate_json (consolidation module). This version
handles the case where ConsolidationPipeline is a Ray actor (ActorClass)
or a plain class. It derives paths relative to this test file.
"""

import os
import json
from copy import deepcopy
import importlib.util
import pytest
import sys
import types

import os

# 1) Tiny fake 'ray' module so @ray.remote doesn't wrap classes into actors during import.
#    This makes consolidate_json define a normal class rather than a Ray ActorClass.
if "ray" not in sys.modules:
    fake_ray = types.ModuleType("ray")

    def _remote(obj=None, **decorator_kwargs):
        """
        Support both usages:
          @ray.remote
          @ray.remote(**kwargs)
        Return the class/function unchanged (identity).
        """
        if obj is None:
            # used as @ray.remote(...)
            def _deco(x):
                return x
            return _deco
        return obj

    fake_ray.remote = _remote
    # minimal hooks used by tests if code calls ray.init() etc.
    fake_ray.init = lambda *a, **k: None
    fake_ray.is_initialized = lambda : False
    fake_ray.get = lambda *a, **k: None
    # Provide a minimal exceptions namespace to avoid attribute errors
    fake_ray.exceptions = types.SimpleNamespace(ActorDiedError=Exception, RayTaskError=Exception)
    sys.modules["ray"] = fake_ray

# 2) If consolidate_json imports stitch_mux (it does), insert tiny fake before import
if "stitch_mux" not in sys.modules:
    _fake_stitch_mux = types.ModuleType("stitch_mux")
    class RenderSettings:
        def __init__(self, *a, **k):
            pass
    def build_and_run_ffmpeg_for_group(*a, **k):
        class DummyRemote:
            @staticmethod
            def remote(*a, **k):
                return "dummy-ray-objref"
        return DummyRemote()
    _fake_stitch_mux.RenderSettings = RenderSettings
    _fake_stitch_mux.build_and_run_ffmpeg_for_group = build_and_run_ffmpeg_for_group
    sys.modules["stitch_mux"] = _fake_stitch_mux



# --- Begin: improved fake azure.storage.blob shim (must run BEFORE loading consolidate_json.py) ---
import sys
import types

if "azure" not in sys.modules:
    azure_mod = types.ModuleType("azure")
    sys.modules["azure"] = azure_mod

if "azure.storage" not in sys.modules:
    storage_mod = types.ModuleType("azure.storage")
    sys.modules["azure.storage"] = storage_mod
    setattr(sys.modules["azure"], "storage", storage_mod)

if "azure.storage.blob" not in sys.modules:
    blob_mod = types.ModuleType("azure.storage.blob")

    class ContentSettings:
        def __init__(self, content_type=None, **kwargs):
            self.content_type = content_type
            for k, v in kwargs.items():
                setattr(self, k, v)
        def __repr__(self):
            return f"ContentSettings(content_type={self.content_type!r})"

    class _FakeBlobDownload:
        def __init__(self, data):
            self._data = data
        def readall(self):
            return self._data

    class _FakeContainerClient:
        def __init__(self, name=None):
            self.name = name
            # in-memory dict: blob_name -> bytes
            self._store = {}

        # container-level exists: return True (container present)
        def exists(self):
            return True

        def upload_blob(self, name, data, overwrite=False, content_settings=None):
            if hasattr(data, "read"):
                val = data.read()
            else:
                val = data
            if isinstance(val, str):
                val = val.encode("utf-8")
            self._store[name] = val
            return types.SimpleNamespace(name=name)

        def download_blob(self, name):
            val = self._store.get(name)
            if val is None:
                raise FileNotFoundError(f"blob {name} not found in container {self.name}")
            return _FakeBlobDownload(val)

        def list_blobs(self, name_starts_with=None):
            for k in list(self._store.keys()):
                if not name_starts_with or k.startswith(name_starts_with):
                    yield types.SimpleNamespace(name=k)

        def get_blob_client(self, blob_name):
            # Return a blob-level client object that wraps this container and name
            return _FakeBlobClient(self, blob_name)

        # convenience: support download_blob_to_bytes(name) if used
        def download_blob_to_bytes(self, name):
            val = self._store.get(name)
            if val is None:
                raise FileNotFoundError(f"blob {name} not found")
            return val

    class _FakeBlobClient:
        """Represents a blob client for one blob (backed by container._store)."""
        def __init__(self, container: _FakeContainerClient, blob_name: str):
            self._container = container
            self.blob_name = blob_name

        def exists(self):
            return self.blob_name in self._container._store

        def upload_blob(self, data, overwrite=False, content_settings=None):
            return self._container.upload_blob(self.blob_name, data, overwrite=overwrite, content_settings=content_settings)

        def download_blob(self):
            val = self._container._store.get(self.blob_name)
            if val is None:
                raise FileNotFoundError(f"blob {self.blob_name} not found")
            return _FakeBlobDownload(val)

        # alias for container.get_blob_client(...).download_blob().readall() patterns
        def readall(self):
            return self.download_blob().readall()

    class _FakeBlobServiceClient:
        @classmethod
        def from_connection_string(cls, conn_str):
            # ignore validation, return instance
            return cls()

        def __init__(self):
            self._containers = {}

        def get_container_client(self, container_name):
            if container_name not in self._containers:
                self._containers[container_name] = _FakeContainerClient(container_name)
            return self._containers[container_name]

        def get_blob_client(self, container_name, blob_name=None):
            c = self.get_container_client(container_name)
            if blob_name is None:
                return c
            return c.get_blob_client(blob_name)

    blob_mod.BlobServiceClient = _FakeBlobServiceClient
    blob_mod.ContentSettings = ContentSettings

    sys.modules["azure.storage.blob"] = blob_mod
# --- End fake azure shim ---




# ----------------------------
# Determine project root from this test file's location (so paths are relative)
# ----------------------------
THIS_TEST_DIR = os.path.dirname(os.path.abspath(__file__))            # .../insta360-video-activity-segmentation/tests
PROJECT_ROOT = os.path.dirname(THIS_TEST_DIR)                         # .../insta360-video-activity-segmentation

# Build important relative paths (project-root relative)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TESTING_JSON_PATHS = [
    os.path.join(DATA_DIR, "testing_json1.json"),
    os.path.join(DATA_DIR, "testing_json2.json"),
]
OUTPUT_DIR = os.path.join(DATA_DIR, "testing_output")
GOLDEN_DIR = os.path.join(DATA_DIR, "golden_json")

# Path to consolidate_json.py (inside project)
_CONSOLIDATE_PY_PATH = os.path.join(PROJECT_ROOT, "video-audio-stiching", "consolidate_json.py")
if not os.path.exists(_CONSOLIDATE_PY_PATH):
    raise FileNotFoundError(f"Expected consolidate_json.py at relative path: {_CONSOLIDATE_PY_PATH}")



# Load consolidate_json.py module from file path
spec = importlib.util.spec_from_file_location("consolidate_json", _CONSOLIDATE_PY_PATH)
consolidate_json = importlib.util.module_from_spec(spec)
spec.loader.exec_module(consolidate_json)

# Expose the needed names from the loaded module
compute_shard_offset = consolidate_json.compute_shard_offset
_format_time = consolidate_json._format_time
# The ConsolidationPipeline symbol may be a Ray ActorClass or a normal class.
RawConsolidationPipeline = consolidate_json.ConsolidationPipeline




# ----------------------------
# Helper to create a pipeline object that is usable synchronously in tests.
# It supports two cases:
#  - RawConsolidationPipeline is a normal class -> instantiate normally.
#  - RawConsolidationPipeline is a Ray ActorClass -> start ray and create actor,
#    return a SyncWrapper that forwards method calls via ray.get(...)
# ----------------------------


def create_pipeline(*args, **kwargs):
    """
    Create a synchronous ConsolidationPipeline instance for tests.

    Strategy:
      1. Try to instantiate RawConsolidationPipeline directly (normal class).
      2. If that fails with the Ray actor TypeError, attempt to recover the
         original underlying class from common wrapper attributes:
         - __wrapped__, _actor_cls, _actor_class, actor_class
      3. If underlying class found, instantiate it directly (preferred).
      4. Only if we cannot recover the underlying class, fall back to starting Ray
         and creating a remote actor and wrapping it (less desirable).
    """
    # 1) Try direct instantiation
    try:
        inst = RawConsolidationPipeline(*args, **kwargs)
        return inst
    except Exception as e:
        # If it's not the Ray-actor TypeError, re-raise
        msg = str(e)
        if "Actors cannot be instantiated directly" not in msg and "use '.remote()'" not in msg:
            # Not the actor-type error. Re-raise to surface unexpected errors.
            raise

    # 2) Attempt to recover underlying class from wrapper attributes
    candidate_attrs = ("__wrapped__", "_actor_cls", "_actor_class", "actor_class", "__wrapped_class__")
    underlying = None
    for attr in candidate_attrs:
        underlying = getattr(RawConsolidationPipeline, attr, None)
        if underlying:
            # If underlying is a function wrapper (callable), it might itself be an ActorClass; try to unwrap recursively
            # If underlying is an ActorClass wrapper, try its __wrapped__ too
            # If underlying is an object with .__wrapped__, keep unwrapping a few times
            tries = 0
            u = underlying
            while tries < 5 and hasattr(u, "__wrapped__") and getattr(u, "__wrapped__", None) is not None:
                u = getattr(u, "__wrapped__")
                tries += 1
            underlying = u
            break

    if underlying and callable(underlying):
        # 3) instantiate underlying directly
        try:
            return underlying(*args, **kwargs)
        except Exception as e:
            # If instantiation failed for other reasons, surface the error
            raise RuntimeError(f"Found underlying class {underlying}, but failed to instantiate it: {e}") from e

    # 4) Last resort: start Ray and create remote actor (only if absolutely necessary)
    import ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)
    try:
        actor = RawConsolidationPipeline.remote(*args, **kwargs)
    except Exception as e:
        raise RuntimeError(
            "Failed to create a Ray actor; tests cannot proceed. "
            "Tried direct instantiation, unwrapping, and remote actor creation. "
            f"Last error: {e}"
        ) from e

    class _SyncWrapper:
        def __init__(self, actor_handle):
            self._actor = actor_handle
        def __getattr__(self, name):
            def method(*a, **k):
                return ray.get(getattr(self._actor, name).remote(*a, **k))
            return method

    return _SyncWrapper(actor)



# ----------------------------
# no_blob_save fixture: replace save_master with harmless stub (works for actor or class)
# ----------------------------
@pytest.fixture
def no_blob_save(monkeypatch):
    """
    Ensure tests do not attempt to contact real Azure but allow the pipeline
    to persist masters into the in-memory fake azure shim so subsequent shards
    can be consolidated correctly.
    """
    
    yield


# ----------------------------
# Helpers for fixtures and I/O
# ----------------------------
def _ensure_test_inputs_exist():
    missing = [p for p in TESTING_JSON_PATHS if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required testing JSON(s). Expected these relative paths (project root):\n"
            + "\n".join(TESTING_JSON_PATHS)
            + "\nMissing: " + ", ".join(missing)
        )


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ----------------------------
# Basic unit tests for pure functions
# ----------------------------
def test_compute_shard_offset_basic():
    assert compute_shard_offset(1) == 0
    assert compute_shard_offset(2) >= 0
    with pytest.raises(ValueError):
        compute_shard_offset(0)


def test_format_time_basic():
    assert _format_time(0) == "00:00:00.000"
    assert _format_time(1.234) == "00:00:01.234"
    assert _format_time(None) is None
    assert _format_time(-5) == "00:00:00.000"


# ----------------------------
# Test globalization helpers if input fixtures contain relevant keys
# ----------------------------
def test_globalise_predictions_and_annotations_if_present():
    _ensure_test_inputs_exist()
    preds_obj = None
    anns_obj = None
    for p in TESTING_JSON_PATHS:
        obj = _load_json(p)
        if preds_obj is None and isinstance(obj, dict) and "predictions" in obj:
            preds_obj = obj["predictions"]
        if anns_obj is None and isinstance(obj, dict) and "annotations" in obj:
            anns_obj = obj["annotations"]

    pipeline = create_pipeline(connection_string="unused", masters_container="unused", final_container="unused", do_stitching=False)

    if preds_obj is not None:
        preds_copy = deepcopy(preds_obj)
        counters = pipeline.globalise_predictions(preds_copy, offset_seconds=5.0)
        assert isinstance(counters, dict)
        # If numeric time fields exist, they should have been shifted
        numeric_found = False

        def scan(obj):
            nonlocal numeric_found
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in ("start", "end", "start_time", "end_time", "timestamp_seconds", "clap_timestamp"):
                        if isinstance(v, (int, float)):
                            numeric_found = True
                            assert v >= 5.0
                    else:
                        scan(v)
            elif isinstance(obj, list):
                for e in obj:
                    scan(e)

        scan(preds_copy)
        if numeric_found:
            assert any(c.get("adjusted", 0) >= 0 for c in counters.values())

    if anns_obj is not None:
        out = pipeline._globalise_annotations(anns_obj, offset_seconds=65.0)
        assert isinstance(out, list)
        taxonomy_found = False
        numeric_shifted = False
        for rec in out:
            if not isinstance(rec, dict):
                continue
            for ann in rec.get("result", []):
                val = ann.get("value") if isinstance(ann.get("value"), dict) else ann
                if isinstance(val, dict) and "taxonomy" in val:
                    taxonomy_found = True
                    t = val["taxonomy"]
                    if isinstance(t, list) and t and isinstance(t[0], list):
                        s = t[0][0]
                        assert isinstance(s, str) and len(s) == 2 and s.isdigit()
                for k in ("start", "end"):
                    if k in val and isinstance(val[k], (int, float)):
                        numeric_shifted = True
                        assert val[k] >= 65.0


# ----------------------------
# validate_shard_json
# ----------------------------
def test_validate_shard_json_positive_and_negative():
    _ensure_test_inputs_exist()
    shard = None
    for p in TESTING_JSON_PATHS:
        obj = _load_json(p)
        if isinstance(obj, dict):
            data = obj.get("data") or {}
            if "video_name" in data and "shard_number" in data:
                shard = obj
                break
    if shard is None:
        pytest.skip("No shard-like JSON found in provided testing JSONs for validate_shard_json test")

    pipeline = create_pipeline(connection_string="unused", masters_container="unused", final_container="unused", do_stitching=False)
    assert pipeline.validate_shard_json(shard) is True

    bad = deepcopy(shard)
    if "data" in bad and "video_name" in bad["data"]:
        del bad["data"]["video_name"]
    assert pipeline.validate_shard_json(bad) is False


# ----------------------------
# Main integration test: consolidate provided shard JSONs and produce final JSON(s)
# ----------------------------
def test_consolidate_shards_and_produce_final_using_returned_master(no_blob_save):
    _ensure_test_inputs_exist()

    # Load the provided testing JSONs
    objs = [_load_json(p) for p in TESTING_JSON_PATHS]

    # Filter shard-like objects
    shard_objects = [o for o in objs if isinstance(o, dict) and isinstance(o.get("data"), dict) and "video_name" in o["data"] and "shard_number" in o["data"]]
    assert shard_objects, "Provided testing JSONs do not appear to be shard JSONs with data.video_name and data.shard_number."

    # Group by video_name
    groups = {}
    for s in shard_objects:
        vid = s["data"]["video_name"]
        groups.setdefault(vid, []).append(s)

    if not groups:
        pytest.skip("No shard groups found in testing JSONs")

    # Ensure output dir exists (project-root relative)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for video_id, shards in groups.items():
        # Sort shards by numeric shard_number
        def _sn(s):
            try:
                return int(s["data"]["shard_number"])
            except Exception:
                return 0

        sorted_shards = sorted(shards, key=_sn)

        pipeline = create_pipeline(connection_string="unused", masters_container="unused", final_container="unused", do_stitching=False)

        last_master = None
        for s in sorted_shards:
            master_after = pipeline.consolidate_shard(s, default_duration=60, overwrite_existing=True)
            assert isinstance(master_after, dict)
            last_master = master_after

        assert last_master is not None, "consolidation returned no master object"

        # Verify consolidated_shards contains expected shard indices
        expected_nums = sorted({int(s["data"]["shard_number"]) for s in sorted_shards})
        got_nums = sorted(last_master.get("consolidated_shards", []))
        assert got_nums == expected_nums, f"Consolidated shards {got_nums} != expected {expected_nums}"

        # Prepare final JSON and write to OUTPUT_DIR
        final_json = pipeline.prepare_final_json(last_master)
        out_file = os.path.join(OUTPUT_DIR, f"{video_id}_final_consolidated.json")
        with open(out_file, "w", encoding="utf-8") as fh:
            json.dump(final_json, fh, indent=2)
        assert os.path.exists(out_file)

        # If golden exists, compare
        golden_file = os.path.join(GOLDEN_DIR, f"{video_id}_golden_final.json")
        if os.path.exists(golden_file):
            with open(golden_file, "r", encoding="utf-8") as gf:
                golden = json.load(gf)
            assert final_json == golden, f"Produced final JSON differs from golden for {video_id}"
        else:
            shard_keys = list(final_json.get("shards", {}).keys())
            got_nums2 = sorted([int(k.replace("shard_", "")) for k in shard_keys])
            assert got_nums2 == expected_nums, f"Final JSON shards {got_nums2} != expected {expected_nums}"

    # print output dir for convenience
    print(f"[test] Final consolidated JSON(s) written to: {os.path.abspath(OUTPUT_DIR)}")


# ----------------------------
# Optional smoke tests for clap extraction & annotation summarization if fixtures include them
# ----------------------------
def test_extract_claps_and_summarize_if_fixtures_present():
    master_obj = None
    anns_obj = None
    for p in TESTING_JSON_PATHS:
        obj = _load_json(p)
        if master_obj is None and isinstance(obj, dict) and obj.get("shards"):
            master_obj = obj
        if anns_obj is None and isinstance(obj, dict) and obj.get("annotations"):
            anns_obj = obj["annotations"]
        if master_obj and anns_obj:
            break

    pipeline = create_pipeline(connection_string="unused", masters_container="unused", final_container="unused", do_stitching=False)

    if master_obj:
        result = pipeline.extract_claps_from_master(master_obj)
        assert isinstance(result, dict)
        assert "first_clap" in result and "trim_suggestions" in result

    if anns_obj:
        summary = pipeline.summarize_annotations(anns_obj)
        assert isinstance(summary, dict)
        assert "run_time" in summary and "domain_detected" in summary
