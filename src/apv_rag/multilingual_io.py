"""Bounded file-save retries for the multilingual Windows runner only."""
import time

from apv_rag.splits import write_json_atomic as original_write

DELAYS=(0.1,0.2,0.4,0.8,1.6,2.0,2.0,2.0,2.0)


def write_json_atomic(path,payload):
    for attempt in range(len(DELAYS)+1):
        try:
            original_write(path,payload)
            return
        except PermissionError as error:
            if attempt==len(DELAYS):
                raise PermissionError(
                    f'Windows still denies saving {path}. Close file previews or another running experiment, '
                    'then rerun the same command. The previous checkpoint was not deleted.') from error
            if attempt==0:print(f'File save denied; retrying: {path.name}',flush=True)
            time.sleep(DELAYS[attempt])


def infer_english(prepared,cache,path,*,root):
    # Keep the earlier producer script unchanged on disk and restore its writer after this call.
    import run_fresh_pipeline
    previous=run_fresh_pipeline.write_json_atomic
    try:
        run_fresh_pipeline.write_json_atomic=write_json_atomic
        return run_fresh_pipeline.infer_missing(prepared,cache,path,root=root)
    finally:
        run_fresh_pipeline.write_json_atomic=previous
