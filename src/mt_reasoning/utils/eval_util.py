from comet import download_model, load_from_checkpoint
from typing import List, Dict

def init_comet_model(
    model_name: str = "Unbabel/wmt22-comet-da",
    gpus: int = 0
):
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    return model

def evaluate_with_comet_ref(
    model,                     
    src: List[str],
    mt: List[str],
    ref: List[str],
    batch_size: int = 16,
    gpus: int = 0,
    num_workers: int = 0,
    progress_bar: bool = False
) -> Dict:
    assert len(src) == len(mt) == len(ref), "src、mt、ref must have the same length."

    data = [{"src": s, "mt": t, "ref": r} for s, t, r in zip(src, mt, ref)]

    output = model.predict(
        data,
        batch_size=batch_size,
        gpus=gpus,              
        num_workers=num_workers,
        progress_bar=progress_bar
    )
    seg_scores = list(output.scores)
    sys_score = float(output.system_score)
    return {"segment_scores": seg_scores, "system_score": sys_score}

