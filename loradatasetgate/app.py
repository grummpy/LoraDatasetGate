"""Local FastAPI UI. Binds to loopback only when started from the CLI."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .analyze import analyze, detect_dataset_faces
from .dataset import IMAGE_EXTS, scan
from .facegate import ExportBlocked, UnknownCrop, decide, export_approved
from .neardup import duplicates_to_csv, move_rejected
from .options import Options
from .report import render_html
from .safety import safe_join

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
ART = ROOT / "art"


class AnalyzeBody(BaseModel):
    path: str
    repeats: int = 1
    reg_per_subject_min: float = 1.0
    reg_per_subject_max: float = 2.0
    phash_threshold: int = 10
    dhash_threshold: int = 8
    target_per_cell: int = 1


class MoveBody(BaseModel):
    path: str
    files: list[str]
    confirm: bool = False


class DecisionBody(BaseModel):
    path: str
    crop_id: str
    decision: str
    crop_box: list[int] | None = None


class ExportBody(BaseModel):
    path: str
    dest: str | None = None


def _options(body: AnalyzeBody) -> Options:
    try:
        return Options.from_dict(body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _dataset(path: str) -> Path:
    folder = Path(path).expanduser()
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail=f"dataset folder not found: {path}")
    return folder.resolve()


def create_app() -> FastAPI:
    app = FastAPI(title="LoraDatasetGate", version=__version__, docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def no_store(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/favicon.ico")
    def favicon():
        return FileResponse(ART / "icon.png")

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "version": __version__,
            "bind": "127.0.0.1",
            "gpu": False,
            "deletes_files": False,
        }

    @app.get("/api/samples")
    def samples():
        folder = ROOT / "samples"
        if not (folder / "subject").is_dir():
            from .samples import generate

            generate(folder)
        return {"path": str(folder.resolve()), "synthetic": True}

    @app.post("/api/analyze")
    def analyze_route(body: AnalyzeBody):
        dataset = _dataset(body.path)
        try:
            return analyze(dataset, _options(body))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/report/html")
    def report_route(body: AnalyzeBody):
        dataset = _dataset(body.path)
        result = analyze(dataset, _options(body))
        return HTMLResponse(render_html(result))

    @app.post("/api/duplicates/csv")
    def csv_route(body: AnalyzeBody):
        dataset = _dataset(body.path)
        result = analyze(dataset, _options(body))
        return Response(
            content=duplicates_to_csv(result["duplicates"]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=keep-drop.csv"},
        )

    @app.post("/api/duplicates/move")
    def move_route(body: MoveBody):
        if not body.confirm:
            raise HTTPException(
                status_code=400,
                detail="confirm must be true. Files are never deleted; they are only renamed into _rejected/.",
            )
        if not body.files:
            raise HTTPException(status_code=400, detail="no files were selected")
        dataset = _dataset(body.path)
        try:
            moved = move_rejected(dataset, body.files)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"moved": moved, "deleted": False}

    @app.post("/api/gate/decision")
    def decision_route(body: DecisionBody):
        dataset = _dataset(body.path)
        items, _errors = scan(dataset)
        faces = detect_dataset_faces(items)
        try:
            return decide(dataset, faces, body.crop_id, body.decision, body.crop_box)
        except UnknownCrop as exc:
            raise HTTPException(status_code=404, detail=f"unknown crop {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/gate/export")
    def export_route(body: ExportBody):
        dataset = _dataset(body.path)
        items, _errors = scan(dataset)
        faces = detect_dataset_faces(items)
        dest = None
        if body.dest:
            dest = Path(body.dest).expanduser()
            if not dest.is_absolute():
                dest = dataset / dest
        try:
            return export_approved(dataset, faces, dest)
        except ExportBlocked as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/media")
    def media(dataset: str, rel: str):
        root = _dataset(dataset)
        try:
            target = safe_join(root, rel)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not target.is_file() or target.suffix.lower() not in IMAGE_EXTS:
            raise HTTPException(status_code=404, detail="image not found")
        return FileResponse(target)

    if ART.is_dir():
        app.mount("/art", StaticFiles(directory=ART), name="art")
    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
