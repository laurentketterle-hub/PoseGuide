from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from poseguide import __version__
from poseguide.config import OUT_DIR
from poseguide.data.loader import list_pose_files, list_scene_files, load_pose, load_scene
from poseguide.eval.metrics import evaluate_scenes
from poseguide.guide.demo import PRESETS, run_demo
from poseguide.guide.recommend import recommend_for_scene_path, recommend_for_tags
from poseguide.guide.score import score_subject_against_pose
from poseguide.models.catalog import get_pose_by_id
from poseguide.render.overlay import (
    VisionUnavailableError,
    render_overlay_png,
    write_guidance_overlay,
)
from poseguide.render.svg import render_pose_svg
from poseguide.train.toy_train import train_toy

app = typer.Typer(
    help="PoseGuide 鈥?photography pose guidance (scene 鈫?standing pose coach).",
    no_args_is_help=True,
)
poses_app = typer.Typer(help="Standing pose catalog")
scenes_app = typer.Typer(help="Background / scene samples")
guide_app = typer.Typer(help="Recommend and score poses")
train_app = typer.Typer(help="Training")
eval_app = typer.Typer(help="Evaluation")
data_app = typer.Typer(help="Data utilities (import/extract)")
app.add_typer(poses_app, name="poses")
app.add_typer(scenes_app, name="scenes")
app.add_typer(guide_app, name="guide")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(data_app, name="data")
console = Console()
POSE_DIFFICULTIES = ("easy", "medium", "hard")


def _normalize_difficulty(value: str | None) -> str | None:
    if value is None:
        return None
    difficulty = value.strip().lower()
    if difficulty not in POSE_DIFFICULTIES:
        console.print(f"[red]--difficulty must be one of: {', '.join(POSE_DIFFICULTIES)}[/red]")
        raise typer.Exit(2)
    return difficulty


def _matches_pose_filters(pose: dict, *, tag: str | None, difficulty: str | None) -> bool:
    pose_tags = {str(value).strip().lower() for value in (pose.get("tags") or [])}
    pose_difficulty = str(pose.get("difficulty") or "medium").strip().lower()
    return (tag is None or tag in pose_tags) and (
        difficulty is None or difficulty == pose_difficulty
    )


@app.command("version")
def version_cmd() -> None:
    console.print(f"PoseGuide {__version__}")
    console.print(f"Poses: {len(list_pose_files())} | Scenes: {len(list_scene_files())}")


@app.command("stats")
def stats_cmd() -> None:
    """Catalog inventory: pose/scene counts and top tags."""
    from collections import Counter

    tags: Counter[str] = Counter()
    standing = 0
    for path in list_pose_files():
        pose = load_pose(path)
        if pose.get("standing"):
            standing += 1
        for t in pose.get("tags") or []:
            tags[str(t).lower()] += 1
    console.print_json(
        data={
            "version": __version__,
            "poses": len(list_pose_files()),
            "scenes": len(list_scene_files()),
            "standing_poses": standing,
            "top_tags": tags.most_common(10),
        }
    )


@app.command("demo")
def demo_cmd(preset: str = typer.Option("beach", "--preset", "-p")) -> None:
    """End-to-end demo: preset scene tags 鈫?pose recommendations + SVG stick figure."""
    try:
        result = run_demo(preset)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print(f"Presets: {', '.join(PRESETS)}")
        raise typer.Exit(1) from exc
    console.print_json(data=result)
    console.print(f"[green]SVG[/green] {result.get('svg_path')}")


@poses_app.command("list")
def poses_list(
    tag: str | None = typer.Option(None, "--tag", "-t", help="Filter by an exact tag."),
    difficulty: str | None = typer.Option(
        None, "--difficulty", "-d", help="Filter by difficulty: easy, medium, hard."
    ),
) -> None:
    tag = tag.strip().lower() if tag and tag.strip() else None
    difficulty = _normalize_difficulty(difficulty)
    poses = [
        pose
        for path in list_pose_files()
        if _matches_pose_filters(pose := load_pose(path), tag=tag, difficulty=difficulty)
    ]
    if not poses:
        console.print("[yellow]No poses match the selected filters[/yellow]")
        raise typer.Exit()
    table = Table(title=f"Standing poses ({len(poses)})")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Tags")
    for pose in poses:
        tags = ", ".join(pose.get("tags") or [])
        table.add_row(
            str(pose.get("id")),
            str(pose.get("name")),
            tags,
        )
    console.print(table)


@poses_app.command("show")
def poses_show(
    pose_id: str = typer.Argument(..., help="Pose ID to display"),
) -> None:
    """Show full details of a pose by its ID with pretty-printed JSON."""
    pose = get_pose_by_id(pose_id)
    if pose is None:
        console.print(f"[red]Pose {pose_id!r} not found[/red]")
        raise typer.Exit(1)
    console.print_json(
        data={
            "id": pose.get("id"),
            "name": pose.get("name"),
            "standing": pose.get("standing"),
            "difficulty": pose.get("difficulty"),
            "tags": pose.get("tags", []),
            "tips": pose.get("tips", []),
            "camera_cues": pose.get("camera_cues", []),
            "joints": pose.get("joints", {}),
        }
    )


@poses_app.command("svg")
def poses_svg(
    pose: str = typer.Option(..., "--pose", "-p"),
    out: Path | None = typer.Option(None, "--out", "-o"),
) -> None:
    out_path = out or (OUT_DIR / f"{pose}.svg")
    try:
        path = render_pose_svg(pose, out_path)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Wrote[/green] {path}")


@poses_app.command("overlay")
def poses_overlay(
    pose: str = typer.Option(..., "--pose", "-p"),
    out: Path | None = typer.Option(None, "--out", "-o"),
    background: Path | None = typer.Option(None, "--bg", exists=True, dir_okay=False),
    width: int = typer.Option(360, "--width", min=64, max=4096),
    height: int = typer.Option(480, "--height", min=64, max=4096),
) -> None:
    """Render a PNG skeleton overlay for a target pose (needs vision extra)."""
    out_path = out or (OUT_DIR / f"{pose}_overlay.png")
    try:
        path = render_overlay_png(pose, out_path, background=background, width=width, height=height)
    except VisionUnavailableError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Wrote[/green] {path}")


@scenes_app.command("list")
def scenes_list() -> None:
    files = list_scene_files()
    if not files:
        console.print("[yellow]No scenes in data/scenes[/yellow]")
        raise typer.Exit()
    table = Table(title=f"Scenes ({len(files)})")
    table.add_column("ID")
    table.add_column("Tags")
    table.add_column("Expected poses")
    for path in files:
        scene = load_scene(path)
        tags = ", ".join(scene.get("tags") or [])
        expected = ", ".join(scene.get("expected_poses") or [])
        table.add_row(str(scene.get("id")), tags, expected)
    console.print(table)


@guide_app.command("recommend")
def guide_recommend(
    scene: Path | None = typer.Option(None, "--scene", "-s", exists=True, dir_okay=False),
    tags: str | None = typer.Option(None, "--tags", "-t"),
    top: int = typer.Option(3, "--top", "-k", min=1, max=20),
    subject: Path | None = typer.Option(None, "--subject", exists=True, dir_okay=False),
    overlay_out: Path | None = typer.Option(None, "--overlay-out"),
    svg: bool = typer.Option(True, "--svg/--no-svg"),
    difficulty: str | None = typer.Option(
        None, "--difficulty", "-d", help="Filter by difficulty: easy, medium, hard"
    ),
) -> None:
    if scene is None and not tags:
        console.print("[red]Provide --scene or --tags[/red]")
        raise typer.Exit(code=1)
    if scene is not None:
        result = recommend_for_scene_path(scene, top_k=top, subject_path=subject)
    else:
        result = recommend_for_tags(tags or "", top_k=top)
    if difficulty:
        recs = result.get("recommendations", [])
        before = len(recs)
        recs = [r for r in recs if r.get("difficulty", "medium") == difficulty]
        result["recommendations"] = recs
        console.print(f"[dim]difficulty={difficulty}[/dim] {before} -> {len(recs)} recommendations")
    console.print_json(data=result)
    out = overlay_out or (OUT_DIR / "last_overlay.json")
    path = write_guidance_overlay(result, out)
    console.print(f"[dim]overlay[/dim] {path}")
    if svg and result.get("recommendations"):
        pose_id = str(result["recommendations"][0]["pose_id"])
        svg_path = render_pose_svg(pose_id, OUT_DIR / f"{pose_id}.svg")
        console.print(f"[dim]svg[/dim] {svg_path}")


@guide_app.command("score")
def guide_score(
    pose: str = typer.Option(..., "--pose", "-p"),
    subject: Path = typer.Option(..., "--subject", "-i", exists=True, dir_okay=False),
) -> None:
    try:
        result = score_subject_against_pose(pose, subject)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print_json(data=result)


@guide_app.command("composition")
def guide_composition(pose: str = typer.Option(..., "--pose", "-p")) -> None:
    """Rule-of-thirds composition analysis for a catalog pose."""
    from poseguide.guide.composition import composition_report

    try:
        console.print_json(data=composition_report(pose))
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@guide_app.command("coach")
def guide_coach(
    pose: str = typer.Option(..., "--pose", "-p"),
    subject: Path | None = typer.Option(None, "--subject", "-i", exists=True, dir_okay=False),
) -> None:
    """Coach mode: composition tips + target SVG (+ optional subject score)."""
    from poseguide.guide.composition import coach_bundle

    try:
        console.print_json(data=coach_bundle(pose, subject_path=subject))
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@guide_app.command("demo")
def guide_demo(preset: str = typer.Option("beach", "--preset", "-p")) -> None:
    try:
        result = run_demo(preset)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print_json(data=result)


@eval_app.command("scenes")
def eval_scenes(
    top: int = typer.Option(3, "--top", "-k", min=1, max=20),
    table: bool = typer.Option(True, "--table/--json", help="Rich per-scene table vs raw JSON"),
    markdown: Path | None = typer.Option(
        None, "--md", "--markdown", help="Export results as Markdown file"
    ),
) -> None:
    """Evaluate hit@k / precision / recall over labeled scenes."""
    import json

    from poseguide.config import RUNS_DIR

    report = evaluate_scenes(top_k=top)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / "eval_scenes.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    console.print(
        f"[green]hit@{top}[/green]={report['hit_at_k']} MRR={report.get('mrr', 'N/A')} "
        f"P@{top}={report.get('precision_at_k')} "
        f"R@{top}={report.get('recall_at_k')} "
        f"n={report['n_labeled']}/{report['n_scenes']}"
    )
    if table and report.get("rows"):
        t = Table(title=f"hit@{top} per scene")
        t.add_column("Scene")
        t.add_column("Hit")
        t.add_column("Top poses")
        t.add_column("Overlap")
        for row in report["rows"]:
            t.add_row(
                str(row.get("scene")),
                "yes" if row.get("hit") else "no",
                ", ".join(str(x) for x in (row.get("top") or [])[:top]),
                ", ".join(str(x) for x in (row.get("overlap") or [])),
            )
        console.print(t)
    else:
        console.print_json(data=report)
    if markdown:
        md_content = _build_markdown_report(report, top)
        markdown.write_text(md_content, encoding="utf-8")
        console.print(f"[green]Markdown report[/green] -> {markdown}")
    console.print(f"Report: {path}")


def _build_markdown_report(report: dict, top: int) -> str:
    """Build a Markdown evaluation report from scene results."""
    lines = [
        "# PoseGuide Evaluation Report",
        "",
        f"**hit@{top}**: {report.get('hit_at_k', 'N/A')}  ",
        f"**Precision@{top}**: {report.get('precision_at_k', 'N/A')}  ",
        f"**Recall@{top}**: {report.get('recall_at_k', 'N/A')}  ",
        f"**Labeled scenes**: {report.get('n_labeled', 0)} / {report.get('n_scenes', 0)}",
        "",
        "## Per-Scene Results",
        "",
        "| Scene | Hit | Top Poses | Overlap |",
        "|-------|-----|-----------|---------|",
    ]
    for row in report.get("rows", []):
        scene = str(row.get("scene", "?"))
        hit = "yes" if row.get("hit") else "no"
        top_poses = ", ".join(str(x) for x in (row.get("top") or [])[:top])
        overlap = ", ".join(str(x) for x in (row.get("overlap") or []))
        lines.append(f"| {scene} | {hit} | {top_poses} | {overlap} |")
    return "\n".join(lines) + "\n"


@poses_app.command("search")
def poses_search(
    query: str | None = typer.Argument(
        None, help="Optional substring over id/name/tags/tips/camera cues"
    ),
    tag: str | None = typer.Option(None, "--tag", "-t", help="Filter by an exact tag."),
    difficulty: str | None = typer.Option(
        None, "--difficulty", "-d", help="Filter by difficulty: easy, medium, hard."
    ),
    limit: int = typer.Option(15, "--limit", "-n", min=1, max=50),
) -> None:
    """Search standing pose catalog by id, name, tags, tips, or camera cues."""
    q = (query or "").strip().lower()
    tag = tag.strip().lower() if tag and tag.strip() else None
    difficulty = _normalize_difficulty(difficulty)
    if not q and tag is None and difficulty is None:
        console.print("[red]Provide a query, --tag, or --difficulty[/red]")
        raise typer.Exit(2)
    hits = []
    for path in list_pose_files():
        pose = load_pose(path)
        if not _matches_pose_filters(pose, tag=tag, difficulty=difficulty):
            continue
        searchable = {
            "id": [str(pose.get("id") or "")],
            "name": [str(pose.get("name") or "")],
            "tags": [str(tag) for tag in (pose.get("tags") or [])],
            "tips": [str(tip) for tip in (pose.get("tips") or [])],
            "camera_cues": [str(cue) for cue in (pose.get("camera_cues") or [])],
        }
        matched_fields = [
            field
            for field, values in searchable.items()
            if q and any(q in value.lower() for value in values)
        ]
        if q and not matched_fields:
            continue
        hits.append((pose, matched_fields or ["filters"]))
        if len(hits) >= limit:
            break
    table = Table(title=f"Pose search: {query or 'filters'} ({len(hits)})")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Tags")
    table.add_column("Matched")
    for pose, matched_fields in hits:
        table.add_row(
            str(pose.get("id")),
            str(pose.get("name")),
            ", ".join((pose.get("tags") or [])[:6]),
            ", ".join(matched_fields),
        )
    console.print(table)


@train_app.command("toy")
def train_toy_cmd(epochs: int = typer.Option(3, "--epochs", "-e", min=1, max=50)) -> None:
    report = train_toy(epochs=epochs)
    last = report["history"][-1]["hit_rate_at_3"]
    console.print(f"[green]Training complete[/green] hit@3={last}")
    console.print(f"Report: {report['report_path']}")


@app.command("e2e")
def e2e_cmd(
    tags: str = typer.Option(..., "--tags", "-t", help="Comma-separated scene tags or preset name"),
    image: Path | None = typer.Option(None, "--image", "-i", exists=True, dir_okay=False),
    top: int = typer.Option(3, "--top", "-k", min=1, max=20),
    subject: Path | None = typer.Option(None, "--subject", exists=True, dir_okay=False),
    png: bool = typer.Option(True, "--png/--no-png", help="Render PNG overlay"),
) -> None:
    """End-to-end product path: scene tags → pose list → coach → overlay."""
    from poseguide.guide.e2e import run_e2e

    try:
        summary = run_e2e(
            tags,
            image=image,
            top_k=top,
            subject_json=subject,
            render_png=png,
        )
        console.print_json(data=summary)
        console.print(f"[green]Done[/green] {summary['run_dir']}")
    except (RuntimeError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@data_app.command("extract")
def data_extract(
    image: Path = typer.Option(..., "--image", "-i", exists=True, dir_okay=False),
    out: Path = typer.Option(..., "--out", "-o"),
) -> None:
    """Extract joints from a photo into a PoseGuide subject JSON (MediaPipe)."""
    from poseguide.data.extract import extract_to_file

    try:
        path = extract_to_file(image, out)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Wrote[/green] {path}")


@train_app.command("report")
def train_report() -> None:
    path = Path("data/runs/toy_train_report.json")
    if not path.exists():
        console.print("[yellow]No report yet. Run: poseguide train toy[/yellow]")
        raise typer.Exit(code=1)
    console.print(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Export commands (#22)
# ---------------------------------------------------------------------------
export_app = typer.Typer(help="Export poses to COCO / MediaPipe formats")
app.add_typer(export_app, name="export")


@export_app.command("poses")
def export_poses(
    fmt: str = typer.Option("mediapipe", "--format", "-f", help="coco or mediapipe"),
    out_dir: Path | None = typer.Option(None, "--out-dir", "-o"),
) -> None:
    """Export all poses to COCO or MediaPipe keypoint format."""
    from poseguide.data.export_format import export_all_poses

    out = export_all_poses(fmt=fmt.lower(), out_dir=out_dir)
    console.print(f"[green]Exported[/green] {len(list(out.glob('*.json')))} files -> {out}")


# ---------------------------------------------------------------------------
# Render batch SVG (#30)
# ---------------------------------------------------------------------------
render_app = typer.Typer(help="Batch render commands")
app.add_typer(render_app, name="render")


@render_app.command("batch-svg")
def render_batch_svg(
    out_dir: Path | None = typer.Option(None, "--out-dir", "-o"),
    limit: int = typer.Option(0, "--limit", "-n", min=0, help="Max poses to render (0=all)"),
) -> None:
    """Render all poses to SVG stick figures in out/ directory."""
    from poseguide.data.loader import list_pose_files
    from poseguide.render.svg import render_pose_svg

    out = out_dir or (OUT_DIR / "svg_batch")
    out.mkdir(parents=True, exist_ok=True)
    files = list_pose_files()
    if limit > 0:
        files = files[:limit]
    count = 0
    for path in files:
        pose_id = path.stem
        try:
            render_pose_svg(pose_id, out / f"{pose_id}.svg")
            count += 1
        except KeyError:
            pass
    console.print(f"[green]Rendered[/green] {count} SVGs -> {out}")


# ---------------------------------------------------------------------------
# Scene tagger (#6)
# ---------------------------------------------------------------------------
tagger_app = typer.Typer(help="Scene tagger from description or preset")
app.add_typer(tagger_app, name="tagger")


@tagger_app.command("tags")
def tagger_tags(
    description: str | None = typer.Option(None, "--desc", "-d"),
    preset: str | None = typer.Option(None, "--preset", "-p"),
) -> None:
    """Produce scene tags from a description or preset name."""
    from poseguide.guide.scene_tagger import tag_scene

    tags = tag_scene(description=description, preset=preset)
    console.print_json(data={"tags": tags, "count": len(tags)})


# ---------------------------------------------------------------------------
# Train embed (#8)
# ---------------------------------------------------------------------------
@train_app.command("embed")
def train_embed(
    epochs: int = typer.Option(5, "--epochs", "-e", min=1, max=50),
    save: bool = typer.Option(True, "--save/--no-save"),
) -> None:
    """Train embedding-based ranker on labeled scenes."""
    from poseguide.config import RUNS_DIR
    from poseguide.data.loader import list_scene_files, load_scene
    from poseguide.models.embed import EmbedPoseRanker

    scenes = [load_scene(p) for p in list_scene_files()]
    labeled = [s for s in scenes if s.get("expected_poses")]
    ranker = EmbedPoseRanker()
    history = ranker.train(labeled, epochs=epochs)
    console.print(f"[green]Trained embed ranker[/green] on {len(labeled)} labeled scenes")
    for h in history:
        console.print(f"  epoch {h['epoch']}: loss={h['loss']}")
    if save:
        path = RUNS_DIR / "embed_ranker.json"
        ranker.save(path)
        console.print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Datasets list (#20)
# ---------------------------------------------------------------------------
@data_app.command("datasets")
def data_datasets(
    domain: str | None = typer.Option(None, "--domain", "-d", help="Filter by domain"),
) -> None:
    """List public pose/photography datasets."""
    from rich.table import Table

    from poseguide.data.datasets import list_datasets

    ds_list = list_datasets(domain=domain)
    table = Table(title=f"Public datasets ({len(ds_list)})")
    table.add_column("Name")
    table.add_column("Domain")
    table.add_column("License")
    table.add_column("Keypoints")
    for d in ds_list:
        table.add_row(d["name"], d["domain"], d["license"], d["keypoints"])
    console.print(table)


if __name__ == "__main__":
    app()
