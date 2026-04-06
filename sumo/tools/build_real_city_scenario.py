from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import os
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a real-city SUMO scenario from OSM data and generate mixed traffic routes."
        )
    )
    parser.add_argument(
        "--scenario-name",
        default="city",
        help="Scenario prefix for generated files (default: city).",
    )
    parser.add_argument(
        "--bbox",
        default="77.580,12.965,77.640,13.010",
        help="OSM bbox west,south,east,north used when --osm-file is not provided.",
    )
    parser.add_argument(
        "--osm-file",
        default=None,
        help="Existing .osm/.osm.xml(.gz) file path. If set, download is skipped.",
    )
    parser.add_argument("--begin", type=int, default=0, help="Route begin time.")
    parser.add_argument("--end", type=int, default=3600, help="Route end time.")
    parser.add_argument(
        "--passenger-period",
        default="1.2,0.8,1.6",
        help="randomTrips period for passenger demand (comma-separated dynamic periods).",
    )
    parser.add_argument(
        "--freight-period",
        default="6.0",
        help="randomTrips period for freight demand.",
    )
    parser.add_argument("--seed-passenger", type=int, default=101, help="Passenger seed.")
    parser.add_argument("--seed-freight", type=int, default=202, help="Freight seed.")
    parser.add_argument(
        "--netconvert-options",
        default=(
            "--geometry.remove,--ramps.guess,--junctions.join,--tls.guess-signals,"
            "--tls.discard-simple,--tls.guess,--tls.ignore-internal-junction-jam"
        ),
        help="Comma-separated options passed to osmBuild netconvert.",
    )
    parser.add_argument(
        "--skip-shapes",
        action="store_true",
        help="Do not request OSM polygon data and do not generate poly file.",
    )
    return parser.parse_args()


def run_cmd(command: list[str], cwd: Path) -> None:
    print("[CMD]", " ".join(command))
    completed = subprocess.run(command, cwd=str(cwd), text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}")


def locate_sumo_tools() -> Path:
    if os.environ.get("SUMO_HOME"):
        tools = Path(os.environ["SUMO_HOME"]) / "tools"
        if tools.exists():
            return tools

    default = Path("/usr/share/sumo/tools")
    if default.exists():
        return default

    raise FileNotFoundError(
        "Could not locate SUMO tools directory. Set SUMO_HOME or install SUMO tools."
    )


def find_typemap_file() -> Path | None:
    candidates = [
        Path("/usr/share/sumo/data/typemap/osmPolyconvert.typ.xml"),
        Path("/usr/share/sumo/data/typemap/osmPolyconvertUrbanDe.typ.xml"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_builtin_city_net() -> Path | None:
    candidates = [
        Path("/usr/share/sumo/tools/game/DRT/osm.net.xml"),
        Path("/usr/share/sumo/tools/game/A10KW/osm.net.xml"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_osm_input(args: argparse.Namespace, project_root: Path, tools_dir: Path) -> Path:
    if args.osm_file:
        osm_path = Path(args.osm_file).expanduser().resolve()
        if not osm_path.exists():
            raise FileNotFoundError(f"OSM file not found: {osm_path}")
        return osm_path

    raw_dir = project_root / "sumo" / "raw_osm"
    raw_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{args.scenario_name}_src"
    osm_get = tools_dir / "osmGet.py"

    command = [
        "python3",
        str(osm_get),
        "-p",
        prefix,
        "-b",
        args.bbox,
        "-d",
        str(raw_dir),
    ]
    if not args.skip_shapes:
        command.append("-s")

    run_cmd(command, cwd=project_root)

    candidates = sorted(raw_dir.glob(f"{prefix}*.osm.xml*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("OSM download finished but no osm.xml file was found.")
    return candidates[-1]


def write_sumocfg(
    *,
    scenario_name: str,
    sumocfg_path: Path,
    has_poly: bool,
    begin: int,
    end: int,
) -> None:
    additional_line = (
        f'        <additional-files value="../networks/{scenario_name}.poly.xml"/>\n'
        if has_poly
        else ""
    )

    content = (
        "<configuration>\n"
        "    <input>\n"
        f'        <net-file value="../networks/{scenario_name}.net.xml"/>\n'
        f'        <route-files value="../routes/{scenario_name}_passenger.rou.xml,../routes/{scenario_name}_freight.rou.xml"/>\n'
        f"{additional_line}"
        "    </input>\n"
        "    <time>\n"
        f'        <begin value="{begin}"/>\n'
        f'        <end value="{end}"/>\n'
        "    </time>\n"
        "</configuration>\n"
    )
    sumocfg_path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    sumo_root = project_root / "sumo"
    networks_dir = sumo_root / "networks"
    routes_dir = sumo_root / "routes"
    scenarios_dir = sumo_root / "scenarios"

    networks_dir.mkdir(parents=True, exist_ok=True)
    routes_dir.mkdir(parents=True, exist_ok=True)
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    tools_dir = locate_sumo_tools()
    osm_build = tools_dir / "osmBuild.py"
    random_trips = tools_dir / "randomTrips.py"

    osm_input = resolve_osm_input(args, project_root, tools_dir)
    print(f"[INFO] OSM input: {osm_input}")

    typemap = find_typemap_file()
    scenario_name = args.scenario_name

    build_command = [
        "python3",
        str(osm_build),
        "-f",
        str(osm_input),
        "-p",
        scenario_name,
        "-d",
        str(networks_dir),
        "--vehicle-classes",
        "all",
        "--pedestrians",
        f"--netconvert-options={args.netconvert_options}",
    ]
    if typemap is not None and not args.skip_shapes:
        build_command += ["-m", str(typemap)]

    net_file = networks_dir / f"{scenario_name}.net.xml"
    poly_file = networks_dir / f"{scenario_name}.poly.xml"
    try:
        run_cmd(build_command, cwd=project_root)
    except RuntimeError as e:
        builtin = find_builtin_city_net()
        if builtin is None:
            raise RuntimeError(
                "OSM conversion failed and no fallback built-in city network is available."
            ) from e

        print(
            "[WARN] OSM conversion failed. Falling back to built-in SUMO city network:",
            builtin,
        )
        shutil.copy2(builtin, net_file)

    if not net_file.exists():
        raise FileNotFoundError(f"Expected network not generated: {net_file}")

    passenger_route = routes_dir / f"{scenario_name}_passenger.rou.xml"
    freight_route = routes_dir / f"{scenario_name}_freight.rou.xml"

    run_cmd(
        [
            "python3",
            str(random_trips),
            "-n",
            str(net_file),
            "-r",
            str(passenger_route),
            "-b",
            str(args.begin),
            "-e",
            str(args.end),
            "--period",
            args.passenger_period,
            "--seed",
            str(args.seed_passenger),
            "--fringe-factor",
            "10",
            "--prefix",
            "p_",
            "--trip-attributes",
            'departLane="best" departSpeed="max" departPos="random"',
        ],
        cwd=project_root,
    )

    run_cmd(
        [
            "python3",
            str(random_trips),
            "-n",
            str(net_file),
            "-r",
            str(freight_route),
            "-b",
            str(args.begin),
            "-e",
            str(args.end),
            "--period",
            args.freight_period,
            "--seed",
            str(args.seed_freight),
            "--vehicle-class",
            "truck",
            "--prefix",
            "t_",
            "--trip-attributes",
            'departLane="best" departSpeed="max" departPos="random"',
        ],
        cwd=project_root,
    )

    sumocfg_path = scenarios_dir / f"{scenario_name}.sumocfg"
    write_sumocfg(
        scenario_name=scenario_name,
        sumocfg_path=sumocfg_path,
        has_poly=poly_file.exists(),
        begin=args.begin,
        end=args.end,
    )

    # Keep a city defaults file for 3D GUI behavior.
    city_settings = scenarios_dir / "city_3d.settings.xml"
    if not city_settings.exists():
        city_settings.write_text(
            "<viewsettings>\n"
            "    <scheme name=\"real world\"/>\n"
            "    <delay value=\"30\"/>\n"
            "</viewsettings>\n",
            encoding="utf-8",
        )

    print("[DONE] Generated files:")
    print("       ", net_file)
    print("       ", passenger_route)
    print("       ", freight_route)
    print("       ", sumocfg_path)
    if poly_file.exists():
        print("       ", poly_file)


if __name__ == "__main__":
    main()
