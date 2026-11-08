from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS


# ============================================================
# TTL FILE PATH
# ============================================================
TTL_FILE = Path(
    "/workspaces/proj/sinergym/data/buildings/"
    "5ZoneAutoDXVAVnw_brick.ttl"
)


# ============================================================
# RDF / BRICK NAMESPACES USED IN THE TTL
# ============================================================
BRICK = Namespace("https://brickschema.org/schema/Brick#")
REF = Namespace("https://brickschema.org/schema/Brick/ref#")
EXT = Namespace("urn:sinergym-brick-ext#")


# ============================================================
# OPTIONAL RUNTIME DATA
# ============================================================
# The TTL contains timeseries references, but not the live values themselves.
# During each Sinergym step, populate this dictionary using the reference name.
#
# Example:
# RUNTIME_DATA = {
#     "sinergym:outdoor_temperature": {
#         "current": 8.4,
#         "history": [7.8, 7.9, 8.0, 8.2, 8.4],
#     },
#     "sinergym:air_temperature_space1": {
#         "current": 22.81,
#         "history": [22.42, 22.53, 22.64, 22.73, 22.81],
#     },
#     "sinergym:htg_setpoint_space1": {
#         "current": 20.0,
#         "history": [20.0, 20.0],
#     },
# }
#
# Leave it empty to test only the static graph retrieval.
RUNTIME_DATA: dict[str, dict[str, Any]] = {}


# ============================================================
# BASIC RDF HELPERS
# ============================================================
def local_name(value: URIRef | BNode | Literal | None) -> str:
    """Return the final readable part of a URI or literal."""
    if value is None:
        return "not available"
    if isinstance(value, Literal):
        return str(value)
    if isinstance(value, BNode):
        return "blank_node"

    text = str(value)
    if "#" in text:
        return text.rsplit("#", 1)[-1]
    return text.rstrip("/").rsplit("/", 1)[-1]


def humanize(value: URIRef | BNode | Literal | None) -> str:
    """Convert URI local names such as HVAC_Zone into HVAC Zone."""
    return local_name(value).replace("_", " ")


def label(graph: Graph, node: URIRef | BNode | Literal | None) -> str:
    """Use rdfs:label when available, otherwise use the URI local name."""
    if node is None:
        return "not available"
    if isinstance(node, Literal):
        return str(node)

    value = graph.value(node, RDFS.label)
    return str(value) if value is not None else humanize(node)


def rdf_types(graph: Graph, node: URIRef | BNode) -> list[str]:
    """Return readable RDF types for one entity."""
    return sorted(humanize(value) for value in graph.objects(node, RDF.type))


def sorted_nodes(
    graph: Graph,
    nodes: Iterable[URIRef | BNode],
) -> list[URIRef | BNode]:
    """Remove duplicates and sort entities by human-readable label."""
    return sorted(set(nodes), key=lambda node: label(graph, node).lower())


def relationship_line(
    graph: Graph,
    relation: str,
    nodes: Iterable[URIRef | BNode],
    indent: int = 2,
) -> str | None:
    """
    Group repeated relations onto one line.

    Example:
      hasPart -> Cooling Coil, Heating Coil, Supply Fan
    """
    nodes = sorted_nodes(graph, nodes)
    if not nodes:
        return None

    values = ", ".join(label(graph, node) for node in nodes)
    return f"{' ' * indent}{relation} -> {values}"


def subjects_with_type(
    graph: Graph,
    predicate: URIRef,
    obj: URIRef | BNode,
    rdf_type: URIRef,
) -> list[URIRef | BNode]:
    """Follow an incoming relationship and retain subjects of one RDF type."""
    return sorted_nodes(
        graph,
        (
            subject
            for subject in graph.subjects(predicate, obj)
            if isinstance(subject, (URIRef, BNode))
            and (subject, RDF.type, rdf_type) in graph
        ),
    )


# ============================================================
# STRUCTURED BRICK PROPERTY HELPERS
# ============================================================
def structured_property(
    graph: Graph,
    entity: URIRef | BNode,
    predicate: URIRef,
) -> str | None:
    """
    Resolve a Brick EntityPropertyValue blank node.

    Example:
      zone brick:volume [
          brick:value 239.2474 ;
          brick:hasUnit unit:M3
      ] .

    Output:
      239.2474 M3
    """
    property_node = graph.value(entity, predicate)
    if property_node is None:
        return None

    if isinstance(property_node, Literal):
        return str(property_node)

    value = graph.value(property_node, BRICK.value)
    unit = graph.value(property_node, BRICK.hasUnit)

    if value is None:
        return label(graph, property_node)

    unit_text = local_name(unit) if unit is not None else ""
    return f"{value} {unit_text}".strip()


# ============================================================
# POINT AND TIMESERIES HELPERS
# ============================================================
def timeseries_reference(graph: Graph, point: URIRef | BNode) -> str | None:
    """Follow ref:hasExternalReference and return its rdf:value or label."""
    reference_node = graph.value(point, REF.hasExternalReference)
    if reference_node is None:
        return None

    value = graph.value(reference_node, RDF.value)
    if value is not None:
        return str(value)

    return label(graph, reference_node)


def runtime_for(
    reference: str | None,
    runtime_data: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Find live data using either the full reference or its suffix."""
    if reference is None:
        return {}

    if reference in runtime_data:
        return runtime_data[reference]

    short_name = reference.split(":", 1)[-1]
    return runtime_data.get(short_name, {})


def format_history(values: Any, limit: int = 5) -> str:
    """Format only the most recent history values."""
    if values is None:
        return "not supplied"
    if isinstance(values, (str, bytes)):
        return str(values)

    try:
        values = list(values)[-limit:]
    except TypeError:
        return str(values)

    return "[" + ", ".join(str(value) for value in values) + "]"


def point_details(
    graph: Graph,
    point: URIRef | BNode,
    runtime_data: Mapping[str, Mapping[str, Any]],
    indent: int = 4,
    is_setpoint: bool = False,
) -> list[str]:
    """Print one point beneath a grouped hasPoint relationship."""
    pad = " " * indent
    child_pad = " " * (indent + 2)

    types = rdf_types(graph, point)
    point_type = ", ".join(types) if types else "not available"
    reference = timeseries_reference(graph, point)
    runtime = runtime_for(reference, runtime_data)
    current = runtime.get("current", "not supplied")
    history = runtime.get("history")

    lines = [f"{pad}Point: {label(graph, point)}"]
    lines.append(f"{child_pad}type: {point_type}")

    lines.append(f"{child_pad}current: {current}")

    if is_setpoint:
        lines.append(f"{child_pad}history_last_5: {format_history(history)}")
    else:
        lines.append(f"{child_pad}history_last_5: {format_history(history)}")

    if reference is not None:
        lines.append(f"{child_pad}hasExternalReference -> {reference}")

    return lines


def grouped_points(
    graph: Graph,
    points: Iterable[URIRef | BNode],
    runtime_data: Mapping[str, Mapping[str, Any]],
    indent: int = 2,
    is_setpoint: bool = False,
) -> list[str]:
    """Print hasPoint once, then print compact details for every point."""
    points = sorted_nodes(graph, points)
    if not points:
        return []

    relation = relationship_line(graph, "hasPoint", points, indent=indent)
    lines = [relation] if relation is not None else []

    for point in points:
        lines.extend(
            point_details(
                graph,
                point,
                runtime_data,
                indent=indent + 2,
                is_setpoint=is_setpoint,
            )
        )

    return lines


# ============================================================
# RETRIEVAL AND FORMATTING SECTIONS
# ============================================================
def weather_section(
    graph: Graph,
    runtime_data: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Retrieve the site, weather station, and outdoor dynamic points."""
    lines = ["[WEATHER AND OUTDOOR CONDITIONS]", ""]

    for site in sorted_nodes(graph, graph.subjects(RDF.type, BRICK.Site)):
        stations = [
            obj
            for obj in graph.objects(site, BRICK.hasPart)
            if isinstance(obj, (URIRef, BNode))
            and (obj, RDF.type, BRICK.Weather_Station) in graph
        ]

        lines.append(f"Site: {label(graph, site)}")
        relation = relationship_line(graph, "hasPart", stations)
        if relation:
            lines.append(relation)
        lines.append("")

    for station in sorted_nodes(
        graph,
        graph.subjects(RDF.type, BRICK.Weather_Station),
    ):
        lines.append(f"Weather Station: {label(graph, station)}")
        points = [
            point
            for point in graph.objects(station, BRICK.hasPoint)
            if isinstance(point, (URIRef, BNode))
        ]
        lines.extend(grouped_points(graph, points, runtime_data))
        lines.append("")

    return lines


def central_hvac_section(
    graph: Graph,
    runtime_data: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """
    Retrieve the complete shared HVAC topology in one place.

    Damper position points are intentionally NOT expanded here. They are printed
    with the corresponding HVAC zone so all zone-specific dynamic points remain
    together and are not duplicated.
    """
    lines = ["[CENTRAL HVAC STRUCTURE]", ""]

    for air_loop in sorted_nodes(graph, graph.subjects(RDF.type, BRICK.Air_Loop)):
        parts = [
            obj
            for obj in graph.objects(air_loop, BRICK.hasPart)
            if isinstance(obj, (URIRef, BNode))
        ]
        points = [
            obj
            for obj in graph.objects(air_loop, BRICK.hasPoint)
            if isinstance(obj, (URIRef, BNode))
        ]

        ahus = [
            part
            for part in parts
            if (part, RDF.type, BRICK.Air_Handling_Unit) in graph
        ]
        vavs = [
            part
            for part in parts
            if (part, RDF.type, BRICK.Variable_Air_Volume_Box_With_Reheat)
            in graph
        ]

        # Collect the central components and all zone reheat coils. These are
        # used to explain what contributes to the total HVAC electricity point.
        central_cooling_coils: list[URIRef | BNode] = []
        central_heating_coils: list[URIRef | BNode] = []
        supply_fans: list[URIRef | BNode] = []

        for ahu in ahus:
            for component in graph.objects(ahu, BRICK.hasPart):
                if not isinstance(component, (URIRef, BNode)):
                    continue
                if (
                    component,
                    RDF.type,
                    BRICK.Direct_Expansion_Cooling_Coil,
                ) in graph:
                    central_cooling_coils.append(component)
                elif (component, RDF.type, BRICK.Heating_Coil) in graph:
                    central_heating_coils.append(component)
                elif (component, RDF.type, BRICK.Supply_Fan) in graph:
                    supply_fans.append(component)

        zone_reheat_coils: list[URIRef | BNode] = []
        for vav in vavs:
            zone_reheat_coils.extend(
                component
                for component in graph.objects(vav, BRICK.hasPart)
                if isinstance(component, (URIRef, BNode))
                and (component, RDF.type, BRICK.Heating_Coil) in graph
            )

        lines.append(f"Air Loop: {label(graph, air_loop)}")

        air_loop_parts = sorted_nodes(
            graph,
            parts + central_cooling_coils + central_heating_coils + supply_fans,
        )
        relation = relationship_line(graph, "hasPart", air_loop_parts)
        if relation:
            lines.append(relation)

        # Print hasPoint only once. The HVAC energy point is then expanded with
        # its type, runtime value, history, timeseries reference, and a concise
        # explanation of the equipment included in the total.
        point_relation = relationship_line(graph, "hasPoint", points)
        if point_relation:
            lines.append(point_relation)

        combined_energy_components = (
            sorted_nodes(graph, central_cooling_coils)
            + sorted_nodes(graph, central_heating_coils)
            + sorted_nodes(graph, supply_fans)
            + sorted_nodes(graph, zone_reheat_coils)
        )

        for point in sorted_nodes(graph, points):
            lines.extend(
                point_details(
                    graph,
                    point,
                    runtime_data,
                    indent=4,
                    is_setpoint=False,
                )
            )

            if (
                point,
                RDF.type,
                BRICK.Electrical_Energy_Usage_Sensor,
            ) in graph and combined_energy_components:
                component_names = ", ".join(
                    label(graph, component)
                    for component in combined_energy_components
                )
                lines.append(
                    "      combinedElectricalEnergyOf -> " + component_names
                )

        lines.append("")

    return lines

def relevant_zone_state_points(
    graph: Graph,
    zone: URIRef | BNode,
    excluded: set[URIRef | BNode],
) -> list[URIRef | BNode]:
    """
    Keep the dynamic points that should be shown with the HVAC zone.

    Damper position is intentionally included here, even though the damper is a
    VAV subcomponent, because the point describes the airflow delivered to this
    specific zone and should sit beside the zone temperature, humidity,
    occupancy, and PMV context.
    """
    allowed_types = {
        BRICK.Zone_Air_Temperature_Sensor,
        BRICK.Zone_Air_Humidity_Sensor,
        BRICK.Occupancy_Sensor,
        BRICK.Damper_Position_Sensor,
        BRICK.Sensor,
    }

    points: list[URIRef | BNode] = []
    for point in graph.objects(zone, BRICK.hasPoint):
        if not isinstance(point, (URIRef, BNode)) or point in excluded:
            continue

        if set(graph.objects(point, RDF.type)).intersection(allowed_types):
            points.append(point)

    return sorted_nodes(graph, points)

def zone_branches_section(
    graph: Graph,
    runtime_data: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """
    Retrieve zone spatial properties, zone-specific dynamic points, and
    thermostat setpoints.

    VAV terminals, dampers, and reheat coils are not printed in this output.
    """
    lines = ["[ZONE SPATIAL AND CONTROL CONTEXT]", ""]

    vavs = sorted_nodes(
        graph,
        graph.subjects(RDF.type, BRICK.Variable_Air_Volume_Box_With_Reheat),
    )

    # Use the VAV hasLocation relationship to identify the served zones and
    # remove duplicates while preserving a predictable label-based order.
    served_zones = sorted_nodes(
        graph,
        (
            zone
            for vav in vavs
            for zone in [graph.value(vav, BRICK.hasLocation)]
            if isinstance(zone, (URIRef, BNode))
        ),
    )

    for zone in served_zones:
        zone_name = label(graph, zone)
        lines.append(f"--- ZONE: {zone_name} ---")
        lines.append("")

        # Find the thermostat through the inverse hasLocation relationship.
        thermostats = subjects_with_type(
            graph,
            BRICK.hasLocation,
            zone,
            BRICK.Thermostat,
        )

        # Setpoint points are printed only under the thermostat, so exclude them
        # from the general zone hasPoint block.
        thermostat_points: set[URIRef | BNode] = set()
        for thermostat in thermostats:
            thermostat_points.update(
                point
                for point in graph.objects(thermostat, BRICK.hasPoint)
                if isinstance(point, (URIRef, BNode))
            )

        lines.append(f"HVAC Zone: {zone_name}")
        lines.append("  type: HVAC Zone")

        for name, predicate in (
            ("grossArea", BRICK.grossArea),
            ("volume", BRICK.volume),
            ("azimuth", BRICK.azimuth),
        ):
            value = structured_property(graph, zone, predicate)
            if value is not None:
                lines.append(f"  {name}: {value}")

        adjacent_zones = [
            obj
            for obj in graph.objects(zone, EXT.adjacentTo)
            if isinstance(obj, (URIRef, BNode))
        ]
        relation = relationship_line(graph, "adjacentTo", adjacent_zones)
        if relation:
            lines.append(relation)

        # This grouped block now includes damper position together with the
        # zone's occupancy, PMV, humidity, and air-temperature points.
        state_points = relevant_zone_state_points(
            graph,
            zone,
            thermostat_points,
        )
        lines.extend(grouped_points(graph, state_points, runtime_data))
        lines.append("")

        # Thermostat and its heating/cooling setpoint points.
        for thermostat in thermostats:
            points = [
                obj
                for obj in graph.objects(thermostat, BRICK.hasPoint)
                if isinstance(obj, (URIRef, BNode))
            ]

            lines.append(f"Thermostat: {label(graph, thermostat)}")
            lines.append("  type: Thermostat")
            lines.append(f"  hasLocation -> {zone_name}")
            lines.extend(
                grouped_points(
                    graph,
                    points,
                    runtime_data,
                    is_setpoint=True,
                )
            )
            lines.append("")

    return lines

def return_air_section(graph: Graph) -> list[str]:
    """Retrieve the return plenum and its explicit feeds relation to the AHU."""
    lines = ["[RETURN AIR PATH]", ""]

    for plenum in sorted_nodes(
        graph,
        graph.subjects(RDF.type, BRICK.Return_Air_Plenum),
    ):
        location = graph.value(plenum, BRICK.hasLocation)
        downstream = [
            obj
            for obj in graph.objects(plenum, BRICK.feeds)
            if isinstance(obj, (URIRef, BNode))
        ]

        lines.append(f"Return Air Plenum: {label(graph, plenum)}")
        lines.append("  type: Return Air Plenum")
        if location is not None:
            lines.append(f"  hasLocation -> {label(graph, location)}")

        relation = relationship_line(graph, "feeds", downstream)
        if relation:
            lines.append(relation)
        lines.append("")

    lines.append(
        "All served HVAC zones -> Return Air Plenum -> AHU "
        "(derived from the air-loop topology stored in the TTL)."
    )
    lines.append("")

    return lines


# ============================================================
# BUILD FINAL LLM-READY CONTEXT
# ============================================================
def build_llm_context(
    graph: Graph,
    runtime_data: Mapping[str, Mapping[str, Any]],
    timestamp: str | None = None,
    simulation_step: int | None = None,
    control_interval_minutes: int | None = None,
) -> str:
    """Build one filtered, linked, non-redundant context string."""
    lines = [
        "=" * 78,
        "BUILDING SPATIO-TEMPORAL CONTROL CONTEXT",
        "=" * 78,
    ]

    if timestamp is not None:
        lines.append(f"timestamp: {timestamp}")
    if simulation_step is not None:
        lines.append(f"simulation_step: {simulation_step}")
    if control_interval_minutes is not None:
        lines.append(f"control_interval: {control_interval_minutes} minutes")

    lines.append("")
    lines.extend(weather_section(graph, runtime_data))
    lines.extend(central_hvac_section(graph, runtime_data))
    lines.extend(zone_branches_section(graph, runtime_data))
    lines.extend(return_air_section(graph))

    return "\n".join(lines).rstrip() + "\n"


# ============================================================
# OPTIONAL SINERGYM ADAPTER
# ============================================================
def runtime_data_from_sinergym(
    current_observation: Mapping[str, Any],
    history_by_variable: Mapping[str, Iterable[Any]],
) -> dict[str, dict[str, Any]]:
    """
    Convert Sinergym variable dictionaries into the format expected above.

    current_observation example:
      {"air_temperature_space1": 22.81, "pmv_space1": -0.18}

    history_by_variable example:
      {"air_temperature_space1": [22.42, 22.53, 22.64, 22.73, 22.81]}
    """
    runtime: dict[str, dict[str, Any]] = {}

    for variable_name, current_value in current_observation.items():
        runtime[f"sinergym:{variable_name}"] = {
            "current": current_value,
            "history": list(history_by_variable.get(variable_name, []))[-5:],
        }

    return runtime


# ============================================================
# MAIN PROGRAM
# ============================================================
def main() -> None:
    """Load the TTL, retrieve the required context, and print it."""
    if not TTL_FILE.exists():
        raise FileNotFoundError(
            f"TTL file was not found:\n{TTL_FILE}\n"
            "Check the file path before running the script."
        )

    graph = Graph()

    print(f"Loading TTL file: {TTL_FILE}")
    graph.parse(TTL_FILE, format="turtle")
    print(f"TTL loaded successfully. Total triples: {len(graph)}\n")

    # Static test:
    runtime_data = RUNTIME_DATA

    # During the Sinergym control loop, use this instead:
    # runtime_data = runtime_data_from_sinergym(
    #     current_observation=observation_dict,
    #     history_by_variable=history_dict,
    # )

    context = build_llm_context(
        graph=graph,
        runtime_data=runtime_data,
        timestamp=None,              # Example: "2026-01-15 10:15"
        simulation_step=None,        # Example: 125
        control_interval_minutes=None,  # Example: 15
    )

    # Print the complete linked context on the terminal.
    print(context)

    # The same context can be inserted directly into the LLM user prompt:
    # user_prompt = f"""
    # {context}
    #
    # Select heating and cooling setpoints for all five zones.
    # """


if __name__ == "__main__":
    main()
