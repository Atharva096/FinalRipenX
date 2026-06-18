"""Export destination recommendation (CLI, API, and frontend)."""

DEFAULT_CULTIVAR = "Alphonso"


def recommend_export_destination(predicted_days: int, cultivar: str) -> tuple[str, str]:
    """
    Map days-to-ready and cultivar to an export lane and logistics rationale.
    predicted_days should match the harvest estimate (post RF + sanity when used from API).
    """
    key = (cultivar or "").strip().casefold()

    if predicted_days < 5:
        dest = "Domestic Market / Processing Plant"
        logistics = (
            "Shelf life too short for international shipping; prioritize local sale, "
            "juice, or processing before spoilage."
        )
        return dest, logistics

    if 5 <= predicted_days <= 8:
        if key == "alphonso":
            dest = "UAE (Dubai) via Air Freight"
            logistics = (
                "About one week of remaining shelf life suits short air legs; "
                "Alphonso premium pricing justifies air freight to Gulf hubs."
            )
        else:
            dest = "Middle East (Saudi Arabia / UAE) via Fast Sea Route"
            logistics = (
                "Moderate shelf life (5-8 days) fits expedited sea service to nearby "
                "Middle East ports; monitor cold chain and vessel schedule."
            )
        return dest, logistics

    if key == "alphonso":
        dest = "United Kingdom / USA via Express Air Freight"
        logistics = (
            "Longer remaining shelf life supports long-haul air for high-value Alphonso "
            "in temperature-controlled ULDs with minimal dwell time."
        )
    elif key in ("kesar", "totapuri"):
        dest = "United Kingdom / European Union via Controlled Atmosphere Sea Reefer"
        logistics = (
            "Extended shelf window allows CA reefer sea freight to Europe/UK; "
            "Kesar/Totapuri bulk and processing demand align with reefer economics."
        )
    else:
        dest = "Global Markets (Japan / South Korea) via Approved Phytosanitary Channels"
        logistics = (
            "Sufficient shelf life for phytosanitary-heavy distant markets; "
            "book certified cold chain and documentation for Japan/South Korea entry."
        )
    return dest, logistics


def get_mandatory_regulatory_compliance(destination: str) -> list[str]:
    """
    Regulatory actions keyed off recommended export destination text.
    Returns one action string per matched market (USA, Japan, EU).
    """
    dest_upper = (destination or "").upper()
    actions: list[str] = []

    if "USA" in dest_upper:
        actions.append(
            "Action Required: Post-harvest Gamma Irradiation Treatment Certificate"
        )
    if "JAPAN" in dest_upper:
        actions.append(
            "Action Required: Vapour Heat Treatment (VHT) at 48°C for 30 minutes"
        )
    if "EUROPEAN UNION" in dest_upper:
        actions.append("Action Required: Hot Water Treatment Compliance Check")

    return actions


def format_regulatory_compliance_block(destination: str) -> str:
    """Multi-line block for CLI / API message append."""
    actions = get_mandatory_regulatory_compliance(destination)
    if not actions:
        return ""
    lines = ["Mandatory Regulatory Compliance:"]
    lines.extend(actions)
    return "\n".join(lines)
