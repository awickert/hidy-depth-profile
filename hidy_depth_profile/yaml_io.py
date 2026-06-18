"""
YAML serialisation and deserialisation for ProfileSettings.
"""
from __future__ import annotations

import yaml


def load_yaml(filename: str):
    """Load a ProfileSettings from a YAML file."""
    from .settings import ProfileSettings, _DistParam

    with open(filename, "r") as fh:
        doc = yaml.safe_load(fh)

    s = ProfileSettings()

    site = doc.get("site", {})
    s.site_name = site.get("name", "")
    s.latitude = site.get("latitude", 0.0)
    s.longitude = site.get("longitude", 0.0)
    s.elevation = site.get("elevation", 0.0)
    s.strike = site.get("strike", 0.0)
    s.dip = site.get("dip", 0.0)

    data = doc.get("data", {})
    if data.get("profile"):
        s.profile_file = data["profile"]
    if data.get("shielding"):
        s.shielding_file = data["shielding"]
    if data.get("density"):
        s.density_file = data["density"]

    prod = doc.get("production", {})
    s.isotope = prod.get("isotope", "Be-10")
    s.production_scheme = prod.get("scheme", "stone2000")
    s.reference_rate = prod.get("reference_rate", 4.086)
    if "constant_rate" in prod:
        s.constant_rate = prod["constant_rate"]
    if "lsdn" in prod:
        lsdn = prod["lsdn"]
        s.lsdn_assumed_age_yr = lsdn.get("assumed_age_yr", 15000)
        s.lsdn_iterate = lsdn.get("iterate", False)
    if "collection_year" in prod:
        s.collection_year = prod["collection_year"]
    if "error" in prod:
        err = prod["error"]
        s.production_error = _DistParam(err["mode"], err["parameters"])
    s.half_life_sigma = prod.get("half_life_sigma", 0.0)

    muons = doc.get("muons", {})
    s.muon_fit_depth_m = muons.get("fit_depth_gcm2", 2000) / (2.7 * 100) \
        if "fit_depth_gcm2" in muons else muons.get("fit_depth_m", 2000.0)
    s.muon_percent_error = muons.get("percent_error", 5.0)

    dens = doc.get("density", {})
    s.density_from_file = dens.get("from_file", False)
    if "error" in dens:
        err = dens["error"]
        s.density_error = _DistParam(err["mode"], err["parameters"])

    shld = doc.get("shielding", {})
    if shld:
        s.shielding_from_file = shld.get("from_file", False)
        s.shielding_value = shld.get("value", 1.0)

    mc = doc.get("monte_carlo", {})
    s.mc_n_solutions = mc.get("n_solutions", 1000)
    if "age" in mc:
        s.mc_age = _DistParam(mc["age"]["mode"], mc["age"]["parameters"])
    if "erosion_rate" in mc:
        s.mc_erosion_rate = _DistParam(mc["erosion_rate"]["mode"], mc["erosion_rate"]["parameters"])
    if "inheritance" in mc:
        s.mc_inheritance = _DistParam(mc["inheritance"]["mode"], mc["inheritance"]["parameters"])
    if "neutron_attenuation" in mc:
        s.mc_neutron_attenuation = _DistParam(
            mc["neutron_attenuation"]["mode"], mc["neutron_attenuation"]["parameters"]
        )
    if "total_erosion_threshold" in mc:
        s.mc_total_erosion_threshold = mc["total_erosion_threshold"]
    if "confidence" in mc:
        s.mc_confidence_mode = mc["confidence"]["mode"]
        s.mc_confidence_value = mc["confidence"]["value"]
    s.mc_n_workers = mc.get("n_workers", 4)

    return s


def save_yaml(settings, filename: str):
    """Serialise a ProfileSettings to a YAML file."""
    def _dp(dp):
        return {"mode": dp.mode, "parameters": dp.parameters}

    doc = {
        "site": {
            "name": settings.site_name,
            "latitude": settings.latitude,
            "longitude": settings.longitude,
            "elevation": settings.elevation,
            "strike": settings.strike,
            "dip": settings.dip,
        },
        "data": {
            "profile": settings.profile_file,
            "shielding": settings.shielding_file,
            "density": settings.density_file,
        },
        "production": {
            "isotope": settings.isotope,
            "scheme": settings.production_scheme,
            "reference_rate": settings.reference_rate,
            "constant_rate": settings.constant_rate,
            "lsdn": {
                "assumed_age_yr": settings.lsdn_assumed_age_yr,
                "iterate": settings.lsdn_iterate,
            },
            "collection_year": settings.collection_year,
            "error": _dp(settings.production_error),
            "half_life_sigma": settings.half_life_sigma,
        },
        "muons": {
            "fit_depth_m": settings.muon_fit_depth_m,
            "percent_error": settings.muon_percent_error,
        },
        "density": {
            "from_file": settings.density_from_file,
            "error": _dp(settings.density_error),
        },
        "shielding": {
            "from_file": settings.shielding_from_file,
            "value": settings.shielding_value,
        },
        "monte_carlo": {
            "n_solutions": settings.mc_n_solutions,
            "age": _dp(settings.mc_age),
            "erosion_rate": _dp(settings.mc_erosion_rate),
            "inheritance": _dp(settings.mc_inheritance),
            "neutron_attenuation": _dp(settings.mc_neutron_attenuation),
            "total_erosion_threshold": settings.mc_total_erosion_threshold,
            "confidence": {
                "mode": settings.mc_confidence_mode,
                "value": settings.mc_confidence_value,
            },
            "n_workers": settings.mc_n_workers,
        },
    }

    with open(filename, "w") as fh:
        yaml.dump(doc, fh, default_flow_style=False, allow_unicode=True)
