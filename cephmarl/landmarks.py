"""Landmark names for ISBI 2015 and Aariz, and the mapping between them."""

ISBI_LANDMARKS = [
    "Sella", "Nasion", "Orbitale", "Porion", "Subspinale (A-point)",
    "Supramentale (B-point)", "Pogonion", "Menton", "Gnathion", "Gonion",
    "Lower Incisal Incision", "Upper Incisal Incision", "Upper Lip", "Lower Lip",
    "Subnasale", "Soft Tissue Pogonion", "Posterior Nasal Spine",
    "Anterior Nasal Spine", "Articulare",
]

# Order follows ANATOMICAL_LANDMARKS in the official Aariz repo (config.py).
AARIZ_LANDMARKS = [
    "A-point", "Anterior Nasal Spine", "B-point", "Menton", "Nasion", "Orbitale",
    "Pogonion", "Posterior Nasal Spine", "Pronasale", "Ramus", "Sella",
    "Articulare", "Condylion", "Gnathion", "Gonion", "Porion",
    "Lower 2nd PM Cusp Tip", "Lower Incisor Tip", "Lower Molar Cusp Tip",
    "Upper 2nd PM Cusp Tip", "Upper Incisor Apex", "Upper Incisor Tip",
    "Upper Molar Cusp Tip", "Lower Incisor Apex", "Labrale inferius",
    "Labrale superius", "Soft Tissue Nasion", "Soft Tissue Pogonion", "Subnasale",
]

# ISBI name -> Aariz name. Matched by anatomical name; check definitions
# (e.g. ISBI "Upper Lip" vs Aariz "Labrale superius") before reporting.
ISBI_TO_AARIZ = {
    "Sella": "Sella",
    "Nasion": "Nasion",
    "Orbitale": "Orbitale",
    "Porion": "Porion",
    "Subspinale (A-point)": "A-point",
    "Supramentale (B-point)": "B-point",
    "Pogonion": "Pogonion",
    "Menton": "Menton",
    "Gnathion": "Gnathion",
    "Gonion": "Gonion",
    "Lower Incisal Incision": "Lower Incisor Tip",
    "Upper Incisal Incision": "Upper Incisor Tip",
    "Upper Lip": "Labrale superius",
    "Lower Lip": "Labrale inferius",
    "Subnasale": "Subnasale",
    "Soft Tissue Pogonion": "Soft Tissue Pogonion",
    "Posterior Nasal Spine": "Posterior Nasal Spine",
    "Anterior Nasal Spine": "Anterior Nasal Spine",
    "Articulare": "Articulare",
}


def map_indices(model_names, data_names):
    """For each model landmark, the index of the same landmark in the dataset (or -1)."""
    if list(model_names) == list(data_names):
        return list(range(len(model_names)))
    lookup = {n: i for i, n in enumerate(data_names)}
    out = []
    for n in model_names:
        target = ISBI_TO_AARIZ.get(n, n)
        if target not in lookup:  # also try the reverse direction
            rev = {v: k for k, v in ISBI_TO_AARIZ.items()}
            target = rev.get(n, n)
        out.append(lookup.get(target, -1))
    return out
