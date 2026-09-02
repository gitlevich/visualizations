#!/usr/bin/env python3
"""Generate deterministic semantic positions and colors for taxonomy nodes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "public" / "taxonomies.json"
DESTINATION = ROOT / "public" / "semantic-layout.json"


def text_value(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(text_value(item) for item in value)
    if isinstance(value, dict):
        return " ".join(text_value(item) for item in value.values())
    return ""


def node_document(node: dict, ancestry: tuple[str, ...]) -> str:
    label = node["label"]
    family = node.get("family") or node.get("region") or ""
    definition = node.get("definition") or node.get("core") or ""
    distinction = node.get("distinction") or node.get("contrast") or ""
    children = " ".join(child["label"] for child in node.get("children", []))
    domains = text_value(node.get("domains", []))
    return " ".join(
        [
            label,
            label,
            label,
            label,
            family,
            family,
            definition,
            definition,
            distinction,
            distinction,
            node.get("role", ""),
            " ".join(ancestry),
            children,
            domains,
        ]
    )


def collect_taxonomy(taxonomy_key: str, root: dict):
    nodes: list[tuple[str, str, dict, tuple[str, ...]]] = []
    groups: dict[str, list[str]] = {}

    def walk(node: dict, ancestry: tuple[str, ...] = ()):
        nodes.append((taxonomy_key, node["id"], node, ancestry))
        children = node.get("children", [])
        if children:
            groups[node["id"]] = [child["id"] for child in children]
        for child in children:
            walk(child, ancestry + (node["label"],))

    walk(root)
    return nodes, groups


def orient_and_normalize(coordinates: np.ndarray) -> np.ndarray:
    coordinates = coordinates - coordinates.mean(axis=0, keepdims=True)
    for dimension in range(coordinates.shape[1]):
        column = coordinates[:, dimension]
        anchor = int(np.argmax(np.abs(column)))
        if column[anchor] < 0:
            coordinates[:, dimension] *= -1
    radius = float(np.linalg.norm(coordinates, axis=1).max(initial=0))
    if radius > 0:
        coordinates /= radius
    return coordinates


def relax_collisions(coordinates: np.ndarray) -> np.ndarray:
    count = len(coordinates)
    if count < 2:
        return coordinates
    minimum = 0.9 if count <= 4 else 0.62 if count <= 8 else 0.48 if count <= 14 else 0.4
    base = coordinates.copy()
    result = coordinates.copy()

    for iteration in range(220):
        for first in range(count):
            for second in range(first + 1, count):
                delta = result[second] - result[first]
                distance = float(np.linalg.norm(delta))
                if distance >= minimum:
                    continue
                if distance < 1e-8:
                    seed = (first + 1) * 73856093 ^ (second + 1) * 19349663
                    direction = np.array(
                        [
                            np.sin(seed * 0.000001),
                            np.cos(seed * 0.0000017),
                            np.sin(seed * 0.0000023 + 1.0),
                        ]
                    )
                    direction /= np.linalg.norm(direction)
                else:
                    direction = delta / distance
                shift = direction * (minimum - distance) * 0.5
                result[first] -= shift
                result[second] += shift
        result = result * 0.992 + base * 0.008
        result -= result.mean(axis=0, keepdims=True)
        radius = float(np.linalg.norm(result, axis=1).max(initial=0))
        if radius > 1:
            result /= radius

    return orient_and_normalize(result)


def semantic_coordinates(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, float, float]:
    count = len(vectors)
    if count == 1:
        return np.zeros((1, 3)), None, 1.0, 0.0

    distances = np.clip(pairwise_distances(vectors, metric="cosine"), 0, 2)
    centering = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * centering @ (distances ** 2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    dimensions = min(4, count - 1)
    raw_coordinates = np.zeros((count, dimensions))
    for output_dimension, source_dimension in enumerate(order[:dimensions]):
        value = max(0.0, float(eigenvalues[source_dimension]))
        raw_coordinates[:, output_dimension] = eigenvectors[:, source_dimension] * np.sqrt(value)

    for dimension in range(raw_coordinates.shape[1]):
        column = raw_coordinates[:, dimension]
        anchor = int(np.argmax(np.abs(column)))
        if column[anchor] < 0:
            raw_coordinates[:, dimension] *= -1

    coordinates = np.zeros((count, 3))
    coordinates[:, : min(3, dimensions)] = raw_coordinates[:, :3]
    color_component = None
    if dimensions == 4 and np.max(np.abs(raw_coordinates[:, 3])) > 1e-9:
        color_component = raw_coordinates[:, 3] / np.max(np.abs(raw_coordinates[:, 3]))

    coordinates = relax_collisions(orient_and_normalize(coordinates))
    upper = np.triu_indices(count, 1)
    embedded_distances = pairwise_distances(coordinates, metric="euclidean")[upper]
    source_distances = distances[upper]
    if np.std(source_distances) < 1e-9 or np.std(embedded_distances) < 1e-9:
        correlation = 1.0
    else:
        correlation = float(np.corrcoef(source_distances, embedded_distances)[0, 1])
    mean_distance = float(source_distances.mean())
    return coordinates, color_component, correlation, mean_distance


def main():
    taxonomies = json.loads(SOURCE.read_text())
    records = []
    taxonomy_groups = {}
    for taxonomy_key, root in taxonomies.items():
        nodes, groups = collect_taxonomy(taxonomy_key, root)
        records.extend(nodes)
        taxonomy_groups[taxonomy_key] = groups

    documents = [node_document(node, ancestry) for _, _, node, ancestry in records]
    word_features = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        max_features=18000,
    ).fit_transform(documents)
    character_features = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=12000,
    ).fit_transform(documents)
    features = hstack([word_features, character_features]).tocsr()
    component_count = min(64, features.shape[0] - 1, features.shape[1] - 1)
    vectors = TruncatedSVD(n_components=component_count, random_state=17).fit_transform(features)
    vectors = normalize(vectors)

    cluster_count = 6
    clustering = KMeans(n_clusters=cluster_count, random_state=17, n_init=24).fit(vectors)
    center_order = np.argsort(np.arctan2(clustering.cluster_centers_[:, 2], clustering.cluster_centers_[:, 1]))
    cluster_remap = {int(old): int(new) for new, old in enumerate(center_order)}

    record_index = {
        (taxonomy_key, node_id): index
        for index, (taxonomy_key, node_id, _, _) in enumerate(records)
    }
    output_taxonomies = {}
    for taxonomy_key, groups in taxonomy_groups.items():
        nodes_output = {}
        for key, index in record_index.items():
            if key[0] != taxonomy_key:
                continue
            nodes_output[key[1]] = {
                "cluster": cluster_remap[int(clustering.labels_[index])]
            }

        groups_output = {}
        for parent_id, child_ids in groups.items():
            indices = [record_index[(taxonomy_key, child_id)] for child_id in child_ids]
            coordinates, color_component, correlation, mean_distance = semantic_coordinates(vectors[indices])
            groups_output[parent_id] = {
                "distanceCorrelation": round(correlation, 4),
                "spread": round(float(np.clip(mean_distance / 0.9, 0.38, 1.0)), 4),
                "children": {
                    child_id: [round(float(value), 6) for value in coordinates[index]]
                    + (
                        [round(float(color_component[index]), 6)]
                        if color_component is not None
                        else []
                    )
                    for index, child_id in enumerate(child_ids)
                },
            }

        output_taxonomies[taxonomy_key] = {
            "nodes": nodes_output,
            "groups": groups_output,
        }

    cluster_examples = {}
    for remapped_cluster in range(cluster_count):
        original_cluster = next(
            original for original, remapped in cluster_remap.items() if remapped == remapped_cluster
        )
        member_indices = np.flatnonzero(clustering.labels_ == original_cluster)
        center = clustering.cluster_centers_[original_cluster]
        ranked = sorted(
            member_indices,
            key=lambda index: float(np.linalg.norm(vectors[index] - center)),
        )[:6]
        cluster_examples[str(remapped_cluster)] = [records[index][2]["label"] for index in ranked]

    output = {
        "version": 1,
        "method": "word-and-character TF-IDF, 64-dimensional LSA, cosine-distance classical MDS",
        "colorEncoding": "fourth local MDS component; global semantic-cluster fallback",
        "clusters": cluster_examples,
        "taxonomies": output_taxonomies,
    }
    DESTINATION.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
