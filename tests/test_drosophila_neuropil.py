import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("rtree")
from simulations.drosophila.neuropil import classify, mesh_from_x3d


def test_three_direction_parity_checks_actual_shape_not_just_bbox():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    mask, uncertain = classify(mesh, np.array([[0., 0., 0.], [.9, .9, .9], [2., 0., 0.]]))
    np.testing.assert_array_equal(mask, [True, False, False])
    assert not uncertain.any()


def test_exact_vertex_merge_preserves_every_triangle():
    original = trimesh.creation.box()
    points = original.vertices[original.faces].reshape(-1, 3)
    xml = "<IndexedTriangleSet index='" + " ".join(map(str, range(len(points)))) + "'><Coordinate point='"
    xml += " ".join(map(str, points.ravel())) + "'/></IndexedTriangleSet>"
    mesh = mesh_from_x3d(xml)
    assert len(mesh.faces) == len(original.faces)
    assert mesh.volume == pytest.approx(original.volume)
    np.testing.assert_array_equal(mesh.vertices[mesh.faces], original.vertices[original.faces])


def test_open_mesh_is_not_silently_repaired():
    with pytest.raises(ValueError, match="closed"):
        mesh_from_x3d("<IndexedTriangleSet index='0 1 2'><Coordinate point='0 0 0 1 0 0 0 1 0'/></IndexedTriangleSet>")


def test_malformed_trailing_numbers_are_not_ignored():
    with pytest.raises(ValueError):
        mesh_from_x3d("<IndexedTriangleSet index='0 1 2 trailing'><Coordinate point='0 0 0 1 0 0 0 1 0'/></IndexedTriangleSet>")


def test_invalid_chunk_size_cannot_report_every_node_outside():
    with pytest.raises(ValueError, match="chunk size"):
        classify(trimesh.creation.box(), np.array([[0, 0, 0]]), chunk_size=-1)
