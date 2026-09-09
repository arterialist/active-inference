import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from simulations.drosophila.amin_anatomy import (
    exponential_tree_sum, sample_tree, taper_distance, tree_order,
)


@pytest.mark.parametrize("seed", [11, 23, 44, 77])
def test_all_source_tree_sum_matches_all_pairs_on_branched_trees(seed):
    rng = np.random.default_rng(seed)
    parents = np.r_[-1, [rng.integers(i) for i in range(1, 61)]]
    distances = np.r_[0., rng.uniform(.01, 5, 60)]
    weights = rng.normal(size=(61, 3))
    c = np.arange(1, 61)
    graph = coo_matrix((np.r_[distances[1:], distances[1:]],
        (np.r_[c, parents[1:]], np.r_[parents[1:], c])), shape=(61, 61)).tocsr()
    pair_distances = dijkstra(graph, directed=False)
    for length in (.1, 1., 20.):
        expected = np.exp(-pair_distances/length) @ weights
        np.testing.assert_allclose(exponential_tree_sum(parents, distances, weights, length), expected,
                                   rtol=1e-11, atol=1e-12)


def test_multiple_samples_split_an_edge_without_changing_integrated_distance():
    parents = np.array([-1, 0, 1])
    xyz = np.array([[0., 0., 0.], [10., 0., 0.], [20., 0., 0.]])
    radius = np.array([1., 4., 2.])
    sample_child = np.array([1, 1, 2])
    # Deliberately out of order within the first edge.
    samples = np.array([[7., 0., 0.], [2., 0., 0.], [12., 0., 0.]])
    p, x, r, distance, order, fraction, error = sample_tree(parents, xyz, radius, sample_child, samples)
    assert p.tolist() == [-1, 3, 5, 4, 0, 1]
    assert len(order) == 6
    assert max(error) == 0
    np.testing.assert_allclose(fraction, [.7, .2, .2])
    assert sum(distance[[4, 3, 1]]) == pytest.approx(taper_distance(10, 1, 4), abs=1e-12)
    assert sum(distance[[5, 2]]) == pytest.approx(taper_distance(10, 4, 2), abs=1e-12)
    np.testing.assert_array_equal(x[:3], xyz)
    np.testing.assert_array_equal(r[:3], radius)


def test_sample_not_on_its_declared_edge_is_rejected():
    with pytest.raises(ValueError, match="inside its declared edge"):
        sample_tree(np.array([-1, 0]), np.array([[0., 0., 0.], [10., 0., 0.]]),
                    np.ones(2), np.array([1]), np.array([[5., 1., 0.]]))


@pytest.mark.parametrize("parents", [[-1, -1], [-1, 2, 1], [-1, 4], [0, 0]])
def test_invalid_tree_cannot_be_repaired_by_traversal(parents):
    with pytest.raises(ValueError):
        tree_order(np.array(parents))


def test_taper_formula_has_finite_equal_radius_limit():
    assert taper_distance(6, 4, 4) == 3
    assert taper_distance(6, 4, 4+1e-12) == pytest.approx(3)
    with pytest.raises(ValueError):
        taper_distance(0, 4, 4)


def test_parent_numbering_need_not_precede_child_numbering():
    p = np.array([2, 0, -1, 2])
    d = np.array([.5, 1., 0., 2.])
    w = np.array([0., 1., 0., 0.])
    np.testing.assert_allclose(exponential_tree_sum(p, d, w, 1.), np.exp(-np.array([1., 0., 1.5, 3.5])))
