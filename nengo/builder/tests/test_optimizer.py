import numpy as np
from numpy.testing import assert_almost_equal
import pytest

import nengo
from nengo.builder.optimizer import SigMerger
from nengo.builder.signal import Signal
from nengo import spa


def test_sigmerger_check():
    # 0-d signals
    assert SigMerger.check([Signal(0), Signal(0)])
    assert not SigMerger.check([Signal(0), Signal(1)])

    # compatible along first axis
    assert SigMerger.check(
        [Signal(np.empty((1, 2))), Signal(np.empty((2, 2)))])

    # compatible along second axis
    assert SigMerger.check(
        [Signal(np.empty((2, 1))), Signal(np.empty((2, 2)))], axis=1)
    assert not SigMerger.check(
        [Signal(np.empty((2, 1))), Signal(np.empty((2, 2)))], axis=0)

    # shape mismatch
    assert not SigMerger.check(
        [Signal(np.empty((2,))), Signal(np.empty((2, 2)))])

    # mixed dtype
    assert not SigMerger.check(
        [Signal(np.empty(2, dtype=int)), Signal(np.empty(2, dtype=float))])

    s1 = Signal(np.empty(5))
    s2 = Signal(np.empty(5))

    # mixed signal and view
    assert not SigMerger.check([s1, s1[:3]])

    # mixed bases
    assert not SigMerger.check([s1[:2], s2[2:]])

    # compatible views
    assert SigMerger.check([s1[:2], s1[2:]])


def test_sigmerger_check_signals():
    # 0-d signals
    SigMerger.check_signals([Signal(0), Signal(0)])
    with pytest.raises(ValueError):
        SigMerger.check_signals([Signal(0), Signal(1)])

    # compatible along first axis
    SigMerger.check_signals(
        [Signal(np.empty((1, 2))), Signal(np.empty((2, 2)))])

    # compatible along second axis
    SigMerger.check_signals(
        [Signal(np.empty((2, 1))), Signal(np.empty((2, 2)))], axis=1)
    with pytest.raises(ValueError):
        SigMerger.check_signals(
            [Signal(np.empty((2, 1))), Signal(np.empty((2, 2)))], axis=0)

    # shape mismatch
    with pytest.raises(ValueError):
        SigMerger.check_signals(
            [Signal(np.empty((2,))), Signal(np.empty((2, 2)))])

    # mixed dtype
    with pytest.raises(ValueError):
        SigMerger.check_signals(
            [Signal(np.empty(2, dtype=int)), Signal(np.empty(2, dtype=float))])

    # compatible views
    s = Signal(np.empty(5))
    with pytest.raises(ValueError):
        SigMerger.check_signals([s[:2], s[2:]])


def test_sigmerger_check_views():
    s1 = Signal(np.empty((5, 5)))
    s2 = Signal(np.empty((5, 5)))

    # compatible along first axis
    SigMerger.check_views([s1[:1], s1[1:]])

    # compatible along second axis
    SigMerger.check_views([s1[0:1, :1], s1[0:1, 1:]], axis=1)
    with pytest.raises(ValueError):
        SigMerger.check_views([s1[0:1, :1], s1[0:1, 1:]], axis=0)

    # shape mismatch
    with pytest.raises(ValueError):
        SigMerger.check_views([s1[:1], s1[1:, 0]])

    # different bases
    with pytest.raises(ValueError):
        SigMerger.check_views([s1[:2], s2[2:]])


def test_sigmerger_merge():
    s1 = Signal(np.array([[0, 1], [2, 3]]))
    s2 = Signal(np.array([[4, 5]]))

    sig, replacements = SigMerger.merge([s1, s2])
    assert np.allclose(sig.initial_value, np.array([[0, 1], [2, 3], [4, 5]]))
    assert np.allclose(replacements[s1].initial_value, s1.initial_value)
    assert np.allclose(replacements[s2].initial_value, s2.initial_value)


def test_sigmerger_merge_views():
    s = Signal(np.array([[0, 1], [2, 3], [4, 5]]))
    v1, v2 = s[:2], s[2:]
    merged, _ = SigMerger.merge_views([v1, v2])

    assert np.allclose(merged.initial_value, s.initial_value)
    assert v1.base is s
    assert v2.base is s


def test_optimizer_does_not_change_result(RefSimulator, plt, seed):
    with spa.SPA(seed=seed) as model:
        model.vision = spa.State(dimensions=16, neurons_per_dimension=80)
        model.vision2 = spa.State(dimensions=16, neurons_per_dimension=80)
        model.motor = spa.State(dimensions=16, neurons_per_dimension=80)
        model.motor2 = spa.State(dimensions=32, neurons_per_dimension=80)

        actions = spa.Actions(
            'dot(vision, A) --> motor=A, motor2=vision*vision2',
            'dot(vision, B) --> motor=vision, motor2=vision*A*~B',
            'dot(vision, ~A) --> motor=~vision, motor2=~vision*vision2'
        )
        model.bg = spa.BasalGanglia(actions)
        model.thalamus = spa.Thalamus(model.bg)

        def input_f(t):
            if t < 0.1:
                return 'A'
            elif t < 0.3:
                return 'B'
            elif t < 0.5:
                return '~A'
            else:
                return '0'
        model.input = spa.Input(vision=input_f, vision2='B*~A')

        input, vocab = model.get_module_input('motor')
        input2, vocab2 = model.get_module_input('motor2')
        p = nengo.Probe(input, 'output', synapse=0.03)
        p2 = nengo.Probe(input2, 'output', synapse=0.03)

    with RefSimulator(model, optimize=False) as sim:
        sim.run(0.5)
    with RefSimulator(model, optimize=True) as sim_opt:
        sim_opt.run(0.5)

    t = sim.trange()
    data = vocab.dot(sim.data[p].T)
    data2 = vocab2.dot(sim.data[p2].T)

    plt.subplot(2, 2, 1)
    plt.plot(t, data.T)
    plt.subplot(2, 2, 2)
    plt.plot(t, data2.T)

    data_opt = vocab.dot(sim_opt.data[p].T)
    data_opt2 = vocab2.dot(sim_opt.data[p2].T)

    plt.subplot(2, 2, 3)
    plt.plot(t, data_opt.T)
    plt.subplot(2, 2, 4)
    plt.plot(t, data_opt2.T)

    assert_almost_equal(sim.data[p], sim_opt.data[p])
    assert_almost_equal(sim.data[p2], sim_opt.data[p2])
