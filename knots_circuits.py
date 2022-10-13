""" """

import numpy as np

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import CXGate
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info.operators import Operator


#
# Knots circuits
#


_KPLUS = np.array([
    [1j, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1j]
])
_KMINUS = np.array([
    [-1j, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1j]
])


def _make_trefoil():
    """
    """
    kplusgate = UnitaryGate(Operator(_KPLUS), label='Kplus')
    circuit = QuantumCircuit(3)
    circuit.append(kplusgate, [0, 1])
    circuit.append(kplusgate, [0, 2])
    circuit.append(kplusgate, [1, 2])
    return circuit


def _make_trefoil_twist():
    """
    """
    kplusgate = UnitaryGate(Operator(_KPLUS), label='Kplus')
    kminusgate = UnitaryGate(Operator(_KMINUS), label='Kminus')
    circuit = QuantumCircuit(4)
    circuit.append(kminusgate, [0, 1])
    circuit.append(kminusgate, [0, 2])
    circuit.append(kminusgate, [1, 2])
    circuit.append(kplusgate, [2, 3])
    return circuit


def _make_arc_trefoil():
    """
    """
    kplusgate = UnitaryGate(Operator(_KPLUS), label='Kplus')
    circuit = QuantumCircuit(2)
    circuit.append(kplusgate, [0, 1])
    return circuit


def _make_arc_trefoil_twist():
    """
    """
    kplusgate = UnitaryGate(Operator(_KPLUS), label='Kplus')
    kminusgate = UnitaryGate(Operator(_KMINUS), label='Kminus')
    circuit = QuantumCircuit(3)
    circuit.append(kplusgate, [0, 1])
    circuit.append(kminusgate, [1, 2])
    return circuit


def _get_unitary_from_circ(circ):
    """ """
    simulator = Aer.get_backend('unitary_simulator')
    result = execute(circ, simulator).result()
    return result.get_unitary(circ)


def _make_controlled_unitary(unitary):
    """ """
    return (
        np.kron(np.diag([1, 0]), np.diag(np.ones(unitary.shape[0])))
        + np.kron(np.diag([0, 1]), unitary)
    )


_KNOT_CIRCS = {
    'trefoil': _make_trefoil(),
    'trefoil-twist': _make_trefoil_twist(),
    'arc-trefoil': _make_arc_trefoil(),
    'arc-trefoil-twist': _make_arc_trefoil_twist(),
}


def get_target_value(knot):
    """
    In this case target value is <+...+|U|+...+> for the knot unitary U
    """
    circ = _KNOT_CIRCS[knot]
    unitary = _get_unitary_from_circ(circ)
    n = int(np.log2(unitary.shape[0]))
    all_plus = 1./np.sqrt(2)**n * np.ones((1, 2**n)).T
    return all_plus.conj().T @ unitary @ all_plus


def make_qiskit_hadamard_test_circuit(part, knot, unrolled=True):
    """
    """
    circ = _KNOT_CIRCS[knot]
    unitary = _get_unitary_from_circ(circ)
    cU = _make_controlled_unitary(unitary)
    n = int(np.log2(cU.shape[0]))

    circ = QuantumCircuit(n)

    # add parameterised component that switches between real and imag part
    if part == 'real':
        real_imag_param = 0
    elif part == 'imag':
        real_imag_param = 1
    else:
        raise ValueError(
            f'Part argument: {part} not recognised. Please choose "real" or "imag".'
        )
    circ.u(np.pi/2., -np.pi/2.*real_imag_param, np.pi/2.*real_imag_param, n-1)

    circ.h(list(range(n-1)))
    # circ.barrier()
    circ.diagonal(np.diag(cU).tolist(), list(range(n)))
    # circ.barrier()
    circ.h(n-1)

    # if unrolled:
    #     pass_ = Unroller(['x', 'rz', 'sx', 'cx'])
    #     pass_manager = PassManager(pass_)
    #     circ = pass_manager.run(circ)

    # this is needed for compatibility with CostWPO and pytket compiler
    circ.barrier()

    return circ


#
# Zero noise extrapolation
#


def multiply_cx(circuit, multiplier):
    """
    A simple implementation of zero-noise extrapolation increases the level of
    noise by some factor `multipler` by adding redundant CNOT's (the main
    source of circuit noise). Copies the circuit so preserves registers,
    parameters and circuit name. Will also match the measurements in the input
    circuit.

    Parameters
    ----------
    circuit : qiskit.QuantumCircuit
        Description
    multiplier : int
        Multiplication factor for CNOT gates

    Returns
    -------
    multiplied_circuit : qiskit circuit
        Copy of input circuit, with CNOTs multiplied by multiplier
    """
    if multiplier == 1:
        return circuit.copy()
    if (not multiplier % 2 == 1) and (not isinstance(multiplier, int)):
        raise ValueError('multiplier must be an odd integer, recieved: '
                         + f'{multiplier}')

    # copy circuit to preserve registers
    multiplied_circuit = circuit.copy()
    # wipe instructions in copy
    multiplied_circuit.data = []

    # iterate over circuit instructions, adding them back to the copy but
    # multiplying CNOT's
    for instuction in circuit.data:
        multiplied_circuit.append(*instuction)
        if isinstance(instuction[0], CXGate):
            for _ in range(multiplier-1):
                multiplied_circuit.append(*instuction)

    return multiplied_circuit
