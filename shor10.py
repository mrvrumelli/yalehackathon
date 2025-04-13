import math
import numpy as np
from matplotlib import pyplot as plt
from QuantumRingsLib import (
    QuantumRegister, ClassicalRegister, QuantumCircuit,
    QuantumRingsProvider, job_monitor
)
from fractions import Fraction
import time
from semiprimes import semiprimes

# We still use Qiskit's UnitaryGate here only for constructing our modular multiplication unitary.
# Its to_matrix() method is used to get the underlying NumPy array.
from qiskit.circuit.library import UnitaryGate

provider = QuantumRingsProvider(
    token='rings-200.cIHZ9beKfAeC8xcVhTAj7sUvHMTMxAdm',
    name='mustafa_mert.ozyilmaz@etu.sorbonne-universite.fr'
)
backend = provider.get_backend("scarlet_quantum_rings")
shots = 1024

def iqft_cct(qc, register, n):
    for i in range(n):
        for j in range(i):
            qc.cu1(-math.pi / 2 ** (i - j), register[j], register[i])
        qc.h(register[i])
    qc.barrier()

def continued_fraction(x, max_den=32):
    return Fraction(x).limit_denominator(max_den).denominator

def modular_mult_gate(multiplier, N, n_working):
    """
    Constructs a unitary gate that performs modular multiplication on the working register.
    For each basis state |x> in the working register (of dimension 2^(n_working)):
      - If x < N, maps |x> to |(multiplier * x) mod N>
      - Otherwise, leaves |x> unchanged.
    This defines a permutation (and hence unitary) as long as multiplier and N are coprime.
    Returns a Qiskit UnitaryGate.
    """
    dim = 2 ** n_working
    U = np.zeros((dim, dim), dtype=complex)
    for x in range(dim):
        if x < N:
            y = (multiplier * x) % N
        else:
            y = x
        U[y, x] = 1.0
    return UnitaryGate(U, label=f"Mult_{multiplier}_mod_{N}")

def attempt_factor(N, a):
    n_count = math.ceil(math.log2(N)) * 2
    n_working = math.ceil(math.log2(N))

    counting = QuantumRegister(n_count, 'counting')
    working = QuantumRegister(n_working, 'working')
    classical = ClassicalRegister(n_count, 'classical')

    qc = QuantumCircuit(counting, working, classical)

    # Initialize counting register with Hadamard gates
    qc.h(counting)
    # Set the working register to |1> (representing the number 1)
    qc.x(working[0])
    qc.barrier()

    # For each counting qubit, apply controlled modular multiplication
    for i in range(n_count):
        multiplier = pow(a, 2**i, N)
        
        # Instead of a big unitary, implement multiplication step-by-step
        # Example: Using Qiskit's arithmetic circuits (if available)
        # Alternatively, manually implement multiplication using CNOTs and Toffolis
        for bit in range(n_working):
            if (multiplier >> bit) & 1:  # If the bit is set, apply a controlled-add
                # Apply a controlled-X (CNOT) for each bit
                qc.cx(counting[i], working[bit])
        qc.barrier()

    # Apply the inverse QFT on the counting register
    iqft_cct(qc, counting, n_count)
    qc.measure(counting, classical)

    job = backend.run(qc, shots=shots)
    job_monitor(job)
    result = job.result()
    counts = result.get_counts()

    for state in counts:
        decimal_result = int(state, 2)
        phase = decimal_result / 2**n_count
        r = continued_fraction(phase)
        if r % 2 == 0 and r != 0:
            guess = pow(a, r // 2, N)
            f1, f2 = math.gcd(guess - 1, N), math.gcd(guess + 1, N)
            if f1 > 1 and f1 < N and f2 > 1 and f2 < N:
                print(f"Success: N = {N}, a = {a}, r = {r}, factors = ({f1}, {f2})")
                return True
    print(f"No valid factors found for N = {N} with a = {a}")
    return False



    

def main():
    for bits, N in semiprimes.items():
        print(f"\nFactoring {N} ({bits}-bit)")
        a = 2
        start_time = time.time()
        while a < N:
            try:
                if math.gcd(a, N) != 1:
                    print(f"Trivial factor found by GCD: {math.gcd(a, N)}")
                    break
                success = attempt_factor(N, a)
                if success:
                    break
                a += 1
            except Exception as e:
                print(f"Error for N = {N}, a = {a}: {e}")
                break

if __name__ == "__main__":
    main()



