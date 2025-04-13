
import math
import numpy as np
from matplotlib import pyplot as plt
from QuantumRingsLib import (
    QuantumRegister, ClassicalRegister, QuantumCircuit,
    QuantumRingsProvider, job_monitor
)
from fractions import Fraction

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

def plot_histogram(counts, title=""):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.bar(counts.keys(), counts.values())
    plt.xlabel("Output State")
    plt.ylabel("Counts")
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def continued_fraction(x, max_den=32):
    return Fraction(x).limit_denominator(max_den).denominator

def shors_algorithm(N, a):
    n_count = math.ceil(math.log2(N)) * 2
    n_working = math.ceil(math.log2(N))

    counting = QuantumRegister(n_count, 'counting')
    working = QuantumRegister(n_working, 'working')
    classical = ClassicalRegister(n_count, 'classical')

    qc = QuantumCircuit(counting, working, classical)

    qc.h(counting)
    qc.x(working[0])
    qc.barrier()

    # Simplified modular exponentiation for small N and a
    # This block must be replaced with a proper reversible modular exponentiation for large N
    for i in range(n_count):
        exp = pow(a, 2**i, N)
        for j in range(n_working):
            if (exp >> j) & 1:
                qc.cx(counting[i], working[j])
    qc.barrier()

    iqft_cct(qc, counting, n_count)
    qc.measure(counting, classical)

    job = backend.run(qc, shots=shots)
    job_monitor(job)
    result = job.result()
    counts = result.get_counts()
    plot_histogram(counts, title=f"Shor’s Algorithm Results (a={a}, N={N})")

    measured = max(counts, key=counts.get)
    decimal_result = int(measured, 2)
    phase = decimal_result / 2**n_count
    r = continued_fraction(phase)

    print(f"Measured: {measured} → phase ≈ {phase:.3f} → estimated r = {r}")

    if r % 2 == 0:
        guess = pow(a, r // 2, N)
        factors = math.gcd(guess - 1, N), math.gcd(guess + 1, N)
        print(f"Non-trivial factors of {N} may be: {factors}")
    else:
        print("Failed to find valid r or r is odd.")

# Example usage
shors_algorithm(N=15, a=7)
