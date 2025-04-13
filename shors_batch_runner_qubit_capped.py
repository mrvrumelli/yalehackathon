
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


provider = QuantumRingsProvider(
    token='rings-200.cIHZ9beKfAeC8xcVhTAj7sUvHMTMxAdm',
    name='mustafa_mert.ozyilmaz@etu.sorbonne-universite.fr'
)
backend = provider.get_backend("scarlet_quantum_rings")

for i in range(backend.num_qubits):
    try:
        props = backend.qubit_properties(i)
        print(f"Qubit {i}:")
        for attr in dir(props):
            if not attr.startswith("_"):
                try:
                    value = getattr(props, attr)
                    print(f"  {attr}: {value}")
                except Exception as e:
                    print(f"  {attr}: Error - {e}")
    except Exception as e:
        print(f"Qubit {i}: Error retrieving properties - {e}")


shots = 1024

TIMEOUT_SECONDS = 120  # Max time per run
MAX_QUBITS = 200       # Hardware limit

def iqft_cct(qc, register, n):
    for i in range(n):
        for j in range(i):
            qc.cu1(-math.pi / 2 ** (i - j), register[j], register[i])
        qc.h(register[i])
    qc.barrier()

def continued_fraction(x, max_den=32):
    return Fraction(x).limit_denominator(max_den).denominator

def attempt_factor(N, a):
    # Estimate register sizes
    full_bits = math.ceil(math.log2(N))
    n_count = 2 * full_bits
    n_working = full_bits

    # Scale down if over hardware limits
    while n_count + n_working > MAX_QUBITS and n_count > 1:
        n_count -= 1

    counting = QuantumRegister(n_count, 'counting')
    working = QuantumRegister(n_working, 'working')
    classical = ClassicalRegister(n_count, 'classical')

    qc = QuantumCircuit(counting, working, classical)

    qc.h(counting)
    qc.x(working[0])
    qc.barrier()

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
                if time.time() - start_time > TIMEOUT_SECONDS:
                    print(f"Timeout reached for N = {N}")
                    break
                a += 1
            except Exception as e:
                print(f"Error for N = {N}, a = {a}: {e}")
                break

if __name__ == "__main__":
    main()
