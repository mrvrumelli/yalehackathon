
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

#add credentials here

provider = QuantumRingsProvider(
    token='',
    name=''
)

backend = provider.get_backend("scarlet_quantum_rings")
shots = 1024

TIMEOUT_SECONDS = 120  # Max time per run

def iqft_cct(qc, register, n):
    for i in range(n):
        for j in range(i):
            qc.cu1(-math.pi / 2 ** (i - j), register[j], register[i])
        qc.h(register[i])
    qc.barrier()

def continued_fraction(x, max_den=32):
    return Fraction(x).limit_denominator(max_den).denominator

def attempt_factor(N, a):
    n_count = math.ceil(math.log2(N)) * 2
    n_working = math.ceil(math.log2(N))

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
