import math
from fractions import Fraction
from QuantumRingsLib import (
    QuantumRegister, ClassicalRegister, QuantumCircuit,
    QuantumRingsProvider, job_monitor
)

def iqft_cct(qc, qreg, n):
    for i in range(n):
        for j in range(i):
            qc.cu1(-math.pi / 2**(i - j), qreg[j], qreg[i])
        qc.h(qreg[i])
    qc.barrier()

def continued_fraction(x, max_den=64):
    return Fraction(x).limit_denominator(max_den).denominator

def controlled_modular_multiply(qc, a, N, ctrl, target, ancilla):
    n = len(target)
    # Use first n ancillas as accumulator
    accum = ancilla[:n]
    
    # Initialize accumulator to 0
    for q in accum:
        qc.reset(q)
    
    # Perform controlled addition for each bit
    for i in range(n):
        # If control is 1 and target bit is 1, add a*2^i mod N
        temp_ctrl = ancilla[n]
        qc.ccx(ctrl, target[i], temp_ctrl)
        
        # Compute a*2^i mod N
        a_shifted = (a << i) % N
        
        # Add to accumulator (simplified version)
        for j in range(n):
            if (a_shifted >> j) & 1:
                qc.ccx(temp_ctrl, accum[j], target[j])
        
        # Reset temp control
        qc.ccx(ctrl, target[i], temp_ctrl)
    
    # Copy result back to target
    for i in range(n):
        qc.cx(accum[i], target[i])
    
    # Reset accumulator
    for q in accum:
        qc.reset(q)

def modular_exponentiation(qc, a, N, ctrl_reg, work_reg, ancilla):
    for i, ctrl in enumerate(ctrl_reg):
        exponent = pow(a, 2**i, N)
        controlled_modular_multiply(qc, exponent, N, ctrl, work_reg, ancilla)

def attempt_factor_15():
    N = 15
    n_count = 4  # Increased counting bits for better accuracy
    n_work = 4   # Enough to hold N=15 (0-15)
    n_anc = 10   # Ancilla qubits
    
    total_qubits = n_count + n_work + n_anc
    if total_qubits > 200:
        print(f"Needs {total_qubits} qubits > 200.")
        return None

    print(f'Using n_count={n_count}, n_work={n_work}, n_anc={n_anc}, total={total_qubits} qubits.')

    q_all = QuantumRegister(total_qubits, 'q')
    c_out = ClassicalRegister(n_count, 'c')
    qc = QuantumCircuit(q_all, c_out)

    q_count = [q_all[i] for i in range(n_count)]
    q_work = [q_all[i + n_count] for i in range(n_work)]
    q_anc = [q_all[i + n_count + n_work] for i in range(n_anc)]

    # Initialize
    for q in q_count:
        qc.h(q)
    qc.x(q_work[0])  # Initialize work register to |1>
    qc.barrier()

    # Modular exponentiation
    a = 7  # A number coprime with 15
    modular_exponentiation(qc, a, N, q_count, q_work, q_anc)
    qc.barrier()

    # Inverse QFT
    iqft_cct(qc, q_count, n_count)

    # Measure counting register
    for i in range(n_count):
        qc.measure(q_count[i], c_out[i])

    return qc

def main():
    token = 'rings-200.cIHZ9beKfAeC8xcVhTAj7sUvHMTMxAdm'
    name = 'mustafa_mert.ozyilmaz@etu.sorbonne-universite.fr'
    provider = QuantumRingsProvider(token=token, name=name)
    backend = provider.get_backend('scarlet_quantum_rings')

    qc = attempt_factor_15()
    if not qc:
        return

    job = backend.run(qc, shots=128)
    job_monitor(job)
    result = job.result()
    counts = result.get_counts()

    N = 15
    a = 7
    n_count = 4
    print("Measurement results:", counts)
    
    # Find the most frequent measurement
    if counts:
        most_frequent = max(counts, key=lambda k: counts[k])
        decimal = int(most_frequent, 2)
        phase = decimal / 2**n_count
        r = continued_fraction(phase, max_den=2**n_count)
        
        print(f"Most frequent measurement: {most_frequent}")
        print(f"Decimal: {decimal}, Phase: {phase}, Possible period: {r}")
        
        if r != 0:
            guess = pow(a, r//2, N)
            f1 = math.gcd(guess - 1, N)
            f2 = math.gcd(guess + 1, N)
            
            print(f"Potential factors: gcd({a}^({r}//2)-1, {N}) = {f1}")
            print(f"Potential factors: gcd({a}^({r}//2)+1, {N}) = {f2}")
            
            if f1 * f2 == N and f1 != 1 and f2 != 1:
                print(f"Success! Factors found: {f1} and {f2}")
            else:
                print("No valid factors found from this measurement.")
        else:
            print("Couldn't determine period from measurement.")

if __name__ == "__main__":
    main()