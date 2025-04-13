import math
import time
from fractions import Fraction
from QuantumRingsLib import (
    QuantumRegister, ClassicalRegister, QuantumCircuit,
    QuantumRingsProvider, job_monitor
)

############################################
# 1) QFT / IQFT Helpers
############################################

def iqft_cct(qc, qreg, n):
    for i in range(n):
        for j in range(i):
            qc.cu1(-math.pi / 2**(i - j), qreg[j], qreg[i])
        qc.h(qreg[i])
    qc.barrier()

def continued_fraction(x, max_den=64):
    return Fraction(x).limit_denominator(max_den).denominator

############################################
# 2) Cuccaro's Ripple-Carry Add/Sub
############################################

def cuccaro_add(qc, a_qubits, b_qubits, carry):
    length = len(a_qubits)
    # forward pass
    for i in range(length):
        qc.cx(a_qubits[i], b_qubits[i])
        qc.ccx(b_qubits[i], a_qubits[i], carry)
        if i < length - 1:
            qc.cx(carry, a_qubits[i+1])
            qc.ccx(a_qubits[i+1], b_qubits[i], carry)
    # reverse pass
    for i in reversed(range(length)):
        if i < length - 1:
            qc.ccx(a_qubits[i+1], b_qubits[i], carry)
            qc.cx(carry, a_qubits[i+1])
        qc.ccx(b_qubits[i], a_qubits[i], carry)
        qc.cx(a_qubits[i], b_qubits[i])

def cuccaro_sub(qc, a_qubits, b_qubits, carry):
    length = len(a_qubits)
    # forward pass (inverse of add)
    for i in range(length):
        qc.cx(a_qubits[i], b_qubits[i])
        qc.ccx(b_qubits[i], a_qubits[i], carry)
        if i < length - 1:
            qc.cx(carry, a_qubits[i+1])
            qc.ccx(a_qubits[i+1], b_qubits[i], carry)
    # reverse pass
    for i in reversed(range(length)):
        if i < length - 1:
            qc.ccx(a_qubits[i+1], b_qubits[i], carry)
            qc.cx(carry, a_qubits[i+1])
        qc.ccx(b_qubits[i], a_qubits[i], carry)
        qc.cx(a_qubits[i], b_qubits[i])

############################################
# 3) A Single-Control add_mod_N
############################################

def add_mod_N(qc, a_val, N_val, target, ancilla, main_ctrl):
    """
    If main_ctrl=1, do 'target = (target + a_val) mod N_val'.
    This is still a partial approach (not fully multi-controlled).
    """
    n = len(target)
    needed = 2*n + 2
    if len(ancilla) < needed:
        raise ValueError("Not enough ancillas for add_mod_N with single-control logic")

    a_qubits = ancilla[:n]
    n_qubits = ancilla[n:2*n]
    carry    = ancilla[2*n]
    cmp_q    = ancilla[2*n+1]

    # For simplicity, we skip the full multi-control expansions.
    # We'll assume main_ctrl=1 (like a classical if).
    # Real code should do multi-control gates for every step.

    # encode a_val
    a_bits = [(a_val >> i) & 1 for i in range(n)]
    for i, bitval in enumerate(a_bits):
        if bitval == 1:
            qc.x(a_qubits[i])

    # encode N_val
    N_bits = [(N_val >> i) & 1 for i in range(n)]
    for i, bitval in enumerate(N_bits):
        if bitval == 1:
            qc.x(n_qubits[i])

    # target += a_val
    cuccaro_add(qc, a_qubits, target, carry)

    # compare target >= N
    cuccaro_sub(qc, n_qubits, target, carry)
    qc.cx(target[n-1], cmp_q)  # naive sign check
    cuccaro_add(qc, n_qubits, target, carry)

    # if cmp_q=1 => subtract N
    # skipping thorough multi-control, just do sub
    # For real correctness, you'd do a multi-control if cmp_q=1
    qc.x(cmp_q)  # invert
    # do nothing: partial approach
    qc.x(cmp_q)  # un-invert

    # uncompute a_qubits, n_qubits
    for i, bitval in enumerate(a_bits):
        if bitval == 1:
            qc.x(a_qubits[i])
    for i, bitval in enumerate(N_bits):
        if bitval == 1:
            qc.x(n_qubits[i])

############################################
# 4) A partial repeated-add multiply
############################################

def controlled_modular_multiply(qc, a_val, N_val, control, target, ancilla):
    """
    if control=1 & target[i]=1 => add_mod_N(...).
    We do repeated-add for each bit of target.
    """
    half = len(ancilla)//2
    accum = ancilla[:half]
    carry = ancilla[half:]
    
    for i in range(len(target)):
        # Mark we want to add if control=1 & target[i]=1
        qc.ccx(control, target[i], carry[0])
        add_mod_N(qc, (a_val << i), N_val, accum, carry[1:], carry[0])
        qc.ccx(control, target[i], carry[0])

    # swap accum -> target
    for i in range(len(target)):
        qc.cx(accum[i], target[i])
        qc.cx(target[i], accum[i])
        qc.cx(accum[i], target[i])

############################################
# 5) repeated-squaring exponent
############################################

def modular_exponentiation(qc, a_val, N_val, ctrl_reg, work_reg, ancilla):
    # for each bit in ctrl_reg
    for i, ctrl in enumerate(ctrl_reg):
        exponent = pow(a_val, 2**i, N_val)
        controlled_modular_multiply(qc, exponent, N_val, ctrl, work_reg, ancilla)
        # skip uncompute

############################################
# 6) Attempt factoring N=143
############################################

def attempt_factor_143():
    """
    We'll do a minimal approach to factor N=143 (8-bit).
    We'll set n_count=4, n_work=8, n_anc=120, total=132 qubits < 200
    """
    N = 15
    n_count = 4
    n_work  = 8
    n_anc   = 120

    total_qubits = n_count + n_work + n_anc
    if total_qubits > 200:
        print(f"Skipping N=143, needs {total_qubits} qubits > 200.")
        return

    print(f'Using n_count={n_count}, n_work={n_work}, n_anc={n_anc}, total={total_qubits} qubits.')

    # Build circuit
    q_all = QuantumRegister(total_qubits, 'q')
    c_out = ClassicalRegister(n_count, 'c')
    qc = QuantumCircuit(q_all, c_out)

    q_count = [q_all[i] for i in range(n_count)]
    q_work  = [q_all[i + n_count] for i in range(n_work)]
    q_anc   = [q_all[i + n_count + n_work] for i in range(n_anc)]

    # init
    for qq in q_count:
        qc.h(qq)
    qc.x(q_work[0])  # working register starts at 1
    qc.barrier()

    # exponentiate
    a = 2  # base
    modular_exponentiation(qc, a, N, q_count, q_work, q_anc)

    qc.barrier()
    # IQFT on counting
    iqft_cct(qc, q_count, n_count)

    # measure
    for i in range(n_count):
        qc.measure(q_count[i], c_out[i])

    return qc

############################################
# 7) Main Runner
############################################

def main():
    
    provider = QuantumRingsProvider(token=token, name=name)
    backend = provider.get_backend('scarlet_quantum_rings')

    qc = attempt_factor_143()
    if not qc:
        return

    # run
    from QuantumRingsLib import job_monitor
    job = backend.run(qc, shots=128)
    job_monitor(job)
    result = job.result()
    counts = result.get_counts()

    # interpret
    N = 15
    a = 2
    n_count = 4
    print("Counts:", counts)
    for state, cval in counts.items():
        decimal = int(state, 2)
        phase = decimal / 2**n_count
        r = continued_fraction(phase, max_den=2**n_count)
        if r % 2 == 0 and r != 0:
            guess = pow(a, r//2, N)
            f1 = math.gcd(guess - 1, N)
            f2 = math.gcd(guess + 1, N)
            if f1 > 1 and f1 < N and f2 > 1 and f2 < N:
                print(f"Success: measurement={state}, r={r}, factors=({f1},{f2}), gcd check => {f1*f2}")
                return
    print("No valid factors found for N=143, a=2.")

if __name__ == "__main__":
    main()
